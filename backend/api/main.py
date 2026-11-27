"""
============================================================
PREDICCIÓN CLIMÁTICA ML · MICRO-ZONA
Módulo: API REST con FastAPI (main.py)
============================================================
Endpoints:
    GET  /           → Info de la API
    GET  /health     → Estado del servicio
    POST /predict    → Predicción de temperatura y variables
    POST /forecast   → Pronóstico 7 días
    GET  /model/info → Métricas y configuración del modelo

Uso:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

El frontend (index.html) consume este endpoint en:
    fetch('http://localhost:8000/predict', { method: 'POST', ... })
============================================================
"""

from __future__ import annotations

import sys
import time
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List

# Windows (cp1252/GBK) no puede imprimir símbolos como ✓/⚠ → UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

# ─── Ruta del modelo serializado ────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / 'model' / 'clima_model.joblib'

# ============================================================
#  ESQUEMAS Pydantic (validación de entrada/salida)
# ============================================================

class WeatherInput(BaseModel):
    """Variables meteorológicas de entrada para la predicción."""
    temp:  float = Field(..., ge=-10, le=50,   description="Temperatura actual (°C)")
    hum:   float = Field(..., ge=0,   le=100,  description="Humedad relativa (%)")
    # Rango amplio: las estaciones están en altura (~750 hPa)
    pres:  float = Field(..., ge=600, le=1100, description="Presión atmosférica (hPa)")
    # Opcionales (compatibilidad con el esquema sintético anterior):
    wind:  float | None = Field(None, description="Velocidad del viento (km/h) — opcional")
    cloud: float | None = Field(None, description="Nubosidad (%) — opcional")
    uv:    float | None = Field(None, description="Índice UV — opcional")
    # Hora local (0-23). Si no se envía, se usa la hora del servidor.
    hora:  int   | None = Field(None, ge=0, le=23, description="Hora local del día (0-23)")

    @field_validator('hum', 'cloud')
    @classmethod
    def clamp_percentage(cls, v):
        return v if v is None else max(0.0, min(100.0, v))


class PredictionOutput(BaseModel):
    """Resultado de la predicción del modelo GBR Ensemble."""
    pred_temp:   float  = Field(..., description="Temperatura predicha por el modelo (°C)")
    rain_prob:   float  = Field(..., description="Probabilidad de lluvia — heurística auxiliar (0-100)")
    sensacion:   float  = Field(..., description="Sensación térmica — estimación heurística (°C)")
    confidence:  float  = Field(..., description="Confianza estimada — heurística, no calculada por el modelo (%)")
    rmse:        float  = Field(..., description="RMSE real de la validación cruzada del modelo (°C)")
    condition:   str    = Field(..., description="Descripción de la condición climática")
    icon:        str    = Field(..., description="Emoji del estado del tiempo")
    hora_usada:  int    = Field(..., description="Hora del día usada como feature del modelo (0-23)")
    model_version: str  = Field(default="1.0.0")
    timestamp:   str    = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ForecastDay(BaseModel):
    day:   str
    icon:  str
    hi:    float
    lo:    float
    rain_prob: float


class ForecastOutput(BaseModel):
    micro_zone: str
    forecast:   List[ForecastDay]
    generated_at: str


class ModelInfo(BaseModel):
    model_type:  str
    features:    List[str]
    r2_score:    float
    rmse:        float
    mae:         float
    n_samples:   int
    version:     str
    mode:        str = Field(default="sintetico", description="real (estaciones BME280) | sintetico")
    arch:        str = Field(default="stacking", description="stacking | xgb (elegido por CV)")
    persist_rmse: float = Field(default=0.0, description="RMSE del baseline naive 't+1h = t' (°C)")
    feature_importances: dict = Field(default_factory=dict, description="Importancia real por feature (%)")
    hyperparameters:     dict = Field(default_factory=dict, description="Mejores hiperparámetros (Optuna)")


# ============================================================
#  CARGA DEL MODELO
# ============================================================

def load_model():
    """Carga el pipeline entrenado desde disco."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modelo no encontrado en {MODEL_PATH}. "
            "Ejecuta primero: python model/train_model.py"
        )
    artifact = joblib.load(MODEL_PATH)
    return artifact

# Carga en startup (None si aún no existe)
try:
    _artifact  = load_model()
    _pipeline  = _artifact['pipeline']
    _features  = _artifact['features']
    _metrics   = _artifact['metrics']
    _n_samples = int(_artifact.get('n_samples', 0))
    _importances = _artifact.get('feature_importances', {}) or {}
    _xgb_params  = _artifact.get('xgb_params', {}) or {}
    _mode        = _artifact.get('mode', 'sintetico')
    _arch        = _artifact.get('arch', 'stacking')
    _persist_rmse = float(_artifact.get('persist_rmse') or 0.0)
    _hum_mean    = float(_artifact.get('hum_mean') or 68.0)
    MODEL_LOADED = True
except Exception as e:
    _pipeline, _features, _metrics = None, None, None
    _n_samples, _importances, _xgb_params = 0, {}, {}
    _mode, _arch, _persist_rmse, _hum_mean = 'sintetico', 'stacking', 0.0, 68.0
    MODEL_LOADED = False
    print(f"⚠ Modelo no cargado: {e}")


# ============================================================
#  FEATURE ENGINEERING (espejo del train_model.py)
# ============================================================

def build_features(inp: WeatherInput) -> np.ndarray:
    """
    Construye el vector de features que espera el modelo, según
    el esquema con el que se entrenó (detectado del artefacto).
    """
    if _mode == 'real':
        # Esquema real (BME280): temperatura, humedad, presión y
        # ciclo horario del día.
        hora = inp.hora if inp.hora is not None else datetime.now().hour
        hour_sin = np.sin(2 * np.pi * hora / 24)
        hour_cos = np.cos(2 * np.pi * hora / 24)
        hum_effect = (inp.hum - _hum_mean) / 100
        heat_index = inp.temp * (1 + inp.hum / 200)
        return np.array([[
            inp.temp, inp.hum, inp.pres,
            hour_sin, hour_cos, hum_effect, heat_index
        ]])

    # Esquema sintético (legacy): 6 variables base + derivadas
    wind  = inp.wind  or 0.0
    cloud = inp.cloud or 0.0
    uv    = inp.uv    or 5.0

    pressure_delta = (inp.pres - 1013) / 40
    hum_effect     = (inp.hum  - 50)   / 100
    cloud_effect   = cloud / 100
    wind_chill     = -(wind - 20) * 0.15 if wind > 20 else 0.0
    heat_index     = inp.temp * (1 + inp.hum / 200)

    return np.array([[
        inp.temp, inp.hum, inp.pres, wind, cloud, uv,
        pressure_delta, hum_effect, cloud_effect, wind_chill, heat_index
    ]])


def classify_condition(pred_temp: float, rain_prob: float,
                       cloud: float | None, wind: float | None):
    """Clasifica la condición climática en descripción + emoji."""
    if rain_prob > 70:  return 'Lluvia probable', '🌧️'
    if rain_prob > 45:  return 'Parcialmente nublado con posible lluvia', '⛅'
    if cloud is not None and cloud > 60:
        return 'Nublado', '☁️'
    if wind is not None and wind > 40:
        return 'Vientoso', '💨'
    if pred_temp > 32:  return 'Caluroso y seco', '🌡️'
    if pred_temp < 5:   return 'Frío · posible helada', '🌨️'
    return 'Despejado y soleado', '☀️'


# ============================================================
#  APLICACIÓN FastAPI
# ============================================================

app = FastAPI(
    title="Clima ML API",
    description="API de predicción climática por micro-zona usando Gradient Boosting Ensemble",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS: permite solicitudes desde el frontend ─────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # En producción: especificar dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
#  ENDPOINTS
# ============================================================

@app.get("/api/info", tags=["Info"])
def api_info():
    """Información general de la API."""
    return {
        "name":    "Clima ML API",
        "version": "1.0.0",
        "model_loaded": MODEL_LOADED,
        "docs":    "/docs",
        "endpoints": ["/predict", "/forecast", "/model/info", "/health"]
    }


@app.get("/health", tags=["Info"])
def health():
    """Estado del servicio."""
    return {
        "status":       "ok" if MODEL_LOADED else "degraded",
        "model_loaded": MODEL_LOADED,
        "timestamp":    datetime.now(timezone.utc).isoformat()
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Predicción"])
def predict(inp: WeatherInput):
    """
    Ejecuta la predicción climática con el modelo GBR Ensemble.

    Recibe las variables meteorológicas de la micro-zona y retorna:
    - Temperatura predicha (°C)      → del modelo ML (t+1 hora)
    - RMSE (°C)                      → real, de la validación cruzada
    - Probabilidad de lluvia, sensación y confianza
      → estimaciones heurísticas auxiliares (NO salen del modelo)
    """
    t0 = time.perf_counter()

    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecuta python model/train_model.py primero."
        )

    try:
        X = build_features(inp)
        pred_temp = round(float(_pipeline.predict(X)[0]), 1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

    # Hora usada como feature (transparencia para el usuario)
    hora_usada = inp.hora if inp.hora is not None else datetime.now().hour

    # ── Heurísticas auxiliares (no son salida del modelo ML) ──
    if _mode == 'real':
        # Sin viento/nubosidad: la humedad es el principal proxy
        rain_prob  = max(0, min(100, round((inp.hum - 40) * 1.6)))
        hum_effect = (inp.hum - _hum_mean) / 100
        sensacion  = round(pred_temp - hum_effect * 2, 1)
        confidence = max(72, min(97, round(94 - abs(inp.hum - 50) / 8)))
    else:
        wind  = inp.wind  or 0.0
        cloud = inp.cloud or 0.0
        pressure_delta = (inp.pres - 1013) / 40
        hum_effect     = (inp.hum  - 50)   / 100
        cloud_effect   = cloud / 100
        wind_chill     = -(wind - 20) * 0.15 if wind > 20 else 0.0

        rain_prob  = max(0, min(100, round(inp.hum * 0.6 + cloud * 0.4 - (inp.pres - 970) * 0.3)))
        sensacion  = round(pred_temp - wind_chill * 0.5 - hum_effect * 2, 1)
        confidence = max(72, min(97, round(94 - abs(pressure_delta) * 5 - cloud_effect * 8)))

    # RMSE REAL de la validación cruzada del modelo (no una fórmula)
    rmse = round(float(_metrics['rmse_mean']), 2)

    condition, icon = classify_condition(pred_temp, rain_prob, inp.cloud, inp.wind)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    print(f"[predict:{_mode}] {pred_temp}°C (hora {hora_usada}) | conf={confidence}% | {latency_ms}ms")

    return PredictionOutput(
        pred_temp=pred_temp,
        rain_prob=float(rain_prob),
        sensacion=sensacion,
        confidence=float(confidence),
        rmse=rmse,
        condition=condition,
        icon=icon,
        hora_usada=int(hora_usada),
    )


@app.post("/forecast", response_model=ForecastOutput, tags=["Predicción"])
def forecast(inp: WeatherInput, zona: str = "micro-zona-1"):
    """
    Genera el pronóstico de 7 días para la micro-zona.

    - Parte de la predicción real del modelo ML (pred_temp).
    - Aplica variaciones diarias deterministas (sin aleatoriedad
      no reproducible) y usa los días de la semana reales.
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    X = build_features(inp)
    base_temp = float(_pipeline.predict(X)[0])

    days_es  = ['Dom', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb']
    today    = datetime.now(timezone.utc).date()
    rng      = np.random.default_rng(42)   # semilla fija → resultados reproducibles

    # Probabilidad base de lluvia según el modo (heurística)
    if _mode == 'real':
        rain_base = max(0, min(100, (inp.hum - 40) * 1.6))
    else:
        rain_base = max(0, min(100, inp.hum * 0.6 + (inp.cloud or 0) * 0.4))

    forecast_list = []
    for i in range(7):
        day_date = today + timedelta(days=i)
        day_name = 'Hoy' if i == 0 else days_es[day_date.weekday()]

        # Variación diaria suave + ruido pequeño reproducible
        variation = np.sin(i * 0.8) * 3 + np.cos(i * 1.2) * 2
        hi = round(base_temp + variation, 1)
        lo = round(hi - 6 - float(rng.uniform(0, 3)), 1)
        rain = max(0, min(100, round(rain_base - i * 2)))

        icon = ('🌧️' if rain > 70 else '⛅' if rain > 45 else
                '☀️' if i % 3 == 0 else '🌤️')

        forecast_list.append(ForecastDay(
            day=day_name, icon=icon, hi=hi, lo=lo, rain_prob=float(rain)
        ))

    return ForecastOutput(
        micro_zone=zona,
        forecast=forecast_list,
        generated_at=datetime.now(timezone.utc).isoformat()
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Modelo"])
def model_info():
    """Retorna las métricas y configuración del modelo entrenado."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    return ModelInfo(
        model_type=(("Stacking Ensemble (XGB + RF + Ridge)" if _arch == 'stacking'
                     else "XGBoost (seleccionado por CV)")
                    + (" · datos reales de estaciones BME280" if _mode == 'real' else "")),
        features=_features,
        r2_score=round(_metrics['r2_mean'], 4),
        rmse=round(_metrics['rmse_mean'], 4),
        mae=round(_metrics['mae_mean'], 4),
        n_samples=_n_samples if _n_samples else 0,
        version="1.0.0",
        mode=_mode,
        arch=_arch,
        persist_rmse=round(_persist_rmse, 4),
        feature_importances=_importances,
        hyperparameters=_xgb_params,
    )


# ============================================================
#  FRONTEND ESTÁTICO (todo en un solo servidor)
# ============================================================
# Disponible en:
#   http://localhost:8000/       → raíz (URL de despliegue)
#   http://localhost:8000/ui/    → alias
#
# Los assets (js/, css/) se montan en la raíz para que las
# rutas relativas del index.html funcionen en ambos casos.
# El orden importa: primero los mounts, luego las rutas de la
# API (FastAPI evalúa en orden de registro).
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / 'frontend'
if FRONTEND_DIR.exists():
    app.mount("/js",  StaticFiles(directory=str(FRONTEND_DIR / 'js'),  check_dir=False), name="js")
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / 'css'), check_dir=False), name="css")
    app.mount("/ui",  StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend-ui")

    @app.get("/", include_in_schema=False)
    def frontend_root():
        """Sirve la interfaz web en la raíz (URL de despliegue)."""
        index = FRONTEND_DIR / 'index.html'
        if index.exists():
            return FileResponse(str(index))
        return {"detail": "Frontend no encontrado", "ui": "/ui/"}


# ============================================================
#  PUNTO DE ENTRADA
# ============================================================

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)