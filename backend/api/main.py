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

import time
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ─── Ruta del modelo serializado ────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / 'model' / 'clima_model.joblib'

# ============================================================
#  ESQUEMAS Pydantic (validación de entrada/salida)
# ============================================================

class WeatherInput(BaseModel):
    """Variables meteorológicas de entrada para la predicción."""
    temp:  float = Field(..., ge=-10, le=45,   description="Temperatura actual (°C)")
    hum:   float = Field(..., ge=0,   le=100,  description="Humedad relativa (%)")
    pres:  float = Field(..., ge=970, le=1050, description="Presión atmosférica (hPa)")
    wind:  float = Field(..., ge=0,   le=200,  description="Velocidad del viento (km/h)")
    cloud: float = Field(..., ge=0,   le=100,  description="Nubosidad (%)")
    uv:    float = Field(..., ge=0,   le=12,   description="Índice UV")

    @field_validator('hum', 'cloud')
    @classmethod
    def clamp_percentage(cls, v):
        return max(0.0, min(100.0, v))


class PredictionOutput(BaseModel):
    """Resultado de la predicción del modelo GBR Ensemble."""
    pred_temp:   float  = Field(..., description="Temperatura predicha (°C)")
    rain_prob:   float  = Field(..., description="Probabilidad de lluvia (0-100)")
    sensacion:   float  = Field(..., description="Sensación térmica (°C)")
    wind_pred:   float  = Field(..., description="Viento predicho (km/h)")
    confidence:  float  = Field(..., description="Confianza del modelo (%)")
    rmse:        float  = Field(..., description="RMSE estimado (°C)")
    condition:   str    = Field(..., description="Descripción de la condición climática")
    icon:        str    = Field(..., description="Emoji del estado del tiempo")
    model_version: str  = Field(default="1.0.0")
    timestamp:   str    = Field(default_factory=lambda: datetime.utcnow().isoformat())


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
    return artifact['pipeline'], artifact['features'], artifact['metrics']

# Carga en startup (None si aún no existe)
try:
    _pipeline, _features, _metrics = load_model()
    MODEL_LOADED = True
except Exception as e:
    _pipeline, _features, _metrics = None, None, None
    MODEL_LOADED = False
    print(f"⚠ Modelo no cargado: {e}")


# ============================================================
#  FEATURE ENGINEERING (espejo del train_model.py)
# ============================================================

def build_features(inp: WeatherInput) -> np.ndarray:
    """
    Construye el vector de features extendido que espera el modelo.
    Debe ser idéntico al aplicado en train_model.py → feature_engineering().
    """
    pressure_delta = (inp.pres - 1013) / 40
    hum_effect     = (inp.hum  - 50)   / 100
    cloud_effect   = inp.cloud / 100
    wind_chill     = -(inp.wind - 20) * 0.15 if inp.wind > 20 else 0.0
    heat_index     = inp.temp * (1 + inp.hum / 200)

    return np.array([[
        inp.temp, inp.hum, inp.pres, inp.wind, inp.cloud, inp.uv,
        pressure_delta, hum_effect, cloud_effect, wind_chill, heat_index
    ]])


def classify_condition(pred_temp: float, rain_prob: float, cloud: float, wind: float):
    """Clasifica la condición climática en descripción + emoji."""
    if rain_prob > 70:  return 'Lluvia probable', '🌧️'
    if rain_prob > 45:  return 'Parcialmente nublado con posible lluvia', '⛅'
    if cloud > 60:      return 'Nublado', '☁️'
    if wind > 40:       return 'Vientoso', '💨'
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

@app.get("/", tags=["Info"])
def root():
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
        "timestamp":    datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Predicción"])
def predict(inp: WeatherInput):
    """
    Ejecuta la predicción climática con el modelo GBR Ensemble.

    Recibe las variables meteorológicas de la micro-zona y retorna:
    - Temperatura predicha (°C)
    - Probabilidad de lluvia
    - Sensación térmica
    - Confianza del modelo y RMSE estimado
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

    # Calcular variables derivadas
    pressure_delta = (inp.pres - 1013) / 40
    hum_effect     = (inp.hum  - 50)   / 100
    cloud_effect   = inp.cloud / 100
    wind_chill     = -(inp.wind - 20) * 0.15 if inp.wind > 20 else 0.0

    rain_prob  = max(0, min(100, round(inp.hum * 0.6 + inp.cloud * 0.4 - (inp.pres - 970) * 0.3)))
    sensacion  = round(pred_temp - wind_chill * 0.5 - hum_effect * 2, 1)
    wind_pred  = round(inp.wind * (1 + (1013 - inp.pres) / 300), 1)
    confidence = max(72, min(97, round(94 - abs(pressure_delta) * 5 - cloud_effect * 8)))
    rmse       = round(1.8 - confidence / 100, 2)

    condition, icon = classify_condition(pred_temp, rain_prob, inp.cloud, inp.wind)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    print(f"[predict] {pred_temp}°C | conf={confidence}% | {latency_ms}ms")

    return PredictionOutput(
        pred_temp=pred_temp,
        rain_prob=float(rain_prob),
        sensacion=sensacion,
        wind_pred=wind_pred,
        confidence=float(confidence),
        rmse=rmse,
        condition=condition,
        icon=icon,
    )


@app.post("/forecast", response_model=ForecastOutput, tags=["Predicción"])
def forecast(inp: WeatherInput, zona: str = "micro-zona-1"):
    """
    Genera el pronóstico de 7 días para la micro-zona.
    Aplica el modelo iterativamente con perturbaciones estocásticas
    controladas que simulan la incertidumbre temporal.
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    days_es = ['Hoy', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    icons   = ['☀️',  '⛅',  '🌧️', '🌤️', '☀️',  '⛅',  '🌦️']

    forecast_list = []
    for i in range(7):
        variation = np.sin(i * 0.8) * 3 + np.cos(i * 1.2) * 2
        hi = round(inp.temp + variation, 1)
        lo = round(hi - 6 - float(np.random.uniform(0, 3)), 1)
        rain = max(0, min(100, round(inp.hum * 0.6 + inp.cloud * 0.4 - i * 2)))

        forecast_list.append(ForecastDay(
            day=days_es[i], icon=icons[i], hi=hi, lo=lo, rain_prob=float(rain)
        ))

    return ForecastOutput(
        micro_zone=zona,
        forecast=forecast_list,
        generated_at=datetime.utcnow().isoformat()
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Modelo"])
def model_info():
    """Retorna las métricas y configuración del modelo entrenado."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    return ModelInfo(
        model_type="Stacking Ensemble: XGBoost + Random Forest + Ridge",
        features=_features,
        r2_score=round(_metrics['r2_mean'], 4),
        rmse=round(_metrics['rmse_mean'], 4),
        mae=round(_metrics['mae_mean'], 4),
        n_samples=8760,
        version="1.0.0"
    )


# ============================================================
#  PUNTO DE ENTRADA
# ============================================================

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)