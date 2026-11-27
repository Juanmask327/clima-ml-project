"""
============================================================
PREDICCIÓN CLIMÁTICA ML · Tests de la API (pytest + httpx)
============================================================
Ejecutar desde backend/:
    pytest tests/ -v

Requiere el modelo entrenado (model/clima_model.joblib).
============================================================
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

# Entrada del esquema REAL (sensor BME280): 3 variables + hora
VALID_INPUT = {
    "temp": 19.0, "hum": 68.0, "pres": 752.0, "hora": 14,
}

# Entrada del esquema sintético (compatibilidad hacia atrás):
# wind/cloud/uv son opcionales y deben seguir siendo aceptados.
LEGACY_INPUT = {
    "temp": 18.5, "hum": 65.0, "pres": 1013.0,
    "wind": 12.0, "cloud": 40.0, "uv": 5.0,
}


# ── Endpoints de información ────────────────────────────────

def test_root_serves_frontend():
    """La raíz sirve la interfaz web (URL pública de despliegue)."""
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Predicción Climática" in r.text


def test_api_info():
    """La información de la API está en /api/info."""
    r = client.get("/api/info")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Clima ML API"
    assert body["model_loaded"] is True


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Predicción ──────────────────────────────────────────────

def test_predict_ok():
    r = client.post("/predict", json=VALID_INPUT)
    assert r.status_code == 200
    body = r.json()

    # El valor principal viene del modelo ML
    assert isinstance(body["pred_temp"], (int, float))
    assert -30 <= body["pred_temp"] <= 60

    # El RMSE es el real de la validación cruzada (no una fórmula ad-hoc)
    assert body["rmse"] > 0
    assert body["condition"]
    assert body["icon"]

    # La hora usada como feature debe ser válida (0-23)
    assert 0 <= body["hora_usada"] <= 23


def test_predict_real_minimal_input():
    """El esquema real (BME280) solo requiere temp/hum/pres."""
    r = client.post("/predict", json={"temp": 19.0, "hum": 68.0, "pres": 752.0})
    assert r.status_code == 200
    assert -30 <= r.json()["pred_temp"] <= 60


def test_predict_real_pressure_altitude():
    """La presión de altura (~750 hPa) debe aceptarse."""
    r = client.post("/predict", json={**VALID_INPUT, "pres": 752.9})
    assert r.status_code == 200
    assert -30 <= r.json()["pred_temp"] <= 60


def test_predict_hora_param():
    """La hora se usa como feature del ciclo diario."""
    a = client.post("/predict", json={**VALID_INPUT, "hora": 3}).json()
    b = client.post("/predict", json={**VALID_INPUT, "hora": 15}).json()
    assert a["hora_usada"] == 3
    assert b["hora_usada"] == 15


def test_predict_legacy_input_compatible():
    """wind/cloud/uv opcionales: el esquema sintético sigue aceptado."""
    r = client.post("/predict", json=LEGACY_INPUT)
    assert r.status_code == 200
    assert -30 <= r.json()["pred_temp"] <= 60


def test_predict_validation_error():
    """Valores fuera de rango deben rechazarse con 422."""
    bad = {**VALID_INPUT, "hum": 150.0}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_missing_field():
    r = client.post("/predict", json={"temp": 20.0})
    assert r.status_code == 422


# ── Pronóstico ──────────────────────────────────────────────

def test_forecast_ok():
    r = client.post("/forecast", json=VALID_INPUT)
    assert r.status_code == 200
    body = r.json()

    assert len(body["forecast"]) == 7
    assert body["forecast"][0]["day"] == "Hoy"
    for day in body["forecast"]:
        assert day["hi"] >= day["lo"] - 30
        assert 0 <= day["rain_prob"] <= 100
    assert body["generated_at"]


def test_forecast_reproducible():
    """Misma entrada → mismo pronóstico (semilla fija)."""
    a = client.post("/forecast", json=VALID_INPUT).json()["forecast"]
    b = client.post("/forecast", json=VALID_INPUT).json()["forecast"]
    assert a == b


# ── Info del modelo ─────────────────────────────────────────

def test_model_info():
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()

    assert 0 <= body["r2_score"] <= 1
    assert body["rmse"] > 0
    assert body["mae"] > 0
    assert body["n_samples"] > 0
    assert len(body["features"]) >= 6

    # Importancia e hiperparámetros reales (no hardcodeados)
    assert body["feature_importances"], "Debe incluir importancia de features"
    assert body["hyperparameters"], "Debe incluir hiperparámetros de Optuna"
    assert "n_estimators" in body["hyperparameters"]

    # El modo indica con qué datos se entrenó
    assert body["mode"] in ("real", "sintetico")


# ── Frontend estático ───────────────────────────────────────

def test_frontend_served():
    r = client.get("/ui/")
    assert r.status_code == 200


def test_static_assets_at_root():
    """Los assets se sirven en la raíz (paths absolutos del index.html)."""
    for path in ("/js/model.js", "/js/ui.js", "/css/styles.css"):
        r = client.get(path)
        assert r.status_code == 200, f"{path} no se sirve en la raíz"
        assert r.content


def test_frontend_uses_relative_api_url():
    """
    El frontend NO debe apuntar a localhost:8000 hardcodeado:
    en la nube el navegador debe usar el mismo origen.
    """
    js = client.get("/js/model.js").text
    assert "window.location.origin" in js, "Debe usar window.location.origin"
    # La asignación no debe apuntar a un host fijo
    for line in js.splitlines():
        if "API_BASE_URL =" in line:
            assert "localhost" not in line, f"API_BASE_URL hardcodeado: {line.strip()}"
