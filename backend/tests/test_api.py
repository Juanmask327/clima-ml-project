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

VALID_INPUT = {
    "temp": 18.5, "hum": 65.0, "pres": 1013.0,
    "wind": 12.0, "cloud": 40.0, "uv": 5.0,
}


# ── Endpoints de información ────────────────────────────────

def test_root():
    r = client.get("/")
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


# ── Frontend estático ───────────────────────────────────────

def test_frontend_served():
    r = client.get("/ui/")
    assert r.status_code == 200
    assert "Predicción Climática" in r.text
