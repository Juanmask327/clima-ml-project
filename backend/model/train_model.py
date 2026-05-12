"""
============================================================
PREDICCIÓN CLIMÁTICA ML · MICRO-ZONA
Módulo: Entrenamiento del modelo (train_model.py)
============================================================
Pipeline completo:
  1. Generación / carga de datos de entrenamiento
  2. Feature engineering y preprocesamiento
  3. Entrenamiento del Ensemble (XGBoost + RF + Ridge)
  4. Optimización bayesiana de hiperparámetros (Optuna)
  5. Evaluación con validación cruzada 5-fold
  6. Serialización del modelo (joblib)

Uso:
    python train_model.py

Dependencias:
    pip install xgboost scikit-learn optuna pandas numpy joblib
============================================================
"""

import numpy as np
import pandas as pd
import joblib
import optuna
import warnings
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── Rutas ──────────────────────────────────────────────────
MODEL_DIR  = Path(__file__).parent
OUTPUT_PATH = MODEL_DIR / 'clima_model.joblib'
DATA_PATH   = MODEL_DIR.parent / 'data' / 'weather_data.csv'


# ============================================================
#  1. GENERACIÓN DE DATOS SINTÉTICOS (demo)
#     En producción: reemplazar por carga real de sensores
# ============================================================

def generate_synthetic_data(n_samples: int = 8760) -> pd.DataFrame:
    """
    Genera datos sintéticos de 1 año (8760 horas) que simulan
    el comportamiento de sensores meteorológicos en micro-zona.

    Features:
        temp_actual   - Temperatura observada (°C)
        humedad       - Humedad relativa (%)
        presion       - Presión atmosférica (hPa)
        viento        - Velocidad del viento (km/h)
        nubosidad     - Porcentaje de nubosidad (%)
        indice_uv     - Índice UV (0-12)

    Target:
        temp_pred     - Temperatura predicha siguiente hora (°C)
    """
    np.random.seed(42)
    t = np.arange(n_samples)

    # Ciclo diario (24h) y estacional (365 días)
    diario    = np.sin(2 * np.pi * t / 24)
    estacional = np.sin(2 * np.pi * t / (24 * 365))

    temp_actual = 18 + 8 * estacional + 5 * diario + np.random.normal(0, 1.5, n_samples)
    humedad     = 65 - 15 * diario + 10 * np.random.randn(n_samples)
    presion     = 1013 + 5 * np.sin(2 * np.pi * t / (24 * 7)) + np.random.normal(0, 2, n_samples)
    viento      = np.abs(12 + 8 * np.random.randn(n_samples))
    nubosidad   = np.clip(40 + 30 * np.random.randn(n_samples), 0, 100)
    indice_uv   = np.clip(5 + 4 * diario + np.random.randn(n_samples), 0, 12)

    # Target: temperatura de la siguiente hora
    pressure_effect = (presion - 1013) / 40 * 2
    hum_effect      = (humedad - 50) / 100 * -1.5
    cloud_effect    = (nubosidad / 100) * -2
    wind_effect     = np.where(viento > 20, -(viento - 20) * 0.15, 0)

    temp_pred = (
        temp_actual + pressure_effect + hum_effect +
        cloud_effect + wind_effect + np.random.normal(0, 0.8, n_samples)
    )

    df = pd.DataFrame({
        'temp_actual': np.round(temp_actual, 2),
        'humedad':     np.clip(np.round(humedad, 1), 0, 100),
        'presion':     np.clip(np.round(presion, 1), 970, 1050),
        'viento':      np.round(viento, 1),
        'nubosidad':   np.clip(np.round(nubosidad, 1), 0, 100),
        'indice_uv':   np.clip(np.round(indice_uv, 1), 0, 12),
        'temp_pred':   np.round(temp_pred, 2)
    })
    return df


# ============================================================
#  2. FEATURE ENGINEERING
# ============================================================

FEATURES = ['temp_actual', 'humedad', 'presion', 'viento', 'nubosidad', 'indice_uv']
TARGET   = 'temp_pred'

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features derivadas para mejorar el modelo."""
    df = df.copy()
    df['pressure_delta'] = (df['presion'] - 1013) / 40
    df['hum_effect']     = (df['humedad'] - 50) / 100
    df['cloud_effect']   = df['nubosidad'] / 100
    df['wind_chill']     = np.where(
        df['viento'] > 20, -(df['viento'] - 20) * 0.15, 0
    )
    df['heat_index'] = df['temp_actual'] * (1 + df['humedad'] / 200)
    return df

FEATURES_EXT = FEATURES + ['pressure_delta', 'hum_effect', 'cloud_effect', 'wind_chill', 'heat_index']


# ============================================================
#  3. OPTIMIZACIÓN BAYESIANA DE HIPERPARÁMETROS (Optuna)
# ============================================================

def optimize_xgboost(X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> dict:
    """
    Busca los mejores hiperparámetros de XGBoost usando
    Bayesian Optimization (TPE Sampler) de Optuna.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 200, 800),
            'max_depth':        trial.suggest_int('max_depth', 3, 8),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 2.0),
            'random_state': 42,
            'verbosity': 0
        }
        model = XGBRegressor(**params)
        scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
        return -scores.mean()

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  ✓ Mejor RMSE (Optuna): {study.best_value:.4f}°C")
    print(f"  ✓ Mejores params: {study.best_params}")
    return study.best_params


# ============================================================
#  4. CONSTRUCCIÓN DEL ENSEMBLE (Stacking)
# ============================================================

def build_ensemble(xgb_params: dict) -> StackingRegressor:
    """
    Construye el Stacking Ensemble:
      - Base learners: XGBoost + Random Forest
      - Meta-learner:  Ridge Regression
    """
    xgb = XGBRegressor(**{**xgb_params, 'verbosity': 0, 'random_state': 42})
    rf  = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    ridge = Ridge(alpha=1.0)

    ensemble = StackingRegressor(
        estimators=[('xgb', xgb), ('rf', rf)],
        final_estimator=ridge,
        cv=5,
        n_jobs=-1
    )
    return ensemble


# ============================================================
#  5. PIPELINE COMPLETO (Scaler + Ensemble)
# ============================================================

def build_pipeline(xgb_params: dict) -> Pipeline:
    """
    Pipeline scikit-learn que integra:
      StandardScaler → Stacking Ensemble
    Permite llamar .predict() directamente con datos sin escalar.
    """
    return Pipeline([
        ('scaler',   StandardScaler()),
        ('ensemble', build_ensemble(xgb_params))
    ])


# ============================================================
#  6. EVALUACIÓN CON VALIDACIÓN CRUZADA
# ============================================================

def evaluate_model(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evalúa el pipeline con K-Fold CV (5 folds).
    Retorna RMSE, MAE y R² con media y desviación estándar.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores, mae_scores, r2_scores = [], [], []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_hat = pipeline.predict(X_val)

        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_hat)))
        mae_scores.append(mean_absolute_error(y_val, y_hat))
        r2_scores.append(r2_score(y_val, y_hat))

    metrics = {
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std':  np.std(rmse_scores),
        'mae_mean':  np.mean(mae_scores),
        'mae_std':   np.std(mae_scores),
        'r2_mean':   np.mean(r2_scores),
        'r2_std':    np.std(r2_scores),
    }
    return metrics


# ============================================================
#  7. FUNCIÓN PRINCIPAL
# ============================================================

def main():
    print("=" * 55)
    print("  PREDICCIÓN CLIMÁTICA ML · Entrenamiento del modelo")
    print("=" * 55)

    # ── Datos ───────────────────────────────────────────────
    print("\n[1/5] Cargando datos...")
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        print(f"  ✓ Cargados {len(df):,} registros desde {DATA_PATH}")
    else:
        print("  ⚠ Archivo de datos no encontrado. Generando datos sintéticos...")
        df = generate_synthetic_data(n_samples=8760)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"  ✓ Generados {len(df):,} registros → {DATA_PATH}")

    # ── Feature engineering ─────────────────────────────────
    print("\n[2/5] Feature engineering...")
    df = feature_engineering(df)
    X = df[FEATURES_EXT].values
    y = df[TARGET].values
    print(f"  ✓ Features: {FEATURES_EXT}")
    print(f"  ✓ Shape: X={X.shape}, y={y.shape}")

    # ── Optimización de hiperparámetros ─────────────────────
    print("\n[3/5] Optimización bayesiana (Optuna) — 50 trials...")
    xgb_params = optimize_xgboost(X, y, n_trials=50)

    # ── Construcción del pipeline ────────────────────────────
    print("\n[4/5] Construyendo pipeline Stacking Ensemble...")
    pipeline = build_pipeline(xgb_params)

    # ── Evaluación ───────────────────────────────────────────
    print("\n[5/5] Evaluación con K-Fold CV (5 folds)...")
    metrics = evaluate_model(pipeline, X, y)

    print("\n  ┌──────────────────────────────┐")
    print(f"  │  RMSE : {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f} °C  │")
    print(f"  │  MAE  : {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f} °C  │")
    print(f"  │  R²   : {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}     │")
    print("  └──────────────────────────────┘")

    # ── Entrenamiento final con todos los datos ──────────────
    pipeline.fit(X, y)

    # ── Serialización ────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': pipeline, 'features': FEATURES_EXT, 'metrics': metrics}, OUTPUT_PATH)
    print(f"\n  ✓ Modelo guardado → {OUTPUT_PATH}")
    print("\n  Listo. Ejecuta 'python api/main.py' para iniciar la API.\n")


if __name__ == '__main__':
    main()