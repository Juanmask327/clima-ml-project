"""
============================================================
PREDICCIÓN CLIMÁTICA ML · Tests del cargador de datos reales
============================================================
Verifica que el pipeline de ingestión de datos de estaciones
meteorológicas reales funciona: alias de columnas, unidades,
construcción del target por shift, limpieza de outliers y
validación de columnas obligatorias.
"""

import numpy as np
import pandas as pd
import pytest

from model.train_model import (
    COLUMN_ALIASES,
    FEATURES,
    TARGET,
    _normalize_column_names,
    _detect_and_convert_units,
    load_real_data,
)


# ── Helpers ─────────────────────────────────────────────────

def _write_csv(tmp_path, df, name="datos.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


BASE = {
    'temp_actual': 20.0, 'humedad': 60.0, 'presion': 1013.0,
    'viento': 10.0, 'nubosidad': 40.0, 'indice_uv': 5.0,
}


# ── Normalización de nombres ────────────────────────────────

def test_normaliza_alias_de_columnas():
    """Acepta nombres típicos de estación y los mapea a los internos."""
    df = pd.DataFrame({
        'Temperatura': [20, 21], 'HR': [60, 61], 'Presion': [1013, 1012],
        'Viento': [10, 12], 'Nubosidad': [40, 30], 'UV': [5, 6],
    })
    out = _normalize_column_names(df)
    for col in FEATURES:
        assert col in out.columns, f"falta {col}"


def test_normaliza_no_altera_columnas_correctas():
    df = pd.DataFrame([BASE])
    out = _normalize_column_names(df)
    assert list(out.columns) == list(BASE.keys())


def test_normaliza_ignora_mayusculas_y_espacios():
    df = pd.DataFrame({'TEMP_ACTUAL': [20], 'HUMEDAD': [60]})
    out = _normalize_column_names(df)
    assert 'temp_actual' in out.columns
    assert 'humedad' in out.columns


# ── Conversión de unidades ──────────────────────────────────

def test_convierte_presion_de_kpa_a_hpa():
    df = pd.DataFrame({'presion': [101.3], 'presion_unit': ['kPa']})
    out = _detect_and_convert_units(df)
    assert out['presion'].iloc[0] == pytest.approx(1013.0, rel=1e-3)
    assert 'presion_unit' not in out.columns


def test_convierte_temperatura_de_fahrenheit_a_celsius():
    df = pd.DataFrame({'temp_actual': [68.0], 'temp_unit': ['F']})
    out = _detect_and_convert_units(df)
    assert out['temp_actual'].iloc[0] == pytest.approx(20.0, rel=1e-3)


def test_unidades_desconocidas_no_rompen():
    df = pd.DataFrame({'presion': [1013.0], 'presion_unit': ['desconocida']})
    out = _detect_and_convert_units(df)
    assert out['presion'].iloc[0] == pytest.approx(1013.0)


# ── load_real_data ──────────────────────────────────────────

def test_carga_csv_con_columnas_estandar(tmp_path):
    df = pd.DataFrame([BASE] * 10)
    df['temp_pred'] = df['temp_actual'] + 0.5
    path = _write_csv(tmp_path, df)

    out = load_real_data(path)
    assert len(out) == 10
    assert TARGET in out.columns
    for col in FEATURES:
        assert col in out.columns


def test_construye_target_con_shift_si_falta(tmp_path):
    """Sin 'temp_pred', el target es la temp de la siguiente hora."""
    df = pd.DataFrame({
        'temp_actual': [10.0, 11.0, 12.0, 13.0],
        'humedad': [60.0] * 4, 'presion': [1013.0] * 4,
        'viento': [10.0] * 4, 'nubosidad': [40.0] * 4, 'indice_uv': [5.0] * 4,
    })
    path = _write_csv(tmp_path, df)

    out = load_real_data(path)
    # La última fila se descarta (no hay siguiente hora) → 3 filas
    assert len(out) == 3
    assert out['temp_pred'].tolist() == [11.0, 12.0, 13.0]


def test_descarta_outliers_fuera_de_rango(tmp_path):
    df = pd.DataFrame([BASE] * 5)
    df['temp_pred'] = df['temp_actual'] + 0.5
    df.loc[2, 'humedad'] = 250.0     # imposible
    df.loc[3, 'presion'] = 5000.0    # imposible
    path = _write_csv(tmp_path, df)

    out = load_real_data(path)
    assert len(out) == 3
    assert out['humedad'].max() <= 100


def test_descarta_nan_y_no_numericos(tmp_path):
    df = pd.DataFrame([BASE] * 4)
    df['temp_pred'] = df['temp_actual'] + 0.5
    df.loc[1, 'temp_actual'] = np.nan
    # Se asigna sobre una columna de tipo object para simular un CSV
    # con texto donde debería haber un número.
    df['viento'] = df['viento'].astype(object)
    df.loc[2, 'viento'] = 'no-numero'
    path = _write_csv(tmp_path, df)

    out = load_real_data(path)
    assert len(out) == 2
    assert out['temp_actual'].notna().all()
    assert out['viento'].notna().all()


def test_acepta_alias_en_el_carga_real(tmp_path):
    """Un CSV con nombres de estación se carga sin tocar el código."""
    df = pd.DataFrame({
        'temperatura': [20.0, 21.0], 'hr': [60.0, 61.0],
        'pressure': [1013.0, 1012.0], 'wind_speed': [10.0, 12.0],
        'cloud': [40.0, 30.0], 'uv_index': [5.0, 6.0],
    })
    path = _write_csv(tmp_path, df)

    out = load_real_data(path)
    # Sin 'temp_pred' en el CSV, el target se construye con shift(-1)
    # y la última fila se descarta (no hay "siguiente hora").
    assert len(out) == 1
    for col in FEATURES:
        assert col in out.columns


def test_error_si_faltan_columnas_obligatorias(tmp_path):
    df = pd.DataFrame({'temp_actual': [20.0], 'humedad': [60.0]})
    path = _write_csv(tmp_path, df)

    with pytest.raises(ValueError) as exc:
        load_real_data(path)
    # El mensaje debe ser accionable: lista columnas y alias
    msg = str(exc.value)
    assert 'presion' in msg
    assert 'Alias' in msg or 'alias' in msg


def test_error_si_no_quedan_filas_validas(tmp_path):
    df = pd.DataFrame([BASE] * 3)
    df['humedad'] = 999.0          # todas fuera de rango
    path = _write_csv(tmp_path, df)

    with pytest.raises(ValueError):
        load_real_data(path)


def test_preserva_orden_temporal(tmp_path):
    """El orden de las filas NO debe alterarse (serie temporal)."""
    df = pd.DataFrame([BASE] * 6)
    df['temp_actual'] = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    df['temp_pred'] = df['temp_actual'].shift(-1)
    path = _write_csv(tmp_path, df)

    out = load_real_data(path)
    # La última fila tiene target NaN (no hay siguiente hora) → se descarta
    assert out['temp_actual'].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]
