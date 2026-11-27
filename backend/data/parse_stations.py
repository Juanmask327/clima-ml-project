"""
============================================================
PREDICCIÓN CLIMÁTICA ML · Parser de datos reales de estaciones
============================================================
Convierte los dumps crudos de las estaciones Raspberry Pi
(sensor BME280) a un CSV limpio y unificado:

    estacion, time, temp_actual, humedad, presion

Fuentes soportadas:
    - Dump phpMyAdmin/MySQL (.sql) con INSERTs de `bme280_data`
    - Export CSV de una sola línea sin cabecera (grupos de 5 campos)

Uso (desde la raíz del proyecto):
    python backend/data/parse_stations.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pandas as pd

# UTF-8 en consola de Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA_DIR = Path(__file__).parent
OUT_PATH = DATA_DIR / 'weather_data_real.csv'

# Tupla de INSERT de MySQL:
# ('rasberry_3', '2023-05-31 19:20:57', 19.8383, 752.972, 52.3622)
RE_SQL_TUPLE = re.compile(
    r"\(\s*'([^']*)'\s*,\s*"          # estacion
    r"'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'\s*,\s*"  # time
    r"(-?[\d.]+)\s*,\s*"               # temperatura
    r"(-?[\d.]+)\s*,\s*"               # presion
    r"(-?[\d.]+)\s*\)"                 # humedad
)

# Nombres de columnas que produce el parser
COLS = ['estacion', 'time', 'temp_actual', 'presion', 'humedad']


def parse_sql_dump(path: Path) -> list[tuple]:
    """Extrae todas las tuplas de un dump phpMyAdmin/MySQL."""
    text = path.read_text(encoding='utf-8', errors='replace')
    return RE_SQL_TUPLE.findall(text)


def parse_one_line_csv(path: Path) -> list[tuple]:
    """
    Parsea el export de una sola línea sin cabecera:
    "rasberry_1","2023-05-31 09:20:57","19.8383","752.972","52.3622", ...
    Grupos repetidos de 5 campos.
    """
    with open(path, encoding='utf-8', errors='replace') as f:
        row = next(csv.reader(f))

    # El export termina con un separador colgante: se recorta
    records = []
    for i in range(0, len(row) - 4, 5):
        est, t, temp, pres, hum = row[i:i + 5]
        if not est or not t or not t[0].isdigit():
            continue
        records.append((est, t, temp, pres, hum))
    return records


def main():
    print("=" * 55)
    print("  PARSER · Datos reales de estaciones (BME280)")
    print("=" * 55)

    all_records: list[tuple] = []

    for f in sorted(DATA_DIR.iterdir()):
        if f.name == 'weather_data.csv' or f.name.startswith('weather_data_real'):
            continue
        try:
            if f.suffix.lower() == '.sql':
                recs = parse_sql_dump(f)
            elif f.suffix.lower() == '.csv':
                recs = parse_one_line_csv(f)
            else:
                continue
        except Exception as e:
            print(f"  ! {f.name}: {e}")
            continue

        if recs:
            all_records.extend(recs)
            ests = {r[0] for r in recs}
            print(f"  {f.name:<38} {len(recs):>8,} registros  estaciones={sorted(ests)}")
        else:
            print(f"  ! {f.name}: sin registros reconocidos")

    if not all_records:
        print("\nNo se reconocieron registros. Revisa los archivos.")
        return

    df = pd.DataFrame(all_records, columns=COLS)
    for c in ['temp_actual', 'presion', 'humedad']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna()

    print(f"\n  Total: {len(df):,} registros de {df['estacion'].nunique()} estaciones")

    # ── Resumen por estación ─────────────────────────────────
    print("\n  Resumen por estación:")
    for est, g in df.groupby('estacion'):
        g = g.sort_values('time')
        dias = (g['time'].max() - g['time'].min())
        print(f"    {est:<12} {len(g):>8,} regs  "
              f"{g['time'].min():%Y-%m-%d} → {g['time'].max():%Y-%m-%d} "
              f"({dias.days} días)")

    # ── Rangos físicos ──────────────────────────────────────
    print("\n  Rangos observados:")
    for c in ['temp_actual', 'presion', 'humedad']:
        print(f"    {c:<12} min={df[c].min():>8.2f}  max={df[c].max():>8.2f}  "
              f"media={df[c].mean():>7.2f}")

    # ── Resolución temporal (respetando la estación) ────────
    print("\n  Resolución temporal (delta más frecuente por estación):")
    for est, g in df.groupby('estacion'):
        g = g.sort_values('time')
        d = g['time'].diff().dropna()
        d = d[d > pd.Timedelta(0)]
        if len(d):
            freq = d.mode().iloc[0]
            print(f"    {est:<12} {freq}")

    # ── Guardar ─────────────────────────────────────────────
    df = df.sort_values(['estacion', 'time'])
    df.to_csv(OUT_PATH, index=False)
    print(f"\n  Guardado → {OUT_PATH}  ({len(df):,} filas)")

    # Muestra
    print("\n  Primeras 5 filas:")
    print(df.head(5).to_string(index=False))


if __name__ == '__main__':
    main()
