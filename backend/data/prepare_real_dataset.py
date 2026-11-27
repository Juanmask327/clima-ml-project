"""
============================================================
PREDICCIÓN CLIMÁTICA ML · Preparación del dataset real
============================================================
A partir de weather_data_real.csv (registros crudos de 5 segundos
de las estaciones BME280) construye el dataset horario de
entrenamiento:

    estacion, time, temp_actual, humedad, presion, temp_next

- Agrega por hora y estación (media de las lecturas).
- Target: temperatura de la SIGUIENTE hora (solo si realmente es
  la hora siguiente, sin saltos).
- Ordenado por tiempo.

Uso (desde la raíz del proyecto):
    python backend/data/prepare_real_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA_DIR = Path(__file__).parent
RAW_PATH  = DATA_DIR / 'weather_data_real.csv'
OUT_PATH  = DATA_DIR / 'weather_data.csv'          # dataset de entrenamiento
SYNT_PATH = DATA_DIR / 'weather_data_synth.csv'    # respaldo del sintético


def main():
    print("=" * 55)
    print("  DATASET REAL · Agregación horaria")
    print("=" * 55)

    if not RAW_PATH.exists():
        print(f"  No existe {RAW_PATH}. Ejecuta antes parse_stations.py")
        return

    df = pd.read_csv(RAW_PATH, parse_dates=['time'])
    print(f"  Crudos: {len(df):,} registros")

    # ── 1. Respaldo del dataset sintético (solo la primera vez) ──
    if OUT_PATH.exists() and not SYNT_PATH.exists():
        OUT_PATH.rename(SYNT_PATH)
        print(f"  Respaldo del sintético → {SYNT_PATH.name}")

    # ── 2. Agregación horaria por estación ───────────────────
    df['hora'] = df['time'].dt.floor('h')
    hourly = (df.groupby(['estacion', 'hora'], as_index=False)
                .agg(temp_actual=('temp_actual', 'mean'),
                     humedad=('humedad', 'mean'),
                     presion=('presion', 'mean')))
    print(f"  Horario: {len(hourly):,} filas "
          f"({df['estacion'].nunique()} estaciones)")

    # ── 3. Target: temperatura de la siguiente hora ──────────
    hourly = hourly.sort_values(['estacion', 'hora']).reset_index(drop=True)
    hourly['temp_next'] = hourly.groupby('estacion')['temp_actual'].shift(-1)
    # Delta temporal CORRECTO: hora siguiente - hora actual
    hora_next = hourly.groupby('estacion')['hora'].shift(-1)
    delta = hora_next - hourly['hora']
    # Solo se conserva el target si la fila siguiente es EXACTAMENTE
    # 1 hora después (sin huecos de muestreo).
    hourly.loc[delta != pd.Timedelta(hours=1), 'temp_next'] = pd.NA
    n_before = len(hourly)
    hourly = hourly.dropna(subset=['temp_next'])
    print(f"  Filas con target válido (t+1h contigua): {len(hourly):,} "
          f"(descartadas {n_before - len(hourly):,})")

    # ── 4. Limpieza: rangos físicos ──────────────────────────
    # Nota: la presión real es de altura (~750 hPa), NO nivel del mar.
    mask = (
        hourly['temp_actual'].between(-10, 50) &
        hourly['humedad'].between(0, 100) &
        hourly['presion'].between(600, 850) &
        hourly['temp_next'].between(-10, 50)
    )
    hourly = hourly[mask]
    print(f"  Tras filtro de rangos físicos: {len(hourly):,}")

    # ── 5. Estadísticas finales ──────────────────────────────
    print("\n  Resumen del dataset de entrenamiento:")
    for est, g in hourly.groupby('estacion'):
        print(f"    {est:<12} {len(g):>7,} horas  "
              f"{g['hora'].min():%Y-%m-%d} → {g['hora'].max():%Y-%m-%d}")

    print("\n  Rango de variables (horarias):")
    for c in ['temp_actual', 'humedad', 'presion', 'temp_next']:
        print(f"    {c:<12} {hourly[c].min():>8.2f} … {hourly[c].max():>8.2f} "
              f"(media {hourly[c].mean():.2f})")

    # ── 6. Persistencia ───────────────────────────────────────
    hourly = hourly.rename(columns={'hora': 'time'})
    hourly = hourly.sort_values('time').reset_index(drop=True)
    hourly.to_csv(OUT_PATH, index=False)
    print(f"\n  Guardado → {OUT_PATH}  ({len(hourly):,} filas)")
    print("  Columnas:", list(hourly.columns))


if __name__ == '__main__':
    main()
