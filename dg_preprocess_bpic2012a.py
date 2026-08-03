# -*- coding: utf-8 -*-
"""
dg_preprocess_bpic2012a.py
Preprocesa el log BPIC2012-A para adaptarlo al pipeline GAN.

Etapas:
  1. Normalizar columnas  : Case ID -> caseid, Activity -> task, Resource -> user
  2. Manejar lifecycle    : cada evento (incluye variantes -START/-COMPLETE/-SCHEDULE)
                            se trata como evento puntual (start_timestamp = end_timestamp)
                            para mantener los 36 tipos de actividad del paper CVAE.
  3. Generar timestamps   : start_timestamp = end_timestamp = time:timestamp
  4. Guardar resultado    : sobreescribe bpic2012_a.csv (backup en bpic2012_a_raw.csv)

Uso:
    python dg_preprocess_bpic2012a.py

El archivo resultante queda listo para dg_training.py: la funcion
preprocess_lifecycle_log detecta start_timestamp/end_timestamp y pasa
sin modificaciones.
"""
import sys
from pathlib import Path

import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
LOG_DIR  = Path('data/0.logs/bpic2012_a')
RAW_CSV  = LOG_DIR / 'bpic2012_a_raw.csv'
OUT_CSV  = LOG_DIR / 'bpic2012_a.csv'


def load_raw() -> pd.DataFrame:
    """Carga el CSV original (separador ;) o el backup si ya existe."""
    src = RAW_CSV if RAW_CSV.exists() else OUT_CSV
    if not src.exists():
        print(f'[ERROR] No se encontro el archivo: {src}')
        sys.exit(1)
    print(f'[Preprocess] Leyendo: {src}')
    df = pd.read_csv(src, sep=';', low_memory=False)
    # Si el separador ya era coma (archivo ya procesado), recargar correctamente
    if len(df.columns) == 1:
        df = pd.read_csv(src, low_memory=False)
    return df


def backup_raw(df_raw: pd.DataFrame):
    """Guarda el CSV original como _raw.csv si aun no existe el backup."""
    if not RAW_CSV.exists():
        df_raw.to_csv(RAW_CSV, sep=';', index=False)
        print(f'[Preprocess] Backup guardado: {RAW_CSV}')
    else:
        print(f'[Preprocess] Backup ya existe: {RAW_CSV}')


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Etapas 1-3: normalizar, asignar timestamps y seleccionar columnas.

    Convencion de eventos puntuales (alineada con CVAE paper):
    - Cada fila del log es un evento discreto en el tiempo.
    - La columna Activity incluye el sufijo de lifecycle (-COMPLETE/-START/-SCHEDULE),
      lo que genera 36 tipos distintos de actividad (igual que en el paper CVAE).
    - start_timestamp = end_timestamp = time:timestamp para todos los eventos.
    - El tiempo entre eventos consecutivos de un caso se captura como wait_time
      en el entrenamiento del GAN.
    """
    # ── Etapa 1: normalizar columnas ─────────────────────────────────────────
    col_map = {
        'Case ID': 'caseid',
        'Activity': 'task',
        'Resource': 'user',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Verificar columnas obligatorias
    required = {'caseid', 'task', 'time:timestamp'}
    missing = required - set(df.columns)
    if missing:
        print(f'[ERROR] Columnas faltantes: {missing}')
        print(f'        Columnas disponibles: {list(df.columns)}')
        sys.exit(1)

    # ── Etapa 2: parsear timestamps ───────────────────────────────────────────
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

    # ── Etapa 3: eventos puntuales (start = end = timestamp) ─────────────────
    df['start_timestamp'] = df['time:timestamp']
    df['end_timestamp']   = df['time:timestamp']

    # Normalizar usuario
    if 'user' not in df.columns:
        df['user'] = 'UNKNOWN'
    df['user'] = df['user'].fillna('UNKNOWN').astype(str)

    # Seleccionar y ordenar
    df_result = (
        df[['caseid', 'task', 'user', 'start_timestamp', 'end_timestamp']]
        .sort_values(['caseid', 'start_timestamp'])
        .reset_index(drop=True)
    )
    return df_result


def print_summary(df: pd.DataFrame):
    """Muestra estadisticas del log preprocesado."""
    n_cases  = df['caseid'].nunique()
    n_events = len(df)
    n_acts   = df['task'].nunique()
    trace_len = df.groupby('caseid').size()

    print()
    print('=' * 58)
    print('  Resumen del log preprocesado')
    print('=' * 58)
    print(f'  Casos           : {n_cases:,}')
    print(f'  Eventos         : {n_events:,}')
    print(f'  Actividades     : {n_acts}')
    print(f'  Eventos/traza   : min={trace_len.min()}  '
          f'med={trace_len.median():.0f}  '
          f'max={trace_len.max()}')
    ts_min = df['start_timestamp'].min()
    ts_max = df['start_timestamp'].max()
    print(f'  Rango temporal  : {ts_min.date()}  a  {ts_max.date()}')

    cycle = (
        df.groupby('caseid')['start_timestamp']
        .agg(lambda x: (x.max() - x.min()).total_seconds() / 86400)
    )
    print(f'  Ciclo (dias)    : min={cycle.min():.1f}  '
          f'med={cycle.median():.1f}  '
          f'media={cycle.mean():.1f}  '
          f'max={cycle.max():.1f}')
    print()
    print('  Actividades unicas:')
    for act in sorted(df['task'].unique()):
        cnt = (df['task'] == act).sum()
        print(f'    {act:<44} {cnt:>7,}')
    print('=' * 58)


def main():
    print('[Preprocess] Iniciando preprocesamiento de BPIC2012-A ...')

    # Cargar
    df_raw = load_raw()

    # Backup del original
    backup_raw(df_raw)

    # Preprocesar
    df_result = preprocess(df_raw)

    # Resumen
    print_summary(df_result)

    # Guardar resultado (formato CSV estandar con coma)
    df_result.to_csv(OUT_CSV, index=False)
    print(f'\n[Preprocess] Archivo listo para el pipeline: {OUT_CSV}')
    print('[Preprocess] Columnas: caseid | task | user | start_timestamp | end_timestamp')
    print('[Preprocess] Listo para ejecutar: python dg_training.py -m transformer_wgan')


if __name__ == '__main__':
    main()
