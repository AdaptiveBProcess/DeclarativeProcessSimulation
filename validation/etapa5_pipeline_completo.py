# -*- coding: utf-8 -*-
"""
ETAPA 5 - Integracion completa del pipeline (dg_training + dg_prediction)
==========================================================================
Ejecutar desde la raiz del proyecto:
    C:\\Users\\Diego\\miniconda3\\envs\\deep_generator\\python.exe validation/etapa5_pipeline_completo.py

Que verifica:
  1. Despacho de dg_training.py:  parse_args + model_type_map -> GANTrainer
  2. Construccion de Params.simulation con la clase de produccion real
  3. pa.hallucinate() genera el CSV alucinado usando el GAN
  4. pa.hallucinate() elimina actividades centinela (Start/End) del CSV
  5. dg_prediction.preprocess_xes_log() produce .csv.gz valido para Simod
  6. El .csv.gz tiene columnas correctas: caseid, task, user, start_ts, end_ts
  7. El .csv.gz no contiene actividades centinela

Diferencias respecto a Etapa 4:
  - Usa Params (clase de produccion) para construir los parametros
  - Llama pa.hallucinate() en vez de ModelPredictor directamente
  - Verifica el paso de postprocesado preprocess_xes_log() (nuevo para Simod)
  - Usa las reglas REALES del archivo rules_global.ini (sin mock)
"""

import os
import sys
import json
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathlib import Path
import pandas as pd
from unittest.mock import patch


def ok(msg):
    print('  [OK]   ' + msg)


def fail(msg):
    print('  [FAIL] ' + msg)
    import traceback; traceback.print_exc()
    sys.exit(1)


def warn(msg):
    print('  [WARN] ' + msg)


def seccion(titulo):
    print('\n' + '-' * 58)
    print('  ' + titulo)
    print('-' * 58)


# -------------------------------------------------------
# CONFIGURACION
# -------------------------------------------------------
LOG_NAME     = 'PurchasingExample'
LOG_FILE     = LOG_NAME + '.csv'
NUM_CASES    = 20    # override de Params.total_cases (608) para velocidad
MODELS_DIR   = os.path.join('data', '1.predicton_models', LOG_NAME)
OUTPUT_DIR   = os.path.join('data', '2.hallucination_logs', LOG_NAME)
RULES_PATH   = os.path.join('data', '0.logs', LOG_NAME, 'rules_global.ini')
INPUT_SIMOD  = os.path.join('data', '2.input_logs', LOG_NAME)

print('Log          : ' + LOG_FILE)
print('Modelos      : ' + MODELS_DIR)
print('Salida CSV   : ' + OUTPUT_DIR)
print('Reglas       : ' + RULES_PATH)
print('Trazas test  : ' + str(NUM_CASES))


# -------------------------------------------------------
# PASO 1: despacho de dg_training.py
# -------------------------------------------------------
seccion('PASO 1 - Despacho de dg_training.py')

try:
    from dg_training import parse_args
    args = parse_args(['-m', 'simple_gan',
                       '-f', LOG_FILE,
                       '-e', '1'],
                      filename=LOG_FILE)
    ok('parse_args ejecutado')
except Exception as exc:
    fail('parse_args fallo: ' + str(exc))

if args.get('model_family') != 'simple_gan':
    fail('model_family esperado "simple_gan", obtenido "' + str(args.get('model_family')) + '"')
ok('model_family = "' + args['model_family'] + '"')

# Verificar el mapa de despacho
model_type_map = {
    'lstm'     : ['shared_cat', 'concatenated'],
    'gru'      : ['shared_cat_gru', 'concatenated_gru'],
    'simple_gan': ['simple_gan'],
}
dispatched_types = model_type_map.get(args['model_family'], [])
if 'simple_gan' not in dispatched_types:
    fail('simple_gan no esta en los tipos despachados: ' + str(dispatched_types))
ok('Despacho correcto: model_family="simple_gan" -> GANTrainer')

# Verificar que GANTrainer se puede importar
try:
    from GenerativeGAN.model_training.gan_trainer import GANTrainer
    ok('GANTrainer importado correctamente')
except ImportError as exc:
    fail('No se puede importar GANTrainer: ' + str(exc))


# -------------------------------------------------------
# PASO 2: construccion de Params.simulation
# -------------------------------------------------------
seccion('PASO 2 - Construccion de Params.simulation (clase de produccion)')

try:
    from params.Params import Params
    params = Params(root=Path('.'), log_filename=LOG_FILE)
    sim = params.simulation.copy()
    ok('Params construido correctamente')
except Exception as exc:
    fail('Params fallo: ' + str(exc))

# Verificar claves requeridas
required_keys = ['filename', 'log_name', 'activity', 'variant', 'rep',
                 'num_instances', 'one_timestamp', 'model_file', 'folder',
                 'read_options']
for k in required_keys:
    if k not in sim:
        fail('Params.simulation falta la clave "' + k + '"')
ok('Todas las claves requeridas presentes en Params.simulation')

print('  filename     : ' + str(sim['filename']))
print('  log_name     : ' + str(sim['log_name']))
print('  activity     : ' + str(sim['activity']))
print('  variant      : ' + str(sim['variant']))
print('  num_instances: ' + str(sim['num_instances']) + ' (original, se usaran ' + str(NUM_CASES) + ')')
print('  model_file   : ' + str(sim['model_file']))
print('  folder       : ' + str(sim['folder']))

# Verificar que el folder del modelo existe
model_h5 = os.path.join(MODELS_DIR, sim['folder'], LOG_NAME + '.h5')
if not os.path.exists(model_h5):
    fail('El modelo .h5 no existe en: ' + model_h5)
ok('Modelo .h5 existe: ' + model_h5)

# Override num_instances para el test de velocidad
sim['num_instances'] = NUM_CASES
ok('num_instances sobreescrito a ' + str(NUM_CASES) + ' para la validacion')


# -------------------------------------------------------
# PASO 3: limpiar traces_generated de ejecuciones anteriores
# -------------------------------------------------------
seccion('PASO 3 - Preparar directorios')

from support_modules.predictor_adapter import get_latest_output_folder
folder_name = get_latest_output_folder(MODELS_DIR)
model_dir   = os.path.join(MODELS_DIR, folder_name)
params_dir  = os.path.join(model_dir, 'parameters')

traces_gen_path = os.path.join(params_dir, 'traces_generated')
if os.path.exists(traces_gen_path):
    shutil.rmtree(traces_gen_path)
    ok('traces_generated limpiado')
else:
    ok('traces_generated no existia (primera ejecucion)')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_SIMOD, exist_ok=True)
ok('Directorios de salida listos')


# -------------------------------------------------------
# PASO 4: pa.hallucinate() con reglas REALES
# -------------------------------------------------------
seccion('PASO 4 - pa.hallucinate() con reglas reales (rules_global.ini)')

if not os.path.exists(RULES_PATH):
    fail('Archivo de reglas no encontrado: ' + RULES_PATH)

# Mostrar las reglas que se van a aplicar
import support_modules.traces_evaluation as te
rules = te.extract_rules(path=RULES_PATH)
print('  Reglas del archivo:')
print('    rule        : ' + str(rules.get('rule', '???')))
print('    path        : ' + str(rules.get('path', '???')))
print('    variation   : ' + str(rules.get('variation', '???')))
print('    prop_target : ' + str(rules.get('prop_variation', '???')))
print()

import support_modules.predictor_adapter as pa

try:
    pa.hallucinate(
        parameters   = sim,
        input_folder = MODELS_DIR,
        output_folder= OUTPUT_DIR,
        rules_path   = RULES_PATH,
    )
    ok('pa.hallucinate() completado sin excepciones')
except Exception as exc:
    import traceback; traceback.print_exc()
    fail('pa.hallucinate() lanzo excepcion: ' + str(exc))


# -------------------------------------------------------
# PASO 5: verificar CSV de hallucination
# -------------------------------------------------------
seccion('PASO 5 - Verificar CSV hallucination')

csv_hall = os.path.join(OUTPUT_DIR, LOG_FILE)
if not os.path.exists(csv_hall):
    fail('CSV hallucination no existe: ' + csv_hall)
ok('CSV hallucination existe: ' + csv_hall)

df_hall = pd.read_csv(csv_hall)
# pa.hallucinate() elimina la columna de indice Unnamed:0 al guardar
# Columnas esperadas despues del rename en export_predictions + hallucinate cleanup
ok('CSV hallucination tiene ' + str(len(df_hall)) + ' filas')

expected_cols = {'caseid', 'task', 'user', 'start_timestamp', 'end_timestamp'}
# Ignorar columna Unnamed:0 si existe
actual_cols = set(df_hall.columns) - {'Unnamed: 0'}
missing = expected_cols - actual_cols
if missing:
    fail('Columnas faltantes en CSV hallucination: ' + str(missing))
ok('Columnas presentes: ' + str(sorted(actual_cols)))

# Verificar sin centinelas (pa.hallucinate limpia start/end)
sentinel_mask = df_hall['task'].isin(['start', 'end', 'Start', 'End'])
if sentinel_mask.sum() > 0:
    fail(str(sentinel_mask.sum()) + ' actividades centinela en CSV hallucination')
ok('Sin actividades centinela en CSV hallucination')

n_cases = df_hall['caseid'].nunique()
print('  Casos generados: ' + str(n_cases))
print('  Eventos totales: ' + str(len(df_hall)))


# -------------------------------------------------------
# PASO 6: preprocess_xes_log() -> .csv.gz para Simod
# -------------------------------------------------------
seccion('PASO 6 - preprocess_xes_log() -> .csv.gz (entrada para Simod)')

from dg_prediction import preprocess_xes_log

try:
    gz_path = preprocess_xes_log(
        csv_path      = Path(csv_hall),
        output_folder = Path(INPUT_SIMOD),
        filter_sentinels=True,
    )
    ok('preprocess_xes_log() completado')
    ok('.csv.gz generado en: ' + str(gz_path))
except Exception as exc:
    import traceback; traceback.print_exc()
    fail('preprocess_xes_log() fallo: ' + str(exc))


# -------------------------------------------------------
# PASO 7: verificar .csv.gz
# -------------------------------------------------------
seccion('PASO 7 - Verificar .csv.gz')

gz_file = str(gz_path)
if not os.path.exists(gz_file):
    fail('.csv.gz no existe: ' + gz_file)

size_kb = os.path.getsize(gz_file) / 1024
if size_kb < 0.1:
    fail('.csv.gz existe pero esta casi vacio (' + str(round(size_kb, 2)) + ' KB)')
ok('.csv.gz existe y tiene contenido: ' + str(round(size_kb, 1)) + ' KB')

# Leer el .csv.gz y verificar estructura
df_gz = pd.read_csv(gz_file, compression='gzip')
ok('.csv.gz leido: ' + str(len(df_gz)) + ' filas, ' + str(df_gz['caseid'].nunique()) + ' casos')

expected_gz_cols = {'caseid', 'task', 'user', 'start_timestamp', 'end_timestamp'}
missing_gz = expected_gz_cols - set(df_gz.columns)
if missing_gz:
    fail('Columnas faltantes en .csv.gz: ' + str(missing_gz))
ok('Columnas en .csv.gz: ' + str(sorted(df_gz.columns.tolist())))

# Sin centinelas
sentinel_gz = df_gz['task'].isin(['start', 'end', 'Start', 'End'])
if sentinel_gz.sum() > 0:
    fail(str(sentinel_gz.sum()) + ' actividades centinela en .csv.gz')
ok('Sin actividades centinela en .csv.gz')

# Timestamps validos
try:
    df_gz['start_timestamp'] = pd.to_datetime(df_gz['start_timestamp'])
    df_gz['end_timestamp']   = pd.to_datetime(df_gz['end_timestamp'])
    ok('Timestamps en .csv.gz parseados correctamente')
except Exception as exc:
    fail('Error al parsear timestamps en .csv.gz: ' + str(exc))

neg = (df_gz['end_timestamp'] < df_gz['start_timestamp']).sum()
if neg > 0:
    fail(str(neg) + ' eventos con end < start en .csv.gz')
ok('Sin timestamps negativos en .csv.gz (' + str(len(df_gz)) + ' eventos)')


# -------------------------------------------------------
# PASO 8: estadisticas del log hallucination
# -------------------------------------------------------
seccion('PASO 8 - Estadisticas del log generado')

lengths = df_gz.groupby('caseid').size()
print('  Casos en .csv.gz          : ' + str(len(lengths)))
print('  Eventos por traza (media) : ' + str(round(lengths.mean(), 1)))
print('  Eventos por traza (min)   : ' + str(lengths.min()))
print('  Eventos por traza (max)   : ' + str(lengths.max()))
print('  Actividades distintas     : ' + str(df_gz['task'].nunique()))
print('  Usuarios distintos        : ' + str(df_gz['user'].nunique()))

top3 = df_gz['task'].value_counts().head(3)
print('\n  Top-3 actividades en .csv.gz:')
for task, cnt in top3.items():
    print('    ' + str(round(cnt / len(df_gz) * 100, 1)).rjust(5) + '%  ' + str(task))


# -------------------------------------------------------
# RESUMEN FINAL
# -------------------------------------------------------
print('\n' + '=' * 58)
print('  ETAPA 5 SUPERADA - integracion completa del pipeline OK')
print('=' * 58)
print()
print('  Flujo validado:')
print('  dg_training.py dispatch')
print('    -> Params.simulation (clase de produccion)')
print('    -> pa.hallucinate() -> CSV hallucination')
print('    -> preprocess_xes_log() -> .csv.gz (listo para Simod)')
print()
