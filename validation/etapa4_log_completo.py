# -*- coding: utf-8 -*-
"""
ETAPA 4 - Generacion del log completo (sin reglas declarativas reales)
=======================================================================
Ejecutar desde la raiz del proyecto:
    python validation/etapa4_log_completo.py

Estrategia:
  Las reglas declarativas se neutralizan con mocks para aislar la generacion
  pura del GAN.  El resto del pipeline corre real:
    - GANPredictor._generate_traces()
    - ModelPredictor.export_predictions()   (escribe CSV)
    - XesWriter                             (escribe XES)
    - get_stats_log_traces                  (lee archivos reales, controla el loop)

Que verifica:
  1. El loop de generacion termina correctamente (no se cuelga)
  2. El CSV final existe con las columnas correctas
  3. El XES final existe y no esta vacio
  4. El CSV no contiene actividades centinela ('start', 'end')
  5. El numero de casos generados esta dentro del rango esperado
  6. Todos los timestamps son parseable y start <= end
"""

import os
import sys
import json
import shutil
from unittest.mock import patch, MagicMock

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd


def ok(msg):
    print('  [OK]   ' + msg)


def fail(msg):
    print('  [FAIL] ' + msg)
    import traceback; traceback.print_exc()
    sys.exit(1)


def warn(msg):
    print('  [WARN] ' + msg)


def seccion(titulo):
    print('\n' + '-' * 55)
    print('  ' + titulo)
    print('-' * 55)


# -------------------------------------------------------
# CONFIGURACION
# -------------------------------------------------------
LOG_NAME     = 'PurchasingExample'
MODELS_DIR   = os.path.join('data', '1.predicton_models', LOG_NAME)
OUTPUT_DIR   = os.path.join('data', '2.hallucination_logs', LOG_NAME)
NUM_CASES    = 20     # trazas a generar en este test

from support_modules.predictor_adapter import get_latest_output_folder
folder_name = get_latest_output_folder(MODELS_DIR)
if folder_name is None:
    print('[FAIL] No hay modelo entrenado. Ejecuta primero etapa1_smoke_test.py')
    sys.exit(1)

MODEL_DIR  = os.path.join(MODELS_DIR, folder_name)
PARAMS_DIR = os.path.join(MODEL_DIR, 'parameters')

print('Usando modelo   : ' + MODEL_DIR)
print('Salida generada : ' + OUTPUT_DIR)
print('Trazas objetivo : ' + str(NUM_CASES))

# -------------------------------------------------------
# LIMPIAR RESTOS DE EJECUCIONES ANTERIORES
# -------------------------------------------------------
traces_gen_path = os.path.join(PARAMS_DIR, 'traces_generated')
if os.path.exists(traces_gen_path):
    shutil.rmtree(traces_gen_path)
    print('Limpiado traces_generated de ejecucion anterior')

# -------------------------------------------------------
# PARAMETROS DE PREDICCION
# -------------------------------------------------------
PRED_PARMS = {
    'filename'       : LOG_NAME + '.csv',
    'log_name'       : LOG_NAME,
    'activity'       : 'pred_log',
    'variant'        : 'Random Choice',
    'rep'            : 1,
    'num_instances'  : NUM_CASES,
    'is_single_exec' : False,
    'one_timestamp'  : False,
    'include_org_log': False,
    'model_file'     : LOG_NAME + '.h5',
    'folder'         : folder_name,
    'read_options'   : {
        'timeformat'      : '%Y-%m-%d %H:%M:%S',
        'one_timestamp'   : False,
        'filter_d_attrib' : False,
        'column_names'    : {
            'Case ID'              : 'caseid',
            'Activity'             : 'task',
            'lifecycle:transition' : 'event_type',
            'Resource'             : 'user',
        },
    },
}

# -------------------------------------------------------
# MOCKS — solo neutralizan la evaluacion de reglas
# -------------------------------------------------------
FAKE_RULES = {
    'rule'           : 'required',
    'path'           : ['Create Purchase Requisition'],
    'variation'      : '+',
    'prop_variation' : 0.5,
}

def fake_extract_rules(path, **kwargs):
    return FAKE_RULES

def fake_evaluate_condition(df, ac_index, path, rule):
    # Todas las trazas "cumplen" la regla => todos se aceptan
    return True

class FakeGenerateStats:
    def __init__(self, *args, **kwargs):
        pass
    def get_stats(self):
        # 0 de 100 trazas originales cumplen la regla => current_prop empieza en 0
        return (0, 100)


# -------------------------------------------------------
# PASO 1: ejecutar ModelPredictor con mocks activos
# -------------------------------------------------------
seccion('PASO 1 - Ejecutar pipeline completo (con reglas neutralizadas)')

from GenerativeLSTM.model_prediction.model_predictor import ModelPredictor
import support_modules.traces_evaluation as te

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    with patch.object(te, 'extract_rules',       fake_extract_rules),  \
         patch.object(te, 'evaluate_condition',  fake_evaluate_condition), \
         patch.object(te, 'GenerateStats',        FakeGenerateStats):

        predictor = ModelPredictor(
            parms        = PRED_PARMS,
            input_folder = MODELS_DIR,
            output_folder= OUTPUT_DIR,
            rules_path   = 'dummy_rules.ini',
        )
    ok('Pipeline completo ejecutado sin excepciones')

except Exception as exc:
    import traceback; traceback.print_exc()
    fail('Pipeline lanzo excepcion: ' + str(exc))


# -------------------------------------------------------
# PASO 2: verificar el CSV generado
# -------------------------------------------------------
seccion('PASO 2 - Verificar CSV generado')

csv_path = os.path.join(OUTPUT_DIR, LOG_NAME + '.csv')
if not os.path.exists(csv_path):
    fail('CSV no existe: ' + csv_path)
ok('CSV existe: ' + csv_path)

df = pd.read_csv(csv_path)
ok('CSV leido: ' + str(len(df)) + ' filas')

expected_cols = {'caseid', 'task', 'user', 'start_timestamp', 'end_timestamp'}
missing = expected_cols - set(df.columns)
if missing:
    fail('Columnas faltantes en el CSV: ' + str(missing))
ok('Columnas correctas: ' + str(sorted(df.columns.tolist())))


# -------------------------------------------------------
# PASO 3: verificar que no hay actividades centinela
# -------------------------------------------------------
seccion('PASO 3 - Sin actividades centinela (start / end)')

sentinel_mask = df['task'].isin(['start', 'end', 'Start', 'End'])
sentinel_count = sentinel_mask.sum()
if sentinel_count > 0:
    fail(str(sentinel_count) + ' filas con actividad centinela encontradas')
ok('Sin actividades centinela en el CSV final')


# -------------------------------------------------------
# PASO 4: verificar numero de casos generados
# -------------------------------------------------------
seccion('PASO 4 - Numero de casos generados')

num_generated = df['caseid'].nunique()
print('  Casos en el CSV: ' + str(num_generated))
print('  Objetivo        : ' + str(NUM_CASES))

# Con reglas neutralizadas, el numero exacto puede variar ligeramente
# porque el loop usa len(files_gen) >= len_log para terminar.
if num_generated == 0:
    fail('El CSV no contiene ningun caso')
elif num_generated < NUM_CASES * 0.5:
    fail('Se generaron muy pocos casos: ' + str(num_generated) +
         ' (esperados ~' + str(NUM_CASES) + ')')
ok(str(num_generated) + ' casos unicos generados')


# -------------------------------------------------------
# PASO 5: verificar integridad de timestamps
# -------------------------------------------------------
seccion('PASO 5 - Integridad de timestamps')

try:
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
    df['end_timestamp']   = pd.to_datetime(df['end_timestamp'])
    ok('Timestamps parseados correctamente')
except Exception as exc:
    fail('Error al parsear timestamps: ' + str(exc))

neg = (df['end_timestamp'] < df['start_timestamp']).sum()
if neg > 0:
    fail(str(neg) + ' eventos con end_timestamp < start_timestamp')
ok('Sin timestamps negativos (' + str(len(df)) + ' eventos validados)')


# -------------------------------------------------------
# PASO 6: verificar el XES generado
# -------------------------------------------------------
seccion('PASO 6 - Verificar XES generado')

xes_path = os.path.join(OUTPUT_DIR, LOG_NAME + '.xes')
if not os.path.exists(xes_path):
    warn('XES no encontrado en: ' + xes_path)
    # Buscar en subdirectorios
    for root_dir, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f.endswith('.xes'):
                xes_path = os.path.join(root_dir, f)
                warn('XES encontrado en: ' + xes_path)
                break

if os.path.exists(xes_path):
    size_kb = os.path.getsize(xes_path) / 1024
    if size_kb < 0.1:
        fail('XES existe pero esta practicamente vacio (' +
             str(round(size_kb, 2)) + ' KB)')
    ok('XES existe y tiene contenido: ' + str(round(size_kb, 1)) + ' KB')
else:
    fail('XES no encontrado en ninguna ubicacion bajo: ' + OUTPUT_DIR)


# -------------------------------------------------------
# PASO 7: estadisticas rapidas del log generado
# -------------------------------------------------------
seccion('PASO 7 - Estadisticas del log generado')

lengths = df.groupby('caseid').size()
print('  Casos generados          : ' + str(len(lengths)))
print('  Eventos por traza (media): ' + str(round(lengths.mean(), 1)))
print('  Eventos por traza (min)  : ' + str(lengths.min()))
print('  Eventos por traza (max)  : ' + str(lengths.max()))
print('  Actividades distintas    : ' + str(df['task'].nunique()))
print('  Usuarios distintos       : ' + str(df['user'].nunique()))

top3 = df['task'].value_counts().head(3)
print('\n  Top-3 actividades:')
for task, cnt in top3.items():
    print('    ' + str(round(cnt/len(df)*100, 1)).rjust(5) + '%  ' + str(task))


# -------------------------------------------------------
# RESUMEN FINAL
# -------------------------------------------------------
print('\n' + '=' * 55)
print('  ETAPA 4 SUPERADA - todos los checks pasaron')
print('=' * 55 + '\n')
