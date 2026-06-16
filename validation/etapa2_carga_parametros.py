# -*- coding: utf-8 -*-
"""
ETAPA 2 - Carga de parametros por ModelPredictor
=================================================
Ejecutar desde la raiz del proyecto:
    python validation/etapa2_carga_parametros.py

Que verifica:
  1. ModelPredictor.load_parameters() lee el JSON del GAN sin errores
  2. parms['model_type'] == 'simple_gan' despues de la carga
  3. parms['index_ac'] tiene claves enteras (conversion int(k) ok)
  4. parms['scale_args'] tiene 'dur' y 'wait' con valores float
  5. parms['model_file'] apunta a un .h5 que existe en disco
  6. read_model_definition('simple_gan') lee models_spec.ini correctamente
  7. interfaces.py despacha GANPredictor para model_type='simple_gan'
     y EventLogPredictor para cualquier otro tipo
"""

import os
import sys
import json
from unittest.mock import patch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def ok(msg):
    print('  [OK]   ' + msg)


def fail(msg):
    print('  [FAIL] ' + str(msg))
    import traceback; traceback.print_exc()
    sys.exit(1)


def seccion(titulo):
    print('\n' + '-' * 55)
    print('  ' + titulo)
    print('-' * 55)


# -------------------------------------------------------
# LOCALIZACION DEL MODELO ENTRENADO EN ETAPA 1
# -------------------------------------------------------
LOG_NAME   = 'PurchasingExample'
MODELS_DIR = os.path.join('data', '1.predicton_models', LOG_NAME)

from support_modules.predictor_adapter import get_latest_output_folder
folder_name = get_latest_output_folder(MODELS_DIR)
if folder_name is None:
    print('[FAIL] No hay modelo entrenado en ' + MODELS_DIR)
    print('       Ejecuta primero: python validation/etapa1_smoke_test.py')
    sys.exit(1)

INPUT_FOLDER  = MODELS_DIR          # carpeta raiz de modelos del log
MODEL_SUBDIR  = folder_name         # subcarpeta con la fecha/uuid
MODEL_DIR     = os.path.join(MODELS_DIR, folder_name)
PARAMS_DIR    = os.path.join(MODEL_DIR, 'parameters')
JSON_PATH     = os.path.join(PARAMS_DIR, 'model_parameters.json')

print('Modelo encontrado: ' + MODEL_DIR)

# Parametros de prediccion (simulan Params.simulation para PurchasingExample)
PRED_PARMS = {
    'filename'       : LOG_NAME + '.csv',
    'log_name'       : LOG_NAME,
    'activity'       : 'pred_log',
    'variant'        : 'Random Choice',
    'rep'            : 1,
    'num_instances'  : 50,
    'is_single_exec' : False,
    'one_timestamp'  : False,
    'include_org_log': False,
    'model_file'     : LOG_NAME + '.h5',   # sera sobreescrito por load_parameters
    'folder'         : MODEL_SUBDIR,
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
# PASO 1: load_parameters() via ModelPredictor
# -------------------------------------------------------
seccion('PASO 1 - ModelPredictor.load_parameters()')

from GenerativeLSTM.model_prediction.model_predictor import ModelPredictor

# Usamos mock para que __init__ NO ejecute la generacion de trazas.
# Solo queremos correr load_parameters() y read_model_definition().
try:
    with patch.object(ModelPredictor, 'execute_predictive_task', return_value=None):
        predictor = ModelPredictor(
            parms       = PRED_PARMS,
            input_folder= INPUT_FOLDER,
            output_folder= os.path.join('data', '2.hallucination_logs', LOG_NAME),
            rules_path  = os.path.join('data', '0.logs', LOG_NAME, 'rules_global.ini'),
        )
    ok('ModelPredictor.__init__ completo (execute_predictive_task omitido)')
except Exception as exc:
    fail('ModelPredictor.__init__ lanzo excepcion: ' + str(exc))


# -------------------------------------------------------
# PASO 2: verificar parms['model_type']
# -------------------------------------------------------
seccion('PASO 2 - Verificar model_type en parms')

mt = predictor.parms.get('model_type')
if mt != 'simple_gan':
    fail('model_type esperado "simple_gan", obtenido "' + str(mt) + '"')
ok('parms["model_type"] = "' + mt + '"')


# -------------------------------------------------------
# PASO 3: verificar index_ac y index_rl (claves enteras)
# -------------------------------------------------------
seccion('PASO 3 - Verificar index_ac / index_rl')

index_ac = predictor.parms.get('index_ac', {})
index_rl = predictor.parms.get('index_rl', {})

if not index_ac:
    fail('index_ac esta vacio')
if not all(isinstance(k, int) for k in index_ac.keys()):
    fail('index_ac no tiene claves enteras: ' + str(list(index_ac.keys())[:5]))
ok('index_ac: ' + str(len(index_ac)) + ' entradas con claves int')

if not index_rl:
    fail('index_rl esta vacio')
if not all(isinstance(k, int) for k in index_rl.keys()):
    fail('index_rl no tiene claves enteras')
ok('index_rl: ' + str(len(index_rl)) + ' entradas con claves int')

# Mostrar muestra de actividades
sample_ac = [v for k, v in sorted(index_ac.items())[:5]]
print('  Primeras 5 actividades: ' + str(sample_ac))


# -------------------------------------------------------
# PASO 4: verificar scale_args
# -------------------------------------------------------
seccion('PASO 4 - Verificar scale_args')

sa = predictor.parms.get('scale_args', {})
if not sa:
    fail('scale_args esta vacio')

for feat in ('dur', 'wait'):
    if feat not in sa:
        fail('scale_args no tiene la clave "' + feat + '"')
    for subk, subv in sa[feat].items():
        if not isinstance(subv, (int, float)):
            fail('scale_args["' + feat + '"]["' + subk + '"] no es numerico: ' + str(subv))
ok('scale_args["dur"]  = ' + str(sa['dur']))
ok('scale_args["wait"] = ' + str(sa['wait']))


# -------------------------------------------------------
# PASO 5: verificar que model_file apunta a un .h5 real
# -------------------------------------------------------
seccion('PASO 5 - Verificar ruta del modelo')

model_file = predictor.parms.get('model_file', '')
model_path = predictor.model_name   # ruta completa calculada por ModelPredictor
if not os.path.exists(model_path):
    fail('El .h5 no existe en la ruta: ' + model_path)
ok('model_file = "' + model_file + '"')
ok('Ruta completa existe: ' + model_path)


# -------------------------------------------------------
# PASO 6: read_model_definition para 'simple_gan'
# -------------------------------------------------------
seccion('PASO 6 - read_model_definition("simple_gan")')

model_def = predictor.model_def
vectorizer        = model_def.get('vectorizer', 'NO_ENCONTRADO')
additional_cols   = model_def.get('additional_columns', 'NO_ENCONTRADO')

if vectorizer == 'NO_ENCONTRADO':
    fail('model_def no tiene "vectorizer"')
if additional_cols == 'NO_ENCONTRADO':
    fail('model_def no tiene "additional_columns"')

ok('vectorizer         = "' + str(vectorizer) + '"')
ok('additional_columns = ' + str(additional_cols))


# -------------------------------------------------------
# PASO 7: despacho correcto en interfaces.py
# -------------------------------------------------------
seccion('PASO 7 - Despacho en interfaces.py')

from GenerativeLSTM.model_prediction.interfaces import PredictionTasksExecutioner
from GenerativeLSTM.model_prediction.event_log_predictor import EventLogPredictor
from GenerativeGAN.model_prediction.gan_predictor import GANPredictor

exec_ = PredictionTasksExecutioner()

pred_gan  = exec_._get_predictor('pred_log', model_type='simple_gan')
pred_lstm = exec_._get_predictor('pred_log', model_type='shared_cat')
pred_none = exec_._get_predictor('pred_log', model_type=None)

if not isinstance(pred_gan, GANPredictor):
    fail('simple_gan deberia despachar GANPredictor, obtuvo ' + type(pred_gan).__name__)
ok('model_type="simple_gan"  ->  ' + type(pred_gan).__name__)

if not isinstance(pred_lstm, EventLogPredictor):
    fail('shared_cat deberia despachar EventLogPredictor, obtuvo ' + type(pred_lstm).__name__)
ok('model_type="shared_cat"  ->  ' + type(pred_lstm).__name__)

if not isinstance(pred_none, EventLogPredictor):
    fail('model_type=None deberia despachar EventLogPredictor, obtuvo ' + type(pred_none).__name__)
ok('model_type=None          ->  ' + type(pred_none).__name__)


# -------------------------------------------------------
# RESUMEN FINAL
# -------------------------------------------------------
print('\n' + '=' * 55)
print('  ETAPA 2 SUPERADA - todos los checks pasaron')
print('=' * 55 + '\n')
