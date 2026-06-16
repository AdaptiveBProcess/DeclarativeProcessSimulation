# -*- coding: utf-8 -*-
"""
ETAPA 1 - Smoke test del loop de entrenamiento GAN
===================================================
Ejecutar desde la raiz del proyecto:
    python validation/etapa1_smoke_test.py

Que verifica:
  1. GANTrainer completa sin excepciones con epochs=2
  2. El modelo .h5 existe y carga con Keras
  3. model_parameters.json existe y tiene todas las claves requeridas
  4. scale_args tiene la estructura correcta (dur / wait)
  5. test_log.csv y <NAME>_ASIS.csv existen y no estan vacios
"""

import os
import sys
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def ok(msg):
    print('  [OK]   ' + msg)


def fail(msg):
    print('  [FAIL] ' + msg)
    sys.exit(1)


def seccion(titulo):
    print('\n' + '-' * 55)
    print('  ' + titulo)
    print('-' * 55)


# -------------------------------------------------------
# CONFIGURACION DEL TEST
# -------------------------------------------------------
LOG_NAME   = 'PurchasingExample'
LOG_FILE   = LOG_NAME + '.csv'
INPUT_DIR  = os.path.join('data', '0.logs', LOG_NAME + '_preprocessed')
OUTPUT_DIR = os.path.join('data', '1.predicton_models', LOG_NAME)

PARAMS = {
    'file_name': LOG_FILE,
    'rp_sim': 0.85,
    'batch_size': 16,
    'norm_method': ['max', 'lognorm'],
    'epochs': 2,
    'latent_dim': 32,
    'n_size': [5],
    'read_options': {
        'timeformat': '%Y-%m-%d %H:%M:%S',
        'column_names': {
            'Case ID':              'caseid',
            'Activity':             'task',
            'lifecycle:transition': 'event_type',
            'Resource':             'user',
        },
        'one_timestamp': False,
    },
}

REQUIRED_JSON_KEYS = [
    'model_type', 'model_file', 'index_ac', 'index_rl',
    'scale_args', 'norm_method', 'max_trace_size',
    'one_timestamp', 'latent_dim',
]


# -------------------------------------------------------
# PASO 1: ejecutar el entrenamiento
# -------------------------------------------------------
seccion('PASO 1 - Ejecutar GANTrainer (epochs=2)')

from GenerativeGAN.model_training.gan_trainer import GANTrainer

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer = GANTrainer(PARAMS, input_folder=INPUT_DIR, output_folder=OUTPUT_DIR)
    ok('GANTrainer completo sin excepciones')
except Exception as exc:
    import traceback
    traceback.print_exc()
    fail('GANTrainer lanzo excepcion: ' + str(exc))


# -------------------------------------------------------
# PASO 2: localizar la carpeta de salida
# -------------------------------------------------------
seccion('PASO 2 - Localizar carpeta de salida')

subfolders = [
    d for d in os.listdir(OUTPUT_DIR)
    if os.path.isdir(os.path.join(OUTPUT_DIR, d))
]
if not subfolders:
    fail('No se encontro ninguna subcarpeta en ' + OUTPUT_DIR)

latest     = max(subfolders,
                 key=lambda d: os.path.getmtime(os.path.join(OUTPUT_DIR, d)))
model_dir  = os.path.join(OUTPUT_DIR, latest)
params_dir = os.path.join(model_dir, 'parameters')
ok('Carpeta creada: ' + model_dir)


# -------------------------------------------------------
# PASO 3: verificar el modelo .h5
# -------------------------------------------------------
seccion('PASO 3 - Verificar modelo guardado')

model_h5 = os.path.join(model_dir, LOG_NAME + '.h5')
if not os.path.exists(model_h5):
    fail('No existe el archivo del modelo: ' + model_h5)
ok('Archivo del modelo existe: ' + os.path.basename(model_h5))

try:
    from tensorflow.keras.models import load_model
    gen = load_model(model_h5)
    ok('Modelo carga correctamente con Keras  input=' + str(gen.input_shape))
except Exception as exc:
    fail('No se pudo cargar el modelo: ' + str(exc))


# -------------------------------------------------------
# PASO 4: verificar model_parameters.json
# -------------------------------------------------------
seccion('PASO 4 - Verificar model_parameters.json')

json_path = os.path.join(params_dir, 'model_parameters.json')
if not os.path.exists(json_path):
    fail('No existe: ' + json_path)
ok('model_parameters.json existe')

with open(json_path) as f:
    mp = json.load(f)

for key in REQUIRED_JSON_KEYS:
    if key not in mp:
        fail('Falta la clave requerida: "' + key + '"')
ok('Todas las claves requeridas presentes')

if mp['model_type'] != 'simple_gan':
    fail('model_type deberia ser "simple_gan", es "' + str(mp['model_type']) + '"')
ok('model_type = "' + mp['model_type'] + '"')

sa = mp['scale_args']
for feat in ('dur', 'wait'):
    if feat not in sa:
        fail('scale_args no tiene la clave "' + feat + '"')
ok('scale_args tiene claves: ' + str(list(sa.keys())))

index_ac = {int(k): v for k, v in mp['index_ac'].items()}
if 0 not in index_ac:
    fail('index_ac no tiene la clave 0 (start)')
ok('index_ac: ' + str(len(index_ac)) + ' actividades (incluye start/end)')

index_rl = {int(k): v for k, v in mp['index_rl'].items()}
ok('index_rl: ' + str(len(index_rl)) + ' roles')

print('')
print('  max_trace_size : ' + str(mp['max_trace_size']))
print('  norm_method    : ' + str(mp['norm_method']))
print('  latent_dim     : ' + str(mp['latent_dim']))
print('  one_timestamp  : ' + str(mp['one_timestamp']))


# -------------------------------------------------------
# PASO 5: verificar archivos de particion
# -------------------------------------------------------
seccion('PASO 5 - Verificar archivos de particion (train/test split)')

import pandas as pd

test_csv = os.path.join(params_dir, 'test_log.csv')
asis_csv = os.path.join(params_dir, LOG_NAME + '_ASIS.csv')

for path, label in [(test_csv, 'test_log.csv'),
                    (asis_csv, LOG_NAME + '_ASIS.csv')]:
    if not os.path.exists(path):
        fail('No existe: ' + label)
    df = pd.read_csv(path)
    if len(df) == 0:
        fail(label + ' esta vacio')
    ok(label + ': ' + str(len(df)) + ' filas, ' +
       str(df['caseid'].nunique()) + ' casos unicos')


# -------------------------------------------------------
# RESUMEN FINAL
# -------------------------------------------------------
print('\n' + '=' * 55)
print('  ETAPA 1 SUPERADA - todos los checks pasaron')
print('=' * 55 + '\n')
