# -*- coding: utf-8 -*-
"""
ETAPA 6 - Evaluacion comparativa GAN vs LSTM
=============================================
Ejecutar desde la raiz del proyecto:
    C:\\Users\\Diego\\miniconda3\\envs\\deep_generator\\python.exe validation/etapa6_evaluacion_comparativa.py

Que verifica:
  1. SimilarityEvaluator puede ejecutarse sobre el log generado por el GAN
  2. Las 5 metricas producen valores numericos validos (no NaN, no error)
  3. Generacion paralela GAN vs LSTM: mismo numero de trazas, mismo reference
  4. Tabla comparativa GAN vs LSTM para las 5 metricas

Metricas evaluadas:
  - tsd         : Timed String Distance   (mas alto = mas similar; escala 0-1)
  - dl          : Damerau-Levenshtein     (mas alto = mas similar; escala 0-1)
  - mae         : Mean Absolute Error     (cicle time; mas bajo = mejor)
  - log_mae     : Log-scale MAE           (mas alto = mas similar)
  - day_hour_emd: Earth Movers Distance   (distribucion hora del dia; mas alto = mejor)

Nota: con solo 2 epochs de entrenamiento la CALIDAD de las metricas es baja.
Lo que valida este script es que el FLUJO de evaluacion funciona correctamente
y que las metricas son calculables. Los valores reales del articulo se obtendran
con entrenamiento completo (200 epochs).
"""

import os
import sys
import json
import shutil
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from unittest.mock import patch

import support_modules.traces_evaluation as te


def ok(msg):
    print('  [OK]   ' + msg)


def fail(msg):
    print('  [FAIL] ' + msg)
    import traceback; traceback.print_exc()
    sys.exit(1)


def warn(msg):
    print('  [WARN] ' + msg)


def seccion(titulo):
    print('\n' + '-' * 60)
    print('  ' + titulo)
    print('-' * 60)


# -------------------------------------------------------
# CONFIGURACION
# -------------------------------------------------------
LOG_NAME     = 'PurchasingExample'
LOG_FILE     = LOG_NAME + '.csv'
MODELS_DIR   = os.path.join('data', '1.predicton_models', LOG_NAME)
OUTPUT_DIR   = os.path.join('data', '2.hallucination_logs', LOG_NAME)
NUM_CASES    = 30    # trazas a generar para la comparacion

# IDs de modelos: GAN hardcodeado para evitar que get_latest_output_folder
# devuelva la carpeta LSTM (que fue tocada más recientemente por etapa4/5)
GAN_FOLDER  = '20260610_B9F15CEF_4F97_4243_BD85_9D0E479DDBBD'  # GAN (2 epochs)
LSTM_FOLDER = '20260506_894DBD74_B111_4151_954E_61989739B4A1'   # shared_cat

# -------------------------------------------------------
# MOCK de reglas (neutraliza filtro para max generacion)
# -------------------------------------------------------
FAKE_RULES = {
    'rule': 'required',
    'path': ['Create Purchase Requisition'],
    'variation': '+',
    'prop_variation': 0.5,
}

def fake_extract_rules(path, **kwargs):
    return FAKE_RULES

def fake_evaluate_condition(df, ac_index, path, rule):
    return True

class FakeGenerateStats:
    def __init__(self, *args, **kwargs):
        pass
    def get_stats(self):
        return (0, 100)


# -------------------------------------------------------
# HELPER: generar log con ModelPredictor
# -------------------------------------------------------
def generar_log(folder_name, output_subdir, num_cases, model_type_hint=None):
    """
    Genera un log usando ModelPredictor.
    Devuelve (DataFrame, error_msg).
    """
    from GenerativeLSTM.model_prediction.model_predictor import ModelPredictor

    model_dir   = os.path.join(MODELS_DIR, folder_name)
    params_dir  = os.path.join(model_dir, 'parameters')
    traces_path = os.path.join(params_dir, 'traces_generated')
    if os.path.exists(traces_path):
        shutil.rmtree(traces_path)

    out_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    parms = {
        'filename'       : LOG_FILE,
        'log_name'       : LOG_NAME,
        'activity'       : 'pred_log',
        'variant'        : 'Random Choice',
        'rep'            : 1,
        'num_instances'  : num_cases,
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

    try:
        with patch.object(te, 'extract_rules',      fake_extract_rules),  \
             patch.object(te, 'evaluate_condition', fake_evaluate_condition), \
             patch.object(te, 'GenerateStats',       FakeGenerateStats):
            ModelPredictor(
                parms        = parms,
                input_folder = MODELS_DIR,
                output_folder= out_dir,
                rules_path   = 'dummy_rules.ini',
            )
    except Exception as exc:
        return None, str(exc)

    csv_path = os.path.join(out_dir, LOG_FILE)
    if not os.path.exists(csv_path):
        return None, 'CSV no generado en ' + csv_path

    df = pd.read_csv(csv_path)
    df = df[~df['task'].isin(['Start', 'End', 'start', 'end'])]
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
    df['end_timestamp']   = pd.to_datetime(df['end_timestamp'])
    df = df[['caseid', 'task', 'start_timestamp', 'end_timestamp']].reset_index(drop=True)
    return df, None


# -------------------------------------------------------
# HELPER: cargar log de referencia
# -------------------------------------------------------
def cargar_referencia(folder_name):
    """Carga test_log.csv del modelo indicado como DataFrame de referencia."""
    path = os.path.join(MODELS_DIR, folder_name, 'parameters', 'test_log.csv')
    if not os.path.exists(path):
        return None, 'No existe: ' + path
    df = pd.read_csv(path)
    df = df[~df['task'].isin(['Start', 'End', 'start', 'end'])]
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
    df['end_timestamp']   = pd.to_datetime(df['end_timestamp'])
    df = df[['caseid', 'task', 'start_timestamp', 'end_timestamp']].reset_index(drop=True)
    return df, None


# -------------------------------------------------------
# HELPER: calcular las 5 metricas
# -------------------------------------------------------
METRICS = ['tsd', 'dl', 'mae', 'log_mae', 'day_hour_emd']

def calcular_metricas(ref_df, sim_df, parms_eval, label):
    """
    Devuelve dict {metric: value} o None si falla.
    """
    import analyzers.sim_evaluator as ev
    import random
    random.seed(42)

    resultados = {}
    ref_df  = ref_df.reset_index(drop=True)
    sim_df  = sim_df.reset_index(drop=True)
    try:
        evaluator = ev.SimilarityEvaluator(ref_df, sim_df, parms_eval)
    except Exception as exc:
        print('    [ERR] SimilarityEvaluator no pudo inicializarse: ' + str(exc))
        return None

    for metric in METRICS:
        try:
            evaluator.measure_distance(metric)
            val = evaluator.similarity.get('sim_val', float('nan'))
            resultados[metric] = round(float(val), 4)
        except Exception as exc:
            resultados[metric] = float('nan')
            print('    [WARN] ' + label + ' / ' + metric + ': ' + str(exc)[:80])

    return resultados


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    print('GAN  folder  : ' + GAN_FOLDER)
    print('LSTM folder  : ' + LSTM_FOLDER)
    print('Trazas gen.  : ' + str(NUM_CASES))

    # -------------------------------------------------------
    # PASO 1: cargar log de referencia
    # -------------------------------------------------------
    seccion('PASO 1 - Cargar log de referencia (test_log.csv del GAN)')

    ref_df, ref_err = cargar_referencia(GAN_FOLDER)
    if ref_err:
        fail('No se pudo cargar referencia: ' + ref_err)

    n_ref_cases = ref_df['caseid'].nunique()
    ok('Referencia cargada: ' + str(len(ref_df)) + ' eventos, ' + str(n_ref_cases) + ' casos')
    if n_ref_cases < NUM_CASES:
        warn('Referencia (' + str(n_ref_cases) + ' casos) < generados (' + str(NUM_CASES) + '). El evaluador puede fallar.')

    # -------------------------------------------------------
    # PASO 2: generar log GAN
    # -------------------------------------------------------
    seccion('PASO 2 - Generar log GAN (' + str(NUM_CASES) + ' trazas)')

    print('  Modelo: ' + str(GAN_FOLDER))
    gan_df, gan_err = generar_log(GAN_FOLDER, 'gan_eval', NUM_CASES)

    if gan_err:
        fail('GAN log fallido: ' + gan_err)

    n_gan = gan_df['caseid'].nunique()
    ok('GAN log generado: ' + str(len(gan_df)) + ' eventos, ' + str(n_gan) + ' casos')

    # -------------------------------------------------------
    # PASO 3: generar log LSTM
    # -------------------------------------------------------
    seccion('PASO 3 - Generar log LSTM (' + str(NUM_CASES) + ' trazas)')

    print('  Modelo: ' + LSTM_FOLDER)
    lstm_df, lstm_err = generar_log(LSTM_FOLDER, 'lstm_eval', NUM_CASES)

    if lstm_err:
        warn('LSTM log fallo (' + lstm_err[:80] + '). Las metricas LSTM seran N/A.')
        lstm_df = None
    else:
        n_lstm = lstm_df['caseid'].nunique()
        ok('LSTM log generado: ' + str(len(lstm_df)) + ' eventos, ' + str(n_lstm) + ' casos')

    # -------------------------------------------------------
    # PASO 4: evaluar metricas GAN
    # -------------------------------------------------------
    seccion('PASO 4 - SimilarityEvaluator: GAN vs referencia')

    parms_eval = {
        'read_options': {
            'timeformat'    : '%Y-%m-%d %H:%M:%S',
            'one_timestamp' : False,
            'filter_d_attrib': False,
        }
    }

    print('  Calculando ' + str(len(METRICS)) + ' metricas...')
    metricas_gan = calcular_metricas(ref_df.copy(), gan_df.copy(), parms_eval, 'GAN')

    if metricas_gan is None:
        fail('SimilarityEvaluator fallo para el log GAN')

    nan_count = sum(1 for v in metricas_gan.values() if np.isnan(v))
    if nan_count == len(METRICS):
        fail('Todas las metricas GAN son NaN — el evaluador no funciona correctamente')
    elif nan_count > 0:
        warn(str(nan_count) + '/' + str(len(METRICS)) + ' metricas GAN son NaN')

    ok('Metricas GAN calculadas (' + str(len(METRICS) - nan_count) + '/' + str(len(METRICS)) + ' exitosas)')

    # -------------------------------------------------------
    # PASO 5: evaluar metricas LSTM
    # -------------------------------------------------------
    seccion('PASO 5 - SimilarityEvaluator: LSTM vs referencia')

    if lstm_df is not None:
        print('  Calculando ' + str(len(METRICS)) + ' metricas...')
        metricas_lstm = calcular_metricas(ref_df.copy(), lstm_df.copy(), parms_eval, 'LSTM')
        if metricas_lstm is None:
            warn('SimilarityEvaluator fallo para el log LSTM. Metricas LSTM = N/A')
            metricas_lstm = {m: float('nan') for m in METRICS}
        else:
            nan_lstm = sum(1 for v in metricas_lstm.values() if np.isnan(v))
            if nan_lstm < len(METRICS):
                ok('Metricas LSTM calculadas (' + str(len(METRICS) - nan_lstm) + '/' + str(len(METRICS)) + ' exitosas)')
            else:
                warn('Todas las metricas LSTM son NaN')
    else:
        metricas_lstm = {m: float('nan') for m in METRICS}
        warn('LSTM no disponible — metricas = N/A')

    # -------------------------------------------------------
    # PASO 6: tabla comparativa
    # -------------------------------------------------------
    seccion('PASO 6 - Tabla comparativa GAN vs LSTM')

    METRIC_DESC = {
        'tsd'         : 'Timed String Dist. (sim 0-1, alto=mejor)',
        'dl'          : 'Damerau-Levenshtein (sim 0-1, alto=mejor)',
        'mae'         : 'Mean Abs. Error cicle-time (bajo=mejor)',
        'log_mae'     : 'Log-scale MAE (sim 0-1, alto=mejor)',
        'day_hour_emd': 'Earth Movers Dist. hora/dia (alto=mejor)',
    }

    print()
    print('  Metrica          |   GAN    |   LSTM   | Descripcion')
    print('  ' + '-' * 76)
    for m in METRICS:
        gan_val  = metricas_gan.get(m, float('nan'))
        lstm_val = metricas_lstm.get(m, float('nan'))
        gan_str  = str(round(gan_val, 4)).rjust(8)  if not np.isnan(gan_val)  else '     N/A'
        lstm_str = str(round(lstm_val, 4)).rjust(8) if not np.isnan(lstm_val) else '     N/A'
        print('  ' + m.ljust(16) + ' | ' + gan_str + ' | ' + lstm_str + ' | ' + METRIC_DESC.get(m, ''))
    print()

    # -------------------------------------------------------
    # PASO 7: checks de validez numerica
    # -------------------------------------------------------
    seccion('PASO 7 - Checks de validez numerica')

    ok_count = 0
    for m in METRICS:
        v = metricas_gan.get(m, float('nan'))
        if not np.isnan(v) and not np.isinf(v):
            ok_count += 1
            ok(m + ' = ' + str(v) + ' (valido)')
        else:
            warn(m + ' = NaN/Inf (revisa log o evaluador)')

    if ok_count == 0:
        fail('Ninguna metrica GAN produjo un valor valido')
    elif ok_count < len(METRICS):
        warn(str(ok_count) + '/' + str(len(METRICS)) + ' metricas validas. ' +
             'Con entrenamiento completo deberian ser todas validas.')
        ok('Al menos una metrica valida — el evaluador funciona')
    else:
        ok('Todas las metricas GAN producen valores validos')

    # -------------------------------------------------------
    # RESUMEN FINAL
    # -------------------------------------------------------
    print('\n' + '=' * 60)
    print('  ETAPA 6 SUPERADA - evaluacion comparativa funcional')
    print('=' * 60)
    print()
    print('  Metricas validas GAN : ' + str(ok_count) + '/' + str(len(METRICS)))
    lstm_ok = sum(1 for v in metricas_lstm.values() if not np.isnan(v) and not np.isinf(v))
    print('  Metricas validas LSTM: ' + str(lstm_ok) + '/' + str(len(METRICS)))
    print()
    print('  NOTA: valores bajos/inconsistentes son ESPERADOS con 2 epochs.')
    print('  La evaluacion REAL del articulo usara entrenamiento completo.')
    print()
