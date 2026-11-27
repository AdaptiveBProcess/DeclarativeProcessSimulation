"""
Generador de Dynamics con la correccion B (regla de aceptacion).

Problema en el codigo publicado (GenerativeLSTM/model_prediction/event_log_predictor.py,
EventLogPredictor.generate_trace): la proporcion actual se calcula como
(cumplen + n) / (total + n), y una traza que cumple solo se acepta si esa
proporcion es MENOR que la pedida. Con la configuracion del paper (variation = =1,
proporcion pedida 1.0), tras guardar la primera traza la proporcion vale 1.0 y se
rechazan todas las demas. Como la generacion es paralela, sale aproximadamente una
traza por proceso (8 en una CPU de 8 nucleos) en vez de len_log. El paquete de
Zenodo de Dynamics muestra lo mismo (4, 3 y 3 casos generados).

Correccion (solo la regla de aceptacion y el presupuesto de intentos; el modelo,
el muestreo, el filtro de reglas y la decodificacion de la traza son los del codigo
original, copiados sin cambios):
  - Meta: len_log trazas, de las cuales round(p * len_log) cumplen la regla.
  - Una traza que cumple se acepta si todavia faltan trazas que cumplen; una que
    no cumple, si todavia faltan trazas que no cumplen.
  - Intentos: PRESUPUESTO * len_log (el mismo presupuesto que usa RULE-GAN).
  - Como los procesos corren en paralelo pueden pasarse un poco de la meta; al
    final se recorta al numero exacto, en orden de caso.

Se usa reemplazando la clase en el modulo original desde el script que genera
(ver generar.py); el submodulo GenerativeLSTM no se modifica.
"""
import os
import traceback
from glob import glob

import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow.keras.losses import mae

from GenerativeLSTM.model_prediction import event_log_predictor as elp
from support_modules import traces_evaluation as te

PRESUPUESTO = 10


def _meta(parms):
    total = int(parms['len_log'])
    cumplen = int(round(parms['new_prop_cases'] * total))
    return total, cumplen


def _guardadas(parms):
    """(trazas guardadas, cuantas cumplen la regla) en la carpeta compartida."""
    df, archivos = te.get_stats_log_traces(parms['traces_gen_path'])
    if len(archivos) == 0:
        return 0, 0
    gs = te.GenerateStats(df, parms['ac_index'], parms['rules']['path'], parms['rules']['rule'])
    cumplen, total = gs.get_stats()
    return total, cumplen


def carpeta_intentos(traces_gen_path):
    """Carpeta donde cada proceso anota sus intentos (fuera de traces_generated, que el
    predictor lee como trazas guardadas)."""
    return os.path.join(os.path.dirname(os.path.normpath(traces_gen_path)), 'intentos')


def _registrar_intento(parms, cond, acepta):
    """Una linea por traza generada: cumple la regla (0/1), aceptada (0/1). Solo mide
    la eficiencia; no cambia la generacion."""
    carpeta = carpeta_intentos(parms['traces_gen_path'])
    os.makedirs(carpeta, exist_ok=True)
    with open(os.path.join(carpeta, f'{os.getpid()}.txt'), 'a') as f:
        f.write(f'{int(bool(cond))},{int(bool(acepta))}\n')


def _recortar(parms):
    """Deja exactamente la meta (en orden de caso) y devuelve los casos conservados."""
    total, cumplen = _meta(parms)
    archivos = sorted(glob(os.path.join(parms['traces_gen_path'], '*.csv')))
    si, no, quitar = [], [], []
    for archivo in archivos:
        df = pd.read_csv(archivo)
        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
        cond = te.evaluate_condition(df, parms['ac_index'], parms['rules']['path'],
                                     parms['rules']['rule'])
        caso = os.path.basename(archivo)[len('gen-'):-len('.csv')]
        if cond and len(si) < cumplen:
            si.append(caso)
        elif not cond and len(no) < total - cumplen:
            no.append(caso)
        else:
            quitar.append(archivo)
    for archivo in quitar:
        os.remove(archivo)
    conservados = set(si) | set(no)
    if len(conservados) < total:
        print(f"[aviso] Solo se alcanzaron {len(conservados)} de {total} trazas "
              f"({len(si)} cumplen) con {PRESUPUESTO}x intentos.")
    return conservados


class EventLogPredictorCorregido(elp.EventLogPredictor):

    def _generate_traces_parallel(self, parms, vectorizer):
        parms = dict(parms)
        parms['num_cases'] = int(parms['len_log']) * PRESUPUESTO
        event_log = super()._generate_traces_parallel(parms, vectorizer)
        conservados = _recortar(parms)
        return [e for e in event_log if e['caseid'] in conservados]

    @staticmethod
    def generate_trace(args):
        def decode_trace(parms, trace, case):
            # Copia sin cambios de EventLogPredictor.generate_trace.decode_trace
            log_trace = list()
            for i, _ in enumerate(trace):
                event = trace[i]
                if parms['index_ac'][event[0]] != 'end':
                    if parms['one_timestamp']:
                        tbtw = elp.EventLogPredictor.rescale(event[2], parms['scale_args'],
                                                             parms['norm_method'])
                        if i == 0:
                            time = parms['start_time']
                        else:
                            time = (log_trace[i-1]['end_timestamp'] + pd.Timedelta(seconds=tbtw))
                        log_trace.append(dict(caseid=case, task=parms['index_ac'][event[0]],
                                              role=parms['index_rl'][event[1]],
                                              end_timestamp=time, tbtw_raw=event[2], tbtw=tbtw))
                    else:
                        dur = elp.EventLogPredictor.rescale(event[2], parms['scale_args']['dur'],
                                                            parms['norm_method'])
                        wait = elp.EventLogPredictor.rescale(event[3], parms['scale_args']['wait'],
                                                             parms['norm_method'])
                        if i == 0:
                            start_time = (parms['start_time'] + pd.Timedelta(seconds=wait))
                        else:
                            start_time = (log_trace[i-1]['end_timestamp'] + pd.Timedelta(seconds=wait))
                        end_time = (start_time + pd.Timedelta(seconds=dur))
                        log_trace.append(dict(caseid=case, task=parms['index_ac'][event[0]],
                                              role=parms['index_rl'][event[1]],
                                              start_timestamp=start_time, end_timestamp=end_time,
                                              dur_raw=event[2], dur=dur, wait_raw=event[3], wait=wait))
            return log_trace

        def gen(cases, parms, model_path, vectorizer):
            generated_event_log = list()
            try:
                model = load_model(model_path, custom_objects={'mae': mae})
                total, cumplen = _meta(parms)
                for case in cases:
                    n_guardadas, n_cumplen = _guardadas(parms)
                    if n_guardadas >= total:                      # CORRECCION: parar en la meta
                        break

                    # ── Generacion de la traza: copia sin cambios del original ──
                    x_trace = list()
                    seq_tasks = []
                    x_ac_ngram = np.zeros((1, parms['n_size']), dtype=np.float32)
                    x_rl_ngram = np.zeros((1, parms['n_size']), dtype=np.float32)
                    if parms['one_timestamp']:
                        x_t_ngram = np.zeros((1, parms['n_size'], 1), dtype=np.float32)
                    else:
                        x_t_ngram = np.zeros((1, parms['n_size'], 2), dtype=np.float32)
                    if vectorizer in ['basic']:
                        inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
                    elif vectorizer in ['inter']:
                        x_inter_ngram = np.zeros((1, parms['n_size'], len(parms['additional_columns'])),
                                                 dtype=np.float32)
                        inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
                    i = 1
                    while i < parms['max_trace_size']:
                        predictions = model.predict(inputs, verbose=0)
                        if parms['variant'] == 'Random Choice':
                            pos = np.random.choice(np.arange(0, len(predictions[0][0])), p=predictions[0][0])
                            pos1 = np.random.choice(np.arange(0, len(predictions[1][0])), p=predictions[1][0])
                        elif parms['variant'] == 'Arg Max':
                            pos = np.argmax(predictions[0][0])
                            pos1 = np.argmax(predictions[1][0])
                        elif parms['variant'] == 'Rules Based Random Choice':
                            possible_tasks = [x for x in parms['index_ac'].keys() if te.evaluate_condition_list(
                                seq_tasks + [x], parms['ac_index'], parms['rules']['path'], parms['rules']['rule'])]
                            if len(possible_tasks) > 0:
                                pos = np.random.choice(possible_tasks)
                            else:
                                pos = np.random.choice(np.arange(0, len(predictions[0][0])), p=predictions[0][0])
                            pos1 = np.random.choice(np.arange(0, len(predictions[1][0])), p=predictions[1][0])
                        elif parms['variant'] == 'Rules Based Arg Max':
                            possible_tasks = [x for x in parms['index_ac'].keys() if te.evaluate_condition_list(
                                seq_tasks + [x], parms['ac_index'], parms['rules']['path'], parms['rules']['rule'])]
                            if len(possible_tasks) > 0:
                                probs_poss_tasks = [[predictions[0][0][x], x] for x in possible_tasks]
                                pos = max(probs_poss_tasks, key=lambda x: x[0])[1]
                            else:
                                pos = np.argmax(predictions[0][0])
                            pos1 = np.argmax(predictions[1][0])
                        seq_tasks.append(pos)
                        if (not x_trace) and (parms['index_ac'][pos] == 'end'):
                            continue
                        else:
                            if parms['one_timestamp']:
                                x_trace.append([pos, pos1, predictions[2][0][0]])
                            else:
                                pred_dur = (predictions[2][0][0] if predictions[2][0][0] >= 0 else 0)
                                pred_wait = (predictions[2][0][1] if predictions[2][0][1] >= 0 else 0)
                                x_trace.append([pos, pos1, pred_dur, pred_wait])
                                pre_times = np.array([[pred_dur, pred_wait]])
                            x_ac_ngram = np.append(x_ac_ngram, [[pos]], axis=1)
                            x_ac_ngram = np.delete(x_ac_ngram, 0, 1)
                            x_rl_ngram = np.append(x_rl_ngram, [[pos1]], axis=1)
                            x_rl_ngram = np.delete(x_rl_ngram, 0, 1)
                            x_t_ngram = np.append(x_t_ngram, [pre_times], axis=1)
                            x_t_ngram = np.delete(x_t_ngram, 0, 1)
                            if vectorizer in ['basic']:
                                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
                            elif vectorizer in ['inter']:
                                x_inter_ngram = np.append(x_inter_ngram, [predictions[3]], axis=1)
                                x_inter_ngram = np.delete(x_inter_ngram, 0, 1)
                                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
                            if parms['index_ac'][pos] == 'end':
                                break
                            i += 1

                    trace = decode_trace(parms, x_trace, case)
                    df_trace = pd.DataFrame.from_records(trace, index=list(trace[0].keys())).reset_index()
                    cond = te.evaluate_condition(df_trace, parms['ac_index'], parms['rules']['path'],
                                                 parms['rules']['rule'])

                    # ── CORRECCION: aceptar mientras falten trazas de su clase ──
                    acepta = (n_cumplen < cumplen) if cond else ((n_guardadas - n_cumplen) < total - cumplen)
                    _registrar_intento(parms, cond, acepta)
                    if acepta:
                        generated_event_log.extend(trace)
                        df_trace['caseid'] = 'gen-' + df_trace['caseid']
                        df_trace.to_csv(os.path.join(parms['traces_gen_path'], 'gen-{}.csv'.format(case)))
                return generated_event_log
            except Exception:
                traceback.print_exc()
        return gen(*args)
