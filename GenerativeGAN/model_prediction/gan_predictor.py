import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mae as keras_mae

from support_modules import traces_evaluation as te
from GenerativeGAN.model_training.models.model_simple_gan import CUSTOM_OBJECTS


class GANPredictor:
    """
    Drop-in replacement for EventLogPredictor that generates business-process
    traces using a trained GAN generator.

    Contract (same as EventLogPredictor.predict):
        predict(params, model_path, examples, imp, vectorizer)
        → list of dicts [{caseid, task, role, start_timestamp, end_timestamp}]

    Key differences from LSTM:
      - The generator produces the full trace at once from a noise vector.
      - Rule-based filtering logic is identical to the LSTM predictor.
    """

    def predict(self, params, model_path, examples, imp, vectorizer):
        return self._generate_traces(params, model_path)

    # ── Core generation loop ──────────────────────────────────────────────────

    def _generate_traces(self, parms, model_path):
        if os.path.isdir(model_path):
            # SavedModel (GANTrainerV2 / transformer_wgan) — sin custom_objects
            # por diseno (ver docstring de GANTrainerV2._export_params).
            model = load_model(model_path)
        else:
            # .h5 (GANTrainer / simple_gan) — necesita las capas custom de
            # model_simple_gan.py registradas para poder deserializar.
            model = load_model(
                model_path,
                custom_objects={'mae': keras_mae, **CUSTOM_OBJECTS})

        latent_dim  = int(parms.get('latent_dim', 100))
        n_ac        = len(parms['index_ac'])
        n_rl        = len(parms['index_rl'])
        num_cases   = parms['num_cases']
        num_digits  = len(str(num_cases))
        target_prop = parms['new_prop_cases']

        # Proporcion inicial estimada a partir del split de entrenamiento
        init_pos = int(parms.get('pos_cases_org', 0))
        init_n   = int(parms.get('total_cases_org', 1)) or 1

        # Contadores en memoria — elimina la lectura O(n^2) de CSVs individuales
        n_accepted  = 0   # trazas aceptadas en total
        n_cond      = 0   # trazas aceptadas que satisfacen la condicion
        event_log   = []

        # Inferencia en batch: una sola llamada al modelo genera BATCH trazas,
        # mucho mas rapido que llamadas individuales (especialmente en CPU).
        BATCH = 32
        max_candidates = num_cases * 10  # presupuesto maximo de candidatos

        pbar = tqdm(total=num_cases, desc='GAN generating traces')
        i = 0
        while n_accepted < num_cases and i < max_candidates:
            b = min(BATCH, max_candidates - i)
            noise     = np.random.normal(0, 1, (b, latent_dim)).astype(np.float32)
            raw_batch = model.predict(noise, verbose=0)  # (b, max_trace_size, feat_dim)

            for j, raw in enumerate(raw_batch):
                if n_accepted >= num_cases:
                    break

                case_id = f'Case{str(i + j).zfill(num_digits)}'
                trace   = self._decode_trace(raw, case_id, parms, n_ac, n_rl)
                if not trace:
                    continue

                df_trace = pd.DataFrame(trace)
                cond = te.evaluate_condition(
                    df_trace, parms['ac_index'],
                    parms['rules']['path'], parms['rules']['rule'])

                # Proporcion actual: mezcla del prior de entrenamiento con
                # los casos generados hasta ahora
                total_seen   = init_n + n_accepted
                cond_seen    = init_pos + n_cond
                current_prop = cond_seen / max(total_seen, 1)

                if cond and current_prop <= target_prop:
                    event_log.extend(trace)
                    n_accepted += 1
                    n_cond     += 1
                    pbar.update(1)
                elif not cond and current_prop >= target_prop:
                    event_log.extend(trace)
                    n_accepted += 1
                    pbar.update(1)

            i += b

        pbar.close()
        achieved = n_cond / max(n_accepted, 1)
        print(f'[GAN] {n_accepted} trazas aceptadas | '
              f'{n_cond} conformes ({achieved:.2%}) | objetivo {target_prop:.2%}')
        return event_log

    # ── Trace decoding ────────────────────────────────────────────────────────

    def _decode_trace(self, generated, case_id, parms, n_ac, n_rl):
        """Convert a GAN output matrix into a list of event dicts."""
        s_ts = parms['start_time']
        trace = []

        for step in generated:
            ac_probs = np.clip(step[:n_ac], 0, None)
            rl_probs = np.clip(step[n_ac:n_ac + n_rl], 0, None)
            dur_norm = float(step[-2])
            wait_norm = float(step[-1])

            variant = parms.get('variant', 'Random Choice')
            if variant in ('Arg Max', 'Rules Based Arg Max'):
                ac_idx = int(np.argmax(ac_probs))
                rl_idx = int(np.argmax(rl_probs))
            else:
                ac_sum = ac_probs.sum()
                rl_sum = rl_probs.sum()
                ac_p = (ac_probs / ac_sum
                        if ac_sum > 0 else np.ones(n_ac) / n_ac)
                rl_p = (rl_probs / rl_sum
                        if rl_sum > 0 else np.ones(n_rl) / n_rl)
                ac_idx = int(np.random.choice(n_ac, p=ac_p))
                rl_idx = int(np.random.choice(n_rl, p=rl_p))

            task = parms['index_ac'].get(ac_idx)
            role = parms['index_rl'].get(rl_idx, 'UNKNOWN')

            if task in (None, 'end'):
                break
            if task == 'start':
                continue

            # Los roles 'start'/'end' son tokens centinela del indice —
            # no son roles reales de proceso.
            if role in ('start', 'end'):
                role = 'UNKNOWN'

            dur_raw = self._rescale(
                dur_norm, parms['scale_args'].get('dur', {}),
                parms['norm_method'])
            wait_raw = self._rescale(
                wait_norm, parms['scale_args'].get('wait', {}),
                parms['norm_method'])
            dur  = 0.0 if (np.isnan(dur_raw)  or np.isinf(dur_raw))  else max(dur_raw,  0.0)
            wait = 0.0 if (np.isnan(wait_raw) or np.isinf(wait_raw)) else max(wait_raw, 0.0)

            start_ts = s_ts + pd.Timedelta(seconds=wait)
            end_ts = start_ts + pd.Timedelta(seconds=dur)

            trace.append({
                'caseid': case_id,
                'task': task,
                'role': role,
                'start_timestamp': start_ts,
                'end_timestamp': end_ts,
            })
            s_ts = end_ts

        return trace

    # ── Rule-proportion helper ────────────────────────────────────────────────

    @staticmethod
    def _current_proportion(df_generated, files_gen, parms):
        n_files = len(files_gen)
        if n_files == 0:
            pos, total = parms['pos_cases_org'], parms['total_cases_org']
            return pos / total if total > 0 else 0.0
        if isinstance(df_generated, pd.DataFrame) and len(df_generated) > 0:
            gs = te.GenerateStats(df_generated, parms['ac_index'],
                                  parms['rules']['path'], parms['rules']['rule'])
            pos, total = gs.get_stats()
            return pos / total if total > 0 else 0.0
        return 0.0

    # ── Persistence ───────────────────────────────────────────────────────────

    @staticmethod
    def _save_trace(df_trace, case_id, parms):
        path = os.path.join(parms['traces_gen_path'], f'gen-{case_id}.csv')
        df_out = df_trace.copy()
        df_out['caseid'] = 'gen-' + df_out['caseid'].astype(str)
        df_out.to_csv(path, index=False)

    # ── Rescaling ─────────────────────────────────────────────────────────────

    @staticmethod
    def _rescale(value, scale_args, norm_method):
        if norm_method == 'lognorm':
            max_v = scale_args.get('max_value', 1.0)
            min_v = scale_args.get('min_value', 0.0)
            if max_v == min_v:
                return float(np.expm1(min_v))
            value = (value * (max_v - min_v)) + min_v
            result = float(np.expm1(value))
            return 0.0 if (np.isnan(result) or np.isinf(result)) else result
        elif norm_method == 'max':
            max_v = scale_args.get('max_value', 1.0)
            return float(np.rint(value * max_v))
        elif norm_method == 'normal':
            max_v = scale_args.get('max_value', 1.0)
            min_v = scale_args.get('min_value', 0.0)
            if max_v == min_v:
                return float(min_v)
            return float((value * (max_v - min_v)) + min_v)
        elif norm_method == 'standard':
            mean = scale_args.get('mean', 0.0)
            std = scale_args.get('std', 1.0)
            return float((value * std) + mean)
        return float(value)
