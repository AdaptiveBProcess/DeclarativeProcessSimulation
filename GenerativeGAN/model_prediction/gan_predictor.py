import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mae as keras_mae

from support_modules import traces_evaluation as te


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
        model = load_model(model_path, custom_objects={'mae': keras_mae})
        latent_dim = int(parms.get('latent_dim', 100))
        n_ac = len(parms['index_ac'])
        n_rl = len(parms['index_rl'])
        num_cases = parms['num_cases']
        num_digits = len(str(num_cases))
        event_log = []

        # Over-sample to reach the required count after rule filtering
        for i in tqdm(range(num_cases * 5), desc='GAN generating traces'):
            df_generated, files_gen = te.get_stats_log_traces(
                parms['traces_gen_path'])

            if len(files_gen) >= parms['len_log']:
                break

            case_id = f'Case{str(i).zfill(num_digits)}'
            noise = np.random.normal(0, 1, (1, latent_dim)).astype(np.float32)
            raw = model.predict(noise, verbose=0)[0]  # (max_trace_size, feat_dim)

            trace = self._decode_trace(raw, case_id, parms, n_ac, n_rl)
            if not trace:
                continue

            df_trace = pd.DataFrame(trace)
            cond = te.evaluate_condition(
                df_trace, parms['ac_index'],
                parms['rules']['path'], parms['rules']['rule'])

            current_prop = self._current_proportion(
                df_generated, files_gen, parms)

            target = parms['new_prop_cases']
            at_target = (
                abs((current_prop - target) / max(target, 1e-9)) <= 0.05
                and len(files_gen) >= parms['len_log'])
            if at_target:
                break

            if cond and current_prop <= target:
                event_log.extend(trace)
                self._save_trace(df_trace, case_id, parms)
            elif not cond and current_prop >= target:
                event_log.extend(trace)
                self._save_trace(df_trace, case_id, parms)

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

            dur = max(self._rescale(
                dur_norm, parms['scale_args'].get('dur', {}),
                parms['norm_method']), 0.0)
            wait = max(self._rescale(
                wait_norm, parms['scale_args'].get('wait', {}),
                parms['norm_method']), 0.0)

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
            value = (value * (max_v - min_v)) + min_v
            return float(np.expm1(value))
        elif norm_method == 'max':
            max_v = scale_args.get('max_value', 1.0)
            return float(np.rint(value * max_v))
        elif norm_method == 'normal':
            max_v = scale_args.get('max_value', 1.0)
            min_v = scale_args.get('min_value', 0.0)
            return float((value * (max_v - min_v)) + min_v)
        elif norm_method == 'standard':
            mean = scale_args.get('mean', 0.0)
            std = scale_args.get('std', 1.0)
            return float((value * std) + mean)
        return float(value)
