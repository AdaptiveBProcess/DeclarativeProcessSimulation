import os
import numpy as np
import pandas as pd
import tensorflow as tf

import readers.log_reader as lr
import utils.support as sup
import readers.log_splitter as ls

from GenerativeLSTM.model_training.features_manager import FeaturesMannager
from GenerativeGAN.model_training.models.model_simple_gan import (
    build_generator, build_discriminator)
from GenerativeGAN.model_training.samples_creator import GANSamplesCreator
from support_modules import traces_evaluation as te


class GANTrainer:
    """
    Trains a simple GAN (Generator + Discriminator) on a business-process event log
    and saves outputs in the same folder structure that ModelPredictor expects.

    Output layout under <output_folder>/<folder_id>/:
        <log_name>.h5               <- saved generator (Keras model)
        parameters/
            model_parameters.json
            test_log.csv
            <log_name>_ASIS.csv
    """

    def __init__(self, params, input_folder='data/0.logs',
                 output_folder='data/1.predicton_models'):
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Resolve norm_method: trainer receives a list from dg_training.py
        norm = params.get('norm_method', 'max')
        self.norm_method = norm[0] if isinstance(norm, list) else norm

        self.latent_dim = int(params.get('latent_dim', 100))
        self.epochs = int(params.get('epochs', 200))
        self.batch_size = int(params.get('batch_size', 32))

        # ── 1. Load & preprocess ──────────────────────────────────────────────
        self.log = self._load_log(params)
        self.log = FeaturesMannager.add_resources(self.log, params['rp_sim'])

        # ── 2. Build activity / role indexes ─────────────────────────────────
        self._build_indexes()

        # ── 3. Time-split into train / (val) / test ──────────────────────────
        split_config = params.get('split_config')
        if split_config:
            self._split_timeline_70_10_20(
                split_config.get('rules_path', ''),
                split_config.get('test_save_path'))
        else:
            self._split_timeline(0.8, params['read_options']['one_timestamp'])
            self.n_test_cases    = None
            self.train_prop      = None
            self.pos_train_cases = None
            self.n_train_cases   = None

        # ── 4. Add dur / wait features to training split ──────────────────────
        fm = FeaturesMannager({
            'model_type': 'simple_gan',
            'one_timestamp': False,
            'norm_method': self.norm_method,
        })
        self.log_train = fm.add_calculated_times(self.log_train)

        # ── 5. Normalize and store scale_args ─────────────────────────────────
        self.log_train, dur_scale = FeaturesMannager.scale_feature(
            self.log_train, 'dur', self.norm_method)
        self.log_train, wait_scale = FeaturesMannager.scale_feature(
            self.log_train, 'wait', self.norm_method)
        self.scale_args = {'dur': dur_scale, 'wait': wait_scale}

        # ── 6. Build training matrix ──────────────────────────────────────────
        self.max_trace_size = int(
            self.log.groupby('caseid')['task'].count().max())
        creator = GANSamplesCreator()
        X = creator.create_samples(
            self.log_train, self.ac_index, self.rl_index, self.max_trace_size)

        # ── 7. Build GAN architecture ─────────────────────────────────────────
        n_ac = len(self.ac_index)
        n_rl = len(self.rl_index)
        self.generator = build_generator(self.latent_dim, self.max_trace_size,
                                         n_ac, n_rl)
        self.discriminator = build_discriminator(self.max_trace_size, n_ac, n_rl)

        # ── 8. Train ──────────────────────────────────────────────────────────
        output_path = os.path.join(self.output_folder, sup.folder_id())
        os.makedirs(output_path, exist_ok=True)
        self._train_gan(X, output_path)

        # ── 9. Save model and parameters ─────────────────────────────────────
        log_name = params['file_name'].rsplit('.', 1)[0]
        model_file = f'{log_name}.h5'
        self.generator.save(os.path.join(output_path, model_file))
        self._export_params(output_path, model_file, log_name)
        print(f'[GANTrainer] Training complete. Model saved to: {output_path}')

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_log(self, params):
        params['read_options']['filter_d_attrib'] = False
        log_reader = lr.LogReader(
            os.path.join(self.input_folder, params['file_name']),
            params['read_options'])
        df = pd.DataFrame(log_reader.data)
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
        df = df[~df['task'].isin(['Start', 'End'])]
        return df.reset_index(drop=True)

    def _build_indexes(self):
        self.ac_index = self._create_index(self.log, 'task')
        self.ac_index['start'] = 0
        self.ac_index['end'] = len(self.ac_index)
        self.index_ac = {v: k for k, v in self.ac_index.items()}

        self.rl_index = self._create_index(self.log, 'role')
        self.rl_index['start'] = 0
        self.rl_index['end'] = len(self.rl_index)
        self.index_rl = {v: k for k, v in self.rl_index.items()}

        self.log['ac_index'] = self.log['task'].map(self.ac_index)
        self.log['rl_index'] = self.log['role'].map(self.rl_index)

    @staticmethod
    def _create_index(df, col):
        vals = sorted({x[0] for x in df[[col]].values.tolist()})
        return {v: i + 1 for i, v in enumerate(vals)}

    def _split_timeline(self, size, one_ts):
        splitter = ls.LogSplitter(self.log)
        train, test = splitter.split_log('timeline_contained', size, one_ts)
        if len(test) < int(len(self.log) * 0.1):
            train, test = splitter.split_log('timeline_trace', size, one_ts)
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        self.log_train = (pd.DataFrame(train)
                          .sort_values(key).reset_index(drop=True))
        self.log_test = (pd.DataFrame(test)
                         .sort_values(key).reset_index(drop=True))

    def _split_timeline_70_10_20(self, rules_path, test_save_path=None):
        """Chronological 70/10/20 split. Rule proportion calculated on train."""
        case_order = (
            self.log.groupby('caseid')['start_timestamp']
            .min()
            .sort_values()
        )
        n       = len(case_order)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.10)

        train_ids = set(case_order.index[:n_train])
        val_ids   = set(case_order.index[n_train:n_train + n_val])
        test_ids  = set(case_order.index[n_train + n_val:])

        self.log_train = self.log[self.log['caseid'].isin(train_ids)].copy()
        self.log_val   = self.log[self.log['caseid'].isin(val_ids)].copy()
        self.log_test  = self.log[self.log['caseid'].isin(test_ids)].copy()

        # Rule satisfaction proportion in training set
        rules     = te.extract_rules(path=rules_path)
        act_paths = rules['path']
        rule_type = rules['rule']

        def _satisfies(grp):
            tasks = set(grp['task'].values)
            if rule_type == 'not_allowed':
                return act_paths[0] not in tasks
            elif rule_type == 'required':
                return act_paths[0] in tasks
            else:
                return all(a in tasks for a in act_paths)

        flags = self.log_train.groupby('caseid').apply(_satisfies)
        self.pos_train_cases = int(flags.sum())
        self.n_train_cases   = int(len(flags))
        self.train_prop      = float(flags.mean())
        self.n_test_cases    = len(test_ids)

        print(f'[GANTrainer] Split 70/10/20: {n_train} train | '
              f'{len(val_ids)} val | {len(test_ids)} test  (total {n})')
        print(f'[GANTrainer] Regla ({rule_type}) en train: '
              f'{self.pos_train_cases}/{self.n_train_cases} = {self.train_prop:.2%}')

        if test_save_path:
            os.makedirs(
                os.path.dirname(os.path.abspath(test_save_path)), exist_ok=True)
            self.log_test.to_csv(test_save_path, index=False)
            print(f'[GANTrainer] Test split guardado: {test_save_path}')

    def _train_gan(self, X, output_path):
        d_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        g_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        self.discriminator.compile(optimizer=d_opt, loss='binary_crossentropy')

        # Combined model: noise → generator → discriminator (frozen)
        noise_in = tf.keras.Input(shape=(self.latent_dim,))
        self.discriminator.trainable = False
        validity = self.discriminator(self.generator(noise_in))
        combined = tf.keras.Model(noise_in, validity, name='gan')
        combined.compile(optimizer=g_opt, loss='binary_crossentropy')
        self.discriminator.trainable = True

        n = len(X)
        half_batch = max(self.batch_size // 2, 1)

        for epoch in range(self.epochs):
            # ── Discriminator step ────────────────────────────────────────────
            idx = np.random.randint(0, n, half_batch)
            real_traces = X[idx]
            noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
            fake_traces = self.generator.predict(noise, verbose=0)

            self.discriminator.trainable = True
            d_real = self.discriminator.train_on_batch(
                real_traces, np.ones((half_batch, 1)) * 0.9)   # label smoothing
            d_fake = self.discriminator.train_on_batch(
                fake_traces, np.zeros((half_batch, 1)))
            d_loss = 0.5 * (d_real + d_fake)

            # ── Generator step ────────────────────────────────────────────────
            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            self.discriminator.trainable = False
            g_loss = combined.train_on_batch(
                noise, np.ones((self.batch_size, 1)))
            self.discriminator.trainable = True

            if epoch % 20 == 0:
                print(f'[GAN] epoch {epoch:04d}/{self.epochs} '
                      f'| D: {d_loss:.4f} | G: {g_loss:.4f}')

    def _export_params(self, output_path, model_file, log_name):
        params_dir = os.path.join(output_path, 'parameters')
        os.makedirs(params_dir, exist_ok=True)

        # Convert numpy scalars to plain Python floats for JSON serialisation
        scale_args_clean = {
            feat: {k: float(v) for k, v in args.items()}
            for feat, args in self.scale_args.items()
        }

        model_params = {
            'model_type':      'simple_gan',
            'model_file':      model_file,
            'index_ac':        self.index_ac,
            'index_rl':        self.index_rl,
            'scale_args':      scale_args_clean,
            'norm_method':     self.norm_method,
            'max_trace_size':  self.max_trace_size,
            'one_timestamp':   False,
            'latent_dim':      self.latent_dim,
            'n_size':          5,
            # Split metadata (None when using legacy 80/20 split)
            'n_test_cases':    self.n_test_cases,
            'train_prop':      self.train_prop,
            'pos_train_cases': self.pos_train_cases,
            'n_train_cases':   self.n_train_cases,
        }
        sup.create_json(model_params,
                        os.path.join(params_dir, 'model_parameters.json'))

        self.log.to_csv(
            os.path.join(params_dir, f'{log_name}_ASIS.csv'),
            index=False, encoding='utf-8')
        self.log_test.to_csv(
            os.path.join(params_dir, 'test_log.csv'),
            index=False, encoding='utf-8')
