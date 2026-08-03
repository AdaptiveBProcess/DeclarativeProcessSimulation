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

        self.latent_dim = int(params.get('latent_dim', 64))
        self.epochs = int(params.get('epochs', 1000))
        self.batch_size = int(params.get('batch_size', 32))

        # ── Hiperparametros de arquitectura (V4: Transformer + Time2Vec) ───────
        self.d_model = int(params.get('d_model', 64))
        self.num_heads = int(params.get('num_heads', 4))
        self.ff_dim = int(params.get('ff_dim', 256))
        self.num_blocks = int(params.get('num_blocks', 3))
        self.dropout = float(params.get('dropout', 0.1))
        self.time2vec_dim = int(params.get('time2vec_dim', 16))

        # ── Hiperparametros de entrenamiento WGAN-GP ────────────────────────────
        self.n_critic = int(params.get('n_critic', 5))
        self.gp_lambda = float(params.get('gp_lambda', 10.0))

        # Guarda un checkpoint del generador cada N epochs (protege corridas
        # largas de interrupciones — ver comentarios-cloud/CLOUD.md #2 sobre
        # archivos borrados por interferencia de sincronizacion de OneDrive).
        self.checkpoint_every = int(params.get('checkpoint_every', 100))

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
        self.generator = build_generator(
            self.latent_dim, self.max_trace_size, n_ac, n_rl,
            d_model=self.d_model, num_heads=self.num_heads, ff_dim=self.ff_dim,
            num_blocks=self.num_blocks, dropout=self.dropout)
        self.discriminator = build_discriminator(
            self.max_trace_size, n_ac, n_rl,
            d_model=self.d_model, num_heads=self.num_heads, ff_dim=self.ff_dim,
            num_blocks=self.num_blocks, time2vec_dim=self.time2vec_dim,
            dropout=self.dropout)

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
        """
        Entrenamiento WGAN-GP (Gulrajani et al., 2017):
          - discriminador = critico (score sin activacion, sin sigmoid)
          - n_critic pasos de D por cada paso de G
          - gradient penalty vía GradientTape anidado (double backprop) dentro del
            mismo tf.function del paso de D — NO se decora _gradient_penalty por
            separado, para que el tape externo pueda diferenciar a traves de ella.
        """
        d_opt = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)
        g_opt = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.0, beta_2=0.9)

        gp_lambda = tf.constant(self.gp_lambda, dtype=tf.float32)

        def _gradient_penalty(real, fake):
            batch_size = tf.shape(real)[0]
            eps = tf.random.uniform((batch_size, 1, 1), 0.0, 1.0)
            interpolated = eps * real + (1.0 - eps) * fake
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                pred = self.discriminator(interpolated, training=True)
            grads = gp_tape.gradient(pred, [interpolated])[0]
            norm = tf.sqrt(
                tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-12)
            return tf.reduce_mean(tf.square(norm - 1.0))

        @tf.function
        def _critic_step(real):
            batch_size = tf.shape(real)[0]
            noise = tf.random.normal((batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake = self.generator(noise, training=True)
                d_real = self.discriminator(real, training=True)
                d_fake = self.discriminator(fake, training=True)
                gp = _gradient_penalty(real, fake)
                d_loss = (tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
                          + gp_lambda * gp)
            grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            d_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))
            return d_loss

        @tf.function
        def _generator_step(batch_size):
            noise = tf.random.normal((batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake = self.generator(noise, training=True)
                d_fake = self.discriminator(fake, training=True)
                g_loss = -tf.reduce_mean(d_fake)
            grads = tape.gradient(g_loss, self.generator.trainable_variables)
            g_opt.apply_gradients(zip(grads, self.generator.trainable_variables))
            return g_loss

        n = len(X)
        X_tf = tf.constant(X, dtype=tf.float32)
        steps_per_epoch = max(n // self.batch_size, 1)
        batch_size_t = tf.constant(self.batch_size, dtype=tf.int32)

        checkpoint_dir = os.path.join(output_path, 'checkpoints')
        if self.checkpoint_every > 0:
            os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(self.epochs):
            d_loss_acc, g_loss_acc = 0.0, 0.0
            for _ in range(steps_per_epoch):
                for _ in range(self.n_critic):
                    idx = np.random.randint(0, n, self.batch_size)
                    real_batch = tf.gather(X_tf, idx)
                    d_loss = _critic_step(real_batch)
                g_loss = _generator_step(batch_size_t)
                d_loss_acc += float(d_loss)
                g_loss_acc += float(g_loss)

            if epoch % 20 == 0:
                print(f'[GAN-WGANGP] epoch {epoch:04d}/{self.epochs} '
                      f'| D: {d_loss_acc / steps_per_epoch:.4f} '
                      f'| G: {g_loss_acc / steps_per_epoch:.4f}')

            if (self.checkpoint_every > 0 and epoch > 0
                    and epoch % self.checkpoint_every == 0):
                ckpt_path = os.path.join(
                    checkpoint_dir, f'generator_epoch{epoch:04d}.h5')
                self.generator.save(ckpt_path)
                print(f'[GAN-WGANGP] checkpoint guardado: {ckpt_path}')

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
            # Arquitectura V4 (documentacion/reproducibilidad — no se usan en
            # inferencia, la arquitectura ya queda fija dentro del .h5)
            'd_model':         self.d_model,
            'num_heads':       self.num_heads,
            'ff_dim':          self.ff_dim,
            'num_blocks':      self.num_blocks,
            'dropout':         self.dropout,
            'time2vec_dim':    self.time2vec_dim,
            'n_critic':        self.n_critic,
            'gp_lambda':       self.gp_lambda,
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
