"""
GANTrainerV2: iteration v2 — Transformer + WGAN-GP.

Changes vs GANTrainer (v1):
  Architecture : GRU → Transformer (Generator + Discriminator)
  Time encoding: scalar → Time2Vec (Discriminator input)
  Categorical   : one-hot → learned dense projection (Discriminator input)
  Training      : Vanilla GAN → WGAN-GP (gradient penalty λ=10, n_critic=5)
  Normalization : max → lognorm
  Latent dim    : 100 → 64
  Epochs        : 200 → 500

External interface is identical to GANTrainer so that dg_prediction.py
and GANPredictor work without modification.  The generator is saved in
TensorFlow SavedModel format (directory, no .h5 extension) to avoid the
need to register custom layer classes at load time.
"""

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

import readers.log_reader as lr
import utils.support as sup
import readers.log_splitter as ls

from GenerativeLSTM.model_training.features_manager import FeaturesMannager
from GenerativeGAN.model_training.models.model_wgan_transformer import (
    build_transformer_generator,
    build_transformer_discriminator,
)
from GenerativeGAN.model_training.samples_creator import GANSamplesCreator
from support_modules import traces_evaluation as te


class GANTrainerV2:
    """
    Trains a Transformer-based GAN with WGAN-GP on a business-process event log.

    Saved output layout (same as GANTrainer):
        <output_folder>/<folder_id>/
            <log_name>/                  ← SavedModel directory (generator)
            parameters/
                model_parameters.json
                test_log.csv
                <log_name>_ASIS.csv
    """

    def __init__(self, params, input_folder='data/0.logs',
                 output_folder='data/1.predicton_models'):
        self.input_folder = input_folder
        self.output_folder = output_folder

        # ── Hyperparameters (v2 defaults) ─────────────────────────────────────
        norm = params.get('norm_method', 'lognorm')
        self.norm_method  = norm[0] if isinstance(norm, list) else norm
        self.latent_dim   = int(params.get('latent_dim', 64))
        self.epochs       = int(params.get('epochs', 500))
        self.batch_size   = int(params.get('batch_size', 32))
        self.n_critic     = int(params.get('n_critic', 5))
        self.gp_lambda    = float(params.get('gp_lambda', 10.0))
        self.d_model      = int(params.get('d_model', 64))
        self.num_heads    = int(params.get('num_heads', 4))
        self.num_blocks   = int(params.get('num_blocks', 2))
        self.ff_dim       = int(params.get('ff_dim', 128))
        self.dropout      = float(params.get('dropout', 0.1))
        self.time2vec_dim = int(params.get('time2vec_dim', 8))
        self.bg_lambda    = float(params.get('bg_lambda', 0.0))
        self.ct_lambda    = float(params.get('ct_lambda', 0.0))

        # ── 1. Load & preprocess ──────────────────────────────────────────────
        self.log = self._load_log(params)
        self.log = FeaturesMannager.add_resources(self.log, params['rp_sim'])

        # ── 2. Build activity / role indexes ─────────────────────────────────
        self._build_indexes()

        # ── 3. Chronological 70/10/20 split ──────────────────────────────────
        split_config = params.get('split_config')
        if split_config:
            self._split_timeline_70_10_20(
                split_config.get('rules_path', ''),
                split_config.get('test_save_path'))
        else:
            self._split_timeline(0.8, params['read_options']['one_timestamp'])
            self.n_test_cases = self.train_prop = \
                self.pos_train_cases = self.n_train_cases = None

        # ── 4. Add dur / wait features to training split ──────────────────────
        fm = FeaturesMannager({
            'model_type':   'simple_gan',
            'one_timestamp': False,
            'norm_method':   self.norm_method,
        })
        self.log_train = fm.add_calculated_times(self.log_train)

        # ── 5. Normalize and store scale_args ─────────────────────────────────
        self.log_train, dur_scale  = FeaturesMannager.scale_feature(
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

        # ── 7. Build Transformer GAN architecture ─────────────────────────────
        n_ac = len(self.ac_index)
        n_rl = len(self.rl_index)
        self.generator = build_transformer_generator(
            self.latent_dim, self.max_trace_size, n_ac, n_rl,
            d_model=self.d_model, num_heads=self.num_heads,
            num_blocks=self.num_blocks, ff_dim=self.ff_dim,
            dropout=self.dropout)
        self.discriminator = build_transformer_discriminator(
            self.max_trace_size, n_ac, n_rl,
            d_model=self.d_model, num_heads=self.num_heads,
            num_blocks=self.num_blocks, ff_dim=self.ff_dim,
            dropout=self.dropout, time2vec_dim=self.time2vec_dim)

        print(f'[GANTrainerV2] Generator params: '
              f'{self.generator.count_params():,}')
        print(f'[GANTrainerV2] Discriminator params: '
              f'{self.discriminator.count_params():,}')

        # ── 8. Train with WGAN-GP ─────────────────────────────────────────────
        output_path = os.path.join(self.output_folder, time.strftime('%Y%m%d_%H%M%S'))
        os.makedirs(output_path, exist_ok=True)
        self._train_wgan_gp(X, output_path)

        # ── 9. Save generator (SavedModel format) and parameters ──────────────
        log_name   = params['file_name'].rsplit('.', 1)[0]
        model_file = 'gen'  # short name avoids Windows MAX_PATH on SavedModel temp files
        self.generator.save(os.path.join(output_path, model_file))
        self._export_params(output_path, model_file, log_name)
        print(f'[GANTrainerV2] Training complete. Model saved to: {output_path}')

    # ── Data helpers ──────────────────────────────────────────────────────────

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
        self.ac_index['end']   = len(self.ac_index)
        self.index_ac = {v: k for k, v in self.ac_index.items()}

        self.rl_index = self._create_index(self.log, 'role')
        self.rl_index['start'] = 0
        self.rl_index['end']   = len(self.rl_index)
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
        self.log_train = pd.DataFrame(train).sort_values(key).reset_index(drop=True)
        self.log_test  = pd.DataFrame(test).sort_values(key).reset_index(drop=True)

    def _split_timeline_70_10_20(self, rules_path, test_save_path=None):
        """Chronological 70/10/20 split with rule proportion on training set."""
        case_order = (
            self.log.groupby('caseid')['start_timestamp']
            .min().sort_values())
        n       = len(case_order)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.10)

        train_ids = set(case_order.index[:n_train])
        val_ids   = set(case_order.index[n_train:n_train + n_val])
        test_ids  = set(case_order.index[n_train + n_val:])

        self.log_train = self.log[self.log['caseid'].isin(train_ids)].copy()
        self.log_val   = self.log[self.log['caseid'].isin(val_ids)].copy()
        self.log_test  = self.log[self.log['caseid'].isin(test_ids)].copy()

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

        print(f'[GANTrainerV2] Split 70/10/20: {n_train} train | '
              f'{len(val_ids)} val | {len(test_ids)} test  (total {n})')
        print(f'[GANTrainerV2] Regla ({rule_type}) en train: '
              f'{self.pos_train_cases}/{self.n_train_cases} = {self.train_prop:.2%}')

        if test_save_path:
            os.makedirs(
                os.path.dirname(os.path.abspath(test_save_path)), exist_ok=True)
            self.log_test.to_csv(test_save_path, index=False)
            print(f'[GANTrainerV2] Test split guardado: {test_save_path}')

    # ── WGAN-GP training loop ─────────────────────────────────────────────────

    def _train_wgan_gp(self, X, output_path):
        X = tf.cast(X, tf.float32)
        n          = len(X)
        half_batch = max(self.batch_size // 2, 1)

        # Dimensions needed for auxiliary losses
        n_ac_val = len(self.ac_index)
        n_rl_val = len(self.rl_index)

        # ── Auxiliary targets (precomputed from training data) ────────────────
        # Bigram target: normalized joint probability matrix of consecutive
        # activities in the training set. Shape: (n_ac, n_ac).
        # Uses einsum 'bti,btj->ij': for each pair of consecutive positions
        # (t, t+1) sum the outer product of their activity vectors over
        # all traces (b) and all positions (t).
        bg_raw    = tf.einsum('bti,btj->ij',
                              X[:, :-1, :n_ac_val],
                              X[:, 1:,  :n_ac_val])
        target_bg = bg_raw / (tf.reduce_sum(bg_raw) + 1e-8)

        # Cycle time target: mean normalized cycle time across training traces.
        dur_tr    = X[:, :, n_ac_val + n_rl_val]
        wait_tr   = X[:, :, n_ac_val + n_rl_val + 1]
        target_ct = tf.reduce_mean(tf.reduce_sum(dur_tr + wait_tr, axis=1))

        d_opt = tf.keras.optimizers.Adam(
            learning_rate=0.0001, beta_1=0.0, beta_2=0.9)
        g_opt = tf.keras.optimizers.Adam(
            learning_rate=0.0002, beta_1=0.0, beta_2=0.9)

        # Local references so @tf.function closures capture TF objects, not self
        generator     = self.generator
        discriminator = self.discriminator
        gp_lambda     = tf.constant(self.gp_lambda, dtype=tf.float32)
        bg_lam        = tf.constant(self.bg_lambda,  dtype=tf.float32)
        ct_lam        = tf.constant(self.ct_lambda,  dtype=tf.float32)
        latent_dim    = self.latent_dim
        batch_size    = self.batch_size

        # Compiled training steps: eliminates Python-per-op overhead (~3-5x faster
        # on CPU vs eager mode). TF traces once on the first call and reuses the
        # compiled graph for all subsequent epochs.
        @tf.function
        def train_disc_step(real_batch):
            hb    = tf.shape(real_batch)[0]
            noise = tf.random.normal((hb, latent_dim))
            fake  = generator(noise, training=False)
            # d_tape wraps GP so ∂(λ·GP)/∂θ_D flows correctly (second-order)
            with tf.GradientTape() as d_tape:
                d_real = discriminator(real_batch, training=True)
                d_fake = discriminator(fake, training=True)
                eps    = tf.random.uniform([hb, 1, 1], 0.0, 1.0)
                interp = eps * real_batch + (1.0 - eps) * fake
                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interp)
                    d_interp = discriminator(interp, training=True)
                gp_grads = gp_tape.gradient(d_interp, interp)
                gp_norm  = tf.sqrt(
                    tf.reduce_sum(tf.square(gp_grads), axis=[1, 2]) + 1e-8)
                gp     = tf.reduce_mean((gp_norm - 1.0) ** 2)
                d_loss = (tf.reduce_mean(d_fake)
                          - tf.reduce_mean(d_real)
                          + gp_lambda * gp)
            d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
            d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))
            return d_loss

        @tf.function
        def train_gen_step():
            noise = tf.random.normal((batch_size, latent_dim))
            with tf.GradientTape() as g_tape:
                fake   = generator(noise, training=True)
                d_fake = discriminator(fake, training=False)
                g_adv  = -tf.reduce_mean(d_fake)

                # Bigram alignment loss: MSE between soft bigram matrix of
                # generated traces and the precomputed training bigram matrix.
                ac_f   = fake[:, :, :n_ac_val]
                bg_f   = tf.einsum('bti,btj->ij',
                                   ac_f[:, :-1, :], ac_f[:, 1:, :])
                bg_f   = bg_f / (tf.reduce_sum(bg_f) + 1e-8)
                bg_loss = tf.reduce_mean(tf.square(bg_f - target_bg))

                # Cycle time alignment loss: L1 between mean generated CT
                # (sum of dur+wait per trace) and the training mean CT.
                dur_f  = fake[:, :, n_ac_val + n_rl_val]
                wait_f = fake[:, :, n_ac_val + n_rl_val + 1]
                ct_f   = tf.reduce_mean(tf.reduce_sum(dur_f + wait_f, axis=1))
                ct_loss = tf.abs(ct_f - target_ct)

                g_loss = g_adv + bg_lam * bg_loss + ct_lam * ct_loss

            g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
            g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
            return g_loss, g_adv, bg_loss, ct_loss

        print(f'[GANTrainerV2] Iniciando WGAN-GP: {self.epochs} epochs, '
              f'n_critic={self.n_critic}, λ={self.gp_lambda}, '
              f'batch={self.batch_size}, latent_dim={self.latent_dim}')
        print('[GANTrainerV2] Compilando pasos de entrenamiento (primera epoch mas lenta)...')
        t0 = time.time()

        for epoch in range(self.epochs):

            # ── Discriminator: n_critic steps ────────────────────────────────
            d_losses = []
            for _ in range(self.n_critic):
                idx    = np.random.randint(0, n, half_batch)
                d_loss = train_disc_step(tf.gather(X, idx))
                d_losses.append(float(d_loss))

            # ── Generator: one step ───────────────────────────────────────────
            g_loss, g_adv, bg_loss, ct_loss = train_gen_step()

            if epoch % 25 == 0 or epoch == self.epochs - 1:
                w_dist    = -np.mean(d_losses)
                elapsed   = time.time() - t0
                avg_s     = elapsed / (epoch + 1)
                remaining = avg_s * (self.epochs - epoch - 1)
                print(f'[WGAN-GP] epoch {epoch:04d}/{self.epochs} '
                      f'| W: {w_dist:.4f} | G_adv: {float(g_adv):.4f} '
                      f'| BG: {float(bg_loss):.4f} | CT: {float(ct_loss):.4f} '
                      f'| {avg_s:.1f}s/ep | resta ~{remaining/60:.1f}min')

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_params(self, output_path, model_file, log_name):
        params_dir = os.path.join(output_path, 'parameters')
        os.makedirs(params_dir, exist_ok=True)

        scale_args_clean = {
            feat: {k: float(v) for k, v in args.items()}
            for feat, args in self.scale_args.items()
        }

        model_params = {
            'model_type':      'transformer_wgan',
            'model_file':      model_file,
            'index_ac':        self.index_ac,
            'index_rl':        self.index_rl,
            'scale_args':      scale_args_clean,
            'norm_method':     self.norm_method,
            'max_trace_size':  self.max_trace_size,
            'one_timestamp':   False,
            'latent_dim':      self.latent_dim,
            'n_size':          5,
            'n_test_cases':    self.n_test_cases,
            'train_prop':      self.train_prop,
            'pos_train_cases': self.pos_train_cases,
            'n_train_cases':   self.n_train_cases,
            # v2 architecture metadata
            'd_model':         self.d_model,
            'num_heads':       self.num_heads,
            'num_blocks':      self.num_blocks,
            'ff_dim':          self.ff_dim,
            'dropout':         self.dropout,
            'time2vec_dim':    self.time2vec_dim,
            'n_critic':        self.n_critic,
            'gp_lambda':       self.gp_lambda,
            'bg_lambda':       self.bg_lambda,
            'ct_lambda':       self.ct_lambda,
        }
        sup.create_json(model_params,
                        os.path.join(params_dir, 'model_parameters.json'))

        self.log.to_csv(
            os.path.join(params_dir, f'{log_name}_ASIS.csv'),
            index=False, encoding='utf-8')
        self.log_test.to_csv(
            os.path.join(params_dir, 'test_log.csv'),
            index=False, encoding='utf-8')
