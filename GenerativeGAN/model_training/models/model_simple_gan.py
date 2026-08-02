import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


class Time2Vec(layers.Layer):
    """
    Time2Vec (Kazemi et al., 2019): representa un escalar de tiempo como un vector
    [w0*t + b0, sin(w1*t + b1), ..., sin(w(k-1)*t + b(k-1))] de dimension `dim`.
    Componente 0 es lineal (captura tendencia), el resto es periodico (captura
    patrones ciclicos). Entrada esperada: (..., 1).
    """

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1, 1), initializer='glorot_uniform', trainable=True)
        self.b0 = self.add_weight(name='b0', shape=(1,), initializer='zeros', trainable=True)
        self.w = self.add_weight(name='w', shape=(1, self.dim - 1), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='b', shape=(self.dim - 1,), initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        linear = tf.matmul(x, self.w0) + self.b0
        periodic = tf.sin(tf.matmul(x, self.w) + self.b)
        return tf.concat([linear, periodic], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'dim': self.dim})
        return config


class PositionalEncoding(layers.Layer):
    """Encoding posicional sinusoidal fijo (no entrenable), Vaswani et al. (2017)."""

    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_encoding = tf.constant(
            self._build(max_len, d_model), dtype=tf.float32)

    @staticmethod
    def _build(max_len, d_model):
        positions = np.arange(max_len)[:, None].astype(np.float32)
        dims = np.arange(d_model)[None, :].astype(np.float32)
        angle_rates = 1.0 / np.power(10000.0, (2 * (dims // 2)) / np.float32(d_model))
        angle_rads = positions * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return angle_rads[None, ...]

    def call(self, x):
        return x + self.pos_encoding

    def get_config(self):
        config = super().get_config()
        config.update({'max_len': self.max_len, 'd_model': self.d_model})
        return config


class TransformerBlock(layers.Layer):
    """
    Bloque Transformer Pre-LN: LayerNorm -> Multi-Head Self-Attention (no-causal)
    -> residual -> LayerNorm -> Feed-Forward -> residual.
    """

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model),
        ])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        x_norm = self.ln1(x)
        attn_out = self.att(x_norm, x_norm, training=training)
        x = x + self.drop1(attn_out, training=training)

        x_norm2 = self.ln2(x)
        ffn_out = self.ffn(x_norm2, training=training)
        x = x + self.drop2(ffn_out, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout_rate,
        })
        return config


# Capas custom que deben registrarse como custom_objects al cargar el modelo con
# tf.keras.models.load_model (ver GenerativeGAN/model_prediction/gan_predictor.py).
CUSTOM_OBJECTS = {
    'Time2Vec': Time2Vec,
    'PositionalEncoding': PositionalEncoding,
    'TransformerBlock': TransformerBlock,
}


def build_generator(latent_dim, max_trace_size, n_activities, n_roles,
                     d_model=64, num_heads=4, ff_dim=256, num_blocks=3, dropout=0.1):
    """
    Input : ruido (latent_dim,)
    Output: traza (max_trace_size, n_activities + n_roles + 2)
              [:n_activities]           -> distribucion actividad (softmax)
              [n_activities:-2]         -> distribucion rol       (softmax)
              [-2]                      -> dur_norm               (sigmoid)
              [-1]                      -> wait_norm               (sigmoid)
    """
    noise = layers.Input(shape=(latent_dim,), name='noise')

    x = layers.Dense(max_trace_size * d_model, activation='relu')(noise)
    x = layers.Reshape((max_trace_size, d_model))(x)
    x = PositionalEncoding(max_trace_size, d_model)(x)

    for i in range(num_blocks):
        x = TransformerBlock(d_model, num_heads, ff_dim, dropout, name=f'gen_block_{i}')(x)

    ac_out = layers.TimeDistributed(
        layers.Dense(n_activities, activation='softmax'), name='ac')(x)
    rl_out = layers.TimeDistributed(
        layers.Dense(n_roles, activation='softmax'), name='rl')(x)
    tm_out = layers.TimeDistributed(
        layers.Dense(2, activation='sigmoid'), name='time')(x)

    out = layers.Concatenate(axis=-1, name='output')([ac_out, rl_out, tm_out])
    return Model(noise, out, name='generator')


def build_discriminator(max_trace_size, n_activities, n_roles,
                         d_model=64, num_heads=4, ff_dim=256, num_blocks=3,
                         time2vec_dim=16, dropout=0.1):
    """
    Input : traza (max_trace_size, n_activities + n_roles + 2)
    Output: score Wasserstein escalar (SIN activacion — no es un discriminador
            binario, es un critico WGAN; ver gan_trainer.py::_train_gan).
    """
    feat_dim = n_activities + n_roles + 2
    cat_dim = n_activities + n_roles
    seq_in = layers.Input(shape=(max_trace_size, feat_dim), name='trace')

    cat_feats = layers.Lambda(
        lambda t: t[:, :, :cat_dim], name='split_cat')(seq_in)
    dur_feat = layers.Lambda(
        lambda t: t[:, :, cat_dim:cat_dim + 1], name='split_dur')(seq_in)
    wait_feat = layers.Lambda(
        lambda t: t[:, :, cat_dim + 1:cat_dim + 2], name='split_wait')(seq_in)

    cat_proj = layers.TimeDistributed(
        layers.Dense(d_model // 2, use_bias=False), name='cat_proj')(cat_feats)
    dur_enc = layers.TimeDistributed(
        Time2Vec(time2vec_dim), name='dur_enc')(dur_feat)
    wait_enc = layers.TimeDistributed(
        Time2Vec(time2vec_dim), name='wait_enc')(wait_feat)

    x = layers.Concatenate(axis=-1)([cat_proj, dur_enc, wait_enc])
    x = layers.Dense(d_model)(x)
    x = PositionalEncoding(max_trace_size, d_model)(x)

    for i in range(num_blocks):
        x = TransformerBlock(d_model, num_heads, ff_dim, dropout, name=f'disc_block_{i}')(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, name='validity')(x)

    return Model(seq_in, out, name='discriminator')
