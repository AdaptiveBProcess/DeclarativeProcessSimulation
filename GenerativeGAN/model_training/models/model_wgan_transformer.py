"""
Transformer-based GAN v2 architecture.

Generator : noise → Dense → Reshape → Transformer blocks → [softmax(ac) | softmax(rl) | sigmoid(time)]
Discriminator: trace → embedding projection + Time2Vec → Transformer blocks → Wasserstein score

Used with WGAN-GP training (no sigmoid on discriminator output).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


# ── Time2Vec ─────────────────────────────────────────────────────────────────

class Time2Vec(tf.keras.layers.Layer):
    """
    Learned temporal encoding for a scalar feature.
    Output: [linear, sin(ω₁t+φ₁), ..., sin(ωₖt+φₖ)]  — shape (..., output_dim)
    """

    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.W = self.add_weight(
            name='W', shape=(1, self.output_dim),
            initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(
            name='b', shape=(1, self.output_dim),
            initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        x = tf.cast(tf.expand_dims(x, -1), tf.float32)        # (..., 1)
        x_lin = self.W[:, :1] * x + self.b[:, :1]             # linear component
        x_sin = tf.math.sin(self.W[:, 1:] * x + self.b[:, 1:])  # periodic components
        return tf.concat([x_lin, x_sin], axis=-1)              # (..., output_dim)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'output_dim': self.output_dim})
        return cfg


# ── Positional Encoding ──────────────────────────────────────────────────────

def positional_encoding(max_len: int, d_model: int) -> tf.Tensor:
    """Sinusoidal positional encoding, shape (1, max_len, d_model)."""
    pos  = np.arange(max_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = pos / np.power(10000.0, (2 * (dims // 2)) / np.float32(d_model))
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, ...], tf.float32)  # (1, max_len, d_model)


# ── Transformer Block ────────────────────────────────────────────────────────

class TransformerBlock(tf.keras.layers.Layer):
    """Pre-LN Transformer block: LayerNorm → MHA → Residual; LayerNorm → FFN → Residual."""

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model   = d_model
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.dropout   = dropout

        key_dim = max(1, d_model // num_heads)
        self.att   = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.ffn   = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        x_n  = self.norm1(x)
        attn = self.att(x_n, x_n, training=training)
        x    = x + self.drop1(attn, training=training)

        x_n  = self.norm2(x)
        ffn  = self.ffn(x_n)
        x    = x + self.drop2(ffn, training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'd_model':   self.d_model,
            'num_heads': self.num_heads,
            'ff_dim':    self.ff_dim,
            'dropout':   self.dropout,
        })
        return cfg


# ── Generator ────────────────────────────────────────────────────────────────

def build_transformer_generator(
    latent_dim: int,
    max_trace_size: int,
    n_activities: int,
    n_roles: int,
    d_model: int = 64,
    num_heads: int = 4,
    num_blocks: int = 2,
    ff_dim: int = 128,
    dropout: float = 0.1,
) -> Model:
    """
    Transformer Generator.
      Input : noise (latent_dim,)
      Output: (max_trace_size, n_activities + n_roles + 2)
                [:n_activities]  → softmax activity distribution
                [n_activities:-2]→ softmax role distribution
                [-2:]            → sigmoid (dur_norm, wait_norm)
    """
    noise = layers.Input(shape=(latent_dim,), name='noise')

    x = layers.Dense(max_trace_size * d_model, activation='relu')(noise)
    x = layers.Reshape((max_trace_size, d_model))(x)

    pos_enc = positional_encoding(max_trace_size, d_model)
    x = x + pos_enc

    for i in range(num_blocks):
        x = TransformerBlock(
            d_model, num_heads, ff_dim, dropout, name=f'gen_block_{i}')(x)

    ac_out = layers.TimeDistributed(
        layers.Dense(n_activities, activation='softmax'), name='ac')(x)
    rl_out = layers.TimeDistributed(
        layers.Dense(n_roles, activation='softmax'), name='rl')(x)
    tm_out = layers.TimeDistributed(
        layers.Dense(2, activation='sigmoid'), name='time')(x)

    out = layers.Concatenate(axis=-1, name='output')([ac_out, rl_out, tm_out])
    return Model(noise, out, name='transformer_generator')


# ── Discriminator ────────────────────────────────────────────────────────────

def build_transformer_discriminator(
    max_trace_size: int,
    n_activities: int,
    n_roles: int,
    d_model: int = 64,
    num_heads: int = 4,
    num_blocks: int = 2,
    ff_dim: int = 128,
    dropout: float = 0.1,
    time2vec_dim: int = 8,
) -> Model:
    """
    Transformer Discriminator for WGAN-GP (linear output, no sigmoid).
      Input : (max_trace_size, n_activities + n_roles + 2)
      Output: unbounded Wasserstein score scalar
    """
    n_cat    = n_activities + n_roles
    feat_dim = n_cat + 2
    seq_in   = layers.Input(shape=(max_trace_size, feat_dim), name='trace')

    # Split categorical vs temporal features
    cat_in  = seq_in[:, :, :n_cat]           # (batch, seq, n_ac+n_rl)
    dur_in  = seq_in[:, :, n_cat]            # (batch, seq)
    wait_in = seq_in[:, :, n_cat + 1]        # (batch, seq)

    # Learned dense projection of categorical features (equivalent to an embedding)
    cat_proj = layers.TimeDistributed(
        layers.Dense(d_model // 2, use_bias=False), name='cat_embed')(cat_in)

    # Time2Vec encoding of scalar temporal features
    dur_enc  = Time2Vec(time2vec_dim, name='dur_t2v')(dur_in)    # (batch, seq, t2v)
    wait_enc = Time2Vec(time2vec_dim, name='wait_t2v')(wait_in)  # (batch, seq, t2v)

    # Fuse and project to d_model
    x = layers.Concatenate(axis=-1)([cat_proj, dur_enc, wait_enc])
    x = layers.Dense(d_model, name='input_proj')(x)

    pos_enc = positional_encoding(max_trace_size, d_model)
    x = x + pos_enc

    for i in range(num_blocks):
        x = TransformerBlock(
            d_model, num_heads, ff_dim, dropout, name=f'disc_block_{i}')(x)

    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, name='score')(x)   # no sigmoid — WGAN

    return Model(seq_in, out, name='transformer_discriminator')
