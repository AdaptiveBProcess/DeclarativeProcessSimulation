import tensorflow as tf
from tensorflow.keras import layers, Model


def build_generator(latent_dim, max_trace_size, n_activities, n_roles):
    """
    Input : noise vector (latent_dim,)
    Output: padded trace (max_trace_size, n_activities + n_roles + 2)
              [:n_activities]           -> activity distribution (softmax)
              [n_activities:-2]         -> role distribution    (softmax)
              [-2]                      -> dur_norm              (sigmoid)
              [-1]                      -> wait_norm             (sigmoid)
    """
    noise = layers.Input(shape=(latent_dim,), name='noise')

    x = layers.Dense(256, activation='relu')(noise)
    x = layers.Dense(max_trace_size * 64, activation='relu')(x)
    x = layers.Reshape((max_trace_size, 64))(x)
    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.GRU(64, return_sequences=True)(x)

    ac_out = layers.TimeDistributed(
        layers.Dense(n_activities, activation='softmax'), name='ac')(x)
    rl_out = layers.TimeDistributed(
        layers.Dense(n_roles, activation='softmax'), name='rl')(x)
    tm_out = layers.TimeDistributed(
        layers.Dense(2, activation='sigmoid'), name='time')(x)

    out = layers.Concatenate(axis=-1, name='output')([ac_out, rl_out, tm_out])
    return Model(noise, out, name='generator')


def build_discriminator(max_trace_size, n_activities, n_roles):
    """
    Input : trace (max_trace_size, n_activities + n_roles + 2)
    Output: validity scalar (sigmoid)
    """
    feat_dim = n_activities + n_roles + 2
    seq_in = layers.Input(shape=(max_trace_size, feat_dim), name='trace')

    x = layers.GRU(128, return_sequences=True)(seq_in)
    x = layers.GRU(64)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid', name='validity')(x)

    return Model(seq_in, out, name='discriminator')
