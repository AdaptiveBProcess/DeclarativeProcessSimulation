import numpy as np
from tensorflow.keras.utils import to_categorical


class GANSamplesCreator:
    """
    Converts a processed event log (with dur_norm and wait_norm columns) into
    a 3-D numpy array ready for GAN training.

    Output shape: (n_traces, max_trace_size, n_activities + n_roles + 2)
      Each time-step encodes: [ac_onehot | rl_onehot | dur_norm | wait_norm]

    Positions after the last event:
      relleno_end=False (GANTrainer v1): zero-padded.
      relleno_end=True  (GANTrainerV2): filled with the 'end' activity and role
        one-hot (dur = wait = 0). The generator ends in a softmax, so it cannot
        produce all-zero rows; with zero padding it never sees 'end' and the
        decoder (which stops at the first 'end') cannot learn where a trace ends.
        Requires max_trace_size >= longest trace + 1 so every trace has an 'end'.

    Events are ordered by start_timestamp with a stable sort (ties keep the order
    of the log).
    """

    def create_samples(self, log, ac_index, rl_index, max_trace_size, relleno_end=False):
        n_ac = len(ac_index)
        n_rl = len(rl_index)
        feat_dim = n_ac + n_rl + 2
        if relleno_end:
            fila_end = np.concatenate([to_categorical(ac_index['end'], num_classes=n_ac),
                                       to_categorical(rl_index['end'], num_classes=n_rl),
                                       [0.0, 0.0]]).astype(np.float32)

        matrices = []
        for _, group in log.groupby('caseid'):
            group = group.sort_values('start_timestamp', kind='mergesort').reset_index(drop=True)
            matrix = np.zeros((max_trace_size, feat_dim), dtype=np.float32)

            n_eventos = 0
            for pos, (_, row) in enumerate(group.iterrows()):
                if pos >= max_trace_size:
                    break
                ac_idx = ac_index.get(row['task'], 0)
                rl_idx = rl_index.get(row['role'], 0)
                ac_vec = to_categorical(ac_idx, num_classes=n_ac)
                rl_vec = to_categorical(rl_idx, num_classes=n_rl)
                dur_n = float(row.get('dur_norm', 0.0))
                wait_n = float(row.get('wait_norm', 0.0))
                matrix[pos] = np.concatenate([ac_vec, rl_vec, [dur_n, wait_n]])
                n_eventos = pos + 1

            if relleno_end:
                matrix[n_eventos:] = fila_end

            matrices.append(matrix)

        return np.array(matrices, dtype=np.float32)
