import numpy as np
from tensorflow.keras.utils import to_categorical


class GANSamplesCreator:
    """
    Converts a processed event log (with dur_norm and wait_norm columns) into
    a 3-D numpy array ready for GAN training.

    Output shape: (n_traces, max_trace_size, n_activities + n_roles + 2)
      Each time-step encodes: [ac_onehot | rl_onehot | dur_norm | wait_norm]
      Shorter traces are zero-padded at the end.
    """

    def create_samples(self, log, ac_index, rl_index, max_trace_size):
        n_ac = len(ac_index)
        n_rl = len(rl_index)
        feat_dim = n_ac + n_rl + 2

        matrices = []
        for _, group in log.groupby('caseid'):
            group = group.sort_values('start_timestamp').reset_index(drop=True)
            matrix = np.zeros((max_trace_size, feat_dim), dtype=np.float32)

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

            matrices.append(matrix)

        return np.array(matrices, dtype=np.float32)
