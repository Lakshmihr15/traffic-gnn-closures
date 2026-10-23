"""METR-LA loading, identical to DCRNN's scripts/generate_training_data.py.

vel.csv holds exactly the values of metr-la.h5 in the same column order (checked against the
test-set ground truth shipped with DCRNN_PyTorch); timestamps start 2012-03-01 00:00, every 5 min.
"""
import pickle
import numpy as np
import pandas as pd

ROOT = '/Users/lakshmihr/Downloads/traffic_closure_mvp/'
N_STEPS = 34272


def load_speed():
    v = pd.read_csv(ROOT + 'stgcn/data/metr-la/vel.csv', header=None).to_numpy(np.float32)
    idx = pd.date_range('2012-03-01', periods=N_STEPS, freq='5min')
    return pd.DataFrame(v, index=idx)


def load_graph():
    with open(ROOT + 'DCRNN_PyTorch/data/sensor_graph/adj_mx.pkl', 'rb') as f:
        ids, id2ind, A = pickle.load(f, encoding='latin1')
    return ids, id2ind, A.astype(np.float64)


def time_of_day(df):
    return ((df.index - df.index.normalize()) / pd.Timedelta(days=1)).to_numpy(np.float32)


def splits():
    """Sample t uses rows t-11..t as input and t+1..t+12 as target, t in [11, 34260)."""
    n = N_STEPS - 23
    n_test, n_train = round(n * 0.2), round(n * 0.7)
    t = np.arange(11, N_STEPS - 12)
    return t[:n_train], t[n_train:n - n_test], t[n - n_test:]


class Windows:
    def __init__(self, df):
        self.v = df.to_numpy(np.float32)
        self.tod = time_of_day(df)
        tr, _, _ = splits()
        # scaler: mean/std of the training x array (zeros included), as in DCRNN
        xtr = np.stack([self.v[t - 11:t + 1] for t in tr[::1]])
        self.mean, self.std = float(xtr.mean()), float(xtr.std())

    def x(self, ts, speed=None):
        """Model input for anchor times ts: (B, 12, N, 2), scaled. `speed` overrides raw speeds."""
        v = self.v if speed is None else speed
        sp = np.stack([v[t - 11:t + 1] for t in ts])
        tod = np.stack([np.repeat(self.tod[t - 11:t + 1, None], v.shape[1], 1) for t in ts])
        return np.stack([(sp - self.mean) / self.std, tod], -1).astype(np.float32)

    def y(self, ts):
        return np.stack([self.v[t + 1:t + 13] for t in ts])

    def unscale(self, p):
        return p * self.std + self.mean
