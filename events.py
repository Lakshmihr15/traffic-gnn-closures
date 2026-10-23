"""Typical speed (training period only) and real non-recurring slowdown events."""
import numpy as np
import pandas as pd
from data import load_speed, splits, ROOT

DROP, MIN_STEPS, BEFORE = 0.7, 4, 0.9


def typical_speed(df):
    tr, _, _ = splits()
    train_rows = tr[-1] + 1                      # rows seen as inputs during training
    spd = df.replace(0, np.nan)
    slot = (df.index.dayofweek * 288 + df.index.hour * 12 + df.index.minute // 5).to_numpy()
    typ = spd.iloc[:train_rows].groupby(slot[:train_rows]).median()   # 2016 x 207
    return typ, slot


def ratio_matrix(df):
    typ, slot = typical_speed(df)
    spd = df.replace(0, np.nan).to_numpy()
    T = typ.to_numpy()[slot]                     # typical speed for every row
    return spd / T, T


def find_events(df, first_row=0):
    """Event = speed < 70% of typical for >= 20 min, after being > 90% of typical 15 min before.
    Returns DataFrame(sensor, onset_row, length, min_ratio)."""
    r, _ = ratio_matrix(df)
    raw = df.to_numpy()
    rows = []
    for c in range(r.shape[1]):
        low = r[:, c] < DROP                      # NaN compares False
        t = first_row
        while t < len(low):
            if low[t] and (t == 0 or not low[t - 1]):
                e = t
                while e < len(low) and low[e]:
                    e += 1
                L = e - t
                ok = L >= MIN_STEPS and t >= 3 and r[t - 3, c] > BEFORE
                seg = raw[max(t - 12, 0):e, c]
                ok = ok and (seg > 0).all() and np.ptp(seg) > 0   # no missing/stuck readings
                if ok:
                    rows.append((c, t, L, float(np.nanmin(r[t:e, c]))))
                t = e
            else:
                t += 1
    return pd.DataFrame(rows, columns=['sensor', 'onset_row', 'length', 'min_ratio'])


if __name__ == '__main__':
    df = load_speed()
    _, _, te = splits()
    ev = find_events(df, first_row=te[0] - 11)
    ev['time'] = df.index[ev.onset_row]
    ev.to_csv(ROOT + 'results/events_test.csv', index=False)
    print(len(ev), 'events,', ev.sensor.nunique(), 'sensors')
    print(ev.describe())
