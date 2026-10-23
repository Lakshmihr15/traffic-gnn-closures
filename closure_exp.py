"""First closure experiment (E2 pilot) on pretrained DCRNN, METR-LA test period.

For each selected real event at sensor c (onset row s), anchor t = s + 5 (event 30 min old):
  observed slowdown  = typical - actual speed, rows t+1..t+12, every sensor            (mph)
  baseline window    = same time of week, one week earlier (or later) when c was normal
  variants on the baseline window, each giving predicted slowdown = pred(base) - pred(variant):
    transplant : c's 12 input speeds replaced by the real event's speeds (graph unchanged)
    both       : transplant + every edge into/out of c removed from the graph
    cut        : graph edges removed only, inputs unchanged
"""
import numpy as np
import pandas as pd
import torch
from data import load_speed, load_graph, Windows, ROOT
from events import ratio_matrix
from dcrnn import DCRNN

torch.set_num_threads(2)
WEEK = 2016
W_NPZ = ROOT + 'dcrnn_la_weights.npz'


def close_sensor(A, c):
    A2 = A.copy()
    A2[c, :] = 0.0
    A2[:, c] = 0.0
    A2[c, c] = A[c, c]
    return A2


def neighbor_sets(A):
    """Upstream: i with A[i,c] > 0 only (i can drive to c). Downstream: j with A[c,j] > 0 only."""
    G = (A > 0) & ~np.eye(len(A), dtype=bool)
    up = [np.nonzero(G[:, c] & ~G[c, :])[0] for c in range(len(A))]
    dn = [np.nonzero(G[c, :] & ~G[:, c])[0] for c in range(len(A))]
    one = [np.nonzero(G[:, c] | G[c, :])[0] for c in range(len(A))]
    U = G | G.T
    two = []
    for c in range(len(A)):
        h1 = set(one[c])
        h2 = set(np.nonzero(U[list(h1)].any(0))[0]) if h1 else set()
        two.append(np.array(sorted(h2 - h1 - {c}), dtype=int))
    return up, dn, two


def summarize(sl, c, up, dn, two):
    """sl: (12, N) slowdown in mph (positive = slower). Horizon-mean per sensor, then grouped."""
    m = np.nanmean(sl, 0)
    others = np.delete(m, c)
    f = lambda idx: float(np.nanmean(m[idx])) if len(idx) else np.nan
    return dict(self=float(m[c]), up1=f(up[c]), dn1=f(dn[c]), hop2=f(two[c]),
                reach5=int(np.nansum(others > 5)))


def select_events(ev, r, raw, up, dn, n_rows):
    ev = ev[(ev.length >= 6) & (ev.min_ratio >= 0.15)].copy()
    ev = ev[[len(up[c]) > 0 and len(dn[c]) > 0 for c in ev.sensor]]
    ev = ev.sort_values(['length', 'onset_row'], ascending=[False, True], kind='mergesort')        # longest valid event per sensor
    keep, done = [], set()
    for _, e in ev.iterrows():
        c, t = int(e.sensor), int(e.onset_row) + 5
        if c in done or t + 12 >= n_rows:
            continue
        for t0 in (t - WEEK, t + WEEK):
            if t0 - 11 < 0 or t0 + 12 >= n_rows:
                continue
            seg = slice(t0 - 11, t0 + 13)
            if (raw[seg, c] > 0).all() and np.nanmin(r[seg, c]) > 0.8:
                keep.append((c, t, t0, int(e.length), float(e.min_ratio)))
                done.add(c)
                break
    return pd.DataFrame(keep, columns=['sensor', 'anchor', 'base_anchor', 'length', 'min_ratio'])


def main():
    df = load_speed()
    W = Windows(df)
    _, _, A = load_graph()
    r, T = ratio_matrix(df)
    raw = df.to_numpy()
    up, dn, two = neighbor_sets(A)
    ev = pd.read_csv(ROOT + 'results/events_test.csv')
    sel = select_events(ev, r, raw, up, dn, len(df))
    print(len(sel), 'events selected', flush=True)

    base_model = DCRNN(W_NPZ, A)
    rows = []
    for k, e in sel.iterrows():
        c, t, t0 = int(e.sensor), int(e.anchor), int(e.base_anchor)
        spd = raw.copy()
        spd[t0 - 11:t0 + 1, c] = raw[t - 11:t + 1, c]          # transplant the real episode
        x_base = W.x([t0])
        x_tr = W.x([t0], speed=spd)
        cut_model = DCRNN(W_NPZ, close_sensor(A, c))
        pb, ptr = (W.unscale(p) for p in base_model(torch.tensor(np.concatenate([x_base, x_tr]))).numpy())
        pcut, pboth = (W.unscale(p) for p in cut_model(torch.tensor(np.concatenate([x_base, x_tr]))).numpy())
        obs = T[t + 1:t + 13] - np.where(raw[t + 1:t + 13] > 0, raw[t + 1:t + 13], np.nan)
        out = dict(sensor=c, anchor=t, base_anchor=t0, length=e.length, min_ratio=e.min_ratio)
        for name, sl in [('obs', obs), ('transplant', pb - ptr), ('both', pb - pboth), ('cut', pb - pcut)]:
            for key, val in summarize(sl, c, up, dn, two).items():
                out[f'{name}_{key}'] = val
        rows.append(out)
        if k % 10 == 0:
            print(k, flush=True)
            pd.DataFrame(rows).to_csv(ROOT + 'results/closure_pilot.csv', index=False)
    res = add_controls(pd.DataFrame(rows), A, raw, T, up, two)
    res.to_csv(ROOT + 'results/closure_pilot.csv', index=False)
    print('done')


def add_controls(c, A, raw, T, up, two):
    """obs_far: observed slowdown at sensors more than 2 hops from the event (background congestion).
    obs_up1_pre: upstream neighbors' slowdown over the 15 min up to the anchor (already slow?)."""
    G = (A > 0) | (A > 0).T
    far, pre = [], []
    for _, e in c.iterrows():
        s, t = int(e.sensor), int(e.anchor)
        near = set(np.nonzero(G[s] | G[:, s])[0]) | set(two[s]) | {s}
        fidx = [i for i in range(len(A)) if i not in near]
        obs = T[t + 1:t + 13] - np.where(raw[t + 1:t + 13] > 0, raw[t + 1:t + 13], np.nan)
        far.append(np.nanmean(np.nanmean(obs, 0)[fidx]))
        p = T[t - 2:t + 1] - np.where(raw[t - 2:t + 1] > 0, raw[t - 2:t + 1], np.nan)
        pre.append(np.nanmean(np.nanmean(p, 0)[up[s]]))
    c['obs_far'], c['obs_up1_pre'] = far, pre
    return c


if __name__ == '__main__':
    main()
