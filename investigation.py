"""Checkpoint additions (answering the proposal feedback), all with the pretrained DCRNN, no training.

A. Graph ablation: how much does the trained DCRNN use its neighbors at all?
   Replace the road graph by the identity at inference (each sensor sees only itself) and re-measure accuracy.
B. Factual replay on ISOLATED real jams (every 1-hop neighbor > 85% of typical speed in the 15 min
   before onset). No simulation: the model sees the real data 15 min into the jam.
   - neighbor-speed accuracy over the next hour vs baselines, incl. an empirical spillover baseline
   - attributable spillover: forecast with the jammed sensor's real history minus forecast with that
     history replaced by its typical speed
C. Consistency check of the simulation: the transplant closure (same jam pasted into a normal window one
   week earlier/later) vs the replay-attributable spillover, event by event.
"""
import json
import os
import numpy as np
import pandas as pd
import torch
from data import load_speed, load_graph, Windows, splits, ROOT
from events import ratio_matrix, find_events
from closure_exp import neighbor_sets, WEEK
from dcrnn import DCRNN

torch.set_num_threads(2)
W_NPZ = ROOT + 'dcrnn_la_weights.npz'
ISO, ANCHOR = 0.85, 3            # isolation threshold; anchor = onset + 3 steps (15 min into the jam)
rng = np.random.default_rng(0)


def boot(x, n=2000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return [np.nan] * 3
    b = np.median(rng.choice(x, (n, len(x))), 1)
    return [float(np.median(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def predict(model, W, ts, speed=None, bs=128):
    out = []
    for i in range(0, len(ts), bs):
        out.append(W.unscale(model(torch.tensor(W.x(ts[i:i + bs], speed))).numpy()))
    return np.concatenate(out)


def isolated(ev, r, up, dn):
    keep = []
    for e in ev.itertuples():
        nb = np.r_[up[e.sensor], dn[e.sensor]]
        w = r[e.onset_row - 3:e.onset_row + 1, nb]
        keep.append(len(nb) > 0 and not np.isnan(w).all() and np.nanmin(w) > ISO)
    return ev[np.array(keep)].copy()


def main():
    df = load_speed(); raw = df.to_numpy(); W = Windows(df)
    _, _, A = load_graph()
    r, T = ratio_matrix(df)
    up, dn, two = neighbor_sets(A)
    tr, _, te = splits()
    nan_raw = np.where(raw > 0, raw, np.nan)
    out = {}

    # ---------------- A. graph ablation on the test set (every 4th window) ----------------
    ev_te = pd.read_csv(ROOT + 'results/events_test.csv')
    in_ev = np.zeros(raw.shape, bool)
    for e in ev_te.itertuples():
        in_ev[e.onset_row:e.onset_row + e.length, e.sensor] = True
    ts = te[::4] if not os.environ.get('SMOKE') else te[:8]
    Y = np.stack([raw[t + 1:t + 13] for t in ts]); E = np.stack([in_ev[t + 1:t + 13] for t in ts])
    abl = {}
    # 'no neighbors': both diffusion matrices replaced by the identity, so each sensor only sees its own
    # history (its own reading is kept, unlike zeroing the terms, which also removes the self-loop).
    # Direction-only versions (one matrix = identity) were tried and dropped: they push the model further
    # out of distribution than removing all neighbors, so they say nothing about direction.
    for name in ['full', 'no neighbors']:
        m = DCRNN(W_NPZ, A)
        if name == 'no neighbors':
            m.enc[0].supports[0] = torch.eye(A.shape[0]); m.enc[0].supports[1] = torch.eye(A.shape[0])  # shared list
        P = predict(m, W, ts)
        res = {}
        for h in (3, 12):
            p, y, e = P[:, h - 1], Y[:, h - 1], E[:, h - 1]
            ok = y > 0
            res[f'{h * 5} min'] = dict(all=float(np.abs(p - y)[ok].mean()),
                                       normal=float(np.abs(p - y)[ok & ~e].mean()),
                                       event=float(np.abs(p - y)[ok & e].mean()))
        abl[name] = res
        print('A', name, res, flush=True)
    out['A_graph_ablation'] = abl
    out['A_windows'] = int(len(ts))

    # ---------------- B. isolated real jams ----------------
    ev_all = find_events(df, first_row=12)
    train_end = tr[-1] + 1
    ev_tr = isolated(ev_all[ev_all.onset_row + ANCHOR + 13 < train_end], r, up, dn)
    ev_iso = isolated(ev_te, r, up, dn)
    ev_iso = ev_iso[ev_iso.onset_row + ANCHOR + 13 < len(df)]
    out['B_n_isolated_train'], out['B_n_isolated_test'] = int(len(ev_tr)), int(len(ev_iso))
    out['B_n_isolated_test_sensors'] = int(ev_iso.sensor.nunique())
    ev_iso.to_csv(ROOT + 'results/events_isolated_085.csv', index=False)

    def obs_slow(t):                                           # (12, N) observed slowdown in mph
        return T[t + 1:t + 13] - nan_raw[t + 1:t + 13]

    # empirical spillover baseline: mean observed neighbor slowdown per horizon, learned on TRAIN jams
    prof = {'up': [], 'dn': []}
    for e in ev_tr.itertuples():
        t = e.onset_row + ANCHOR; o = obs_slow(t)
        prof['up'].append(np.nanmean(o[:, up[e.sensor]], 1)); prof['dn'].append(np.nanmean(o[:, dn[e.sensor]], 1))
    prof = {k: np.nanmean(np.stack(v), 0) for k, v in prof.items()}       # (12,) each
    out['B_empirical_profile_mph'] = {k: [round(float(x), 2) for x in v] for k, v in prof.items()}

    # replay: real input, and the same input with the jammed sensor's history set to its typical speed
    ts_e = (ev_iso.onset_row + ANCHOR).to_numpy()
    base = DCRNN(W_NPZ, A)
    P_real = predict(base, W, ts_e)
    P_nojam = []
    for e, t in zip(ev_iso.itertuples(), ts_e):
        spd = raw.copy(); spd[t - 11:t + 1, e.sensor] = T[t - 11:t + 1, e.sensor]
        P_nojam.append(predict(base, W, [t], spd)[0])
    P_nojam = np.stack(P_nojam)

    # C. transplant (simulation) for the same isolated jams, into a normal window a week away
    P_tb, P_tt, has_t = [], [], []
    for e, t in zip(ev_iso.itertuples(), ts_e):
        c = e.sensor; ok = False
        for t0 in (t - WEEK, t + WEEK):
            if t0 - 11 < 0 or t0 + 12 >= len(df):
                continue
            seg = slice(t0 - 11, t0 + 13)
            if (raw[seg, c] > 0).all() and np.nanmin(r[seg, c]) > 0.8:
                spd = raw.copy(); spd[t0 - 11:t0 + 1, c] = raw[t - 11:t + 1, c]
                pb, pt = predict(base, W, [t0, t0], None)[0], predict(base, W, [t0], spd)[0]
                P_tb.append(pb); P_tt.append(pt); ok = True
                break
        if not ok:
            P_tb.append(np.full((12, raw.shape[1]), np.nan)); P_tt.append(np.full((12, raw.shape[1]), np.nan))
        has_t.append(ok)

    rows, err = [], {k: {'up': [], 'dn': []} for k in ['DCRNN replay', 'Last value', 'Historical average', 'Empirical spillover']}
    for k, (e, t) in enumerate(zip(ev_iso.itertuples(), ts_e)):
        c = e.sensor; y = nan_raw[t + 1:t + 13]; o = obs_slow(t)
        last = np.repeat(np.where(raw[t] > 0, raw[t], T[t])[None], 12, 0)
        HA = T[t + 1:t + 13]
        preds = {'DCRNN replay': P_real[k], 'Last value': last, 'Historical average': HA}
        for side, nb in (('up', up[c]), ('dn', dn[c])):
            preds_s = dict(preds); preds_s['Empirical spillover'] = HA - prof[side][:, None]
            for name, p in preds_s.items():
                err[name][side].append(float(np.nanmean(np.abs(p[:, nb] - y[:, nb]))))
        attr = P_nojam[k] - P_real[k]                         # slowdown the model attributes to the jam
        tran = P_tb[k] - P_tt[k]
        f = lambda m, idx: float(np.nanmean(m[:, idx])) if len(idx) else np.nan
        rows.append(dict(sensor=c, anchor=int(t), has_transplant=has_t[k],
                         obs_self=f(o, [c]), obs_up=f(o, up[c]), obs_dn=f(o, dn[c]), obs_hop2=f(o, two[c]),
                         pred_self=f(HA - P_real[k], [c]), pred_up=f(HA - P_real[k], up[c]), pred_dn=f(HA - P_real[k], dn[c]),
                         attr_self=f(attr, [c]), attr_up=f(attr, up[c]), attr_dn=f(attr, dn[c]), attr_hop2=f(attr, two[c]),
                         tran_self=f(tran, [c]), tran_up=f(tran, up[c]), tran_dn=f(tran, dn[c])))
    R = pd.DataFrame(rows); R.to_csv(ROOT + 'results/replay_isolated.csv', index=False)
    out['B_neighbor_MAE_mph'] = {n: {s: boot(v) for s, v in d.items()} for n, d in err.items()}
    out['B_slowdown_medians'] = {k: boot(R[k]) for k in R.columns if k.split('_')[0] in ('obs', 'pred', 'attr', 'tran')}
    out['B_share_obs_up_gt_dn'] = float((R.obs_up > R.obs_dn).mean())
    Rt = R[R.has_transplant]
    out['C_n'] = int(len(Rt))
    out['C_spearman_self_attr_vs_tran'] = float(Rt[['attr_self', 'tran_self']].corr('spearman').iloc[0, 1])
    out['C_spearman_up_attr_vs_tran'] = float(Rt[['attr_up', 'tran_up']].corr('spearman').iloc[0, 1])
    json.dump(out, open(ROOT + 'results/investigation.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
