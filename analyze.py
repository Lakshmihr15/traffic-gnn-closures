"""Level 1 accuracy + baselines, Level 2 (event vs normal) accuracy, and closure pilot summary."""
import json
import numpy as np
import pandas as pd
from data import load_speed, splits, Windows, ROOT
from events import ratio_matrix

rng = np.random.default_rng(0)


def masked(p, y):
    m = y > 0
    e = p[m] - y[m]
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                MAPE=float((np.abs(e) / y[m]).mean() * 100))


def boot_median(x, n=2000):
    x = np.asarray(x)[~np.isnan(x)]
    b = np.median(rng.choice(x, (n, len(x))), 1)
    return [float(np.median(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    df = load_speed()
    raw = df.to_numpy()
    _, _, te = splits()
    _, T = ratio_matrix(df)
    Y = np.stack([raw[t + 1:t + 13] for t in te])                    # (S, 12, N)
    P = np.load(ROOT + 'results/test_pred.npy')
    HA = np.stack([T[t + 1:t + 13] for t in te])                      # historical average
    last = raw[te][:, None, :].repeat(12, 1)                          # copy last value
    last = np.where(last > 0, last, HA)
    out = {'level1': {}}
    for name, pred in [('DCRNN (pretrained)', P), ('Historical average', HA), ('Last value', last)]:
        out['level1'][name] = {f'{h * 5} min': masked(pred[:, h - 1], Y[:, h - 1]) for h in (3, 6, 12)}

    # Level 2: target cells inside a real event vs all other cells
    ev = pd.read_csv(ROOT + 'results/events_test.csv')
    in_ev = np.zeros(raw.shape, bool)
    for _, e in ev.iterrows():
        in_ev[e.onset_row:e.onset_row + e.length, e.sensor] = True
    E = np.stack([in_ev[t + 1:t + 13] for t in te])
    out['level2'] = {}
    for name, pred in [('DCRNN (pretrained)', P), ('Historical average', HA), ('Last value', last)]:
        res = {}
        for h in (3, 6, 12):
            p, y, e = pred[:, h - 1], Y[:, h - 1], E[:, h - 1]
            ok = y > 0
            res[f'{h * 5} min'] = dict(event=float(np.abs(p - y)[ok & e].mean()),
                                       normal=float(np.abs(p - y)[ok & ~e].mean()))
        out['level2'][name] = res
    out['level2_share_event_cells'] = float(E[Y > 0].mean())
    out['n_events'] = int(len(ev))
    out['n_event_sensors'] = int(ev.sensor.nunique())

    # Closure pilot
    c = pd.read_csv(ROOT + 'results/closure_pilot.csv')
    out['n_closures'] = int(len(c))
    summ = {}
    for v in ['obs', 'transplant', 'both', 'cut']:
        summ[v] = {k: boot_median(c[f'{v}_{k}']) for k in ['self', 'up1', 'dn1', 'hop2', 'reach5']}
        d = (c[f'{v}_up1'] - c[f'{v}_dn1']).dropna()
        summ[v]['share_up_gt_dn'] = float((d > 0).mean())
        summ[v]['up_minus_dn'] = boot_median(d)
    summ['obs']['far'] = boot_median(c['obs_far'])               # sensors > 2 hops away: background
    summ['obs']['up1_at_anchor'] = boot_median(c['obs_up1_pre'])
    for k in ['up1', 'dn1', 'hop2']:
        summ['obs'][f'{k}_minus_far'] = boot_median(c[f'obs_{k}'] - c['obs_far'])
    out['closure'] = summ
    out['corr_up1_obs_vs_transplant'] = float(c[['obs_up1', 'transplant_up1']].corr('spearman').iloc[0, 1])
    out['corr_up1_obs_vs_both'] = float(c[['obs_up1', 'both_up1']].corr('spearman').iloc[0, 1])
    json.dump(out, open(ROOT + 'results/summary.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
