"""Learned-recovery test: does the DCRNN expect every slowdown to clear the way everyday congestion does?

    python recovery_prior.py checkpoints/dcrnn_check_plain_ft.pt checkpoints/dcrnn_check_nograph_ft.pt \
        checkpoints/dcrnn_check_slowdown_ft.pt

Explanation under test (after graph gating ruled out the neighbours, graph_gating.py): most slow readings the
model trains on are congestion that clears on schedule, so it learns "slow now -> faster soon" from the
current speed and time of day, and applies it to closures, which do not clear on schedule.

A forecast origin is a (time, sensor) where the sensor is slowed (below 50 mph) at the last input step. Three
groups, decided at the origin with information available then:
  usual congestion   within 5 mph of the typical speed for that time of week, no incident active
  unusual slowdown   10+ mph below typical, no recorded incident active (any other cause)
  closure            10+ mph below typical, freeway closure active at the sensor
For each origin: actual recovery = speed 60 min later minus speed now; predicted recovery = the forecast for
60 min minus speed now; base rate = mean recovery in the TRAINING period of all readings at the same speed
(5 mph bin) and hour of day, i.e. what "slow now -> faster soon" from speed and time alone predicts.

Predictions of the explanation:
  P1 (data)  In training, usual congestion and the base rate recover; closures and unusual slowdowns recover
             much less, so a model that learns the base rate is too optimistic for them.
  P2         For closures, the model's predicted recovery is close to the base rate and well above the actual.
  P3         The too-fast bias is near zero in usual congestion and large in unusual slowdowns and closures
             (the model is wrong where the slowdown is not the usual one, not on slow traffic as such).
  P4 (fix)   Fine-tuning with unusual slowdowns weighted 5x (finetune.py --slowdown-weight 5; only 0.15% of
             those readings are closures) lowers the closure bias, against the same fine-tuning without weights.

Intervals: 95%, resampling closures (closure group) or sensor-days (other groups).
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from baselines import LastValue
from closure_errors import DISRUPTED_MPH
from config import EVENTS, PROCESSED, RESULTS
from graph_gating import bootstrap
from incident_eval import KINDS, Tally
from tgc.data import TrafficData
from tgc.incidents import event_mask
from tgc.training import load_model, pick_device

SLOW_MPH, USUAL_MPH, UNUSUAL_MPH = 50.0, 5.0, 10.0
GROUPS = ["usual congestion", "unusual slowdown", "closure"]
SPEED_BINS = np.arange(0, 75, 5)
STEPS_PER_DAY = 288


def origin_groups(speed, typ_now, active_any, closure_on):
    """(T, N) int8: -1 = not an origin, else index into GROUPS."""
    deficit = typ_now - speed
    slow = (speed > 0) & (speed < SLOW_MPH)
    g = np.full(speed.shape, -1, np.int8)
    g[slow & (deficit < USUAL_MPH) & ~active_any] = 0
    g[slow & (deficit >= UNUSUAL_MPH) & ~active_any] = 1
    g[slow & (deficit >= UNUSUAL_MPH) & closure_on] = 2
    return g


def base_rate(speed, slots, train_t, groups, typical, h=12):
    """Training period: mean 60-min recovery by speed bin x hour (all readings), and per group."""
    t0 = train_t[train_t + h < len(speed)]
    s0, s1 = speed[t0], speed[t0 + h]
    ok = (s0 > 0) & (s1 > 0)
    rec = (s1 - s0)
    sb = np.clip(np.digitize(s0, SPEED_BINS) - 1, 0, len(SPEED_BINS) - 1)
    hour = np.broadcast_to(((slots[t0] % STEPS_PER_DAY) // 12)[:, None], s0.shape)
    table = np.full((len(SPEED_BINS), 24), np.nan, np.float32)
    df = pd.DataFrame({"sb": sb[ok], "hour": hour[ok], "rec": rec[ok]})
    m = df.groupby(["sb", "hour"]).rec.mean()
    table[m.index.get_level_values(0), m.index.get_level_values(1)] = m.to_numpy()
    # fill empty cells with the speed-bin mean
    sbm = df.groupby("sb").rec.mean()
    for b in range(len(SPEED_BINS)):
        table[b, np.isnan(table[b])] = sbm.get(b, 0.0)
    g = groups[t0]
    typ1 = typical[slots[t0 + h]]
    out = {}
    for i, name in enumerate(GROUPS):
        sel = ok & (g == i)
        out[name] = {"origins": int(sel.sum()), "actual_recovery_60": float(rec[sel].mean()),
                     "base_rate_recovery_60": float(table[sb[sel], hour[sel]].mean()),
                     "recovered_share": float((s1[sel] >= typ1[sel] - USUAL_MPH).mean())}
    return table, out


@torch.no_grad()
def run(model, data, groups, cluster, table, since, event_id, masks, batch_size=64):
    """One pass over the test windows: per-origin recovery rows, closure reading rows, per-kind tallies."""
    if isinstance(model, torch.nn.Module):
        model.eval()
    orows, crows, tallies = [], [], {k: Tally() for k in masks}
    table_t = torch.as_tensor(table, device=data.device)
    for x, y, y_idx in data.batches("test", batch_size):
        pred = model.predict(x, y_idx, data) if hasattr(model, "predict") else data.unscale(model(x))[..., 0]
        y = y[..., 0]
        valid = y != 0
        ae = (pred - y).abs()
        for k, m in masks.items():
            tallies[k].add(ae, valid & m[y_idx])
        # Origins
        t0 = y_idx[:, 0] - 1
        g = groups[t0]                                            # (B, N)
        s0 = data.speed[t0]
        sel = (g >= 0) & valid[:, -1]
        if sel.any():
            b, n = sel.nonzero(as_tuple=True)
            sb = torch.clamp(torch.bucketize(s0[b, n], torch.as_tensor(SPEED_BINS[1:], device=s0.device,
                                                                          dtype=s0.dtype), right=True), 0, len(SPEED_BINS) - 1)
            hour = (data.slots[t0[b]] % STEPS_PER_DAY) // 12
            err = (pred - y)[b, :, n]                             # (R, H)
            v = valid[b, :, n].float()
            orows.append(torch.stack([g[b, n].float(), cluster[t0[b], n].float(), s0[b, n],
                                      y[b, -1, n] - s0[b, n], pred[b, -1, n] - s0[b, n], table_t[sb, hour],
                                      (err * v).sum(1) / v.sum(1).clamp(min=1)], 1).cpu())
        # Closure readings (same as closure_errors.py)
        s = since[y_idx]
        csel = (s >= 0) & valid
        if csel.any():
            b, h, n = csel.nonzero(as_tuple=True)
            t = y_idx[b, h]
            crows.append(torch.stack([event_id[t, n].float(), y[b, h, n], pred[b, h, n],
                                      data.typical[data.slots[t], n]], 1).cpu())
    o = pd.DataFrame(torch.cat(orows).numpy(), columns=["group", "cluster", "speed_now", "actual_recovery_60",
                                                        "pred_recovery_60", "base_rate_recovery_60", "bias"])
    c = pd.DataFrame(torch.cat(crows).numpy(), columns=["event", "actual", "pred", "typical"])
    return o, c, {k: t.result() for k, t in tallies.items()}


def ci(df, col, by):
    return bootstrap(df.rename(columns={by: "event"}), col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stride", type=int, default=1, help="use every n-th test window (quick checks only)")
    ap.add_argument("--out", default=str(RESULTS / "recovery_prior.json"))
    args = ap.parse_args()

    device = pick_device(args.device)
    data = TrafficData(PROCESSED, device)
    data.splits["test"] = data.splits["test"][::args.stride]
    T, N = data.speed.shape
    events = pd.read_csv(EVENTS)
    speed, slots, typical = data.speed.cpu().numpy(), data.slots.cpu().numpy(), data.typical.cpu().numpy()
    active = event_mask(events, T, N, events.kind.unique())
    closure_on = event_mask(events, T, N, ["closure"])
    groups = origin_groups(speed, typical[slots], active, closure_on)

    since = np.full((T, N), -1, dtype=np.int32)
    event_id = np.full((T, N), -1, dtype=np.int32)
    for i, e in enumerate(events[(events.kind == "closure") & ~events.on_ramp].itertuples()):
        since[e.start_step:e.end_step, e.node] = np.arange(e.end_step - e.start_step) * 5
        event_id[e.start_step:e.end_step, e.node] = i
    # Resampling unit: the closure for closure origins, the sensor-day otherwise
    day = (np.arange(T) // STEPS_PER_DAY)[:, None]
    cluster = np.where(groups == 2, event_id, 1_000_000 + np.arange(N)[None, :] * 400 + day).astype(np.int64)

    table, p1 = base_rate(speed, slots, data.splits["train"][::2], groups, typical)   # every other window: memory
    print("P1, training period: 60-min recovery after a slowed reading (mph)")
    print(f"{'group':20s}{'origins':>10s}{'actual':>9s}{'base rate':>11s}{'recovered':>11s}")
    for k, r in p1.items():
        print(f"{k:20s}{r['origins']:>10,d}{r['actual_recovery_60']:>+9.2f}{r['base_rate_recovery_60']:>+11.2f}"
              f"{r['recovered_share']:>10.0%}")

    to = lambda a: torch.from_numpy(a).to(device)
    groups_t, cluster_t, since_t, event_t = to(groups), to(cluster), to(since), to(event_id)
    masks = {k: to(event_mask(events, T, N, [k])) for k in KINDS}
    masks["normal"] = ~to(active)

    models = {"last value": LastValue()}
    for path in args.checkpoints:
        models[path.split("/")[-1].removesuffix(".pt")] = load_model(path, data, device)[0]

    report = {"p1_training": p1, "groups": {}, "closure_disrupted": {}, "per_kind_mae": {}}
    for name, model in models.items():
        o, c, kinds = run(model, data, groups_t, cluster_t, table, since_t, event_t, masks)
        report["per_kind_mae"][name] = kinds
        r = report["groups"][name] = {}
        for i, gname in enumerate(GROUPS):
            d = o[o.group == i]
            r[gname] = {"origins": int(len(d)), "clusters": int(d.cluster.nunique()),
                        "actual_recovery_60": float(d.actual_recovery_60.mean()),
                        "pred_recovery_60": float(d.pred_recovery_60.mean()),
                        "base_rate_recovery_60": float(d.base_rate_recovery_60.mean()),
                        "bias": float(d.bias.mean()), "bias_ci": ci(d, "bias", "cluster"),
                        "pred_recovery_ci": ci(d, "pred_recovery_60", "cluster")}
        c = c.assign(err=c.pred - c.actual)
        dis = c[c.actual <= c.typical - DISRUPTED_MPH]
        report["closure_disrupted"][name] = {"readings": int(len(dis)), "mae": float(dis.err.abs().mean()),
                                             "bias": float(dis.err.mean()), "bias_ci": bootstrap(dis, "err"),
                                             "_err": dis.err.to_numpy(), "_event": dis.event.to_numpy()}
        print(f"done: {name}", flush=True)

    # Paired difference of the slowdown-weighted model vs the same fine-tuning without weights
    cd = report["closure_disrupted"]
    pairs = [(a, b) for a in cd for b in cd if "slowdown" in a and "plain" in b]
    for a, b in pairs:
        d = pd.DataFrame({"event": cd[a]["_event"], "d_abs": np.abs(cd[a]["_err"]) - np.abs(cd[b]["_err"]),
                          "d_bias": cd[a]["_err"] - cd[b]["_err"]})
        cd[a][f"vs {b}"] = {"d_mae": float(d.d_abs.mean()), "d_mae_ci": bootstrap(d, "d_abs"),
                            "d_bias": float(d.d_bias.mean()), "d_bias_ci": bootstrap(d, "d_bias")}
    for v in cd.values():
        v.pop("_err"), v.pop("_event")

    RESULTS.mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    fmt = lambda c: f"[{c[0]:+.2f}, {c[1]:+.2f}]"
    print("\nP2/P3, test period: 60-min recovery (mph) and mean signed error over the hour (> 0 = too fast)")
    for gname in GROUPS:
        first = report["groups"]["last value"][gname]
        print(f"--- {gname}: {first['origins']:,} origins, {first['clusters']:,} "
              f"{'closures' if gname == 'closure' else 'sensor-days'}; actual recovery "
              f"{first['actual_recovery_60']:+.2f}, training base rate {first['base_rate_recovery_60']:+.2f}")
        print(f"{'model':30s}{'predicted recovery':>20s}{'bias':>8s}  {'bias 95% CI':18s}")
        for name, r in report["groups"].items():
            s = r[gname]
            print(f"{name:30s}{s['pred_recovery_60']:>+20.2f}{s['bias']:>+8.2f}  {fmt(s['bias_ci'])}")
    print("\nP4: disrupted closure readings (as in closure_errors.py)")
    for name, s in cd.items():
        extra = "".join(f"   vs {k[3:]}: dMAE {v['d_mae']:+.2f} {fmt(v['d_mae_ci'])}, dbias {v['d_bias']:+.2f} "
                        f"{fmt(v['d_bias_ci'])}" for k, v in s.items() if k.startswith("vs "))
        print(f"{name:30s} MAE {s['mae']:.2f}  bias {s['bias']:+.2f} {fmt(s['bias_ci'])}{extra}")
    print("--- MAE by situation")
    gl = ["normal"] + KINDS
    print(f"{'model':30s}" + "".join(f"{g:>11s}" for g in gl))
    for name, r in report["per_kind_mae"].items():
        print(f"{name:30s}" + "".join(f"{r[g]['mae']:>11.2f}" for g in gl))


if __name__ == "__main__":
    main()
