"""E2 pilot: does DCRNN's simulated spillover look like the spillover real closures caused?

    python e2_pilot.py --checkpoint checkpoints/dcrnn_quick.pt

Uses the recorded road closures whose forecast time falls in the validation or test period (never trained on).
Writes one row per (closure, variant, neighbour) to results/e2_pilot.csv and prints a summary with bootstrap
95% intervals over closures. A closure is "verified" if its own sensor ran >= 10 mph below typical in the
first 30 minutes.
"""
import argparse
import json

import numpy as np
import pandas as pd

from config import EVENTS, PROCESSED, RESULTS, DataConfig
from tgc import closures as C
from tgc.data import TrafficData
from tgc.training import load_model, pick_device

LEAD = 3          # forecast 15 minutes after the closure starts
MAX_KM = 5.0      # neighbours within 5 km along the road
VERIFY_MPH = 10.0


def held_out_closures(data, events):
    first = data.splits["val"][0]
    c = events[events.kind.isin(DataConfig.incident_kinds) & ~events.on_ramp].copy()
    c["t0"] = c.start_step + LEAD
    c = c[(c.t0 >= first) & (c.t0 + data.horizon < len(data.speed))]
    return c.drop_duplicates(["node", "start_step"])


def bootstrap(values, n=2000, seed=0):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (n, len(values))).mean(1)
    return values.mean(), *np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/dcrnn_base.pt")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    data = TrafficData(PROCESSED, device)
    dist = np.load(PROCESSED)["dist_km"]
    model, _ = load_model(args.checkpoint, data, device)
    model.eval()
    closures = held_out_closures(data, pd.read_csv(EVENTS))
    print(f"{len(closures)} held-out closures")

    rows = []
    for ev in closures.itertuples():
        obs = C.observed(data, ev.t0)
        own_drop = np.nanmean(obs[:6, ev.node])
        up, down = C.neighbours(dist, ev.node, MAX_KM)
        for variant in C.VARIANTS:
            pred = C.simulate(model, data, ev.node, ev.start_step, ev.t0, variant)
            for side, nodes in [("closed", [ev.node]), ("upstream", up), ("downstream", down)]:
                for j in nodes:
                    rows.append({"incident_id": ev.incident_id, "node": ev.node, "time": ev.time,
                                 "variant": variant, "side": side, "neighbour": j,
                                 "km": 0.0 if side == "closed" else float(min(dist[j, ev.node], dist[ev.node, j])),
                                 "pred_mph": float(pred[:, j].mean()), "obs_mph": float(np.nanmean(obs[:, j])),
                                 "verified": bool(own_drop >= VERIFY_MPH)})
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "e2_pilot.csv", index=False)

    summary = {}
    for subset, d in [("all", df), ("verified", df[df.verified])]:
        n = d.incident_id.nunique()
        print(f"\n=== {subset} closures (n={n}); mean slowdown in mph [95% CI], averaged per closure")
        print(f"{'variant':12s} {'side':10s} {'predicted':>22s} {'observed':>22s}  sign agree  corr")
        for (variant, side), g in d.groupby(["variant", "side"], sort=False):
            per = g.groupby("incident_id")[["pred_mph", "obs_mph"]].mean()
            p, o = bootstrap(per.pred_mph), bootstrap(per.obs_mph)
            ok = g.dropna(subset=["obs_mph"])
            agree = float((np.sign(ok.pred_mph) == np.sign(ok.obs_mph)).mean()) if len(ok) else np.nan
            corr = float(ok.pred_mph.corr(ok.obs_mph)) if len(ok) > 2 else np.nan
            print(f"{variant:12s} {side:10s} {p[0]:6.2f} [{p[1]:5.2f},{p[2]:5.2f}]  "
                  f"{o[0]:6.2f} [{o[1]:5.2f},{o[2]:5.2f}]  {agree:9.0%}  {corr:5.2f}")
            summary[f"{subset}/{variant}/{side}"] = {"closures": int(per.shape[0]), "pred": p, "obs": o,
                                                      "sign_agree": agree, "corr": corr}
    (RESULTS / "e2_pilot_summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
