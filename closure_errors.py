"""Failure analysis: where inside a closure do forecasts go wrong, and in which direction?

    python closure_errors.py checkpoints/a.pt checkpoints/b.pt

Every test reading taken while a freeway closure is active at its sensor is split by
  - minutes since the closure was reported (0-15, 15-30, 30-60, 60+),
  - whether traffic is actually disrupted (speed at least 5 mph below typical for that time of week),
  - day (6-20h) or night,
  - forecast horizon (5, 15, 30, 60 min ahead),
and scored with MAE and the mean signed error (prediction minus truth; > 0 means the forecast is too fast,
i.e. it expects traffic to recover sooner than it does).
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from baselines import IncidentBlend, LastValue
from config import EVENTS, PROCESSED, RESULTS
from tgc.data import TrafficData
from tgc.incidents import event_mask
from tgc.training import load_model, pick_device

DISRUPTED_MPH = 5.0


@torch.no_grad()
def closure_rows(model, data, since, split="test", batch_size=64):
    """One row per (window, horizon) reading at a sensor with an active closure."""
    if isinstance(model, torch.nn.Module):
        model.eval()
    rows = []
    for x, y, y_idx in data.batches(split, batch_size):
        pred = model.predict(x, y_idx, data) if hasattr(model, "predict") else data.unscale(model(x))[..., 0]
        y = y[..., 0]
        s = since[y_idx]                                         # (B, H, N) minutes since report, -1 = none
        sel = (s >= 0) & (y != 0)
        if not sel.any():
            continue
        b, h, n = sel.nonzero(as_tuple=True)
        t = y_idx[b, h]
        rows.append(torch.stack([h.float() + 1, t.float(), n.float(), s[b, h, n].float(), y[b, h, n],
                                 pred[b, h, n], data.typical[data.slots[t], n]], 1).cpu())
    cols = ["horizon", "t", "node", "minutes_since_report", "actual", "pred", "typical"]
    return pd.DataFrame(torch.cat(rows).numpy(), columns=cols)


def summarize(df):
    df = df.assign(err=df.pred - df.actual)
    df["abs_err"] = df.err.abs()
    df["disrupted"] = np.where(df.actual <= df.typical - DISRUPTED_MPH, "disrupted", "not disrupted")
    df["phase"] = pd.cut(df.minutes_since_report, [-1, 15, 30, 60, np.inf],
                         labels=["0-15 min", "15-30 min", "30-60 min", "60+ min"])
    minute_of_day = (df.slot * 5) % 1440
    df["time_of_day"] = np.where((minute_of_day >= 360) & (minute_of_day < 1200), "day", "night")
    df["ahead"] = df.horizon.map({1: "5 min", 3: "15 min", 6: "30 min", 12: "60 min"})
    out = {"all": {"readings": int(len(df)), "mae": float(df.abs_err.mean()), "bias": float(df.err.mean())}}
    for by in ["disrupted", "phase", "time_of_day", "ahead"]:
        g = df.dropna(subset=[by]).groupby(by, observed=True)
        out[by] = {k: {"readings": int(len(v)), "mae": float(v.abs_err.mean()), "bias": float(v.err.mean())}
                   for k, v in g}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(RESULTS / "closure_errors.json"))
    args = ap.parse_args()

    device = pick_device(args.device)
    data = TrafficData(PROCESSED, device)
    T, N = data.speed.shape
    events = pd.read_csv(EVENTS)
    since = np.full((T, N), -1, dtype=np.int32)
    for e in events[(events.kind == "closure") & ~events.on_ramp].itertuples():
        since[e.start_step:e.end_step, e.node] = np.arange(e.end_step - e.start_step) * 5
    since = torch.from_numpy(since).to(device)
    active = torch.from_numpy(event_mask(events, T, N, events.kind.unique())).to(device)
    slots = data.slots.cpu().numpy()

    models = {"last value": LastValue()}
    for path in args.checkpoints:
        name, model = path.split("/")[-1].removesuffix(".pt"), load_model(path, data, device)[0]
        models[name] = model
        models[name + "+blend"] = IncidentBlend(model, active)

    report = {}
    for name, model in models.items():
        df = closure_rows(model, data, since)
        df["slot"] = slots[df.t.astype(int)]
        report[name] = summarize(df)
        print(f"done: {name}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nTest readings during closures: MAE (mean signed error; > 0 = forecast too fast), mph")
    for by in ["disrupted", "phase", "time_of_day", "ahead"]:
        groups = list(next(iter(report.values()))[by])
        print(f"--- by {by}")
        print(f"{'model':30s}" + "".join(f"{g:>20s}" for g in groups))
        print(f"{'readings':30s}" + "".join(f"{next(iter(report.values()))[by][g]['readings']:>20,d}" for g in groups))
        for name, r in report.items():
            print(f"{name:30s}" + "".join(f"{r[by][g]['mae']:>11.2f} ({r[by][g]['bias']:+5.2f})" for g in groups))


if __name__ == "__main__":
    main()
