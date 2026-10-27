"""Part B: forecast error per incident type, for baselines and trained DCRNN checkpoints.

    python incident_eval.py checkpoints/dcrnn_a.pt checkpoints/dcrnn_b.pt

Following the evaluation advice in "Why Traffic Forecasting Benchmarks Hide Nonrecurrent Failures": errors are
reported separately for each incident type and for normal traffic (no incident active), against persistence
("last value") and a seasonal baseline ("typical"), per horizon and with the 95th-percentile error. A reading
counts for an incident type while that incident is active at its own sensor.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from baselines import IncidentBlend, LastValue, TypicalSpeed
from config import EVENTS, PROCESSED, RESULTS
from tgc.data import TrafficData
from tgc.incidents import event_mask
from tgc.training import load_model, pick_device

KINDS = ["closure", "collision", "hazard", "other", "sigalert"]
BINS = np.linspace(0, 100, 2001)          # 0.05-unit histogram for the 95th percentile


class Tally:
    """Sum, count and histogram of absolute errors for one group of readings."""

    def __init__(self):
        self.sum = self.n = 0.0
        self.sum_h = {3: 0.0, 12: 0.0}
        self.n_h = {3: 0.0, 12: 0.0}
        self.hist = np.zeros(len(BINS) - 1)

    def add(self, ae, mask):
        sel = ae[mask]
        self.sum += float(sel.sum())
        self.n += float(mask.sum())
        for h in self.sum_h:
            self.sum_h[h] += float(ae[:, h - 1][mask[:, h - 1]].sum())
            self.n_h[h] += float(mask[:, h - 1].sum())
        self.hist += np.histogram(sel.clamp(max=BINS[-1] - 1e-6).cpu().numpy(), BINS)[0]

    def result(self):
        cdf = np.cumsum(self.hist) / max(self.hist.sum(), 1)
        return {"readings": int(self.n), "mae": self.sum / max(self.n, 1),
                "mae@15min": self.sum_h[3] / max(self.n_h[3], 1), "mae@60min": self.sum_h[12] / max(self.n_h[12], 1),
                "p95": float(BINS[1:][np.searchsorted(cdf, 0.95)])}


@torch.no_grad()
def errors_by_kind(model, data, masks, split="test", batch_size=64):
    if isinstance(model, torch.nn.Module):
        model.eval()
    tallies = {k: Tally() for k in masks}
    for x, y, y_idx in data.batches(split, batch_size):
        pred = model.predict(x, y_idx, data) if hasattr(model, "predict") else data.unscale(model(x))[..., 0]
        y = y[..., 0]
        ae = (pred - y).abs()
        valid = y != 0
        for k, m in masks.items():
            tallies[k].add(ae, valid & m[y_idx])
    return {k: t.result() for k, t in tallies.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--blend", action="store_true",
                    help="also score each checkpoint blended with last value during incidents")
    ap.add_argument("--out", default=str(RESULTS / "incident_eval.json"))
    args = ap.parse_args()

    device = pick_device(args.device)
    data = TrafficData(PROCESSED, device)
    T, N = data.speed.shape
    events = pd.read_csv(EVENTS)
    masks = {k: torch.from_numpy(event_mask(events, T, N, [k])).to(device) for k in KINDS}
    any_incident = event_mask(events, T, N, events.kind.unique())
    masks["normal"] = torch.from_numpy(~any_incident).to(device)

    models = {"last value": (LastValue(), data), "typical": (TypicalSpeed(), data)}
    data_inc = None
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt["model_cfg"].get("incident_inputs"):
            data_inc = data_inc or TrafficData(PROCESSED, device, incident_inputs=True)
            d = data_inc
        else:
            d = data
        name, model = path.split("/")[-1].removesuffix(".pt"), load_model(path, d, device)[0]
        models[name] = (model, d)
        if args.blend:
            models[name + "+blend"] = (IncidentBlend(model, ~masks["normal"]), d)

    report = {}
    for name, (model, d) in models.items():
        report[name] = errors_by_kind(model, d, masks, args.split)
        print(f"done: {name}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    groups = ["normal"] + KINDS
    print(f"\nMAE in the forecast hour, by what is happening at the sensor ({args.split} period)")
    print(f"{'model':30s}" + "".join(f"{g:>11s}" for g in groups))
    first = next(iter(report.values()))
    print(f"{'readings':30s}" + "".join(f"{first[g]['readings']:>11,d}" for g in groups))
    for metric in ("mae", "mae@15min", "mae@60min", "p95"):
        print(f"--- {metric}")
        for name, r in report.items():
            print(f"{name:30s}" + "".join(f"{r[g][metric]:>11.2f}" for g in groups))


if __name__ == "__main__":
    main()
