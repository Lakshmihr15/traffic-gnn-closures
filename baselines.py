"""Simple baselines, scored by the same evaluate() as DCRNN, next to a trained checkpoint.

    python baselines.py                                    # baselines only
    python baselines.py --checkpoint checkpoints/dcrnn_quick.pt

A useful model must beat both. During closures, "last value" is the bar that matters: it already knows
the current slowdown and only assumes nothing changes.
"""
import argparse
import json

import torch
from torch import nn

from config import PROCESSED, RESULTS
from tgc.data import TrafficData
from tgc.training import evaluate, format_metrics, load_model, pick_device


class LastValue(nn.Module):
    """Every future step equals the last observed reading."""

    def predict(self, x, y_idx, data):
        return data.unscale(x[:, -1:, :, 0]).expand(-1, y_idx.shape[1], -1)


class TypicalSpeed(nn.Module):
    """Median speed for that sensor and 5-minute slot of the week, from the training period."""

    def predict(self, x, y_idx, data):
        return data.typical[data.slots[y_idx]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    data = TrafficData(PROCESSED, device)
    models = {"last value": LastValue(), "typical speed": TypicalSpeed()}
    if args.checkpoint:
        models["DCRNN"] = load_model(args.checkpoint, data, device)[0]

    report = {}
    for name, model in models.items():
        report[name] = evaluate(model, data, args.split)
        print(f"{name:14s}", format_metrics(report[name]))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"baselines_{args.split}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    with torch.no_grad():
        main()
