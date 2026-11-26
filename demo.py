"""Demo: forecast one real freeway closure from the test period and compare models.

    python demo.py                 # the test closure with the deepest slowdown at its own sensor
    python demo.py --rank 3        # the 3rd deepest
    python demo.py --list          # list the test closures

The forecast is made 15 minutes after the closure was reported (the input hour already holds 3 closure
readings). Shown: the sensor's real speed, its typical speed for that time of week, and the next-hour
forecasts of last value, DCRNN (graph), DCRNN with the closure sensor cut off from its neighbours (graph gate,
tgc/gating.py) and DCRNN trained without a graph. Saves results/demo_closure.png (or _rank<n>.png).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from baselines import LastValue
from config import EVENTS, PROCESSED, RESULTS
from tgc.data import TrafficData
from tgc.gating import Gated
from tgc.incidents import event_mask
from tgc.training import load_model, pick_device

LEAD = 3   # forecast origin = report + 15 minutes

COLORS = {"actual": "#0b0b0b", "typical": "#9a9993", "last value": "#52514e",
          "DCRNN (graph)": "#2a78d6", "DCRNN, closure sensor gated": "#eb6834", "DCRNN, no graph": "#1baf7a"}


def test_closures(data, events):
    t_test = set(data.splits["test"].tolist())
    c = events[(events.kind == "closure") & ~events.on_ramp].copy()
    c["t0"] = c.start_step + LEAD
    c = c[c.t0.isin(t_test)]
    speed, typ, slots = data.speed.cpu().numpy(), data.typical.cpu().numpy(), data.slots.cpu().numpy()
    drops = []
    for e in c.itertuples():
        t = np.arange(e.t0 + 1, e.t0 + 1 + data.horizon)
        v = speed[t, e.node]
        drops.append(np.nanmean(np.where(v > 0, typ[slots[t], e.node] - v, np.nan)))
    c["next_hour_drop_mph"] = drops
    return c.sort_values("next_hour_drop_mph", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=1, help="1 = closure with the deepest slowdown")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--graph", default="checkpoints/dcrnn_check_plain_ft.pt")
    ap.add_argument("--gated", default="checkpoints/dcrnn_check_gated_ft.pt")
    ap.add_argument("--nograph", default="checkpoints/dcrnn_check_nograph_ft.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="default results/demo_closure.png (rank 1), else ..._rank<n>.png")
    args = ap.parse_args()

    device = pick_device(args.device)
    data = TrafficData(PROCESSED, device)
    events = pd.read_csv(EVENTS)
    closures = test_closures(data, events)
    if args.list:
        print(closures[["time", "LOCATION", "duration", "node", "next_hour_drop_mph"]].to_string())
        return
    e = closures.iloc[args.rank - 1]
    t0, n = int(e.t0), int(e.node)
    print(f"Closure: {e.DESCRIPTION} at {e.LOCATION}, reported {e.time}, recorded duration {e.duration:.0f} min")
    print(f"Sensor {e.station_id} (I/SR-{e.Fwy} {e.Direction}); forecast made 15 min after the report")

    T, N = data.speed.shape
    active = torch.from_numpy(event_mask(events, T, N, events.kind.unique())).to(device)
    graph = load_model(args.graph, data, device)[0].eval()
    models = {"last value": LastValue(), "DCRNN (graph)": graph,
              "DCRNN, closure sensor gated": Gated(load_model(args.gated, data, device)[0].eval(), active),
              "DCRNN, no graph": load_model(args.nograph, data, device)[0].eval()}

    tb = torch.tensor([t0], device=device)
    x = data.features[tb[:, None] + data.x_off]
    y_idx = tb[:, None] + data.y_off
    preds = {}
    with torch.no_grad():
        for name, m in models.items():
            p = m.predict(x, y_idx, data) if hasattr(m, "predict") else data.unscale(m(x))[..., 0]
            preds[name] = p[0, :, n].cpu().numpy()

    speed, typ, slots = data.speed.cpu().numpy(), data.typical.cpu().numpy(), data.slots.cpu().numpy()
    past = np.arange(t0 - data.seq_len + 1, t0 + 1)
    fut = np.arange(t0 + 1, t0 + 1 + data.horizon)
    actual_f = speed[fut, n]
    print(f"\n{'model':30s}{'MAE next hour':>15s}{'bias':>8s}   (mph; bias > 0 = too fast)")
    for name, p in preds.items():
        ok = actual_f > 0
        err = p[ok] - actual_f[ok]
        print(f"{name:30s}{np.abs(err).mean():15.2f}{err.mean():+8.2f}")

    minutes = lambda t: (t - (int(e.start_step))) * 5
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    allt = np.concatenate([past, fut])
    ax.plot(minutes(allt), typ[slots[allt], n], color=COLORS["typical"], lw=1.5, ls="--", label="typical speed")
    v = np.where(speed[allt, n] > 0, speed[allt, n], np.nan)
    ax.plot(minutes(allt), v, color=COLORS["actual"], lw=2, label="actual speed")
    for name, p in preds.items():
        ax.plot(minutes(fut), p, color=COLORS[name], lw=2, ls=":" if name == "last value" else "-", label=name)
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.axvline(minutes(t0), color="#c3c2b7", lw=1)
    ymax = ax.get_ylim()[1]
    ax.text(0, ymax, "closure reported ", va="top", ha="right", fontsize=7, color="#52514e")
    ax.text(minutes(t0), ymax, " forecast made", va="top", fontsize=7, color="#52514e")
    ax.set_xlabel("minutes since the closure was reported")
    ax.set_ylabel("speed (mph)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    out = args.out or str(RESULTS / ("demo_closure.png" if args.rank == 1 else f"demo_closure_rank{args.rank}.png"))
    fig.savefig(out)
    print(f"\nplot saved to {out}")


if __name__ == "__main__":
    main()
