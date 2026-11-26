"""Graph-gating experiment: do normal-looking neighbours pull closure forecasts toward recovery?

    python graph_gating.py --graph checkpoints/dcrnn_check_plain_ft.pt \
        --nograph checkpoints/dcrnn_check_nograph_ft.pt --gated checkpoints/dcrnn_check_gated_ft.pt

Hypothesis (from closure_errors.py): during a closure the DCRNN forecasts too fast because sensors around the
closure still look normal and their graph messages dominate. Tests, all on test readings at a sensor with an
active freeway closure (same readings as closure_errors.py), gate rule from tgc/gating.py (sensors with any
active incident at the forecast origin stop hearing their neighbours):

  1. Dose-response: the same trained graph model with the gate at g = 1 (normal), 0.5, 0. If neighbours
     cause the fast bias, bias on disrupted readings falls as g goes to 0.
  2. Direction: gate only upstream or only downstream neighbours. Downstream of a closure the road is
     emptier and faster, so the hypothesis expects the downstream messages to carry the fast bias.
  3. Trained gate: the model fine-tuned with the gate (finetune.py --gate-incidents), so a gain is not hidden
     by the model never having seen a gated sensor during training.
  4. Mechanism: bias by how much faster the neighbours look than the closure sensor at the forecast origin.
     The hypothesis predicts the graph model's extra bias (vs the no-graph model) grows with that gap, and
     that gating removes it.
  5. Placebo: gate every sensor everywhere. If that also helps normal traffic, the gain is not specific to
     incidents (the graph would simply be unhelpful).

Uncertainty: 95% intervals from resampling closure events (about 35 in the test period), not readings.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from baselines import LastValue
from closure_errors import DISRUPTED_MPH
from config import EVENTS, PROCESSED, RESULTS
from incident_eval import KINDS, Tally
from tgc.data import TrafficData
from tgc.gating import DOWNSTREAM, UPSTREAM, Gated
from tgc.incidents import event_mask
from tgc.training import load_model, pick_device

GAP_BINS = [-np.inf, 0, 5, 15, np.inf]
GAP_LABELS = ["neighbours slower", "0-5 mph faster", "5-15 mph faster", ">15 mph faster"]


@torch.no_grad()
def run(model, data, since, event_id, masks, batch_size=64):
    """One pass over the test windows: closure reading rows (as closure_errors.py) and per-kind tallies."""
    if isinstance(model, torch.nn.Module):
        model.eval()
    rows, tallies = [], {k: Tally() for k in masks}
    for x, y, y_idx in data.batches("test", batch_size):
        pred = model.predict(x, y_idx, data) if hasattr(model, "predict") else data.unscale(model(x))[..., 0]
        y = y[..., 0]
        valid = y != 0
        ae = (pred - y).abs()
        for k, m in masks.items():
            tallies[k].add(ae, valid & m[y_idx])
        s = since[y_idx]
        sel = (s >= 0) & valid
        if sel.any():
            b, h, n = sel.nonzero(as_tuple=True)
            t = y_idx[b, h]
            rows.append(torch.stack([h.float() + 1, t.float(), n.float(), s[b, h, n].float(),
                                     event_id[t, n].float(), y[b, h, n], pred[b, h, n],
                                     data.typical[data.slots[t], n]], 1).cpu())
    cols = ["horizon", "t", "node", "minutes_since_report", "event", "actual", "pred", "typical"]
    return pd.DataFrame(torch.cat(rows).numpy(), columns=cols), {k: t.result() for k, t in tallies.items()}


def neighbour_gaps(df, speed, adj):
    """Mean speed of upstream / downstream neighbours minus the sensor's own speed, at the forecast origin."""
    origin = (df.t - df.horizon).astype(int).to_numpy()
    node = df.node.astype(int).to_numpy()
    v = speed[origin]                                          # (R, N) speeds of all sensors at the origin
    ok = v > 0
    off = adj * (1 - np.eye(len(adj), dtype=adj.dtype))        # no self-loops
    own = v[np.arange(len(node)), node]
    out = {}
    for name, w in [("gap_up", off[:, node].T), ("gap_down", off[node, :])]:   # w (R, N) neighbour weights
        w = w * ok
        mean = (w * v).sum(1) / np.where(w.sum(1) > 0, w.sum(1), np.nan)
        out[name] = np.where(own > 0, mean - own, np.nan)
    return out


def bootstrap(df, col, n=2000, seed=0):
    """95% interval for the mean of `col`, resampling closure events."""
    g = df.groupby("event")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), (n, len(g)))
    est = s[idx].sum(1) / c[idx].sum(1)
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def stats(d, ref=None):
    """MAE and bias (> 0 = too fast), with event-level intervals; differences vs `ref` rows if given."""
    out = {"readings": int(len(d)), "events": int(d.event.nunique()),
           "mae": float(d.abs_err.mean()), "bias": float(d.err.mean()),
           "bias_ci": bootstrap(d, "err")}
    if ref is not None:
        r = ref.loc[d.index]
        dd = d.assign(d_mae=d.abs_err - r.abs_err, d_bias=d.err - r.err)
        out.update({"d_mae": float(dd.d_mae.mean()), "d_mae_ci": bootstrap(dd, "d_mae"),
                    "d_bias": float(dd.d_bias.mean()), "d_bias_ci": bootstrap(dd, "d_bias")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="checkpoints/dcrnn_check_plain_ft.pt")
    ap.add_argument("--nograph", default="checkpoints/dcrnn_check_nograph_ft.pt")
    ap.add_argument("--gated", default=None, help="checkpoint from finetune.py --gate-incidents")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stride", type=int, default=1, help="use every n-th test window (quick checks only)")
    ap.add_argument("--out", default=str(RESULTS / "graph_gating.json"))
    args = ap.parse_args()

    device = pick_device(args.device)
    data = TrafficData(PROCESSED, device)
    data.splits["test"] = data.splits["test"][::args.stride]
    T, N = data.speed.shape
    events = pd.read_csv(EVENTS)
    since = np.full((T, N), -1, dtype=np.int32)
    event_id = np.full((T, N), -1, dtype=np.int32)
    for i, e in enumerate(events[(events.kind == "closure") & ~events.on_ramp].itertuples()):
        since[e.start_step:e.end_step, e.node] = np.arange(e.end_step - e.start_step) * 5
        event_id[e.start_step:e.end_step, e.node] = i
    since, event_id = torch.from_numpy(since).to(device), torch.from_numpy(event_id).to(device)
    active = torch.from_numpy(event_mask(events, T, N, events.kind.unique())).to(device)
    everywhere = torch.ones_like(active)
    masks = {k: torch.from_numpy(event_mask(events, T, N, [k])).to(device) for k in KINDS}
    masks["normal"] = ~active

    graph = load_model(args.graph, data, device)[0]
    models = {
        "last value": LastValue(),
        "no graph": load_model(args.nograph, data, device)[0],
        "graph": graph,
        "graph, gate 0.5": Gated(graph, active, 0.5),
        "graph, gate 0": Gated(graph, active, 0.0),
        "graph, gate upstream only": Gated(graph, active, 0.0, (UPSTREAM,)),
        "graph, gate downstream only": Gated(graph, active, 0.0, (DOWNSTREAM,)),
        "graph, gate every sensor (placebo)": Gated(graph, everywhere, 0.0),
    }
    if args.gated:
        models["gate-trained, gate 0"] = Gated(load_model(args.gated, data, device)[0], active, 0.0)

    frames, kinds = {}, {}
    for name, model in models.items():
        frames[name], kinds[name] = run(model, data, since, event_id, masks)
        print(f"done: {name}", flush=True)

    base = frames["graph"]
    gaps = neighbour_gaps(base, data.speed.cpu().numpy(), data.adj)
    for name, df in frames.items():
        assert len(df) == len(base) and (df.t.to_numpy() == base.t.to_numpy()).all()
        df["err"] = df.pred - df.actual
        df["abs_err"] = df.err.abs()
        df["disrupted"] = df.actual <= df.typical - DISRUPTED_MPH
        for k, v in gaps.items():
            df[k] = v
            df[k + "_bin"] = pd.cut(v, GAP_BINS, labels=GAP_LABELS)
    ref = frames["graph"]

    report = {"per_kind_mae": kinds, "closure": {}, "by_gap_down": {}, "by_gap_up": {}}
    for name, df in frames.items():
        r = report["closure"][name] = {}
        for sub, d in [("all", df), ("disrupted", df[df.disrupted]), ("not disrupted", df[~df.disrupted])]:
            r[sub] = stats(d, None if name == "graph" else ref)
        for gap in ("gap_down", "gap_up"):
            dd = df[df.disrupted]
            report[f"by_{gap}"][name] = {str(k): stats(v) for k, v in dd.groupby(gap + "_bin", observed=True)}
    # Graph model's extra fast bias over the no-graph model, by how much faster the downstream sensors look
    keep = ref.disrupted & ref.gap_down.notna()
    extra = ref.err[keep] - frames["no graph"].err[keep]
    report["extra_bias_vs_gap_down"] = {
        "readings": int(keep.sum()),
        "slope_mph_per_mph": float(np.polyfit(ref.gap_down[keep], extra, 1)[0]),
        "corr": float(np.corrcoef(ref.gap_down[keep], extra)[0, 1]),
    }
    RESULTS.mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    def fmt_ci(ci):
        return f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"

    first = report["closure"]["graph"]
    print(f"\nClosure readings in the test period: {first['all']['readings']:,} "
          f"({first['disrupted']['readings']:,} disrupted), {first['all']['events']} closures")
    print("MAE / bias in mph (bias > 0 = forecast too fast); change vs the plain graph model with 95% event CI")
    for sub in ("disrupted", "not disrupted", "all"):
        print(f"--- {sub}")
        print(f"{'model':38s}{'MAE':>7s}{'bias':>8s}   {'bias 95% CI':18s}{'dMAE':>7s}  {'dMAE 95% CI':18s}")
        for name, r in report["closure"].items():
            s = r[sub]
            d = (f"{s['d_mae']:+7.2f}  {fmt_ci(s['d_mae_ci']):18s}") if "d_mae" in s else ""
            print(f"{name:38s}{s['mae']:7.2f}{s['bias']:+8.2f}   {fmt_ci(s['bias_ci']):18s}{d}")
    for gap in ("gap_down", "gap_up"):
        print(f"--- disrupted readings, bias by {gap} (neighbour mean speed minus own speed at forecast origin)")
        print(f"{'model':38s}" + "".join(f"{g:>20s}" for g in GAP_LABELS))
        n = report[f"by_{gap}"]["graph"]
        print(f"{'readings (events)':38s}" + "".join(
            f"{(str(n[g]['readings']) + ' (' + str(n[g]['events']) + ')') if g in n else '-':>20s}" for g in GAP_LABELS))
        for name, r in report[f"by_{gap}"].items():
            print(f"{name:38s}" + "".join(f"{r[g]['bias']:>+20.2f}" if g in r else f"{'-':>20s}" for g in GAP_LABELS))
    e = report["extra_bias_vs_gap_down"]
    print(f"graph minus no-graph bias vs downstream gap: slope {e['slope_mph_per_mph']:+.3f} mph/mph, "
          f"corr {e['corr']:+.2f}")
    print("--- MAE by what is happening at the sensor (placebo check: normal traffic)")
    groups = ["normal"] + KINDS
    print(f"{'model':38s}" + "".join(f"{g:>11s}" for g in groups))
    for name, r in kinds.items():
        print(f"{name:38s}" + "".join(f"{r[g]['mae']:>11.2f}" for g in groups))


if __name__ == "__main__":
    main()
