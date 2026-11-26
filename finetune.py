"""Step 10b: start from the trained weights and keep training with a disruption focus.

    python finetune.py --incident-weight 5                 # upweight recorded incidents in the loss
    python finetune.py --incident-weight 5 --edge-drop 0.3 # also cut random sensors out of the graph
    python finetune.py --incident-weight 1 --gate-incidents # train with the graph gate (tgc/gating.py)

Two optional changes, each a hook into the same training loop as Step 9:
  --incident-weight w  readings at a sensor during a recorded CHP incident (closure, SigAlert, collision)
                       count w times in the loss: the rare cases the base model sees too little of.
  --edge-drop p        with probability p per batch, one random sensor loses all its graph edges. This makes
                       the model robust to a missing link; it does not by itself teach closure physics.
  --slowdown-weight w  readings in an unusual slowdown (speed 30% below typical for 20+ minutes, any cause; mostly
                       not incidents) count w times. Tests whether seeing more slowdowns that do not clear on
                       schedule removes the too-fast bias during closures (recovery_prior.py).
  --gate-incidents     sensors with any recorded freeway incident active at the forecast origin stop hearing
                       their graph neighbours (gate g = 0), so the model learns to forecast them from their
                       own history. Evaluate the result with the same gate (graph_gating.py).

The best epoch is the one with the lowest incident MAE whose overall validation MAE stays within
--max-regress of the base model. The base checkpoint is never overwritten.
"""
import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import CHECKPOINTS, EVENTS, PROCESSED, RESULTS, TrainConfig
from tgc.data import TrafficData
from tgc.gating import Gated, incident_gate
from tgc.incidents import event_mask
from tgc.model import build_supports
from tgc.training import evaluate, format_metrics, load_model, pick_device, save_checkpoint, train_epoch
from train import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(CHECKPOINTS / "dcrnn_base.pt"))
    ap.add_argument("--out", default=str(CHECKPOINTS / "dcrnn_finetuned.pt"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--incident-weight", type=float, default=5.0)
    ap.add_argument("--slowdown-weight", type=float, default=1.0)
    ap.add_argument("--edge-drop", type=float, default=0.0)
    ap.add_argument("--gate-incidents", action="store_true",
                    help="gate graph messages into sensors with an active incident (train and validate with it)")
    ap.add_argument("--max-regress", type=float, default=0.02, help="allowed relative rise in overall val MAE")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size, help="16 fits a 16 GB Mac")
    ap.add_argument("--stride", type=int, default=1, help="use every n-th train/val window")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    base_cfg = torch.load(args.base, map_location="cpu", weights_only=False)["model_cfg"]
    data = TrafficData(PROCESSED, device, incident_inputs=base_cfg.get("incident_inputs", False))
    for split in ("train", "val"):
        data.splits[split] = data.splits[split][::args.stride]
    model, ckpt = load_model(args.base, data, device)
    gate_fn = None
    if args.gate_incidents:
        events = pd.read_csv(EVENTS)
        active = torch.from_numpy(event_mask(events, *data.speed.shape, events.kind.unique())).to(device)
        gate_fn = lambda y_idx: incident_gate(active, y_idx, 0.0, num_supports=len(model.supports))
    scored = (lambda m: Gated(m, active)) if args.gate_incidents else (lambda m: m)   # validate as deployed
    tcfg = replace(TrainConfig(), base_lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)
    seed_everything(tcfg.seed)
    rng = np.random.default_rng(tcfg.seed)

    base_val = evaluate(scored(model), data, "val", max_batches=args.max_batches)
    print("base model, validation:", format_metrics(base_val))
    limit = base_val["mae_all"] * (1 + args.max_regress)

    def weight_fn(y_idx):
        return (1 + (args.incident_weight - 1) * data.incident[y_idx].float()) * \
               (1 + (args.slowdown_weight - 1) * data.slowdown[y_idx].float())

    def supports_fn():
        if rng.random() >= args.edge_drop:
            return None
        adj = data.adj.copy()
        node = rng.integers(len(adj))
        adj[node, :] = adj[:, node] = 0
        adj[node, node] = 1
        return build_supports(adj).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=tcfg.base_lr, eps=tcfg.epsilon)
    # Teacher forcing has already decayed in the base run; start fine-tuning past that point
    batches_seen = 10 ** 6
    best, history = None, []
    for epoch in range(tcfg.epochs):
        start = time.time()
        train_mae, batches_seen = train_epoch(model, data, opt, tcfg, rng, batches_seen,
                                              weight_fn=weight_fn if (args.incident_weight, args.slowdown_weight) != (1, 1) else None,
                                              supports_fn=supports_fn if args.edge_drop > 0 else None,
                                              gate_fn=gate_fn,
                                              max_batches=args.max_batches)
        val = evaluate(scored(model), data, "val", max_batches=args.max_batches)
        ok = val["mae_all"] <= limit
        improved = ok and (best is None or val["mae_incident"] < best)
        print(f"epoch {epoch:2d}  train (weighted) {train_mae:.3f}  val MAE {val['mae_all']:.3f}  "
              f"incident {val['mae_incident']:.3f}  {time.time() - start:.0f}s"
              f"{'' if ok else '  (overall MAE regressed)'}{'  *saved' if improved else ''}")
        history.append({"epoch": epoch, **val})
        if improved:
            best = val["mae_incident"]
            save_checkpoint(Path(args.out), model, {
                "train_cfg": asdict(tcfg), "epoch": epoch, "val": val, "base": args.base,
                "finetune": {k: getattr(args, k) for k in ("incident_weight", "slowdown_weight", "edge_drop", "lr",
                                                                  "gate_incidents")}})

    if best is None:
        print("no epoch beat the base model within the allowed regression; nothing saved")
        return
    print("Test accuracy, base vs fine-tuned")
    report = {}
    for name, path in [("base", args.base), ("finetuned", args.out)]:
        m, _ = load_model(path, data, device)
        report[name] = evaluate(scored(m), data, "test", max_batches=args.max_batches)
        print(f"{name:10s}", format_metrics(report[name]))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"finetune_{Path(args.out).stem}.json").write_text(
        json.dumps({"args": vars(args), "history": history, "test": report}, indent=2))


if __name__ == "__main__":
    main()
