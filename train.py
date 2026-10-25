"""Steps 8-9: build DCRNN, train it, keep the best weights on validation, then report test accuracy.

    python train.py                              # full DCRNN recipe (100 epochs, early stopping)
    python train.py --epochs 1 --max-batches 20  # smoke test
"""
import argparse
import json
import math
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from config import CHECKPOINTS, PROCESSED, RESULTS, ModelConfig, TrainConfig
from tgc.data import TrafficData
from tgc.model import DCRNN
from tgc.training import evaluate, format_metrics, load_model, pick_device, save_checkpoint, train_epoch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit(data, mcfg, tcfg, out_path, device, max_batches=None, log=print):
    """Train from random weights; save the epoch with the lowest validation MAE. Returns the best val MAE."""
    seed_everything(tcfg.seed)
    model = DCRNN(mcfg, data.adj).to(device)
    log(f"Step 8: DCRNN with {sum(p.numel() for p in model.parameters()):,} weights on {device}")
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.base_lr, eps=tcfg.epsilon)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, list(tcfg.lr_milestones), tcfg.lr_decay)
    rng = np.random.default_rng(tcfg.seed)

    best, waited, batches_seen = float("inf"), 0, 0
    for epoch in range(tcfg.epochs):
        start = time.time()
        train_mae, batches_seen = train_epoch(model, data, opt, tcfg, rng, batches_seen, max_batches=max_batches)
        sched.step()
        val = evaluate(model, data, "val", max_batches=max_batches)
        improved = val["mae_all"] < best
        log(f"epoch {epoch:3d}  train MAE {train_mae:.3f}  val MAE {val['mae_all']:.3f}  "
            f"lr {opt.param_groups[0]['lr']:.0e}  {time.time() - start:.0f}s{'  *saved' if improved else ''}")
        if improved:
            best, waited = val["mae_all"], 0
            save_checkpoint(out_path, model, {"train_cfg": asdict(tcfg), "epoch": epoch, "val": val})
        else:
            waited += 1
            if waited >= tcfg.patience:
                log("early stopping")
                break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--lr", type=float, default=TrainConfig.base_lr)
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size,
                    help="64 needs a large GPU; 16 fits a 16 GB Mac")
    ap.add_argument("--units", type=int, default=ModelConfig.rnn_units)
    ap.add_argument("--K", type=int, default=ModelConfig.max_diffusion_step)
    ap.add_argument("--seed", type=int, default=TrainConfig.seed)
    ap.add_argument("--max-batches", type=int, default=None, help="cap batches per epoch (quick tests)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stride", type=int, default=1,
                    help="use every n-th train/val window; neighbouring windows overlap by 55 of 60 minutes")
    ap.add_argument("--out", default=str(CHECKPOINTS / "dcrnn_base.pt"))
    args = ap.parse_args()

    device = pick_device(args.device)
    data = TrafficData(PROCESSED, device)
    for split in ("train", "val"):
        data.splits[split] = data.splits[split][::args.stride]
    mcfg = ModelConfig(num_nodes=data.num_nodes, input_dim=data.input_dim, rnn_units=args.units,
                       max_diffusion_step=args.K, horizon=data.horizon)
    tcfg = replace(TrainConfig(), epochs=args.epochs, base_lr=args.lr, seed=args.seed, batch_size=args.batch_size)
    if args.epochs < tcfg.lr_milestones[0]:
        # Short run: fit the learning-rate decay and the decoder's switch from true to predicted inputs
        # inside the run (with the paper's settings both happen only after ~20 epochs)
        e = args.epochs
        total_batches = e * math.ceil(len(data.splits["train"]) / tcfg.batch_size)
        tcfg = replace(tcfg, lr_milestones=(int(e * .5), int(e * .7), int(e * .9)),
                       cl_decay_steps=max(50, total_batches // 12))

    out = Path(args.out)
    fit(data, mcfg, tcfg, out, device, args.max_batches)

    print("Test accuracy of the best checkpoint (TraffiDent paper, DCRNN on flow: MAE 11.90 @5min, "
          "13.41 @15min, 14.84 @30min)")
    model, _ = load_model(out, data, device)
    test = evaluate(model, data, "test", max_batches=args.max_batches)
    print(format_metrics(test))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"test_metrics_{out.stem}.json").write_text(json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
