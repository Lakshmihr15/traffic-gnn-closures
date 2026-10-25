"""Step 10a: hyperparameter search. Short runs over a small grid, ranked by validation MAE.

    python tune.py --epochs 10
Then retrain the winner in full:  python train.py --units U --K K --lr LR
"""
import argparse
import csv
import itertools
from dataclasses import replace

from config import CHECKPOINTS, PROCESSED, RESULTS, ModelConfig, TrainConfig
from tgc.data import TrafficData
from tgc.training import pick_device
from train import fit

GRID = {"rnn_units": [32, 64], "max_diffusion_step": [1, 2, 3], "base_lr": [0.01, 0.005]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    data = TrafficData(PROCESSED, device)
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for units, K, lr in itertools.product(*GRID.values()):
        name = f"u{units}_K{K}_lr{lr}"
        print(f"=== {name}")
        mcfg = ModelConfig(num_nodes=data.num_nodes, input_dim=data.input_dim, rnn_units=units,
                           max_diffusion_step=K, horizon=data.horizon)
        # Milestones scaled to the short budget so the learning rate still decays
        e = args.epochs
        tcfg = replace(TrainConfig(), epochs=e, base_lr=lr, patience=e,
                       lr_milestones=(int(e * .5), int(e * .7), int(e * .9)))
        val = fit(data, mcfg, tcfg, CHECKPOINTS / "tune" / f"{name}.pt", device, args.max_batches)
        rows.append({"rnn_units": units, "K": K, "lr": lr, "val_mae": round(val, 4)})

    rows.sort(key=lambda r: r["val_mae"])
    with open(RESULTS / "tune.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("best:", rows[0])


if __name__ == "__main__":
    main()
