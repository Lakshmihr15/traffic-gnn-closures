# Traffic GNNs under road closures

Does a trained traffic GNN (DCRNN) predict the congestion spillover that real road closures cause?
Traffic: TraffiDent (California PeMS, 2023). Ground truth: recorded CHP "CLOSURE of a Road" incidents.
Everything runs on a 16 GB Mac (Apple GPU); full training is better on a CUDA GPU.

## Checkpoint results (quick DCRNN run, test period May 25 – Jun 30, 2023)

| Method | MAE overall (mph) | Normal traffic | During recorded closures |
| --- | --- | --- | --- |
| Last value | 2.02 | 1.89 | 2.89 |
| Typical speed for time of week | 2.77 | 2.33 | 7.56 |
| DCRNN | 1.79 | 1.64 | 3.12 |

E2 pilot on 69 held-out closures (21 verified): DCRNN predicts −0.1 mph slowdown upstream within 5 km where
9.1 mph was observed, and 0.7 vs 13.6 mph downstream. Details in `results/`.

## Setup

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
```

Download TraffiDent from Kaggle (`gpxlcj/xtraffic`, CC BY-NC 4.0) and put into `data/raw/traffident/`:
`sensor_meta_feature.csv`, `node_order.npy`, `incidents_y2023.csv`, and `year_2023/year_2023/2023_p01.npy` …
`2023_p06.npy` (Jan–Jun). From the full `archive.zip`:

```bash
unzip archive.zip 'year_2023/year_2023/2023_p0[1-6].npy' incidents_y2023.csv sensor_meta_feature.csv node_order.npy -d data/raw/traffident
```

## Run

```bash
python -m pytest tests                       # graph direction, isolation, incident parsing (no data needed)
python explore_traffident.py                 # counts of sensors and incidents per district
python prepare_data.py                       # Steps 1-7 + incident matching -> data/processed/   (~1 min)
python train.py --batch-size 16              # full DCRNN recipe -> checkpoints/dcrnn_base.pt   (days on a Mac)
python train.py --epochs 6 --stride 4 --batch-size 16 --out checkpoints/dcrnn_quick.pt   # quick run (~2 h)
python baselines.py --checkpoint checkpoints/dcrnn_quick.pt   # E1: last value, typical speed, DCRNN
python e2_pilot.py  --checkpoint checkpoints/dcrnn_quick.pt   # E2: simulated vs observed spillover
python finetune.py --base checkpoints/dcrnn_quick.pt          # Step 10: upweight closures in the loss
```

Settings (district, months, target channel, thresholds, model size) are in `config.py`.

## Files

| File | What it does |
| --- | --- |
| `config.py` | Paths and all settings |
| `tgc/data.py` | Steps 1–7: sensors, traffic, cleaning, directed road graph from postmiles, time features, windows, split, scaling |
| `tgc/incidents.py` | Classifies CHP incidents and matches them to the nearest sensor on the same freeway and direction |
| `tgc/model.py` | DCRNN; the graph enters only through `supports`, so closures can be applied at inference |
| `tgc/training.py` | Masked metrics, training epoch, evaluation (overall, per horizon, closures vs normal) |
| `tgc/closures.py` | Closure simulator (input / graph / both) and spillover measures |
| `prepare_data.py`, `train.py`, `tune.py`, `finetune.py`, `baselines.py`, `e2_pilot.py` | Scripts |
| `results/` | Metrics from the runs below |

Conventions: `adj[i, j] > 0` means traffic flows from sensor i to j (i is upstream). Slowdowns are in mph,
positive = slower. Speed 0 = missing reading.
