# Traffic GNNs under road closures

How do recorded road closures change traffic compared with other incidents, and how well does a traffic GNN
(DCRNN) forecast through them? Traffic: TraffiDent (California PeMS, District 8, 898 mainline sensors,
Jan–Jun 2023). Ground truth: recorded CHP incidents (closure, SigAlert, collision, hazard, other).
Everything runs on a 16 GB Mac (Apple GPU); full training is better on a CUDA GPU.

## Checkpoint results (test period May 25 – Jun 30, 2023)

**A. Speed change in the first hour after an incident report** (`incident_impact.py`, 17,062 incidents,
16,933 no-incident controls at the same sensors ±1 week)

| Incident type | Mean speed drop (mph) | Share with a visible slowdown |
| --- | --- | --- |
| Closure | 3.57 (95% CI 2.03–5.29) | 25% |
| Other | 1.12 | |
| Collision | 0.98 | 19% |
| Hazard | 0.48 | |
| No incident (control) | 0.06 | 9% |

**B. Forecast error over the next hour, MAE in mph, by what is happening at the sensor** (`incident_eval.py`).
DCRNN runs are short (3 epochs on every 8th training window, then 2 fine-tuning epochs).

| Model | Normal | Closure | Collision | Hazard | SigAlert |
| --- | --- | --- | --- | --- | --- |
| Last value | 2.01 | 2.89 | 4.21 | 3.22 | 3.53 |
| Typical speed for time of week | 2.76 | 7.56 | 6.15 | 4.58 | 8.99 |
| DCRNN | 1.85 | 3.04 | 4.14 | 3.07 | 3.86 |
| DCRNN + 2 epochs | 1.84 | 3.01 | 4.09 | 3.01 | 3.82 |
| DCRNN + 2 epochs, blended with last value during incidents | 1.84 | 2.87 | 4.08 | 3.04 | 3.70 |

**C. Incident information.** Incident inputs plus 20× closure weighting (3.02 mph on closures) did no better
than the same 2 extra epochs without any incident information (3.01): at this budget the gain is from
training longer. With 35 test closures, differences under ~0.05 mph are treated as ties.
All numbers in `results/incident_eval_compare.json` and `results/incident_impact_summary.csv`.

An earlier closure-simulation pilot (`e2_pilot.py`, results in `results/e2_pilot*`) found the model predicts
almost no spillover; it is kept for reference but no longer part of the study.

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
python train.py --epochs 3 --stride 8 --batch-size 16 --incident-inputs --out checkpoints/dcrnn_check_incident.pt
python finetune.py --base checkpoints/dcrnn_check_incident.pt --out checkpoints/dcrnn_check_incident_ft.pt \
    --epochs 2 --lr 1e-4 --incident-weight 20 --stride 8 --batch-size 16   # upweight closures in the loss
python incident_impact.py                    # A: speed and flow change per incident type vs controls
python incident_eval.py checkpoints/*.pt --blend   # B: error per incident type, baselines + checkpoints
python baselines.py --checkpoint checkpoints/dcrnn_quick.pt   # last value, typical speed, DCRNN (overall)
python e2_pilot.py  --checkpoint checkpoints/dcrnn_quick.pt   # earlier pilot: simulated vs observed spillover
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
| `incident_impact.py` | Speed and flow change after each incident type, against no-incident controls |
| `incident_eval.py` | Forecast error per incident type (MAE, MAE@15/60 min, 95th percentile) |
| `baselines.py` | Last value, typical speed, and the incident blend (model + last value during incidents) |
| `prepare_data.py`, `train.py`, `tune.py`, `finetune.py`, `e2_pilot.py` | Scripts |
| `results/` | Metrics from the runs below |

Conventions: `adj[i, j] > 0` means traffic flows from sensor i to j (i is upstream). Slowdowns are in mph,
positive = slower. Speed 0 = missing reading.
