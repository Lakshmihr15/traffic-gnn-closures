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

**D. Graph ablation** (`train.py --no-graph`: same weights and budget, each sensor sees only its own history).
The graph helps in normal traffic (1.84 vs 1.88) and mild incidents, but hurts in severe ones: without it,
closures 3.01 → 2.94 and SigAlerts 3.82 → 3.57 (95th-percentile error 19.8 → 14.6). Results in
`results/incident_eval_nograph.json`.

**E. Closure error breakdown** (`closure_errors.py`). The DCRNN beats last value on closure readings where traffic is
not actually slowed (1.50 vs 1.65) and loses where it is (7.24 vs 6.38), forecasting 5.4 mph too fast on average
(last value +2.9, no-graph model +4.5). It loses most 30–60 min after the report and in daytime. Results in
`results/closure_errors.json`. The first explanation, that neighbours which still look normal pull the forecast
toward recovery, did not hold up when tested (F); a learned expectation of recovery did (G).

**F. Graph gating** (`graph_gating.py`, gate in `tgc/gating.py`). Tests whether neighbours cause the fast bias.
At sensors with an active incident at forecast time, the graph model hears its own readings in place of its
neighbours' (gate g = 0), for one direction or both, with and without fine-tuning under the gate. Disrupted
closure readings, 35 closures; intervals resample closures.

| Model | MAE | Bias (95% CI) | MAE change vs graph (95% CI) |
| --- | --- | --- | --- |
| Graph | 7.24 | +5.39 (2.55 to 10.64) | |
| Gate both directions | 7.75 | +5.84 (2.91 to 10.83) | +0.51 (−0.07 to +1.21) |
| Gate both, fine-tuned with the gate | 7.63 | +5.79 (2.87 to 10.80) | +0.39 (−0.17 to +1.10) |
| Gate upstream only | 7.31 | +5.17 (2.29 to 10.55) | +0.07 (−0.29 to +0.52) |
| Gate downstream only | 7.97 | +6.72 (3.35 to 12.61) | +0.73 (+0.27 to +1.45) |
| No graph (trained without one) | 6.89 | +4.54 (2.20 to 8.83) | −0.35 (−1.07 to +0.19) |

Cutting the closure sensor off from its neighbours does not remove the fast bias; it raises it slightly, and
also after fine-tuning with the gate. Downstream neighbours, which the explanation blamed, make the forecast
*slower*: cutting them raises the bias most. The graph model's extra bias over the no-graph model does not grow
with how much faster the downstream sensors look (correlation −0.03), and the bias is largest (+11 mph) where
neighbours are already *slower* than the closure sensor. So the no-graph model's lower bias comes from how it was
trained, not from neighbour messages at forecast time; the fast bias is shared by every learned model (and, less,
by last value), consistent with a learned expectation that slowdowns recover. A gate of 0.5 lowered the bias
(+4.53) but 0 raised it, so there is no dose-response; this is treated as noise. Gating every sensor (placebo)
costs 0.10 mph in normal traffic, so the graph is useful outside incidents. Results in `results/graph_gating.json`.

**G. Learned recovery** (`recovery_prior.py`, `finetune.py --slowdown-weight 5`). Tests the second explanation:
the model learned how fast slowed traffic usually recovers, and closures are the exception. At each moment a
sensor is slowed (below 50 mph), compare the next hour's actual speed gain, the model's predicted gain, and the
training-period average gain for the same speed and hour (the "base rate").

| Slowed traffic | Training: back to normal within 1 h | Test: actual / base rate / DCRNN predicted (mph) | DCRNN bias |
| --- | --- | --- | --- |
| Usual congestion | 83% | +5.4 / +7.1 / +6.0 | +0.59 |
| Unusual slowdown, no incident | 43% | +9.2 / +7.1 / +8.4 | −0.96 |
| Closure (11 closures) | 17% | +3.6 / +10.2 / +8.5 | +2.90 |

On closures the model predicts close to the base rate and far above what happens. It is not too fast on unusual
slowdowns in general (those recover quickly), only on closures, which it cannot tell apart from speed history.
The no-graph model learned a weaker recovery expectation everywhere (+4.9 on closures), which explains its lower
closure bias. Fine-tuning with unusual slowdowns weighted 5× (only 0.15% of them are closures) lowered the bias on
disrupted closure readings from +5.39 to +4.09 mph (−1.30, 95% CI −2.47 to −0.48; MAE −0.56, CI −1.30 to −0.18),
against the same fine-tuning without weights. But it does so by expecting less recovery everywhere (everyday
congestion +0.59 → −1.41 mph), and it still loses to last value on disrupted closures (6.68 vs 6.38). Results in
`results/recovery_prior.json`.

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
python train.py --epochs 3 --stride 8 --batch-size 16 --no-graph --out checkpoints/dcrnn_check_nograph.pt   # D: ablation
python closure_errors.py checkpoints/dcrnn_check_plain_ft.pt checkpoints/dcrnn_check_nograph_ft.pt   # E: closure errors
python finetune.py --base checkpoints/dcrnn_check_plain.pt --out checkpoints/dcrnn_check_gated_ft.pt \
    --epochs 2 --lr 1e-4 --incident-weight 1 --gate-incidents --stride 8 --batch-size 16   # F: train with the gate
python graph_gating.py --gated checkpoints/dcrnn_check_gated_ft.pt   # F: graph gating
python finetune.py --base checkpoints/dcrnn_check_plain.pt --out checkpoints/dcrnn_check_slowdown_ft.pt \
    --epochs 2 --lr 1e-4 --incident-weight 1 --slowdown-weight 5 --stride 8 --batch-size 16   # G: weight slowdowns
python recovery_prior.py checkpoints/dcrnn_check_plain_ft.pt checkpoints/dcrnn_check_nograph_ft.pt \
    checkpoints/dcrnn_check_slowdown_ft.pt   # G: learned recovery
python baselines.py --checkpoint checkpoints/dcrnn_quick.pt   # last value, typical speed, DCRNN (overall)
python e2_pilot.py  --checkpoint checkpoints/dcrnn_quick.pt   # earlier pilot: simulated vs observed spillover
python demo.py                               # one real test closure: forecasts of every model -> results/demo_closure.png
python report/make_report.py                 # final report -> report/final_report.pdf
```

Settings (district, months, target channel, thresholds, model size) are in `config.py`.

## Files

| File | What it does |
| --- | --- |
| `config.py` | Paths and all settings |
| `tgc/data.py` | Steps 1–7: sensors, traffic, cleaning, directed road graph from postmiles, time features, windows, split, scaling |
| `tgc/incidents.py` | Classifies CHP incidents and matches them to the nearest sensor on the same freeway and direction |
| `tgc/model.py` | DCRNN; the graph enters only through `supports` (and an optional per-sensor gate), so closures can be applied at inference |
| `tgc/training.py` | Masked metrics, training epoch, evaluation (overall, per horizon, closures vs normal) |
| `tgc/closures.py` | Closure simulator (input / graph / both) and spillover measures |
| `incident_impact.py` | Speed and flow change after each incident type, against no-incident controls |
| `incident_eval.py` | Forecast error per incident type (MAE, MAE@15/60 min, 95th percentile) |
| `closure_errors.py` | Closure errors split by disruption, time since report, day/night, horizon, with signed bias |
| `tgc/gating.py`, `graph_gating.py` | Graph gate at incident sensors; tests whether neighbours cause the closure bias |
| `recovery_prior.py` | Predicted vs actual vs base-rate recovery from slowed traffic; tests the learned-recovery explanation |
| `baselines.py` | Last value, typical speed, and the incident blend (model + last value during incidents) |
| `prepare_data.py`, `train.py`, `tune.py`, `finetune.py`, `e2_pilot.py` | Scripts |
| `demo.py` | Demo: forecasts of each model for one test closure, with a plot |
| `report/` | Final report (`final_report.pdf`), its generator and figures, and the demo recording script |
| `results/` | Metrics from the runs below |

Conventions: `adj[i, j] > 0` means traffic flows from sensor i to j (i is upstream). Slowdowns are in mph,
positive = slower. Speed 0 = missing reading.
