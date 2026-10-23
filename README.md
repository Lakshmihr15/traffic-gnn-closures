# Traffic GNNs under road closures — MVP code

Pretrained DCRNN on METR-LA, a closure simulator, a real-event extractor and spillover metrics.
Everything runs on CPU.

## Setup

```bash
python -m venv venv && . venv/bin/activate
pip install torch numpy pandas scipy
git clone --depth 1 https://github.com/chnsh/DCRNN_PyTorch.git     # road graph + pretrained TF checkpoint
git clone --depth 1 https://github.com/hazdzz/STGCN.git stgcn      # data/metr-la/vel.csv (METR-LA speeds)
```

`vel.csv` holds the same values as DCRNN's `metr-la.h5`, in the same column order (checked against the
test-set truth shipped in `DCRNN_PyTorch/data/dcrnn_predictions_pytorch.npz`).
Set `ROOT` in `data.py` to the folder that holds both clones.

The authors' pretrained weights are a TensorFlow checkpoint. Export them once to NumPy (needs `tensorflow`):

```python
import tensorflow as tf, numpy as np
r = tf.train.load_checkpoint('DCRNN_PyTorch/data/model/pretrained/METR-LA/models-2.7422-24375')
np.savez('dcrnn_la_weights.npz', **{k.replace('DCRNN/DCRNN_SEQ/', ''): r.get_tensor(k)
         for k in r.get_variable_to_shape_map() if k.startswith('DCRNN/DCRNN_SEQ') and 'Adam' not in k})
```

## Run

```bash
python eval_test.py      # pretrained DCRNN on the 6,850 test windows -> results/test_pred.npy (~20 min CPU)
python events.py         # typical speed + real events in the test period -> results/events_test.csv
python closure_exp.py    # E2 pilot: 3 closure variants x 145 events -> results/closure_pilot.csv
python analyze.py        # accuracy tables, event vs normal error, spillover summary -> results/summary.json
```

## Files

| File | What it does |
| --- | --- |
| `dcrnn.py` | Minimal PyTorch DCRNN matching the original TF cell; the graph only enters through `build_supports(adj)` |
| `data.py` | Loading, splits and scaling, identical to DCRNN's `generate_training_data.py` |
| `events.py` | Typical speed (training period only) and the event extractor |
| `closure_exp.py` | Closure variants (cut, transplant, both), counterfactual runs, spillover summary per event |
| `analyze.py` | Level 1–3 metrics, baselines, bootstrap intervals |

Convention: A[i, j] > 0 means i is upstream of j (driving distance from i to j). Slowdowns are in mph, positive = slower.
