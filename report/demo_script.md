# Demo recording script (about 8 minutes)

Screen: the repository open in an editor, a terminal in the repo root with the venv active, and
`report/final_report.pdf` open in a PDF viewer. Every command below runs in seconds except where noted. Run
each once before recording so the data is cached.

---

## 1. The question (0:00–0:45), show report page 1

> "Traffic forecasting GNNs are judged on average error, but averages are mostly normal traffic. We asked
> what happens during road closures: how much do they change traffic, how well does DCRNN forecast through
> them, and when and why does it fail?"

- Data: TraffiDent, PeMS District 8, 898 freeway sensors, January–June 2023, with CHP incident records
  as ground truth.
- Mention the pivot: we started on METR-LA, but it has no incident labels, so we switched datasets.

## 2. The pipeline and a sanity check (0:45–1:45), terminal + editor

```bash
python -m pytest tests -q
```

> "These tests need no data. They check that the road graph points in the direction of traffic, that cutting
> a sensor's edges isolates it, and that our graph gate leaves the model unchanged when open and isolates the
> sensor when closed."

Open `tgc/model.py` and point at `DiffusionConv.forward`: the `diffuse` function and the line
`g * out + (1 - g) * v`. That line is the whole gating experiment.

## 3. Closures matter and the model loses on them (1:45–3:00), report pages 3–4

- Figure 1: closures slow traffic most (3.6 mph vs 0.06 for controls).
- Table 1 / Figure 2: DCRNN is best in normal traffic and loses to last value during closures. "Overall
  MAE hides this."
- Section 4.5: incident inputs looked like a gain until the equal-epochs control showed the same gain from
  training longer.

## 4. Live: one real closure (3:00–4:30), terminal

```bash
python demo.py --list | head -8
python demo.py --rank 1
```

Open `results/demo_closure.png`.

> "This is a real closure on I-15 North. The forecast is made 15 minutes after the report. The black line is
> what actually happened: traffic stays stopped. DCRNN, in blue, forecasts a fast recovery and is off by 22 mph.
> Simply repeating the last reading is off by under 3."

> "Here the orange model, which cannot hear its neighbours, does better. That looks like proof that the
> neighbours are to blame. It isn't. It's one closure."

Optionally run `python demo.py --rank 3` to show a second closure.

## 5. The hypothesis tests (4:30–7:00), report pages 4–9

- Figure 3: the failure has a direction, +5.4 mph too fast on disrupted readings, i.e. expecting recovery too early.
- Our explanation was that normal-looking neighbours pull the forecast up. We wrote down four predictions
  that could fail (Section 4.8).
- Table 2 / Figure 4: across all 35 test closures, cutting the neighbours does **not** reduce the bias. Cutting
  the downstream ones makes it worse. Fine-tuning with the gate doesn't help.
- Figure 5: the bias is largest where neighbours are already slower. That's the opposite of what the
  explanation predicts.

> "So the explanation was wrong. The bias is shared by every learned model, which pointed to a second
> explanation: the model learned how fast slowed traffic usually recovers."

- Figure 7 / Table 3 (report page 8): on closures the model predicts +8.5 mph of recovery, close to the
  training base rate (+10.2) and far above what actually happened (+3.6). In training, only 17% of slowed
  closure moments were back to normal within the hour, against 83% for everyday congestion.
- Fine-tuning with unusual slowdowns weighted more cut the closure bias by 1.3 mph, but only by expecting less
  recovery everywhere. "A real fix needs the model to recognise a closure."

To show the evidence live without re-running it, print the saved result (`graph_gating.py` itself takes
hours on a laptop):

```bash
python -c "import json; d=json.load(open('results/graph_gating.json'))['closure']; [print(f'{k:38s} bias {v[\"disrupted\"][\"bias\"]:+.2f}') for k,v in d.items()]"
```

## 6. What we learned (7:00–8:00), report pages 9–10

- Average error hides the failure that matters; compare against persistence, per situation.
- Signed error turned "worse" into a specific behaviour.
- Controls decided the conclusions: equal epochs, placebo, resampling whole closures.
- A plausible mechanism, backed by a vivid example, was refuted by a test designed to be able to fail.
- At our budget, a simple blend with last value beat every learned fix.
