# Demo recording script: 3 speakers, about 8 minutes

The demo is split into three parts of about 2.5–3 minutes each, one per team member. Each part tells one
complete piece of the story and ends with a one-line handover to the next speaker.

| Part | Speaker | Time | What it covers | Report |
|---|---|---|---|---|
| 1 | Speaker 1: ________ | 0:00–2:45 | The question, the data, the pipeline, and the first finding: DCRNN loses during closures | pages 1–4 |
| 2 | Speaker 2: ________ | 2:45–5:30 | A live closure forecast, and the test of the neighbour explanation, which turned out wrong | pages 4–6 |
| 3 | Speaker 3: ________ | 5:30–8:15 | The explanation that held (learned recovery), the partial fix, and what we learned | pages 6–8 |

**Screen setup (everyone):** the repository open in an editor, a terminal in the repo root with the venv
active, and `report/final_report.pdf` open in a PDF viewer. Every command below runs in seconds. Run each one
once before recording so the data is cached.

**Recording:** record in one call with screen sharing passed between speakers, or record each part
separately and join the clips. Either way, keep the handover lines so the parts connect.

**Running the commands:** `demo.py` needs the processed data and the checkpoints, which are not on GitHub.
Whoever has them on their laptop should run the commands in Part 2. If that isn't Speaker 2, Speaker 2 can
show the saved image `results/demo_closure.png` instead.

---

## Part 1: Speaker 1, the question and the first finding (0:00–2:45)

### 1a. The question (0:00–0:45), report page 1

> "Hi, we are group Machine Learning INET - 6. Traffic forecasting models like DCRNN are judged on average
> error, but the average is mostly normal traffic. We asked what happens during road closures: how much do
> closures change traffic, how well does DCRNN forecast through them, and when and why does it fail?"

- Data: TraffiDent, PeMS District 8 in California, 898 freeway sensors, January–June 2023, with California
  Highway Patrol incident records as ground truth.
- The pivot: "We started on METR-LA, but it has no incident labels, so we couldn't tell which slowdowns were
  closures. We switched to a dataset with recorded closures."

### 1b. The pipeline and a sanity check (0:45–1:30), terminal + editor

```bash
python -m pytest tests -q
```

> "These tests need no data. They check that the road graph points in the direction of traffic, that cutting
> a sensor's edges isolates it, and that our graph gate, which my teammate will use, leaves the model
> unchanged when open and isolates the sensor when closed."

Open `tgc/model.py` and briefly show `DCRNN`: 2 layers, 64 units, diffusion over the road graph in both
directions. The model forecasts the next hour of speeds from the past hour.

### 1c. Closures matter and the model loses on them (1:30–2:45), report pages 3–4

- Figure 1: closures slow traffic the most, 3.6 mph below typical against 0.06 mph for matched control times.
- Table 1 / Figure 2: DCRNN is the best model in normal traffic, but during closures it loses to simply
  repeating the last reading. "The overall error hides this."
- Section 4.3: adding incident information as an input looked like a gain, until a control trained for the
  same number of epochs showed the same gain. "The gain came from training longer, not from the incident
  information. That control changed our conclusion."

**Handover:**
> "So DCRNN fails on closures. [Speaker 2] will show you what that looks like on a real closure, and what we
> first thought was causing it."

---

## Part 2: Speaker 2, a real closure and the neighbour explanation (2:45–5:30)

### 2a. Live: one real closure (2:45–3:45), terminal

```bash
python demo.py --list | head -8
python demo.py --rank 1
```

Open `results/demo_closure.png`.

> "This is a real closure on I-15 North. The forecast is made 15 minutes after the closure was reported. The
> black line is what actually happened: traffic stays stopped. DCRNN, in blue, forecasts a fast recovery and
> is off by 22 mph. Simply repeating the last reading is off by under 3."

> "The green model, which was trained without the road graph and so cannot hear its neighbours, does better
> here. That looks like proof that the neighbours are to blame. But it's only one closure."

### 2b. The failure has a direction (3:45–4:15), report Figure 3

- On disrupted closure readings DCRNN is +5.4 mph too fast on average: it expects traffic to recover too early.
- "Our feedback on the checkpoint suggested a reason: the roads around a closure still look normal, and the
  model leans on them. We decided to test that properly."

### 2c. The test: cut the neighbours (4:15–5:30), report Section 5, pages 5–6

Open `tgc/model.py` at `DiffusionConv.forward` and point at the line `g * out + (1 - g) * v`.

> "This one line is the experiment. At sensors with an incident, we can turn the neighbours down or off. If
> the neighbours cause the error, cutting them should reduce it."

- We wrote down four predictions that could fail before running it.
- Table 2 / Figure 4: across all 35 test closures, cutting the neighbours does **not** reduce the bias
  (+5.8 vs +5.4 mph). Cutting only the downstream neighbours makes it worse. A model fine-tuned with the gate
  doesn't help either.
- Figure 5: the bias is largest where the neighbours are already slower than the closure sensor. That is the
  opposite of what the explanation predicts.
- Placebo: cutting the neighbours everywhere makes normal traffic worse, so the graph helps in general.

To show the saved numbers live (the full experiment takes hours on a laptop):

```bash
python -c "import json; d=json.load(open('results/graph_gating.json'))['closure']; [print(f'{k:38s} bias {v[\"disrupted\"][\"bias\"]:+.2f}') for k,v in d.items()]"
```

**Handover:**
> "So the neighbour explanation was wrong, even though the single example made it look right. But every
> learned model shared the bias, with or without the graph. [Speaker 3] will show what that pointed to."

---

## Part 3: Speaker 3, the explanation that held, the fix, and what we learned (5:30–8:15)

### 3a. The second explanation: learned recovery (5:30–6:45), report Section 6, pages 6–7

> "If the graph isn't the cause, maybe the model simply learned how fast slowed traffic usually recovers,
> and closures don't recover like that."

- In training, only 17% of slowed closure moments were back to normal within the hour, against 83% for
  everyday congestion.
- Figure 7 / Table 3: on test closures, DCRNN predicts +8.5 mph of recovery in the next hour. That's close to
  the normal recovery for traffic at that speed and hour (+10.2), and far above what actually happened (+3.6).
- This also explains the no-graph model: it learned a weaker recovery everywhere (+4.9 on closures), which
  is why it did better on closures.
- One prediction failed: we expected the model to be too fast on every unusual slowdown. It isn't, because
  those recover quickly. It is too fast only on closures. "From speed history alone, the model can't tell a
  closure from an ordinary jam."

### 3b. A partial fix (6:45–7:15)

- We fine-tuned with unusual slowdowns weighted 5 times more. The too-fast bias on closures fell from +5.4 to
  +4.1 mph, and the confidence interval excludes zero.
- The cost: the model now expects less recovery everywhere, so everyday congestion is forecast too slow.
- It still doesn't beat repeating the last reading during closures.
- "A real fix needs the model to recognise a closure. Training had only 428 slowed closure moments against
  about 720,000 other slowdowns."

### 3c. What we learned (7:15–8:15), report Sections 7–8, pages 7–8

- Average error hides the failure that matters. Compare against simple baselines, situation by situation.
- Signed error turned "the model is worse" into a specific behaviour: it expects recovery too early.
- Controls decided our conclusions: equal training time, a placebo, and resampling whole closures.
- A plausible explanation, backed by a vivid example, was refuted by a test designed to be able to fail.
- At our training budget, a simple blend with the last reading beat every learned fix.

> "Thank you. The code, results and report are all in our repository."

---

## Questions each speaker should be ready for

**Speaker 1**
- Why not METR-LA? It has no incident labels, so closures can't be identified.
- What is "last value"? Repeating the most recent reading for the whole next hour.
- Why train only 3 epochs? Laptop budget; the equal-epochs control keeps comparisons fair.

**Speaker 2**
- What does the gate do exactly? It mixes each graph step with the sensor's own value: g = 1 is normal,
  g = 0 means the sensor only sees itself.
- Why resample whole closures for the intervals? Readings from one closure aren't independent; there are
  only 35 closures.
- Did the model ever see gated sensors in training? Yes, the gate-trained model, and it didn't help.

**Speaker 3**
- What is the "base rate"? The average one-hour recovery in training for the same speed band and hour of day.
- Why not just weight closures more? There are too few slowed closure moments (428) to learn from.
- What would you do next? Give the model a reliable closure signal, or combine it with last value when an
  incident is active.
