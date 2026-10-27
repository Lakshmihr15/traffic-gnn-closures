"""Part A: what each incident type does to speed and flow at its sensor (data only, no model).

    python incident_impact.py

For every recorded freeway incident with a recorded duration, speed and flow are compared with the sensor's
typical value for that 5-minute slot of the week, from 30 minutes before to 90 minutes after the incident starts.
The effect is measured as the change from the 30 minutes before, so congestion that was already there is not
counted. Incidents with another incident at the same sensor in the hour before are skipped (their "before"
period is not clean). Each incident gets a no-incident control: the same sensor and time of week one week
earlier or later, with no incident around it. Random fluctuation alone produces a positive "peak drop", so
an incident type's effect is what it shows beyond the control.

Writes results/incident_impact_events.csv (one row per incident) and results/incident_impact_summary.csv.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from config import EVENTS, RESULTS, DataConfig
from prepare_data import time_axis
from tgc import data as D

BEFORE, AFTER = 6, 18          # 30 minutes before, 90 minutes after (5-minute steps)
FIRST_HOUR = slice(0, 12)
VISIBLE_MPH = 10.0
KINDS = ["closure", "sigalert", "collision", "hazard", "other", "no incident"]


def typical(x, slots, exclude):
    """Median per sensor and slot of the week, leaving out missing readings and readings during incidents."""
    y = np.where(exclude | ~np.isfinite(x), np.nan, x)
    return pd.DataFrame(y).groupby(slots).median().reindex(range(slots.max() + 1)).to_numpy(np.float32)


def bootstrap_mean(v, n=2000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan, np.nan, np.nan
    means = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    return v.mean(), *np.percentile(means, [2.5, 97.5])


def main(cfg=DataConfig()):
    sensors = D.load_sensors(cfg)
    start, T = time_axis(cfg)
    stamps = pd.date_range(start, periods=T, freq=f"{cfg.freq_minutes}min")
    slots = D.slot_of_week(stamps, cfg.freq_minutes)
    print(f"{len(sensors)} sensors, {T} steps; loading speed and flow")
    speed = D.load_traffic(sensors, cfg)
    speed[speed <= 0] = np.nan
    flow = D.load_traffic(sensors, replace(cfg, target="flow"))

    events = pd.read_csv(EVENTS, parse_dates=["time"])
    events = events[~events.on_ramp].sort_values("start_step").reset_index(drop=True)
    active = np.zeros((T, len(sensors)), bool)
    for e in events.itertuples():
        active[e.start_step:e.end_step, e.node] = True
    typ_speed, typ_flow = typical(speed, slots, active), typical(flow, slots, active)

    def measure(s, n):
        steps = np.arange(s - BEFORE, s + AFTER)
        dev_s = speed[steps, n] - typ_speed[slots[steps], n]
        dev_f = flow[steps, n] - typ_flow[slots[steps], n]
        pre_s, pre_f = np.nanmean(dev_s[:BEFORE]), np.nanmean(dev_f[:BEFORE])
        drop = -(dev_s[BEFORE:] - pre_s)                     # mph, positive = slower than before
        flow_chg = dev_f[BEFORE:] - pre_f                    # vehicles per 5 min
        typ_f = typ_flow[slots[steps[BEFORE:]], n]
        if np.isnan(drop[FIRST_HOUR]).all():
            return None
        return {
            "speed_drop_mph": np.nanmean(drop[FIRST_HOUR]),
            "peak_speed_drop_mph": np.nanmax(drop[FIRST_HOUR]),
            "minutes_to_peak": 5 * int(np.nanargmax(drop[FIRST_HOUR])),
            "flow_change_veh": np.nanmean(flow_chg[FIRST_HOUR]),
            "flow_change_pct": 100 * np.nanmean(flow_chg[FIRST_HOUR]) / max(np.nanmean(typ_f[FIRST_HOUR]), 1),
            "visible": np.nanmax(drop[FIRST_HOUR]) >= VISIBLE_MPH,
            **{f"drop_t{5 * k:03d}": drop[k] for k in range(AFTER)},
        }

    def clean(s, n):
        """Window inside the data and no incident at this sensor from 1 h before to the end of the window."""
        return s - BEFORE - 12 >= 0 and s + AFTER < T and not active[s - BEFORE - 12:s + AFTER, n].any()

    rows = []
    last_start = {}
    week = 7 * 24 * 60 // cfg.freq_minutes
    for e in events.itertuples():
        s, n = e.start_step, e.node
        prev = last_start.get(n)
        last_start[n] = s
        if s - BEFORE < 0 or s + AFTER >= T or (prev is not None and s - prev < 12):
            continue
        base = {"incident_id": e.incident_id, "time": e.time, "node": n, "daytime": 6 <= e.time.hour < 20,
                "duration_min": e.duration}
        m = measure(s, n)
        if m is None:
            continue
        rows.append({**base, "kind": e.kind, **m})
        # Control: same sensor and time of week, one week earlier or later, with no incident around it
        for shift in (-week, week):
            if clean(s + shift, n) and (c := measure(s + shift, n)) is not None:
                rows.append({**base, "kind": "no incident", "duration_min": np.nan, **c})
                break
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "incident_impact_events.csv", index=False)

    summary = []
    for (label, sub) in [("all hours", df), ("daytime 6-20h", df[df.daytime]), ("night", df[~df.daytime])]:
        for kind in KINDS:
            g = sub[sub.kind == kind]
            if len(g) == 0:
                continue
            sd, fp = bootstrap_mean(g.speed_drop_mph), bootstrap_mean(g.flow_change_pct)
            summary.append({
                "hours": label, "kind": kind, "incidents": len(g),
                "median_duration_min": g.duration_min.median(),
                "speed_drop_mph": sd[0], "speed_drop_lo": sd[1], "speed_drop_hi": sd[2],
                "peak_drop_mph": g.peak_speed_drop_mph.mean(),
                "visible_pct": 100 * g.visible.mean(),
                "flow_change_pct": fp[0], "flow_change_lo": fp[1], "flow_change_hi": fp[2],
                "median_minutes_to_peak": g.minutes_to_peak.median(),
            })
    s = pd.DataFrame(summary)
    s.to_csv(RESULTS / "incident_impact_summary.csv", index=False)

    n_inc = int((df.kind != "no incident").sum())
    print(f"\n{n_inc} incidents with a clean hour before them (of {len(events)} on the freeway), "
          f"{len(df) - n_inc} no-incident controls")
    for label, g in s.groupby("hours", sort=False):
        print(f"\n=== {label}: change in the first hour vs the 30 min before, relative to typical")
        print(f"{'kind':10s} {'n':>5s} {'dur':>5s} {'speed drop mph [95% CI]':>26s} {'peak':>6s} {'visible':>8s} "
              f"{'flow change % [95% CI]':>26s} {'to peak':>8s}")
        for r in g.itertuples():
            print(f"{r.kind:10s} {r.incidents:5d} {r.median_duration_min:5.0f} "
                  f"{r.speed_drop_mph:7.2f} [{r.speed_drop_lo:6.2f},{r.speed_drop_hi:6.2f}] "
                  f"{r.peak_drop_mph:6.1f} {r.visible_pct:7.0f}% "
                  f"{r.flow_change_pct:7.1f} [{r.flow_change_lo:6.1f},{r.flow_change_hi:6.1f}] "
                  f"{r.median_minutes_to_peak:6.0f}m")


if __name__ == "__main__":
    main()
