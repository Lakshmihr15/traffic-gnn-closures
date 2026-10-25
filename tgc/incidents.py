"""Phase 3: turn CHP incident records into events at our sensors.

An event is a recorded incident (closure, SigAlert, collision, ...) matched to the nearest mainline sensor
on the same freeway and direction, with a start step and an end step in our time axis.
"""
import re

import numpy as np
import pandas as pd

from config import RAW

COLLISION_CODES = ["1179", "1180", "1181", "1182", "1183", "20001", "20002"]
HAZARD_CODES = ["1125", "1125A"]


def load_incidents(year):
    """The full file from the Kaggle archive if present, else a column subset downloaded from the website."""
    full = RAW / f"incidents_y{year}.csv"
    path = full if full.exists() else next(RAW.glob(f"incidents_y{year}*.csv"))
    with open(path) as f:
        sep = "\t" if "\t" in f.readline() else ","   # the archive's file is tab-separated
    return pd.read_csv(path, sep=sep, low_memory=False)


def categorize(description):
    """closure / sigalert / collision / hazard / other, from the CHP description text."""
    d = description.fillna("").str.upper()
    code = d.str.split("-").str[0].str.strip()
    return np.select(
        [d.str.startswith("CLOSURE"), d.str.contains(r"SIG ?ALERT"),
         code.isin(COLLISION_CODES), code.isin(HAZARD_CODES) | d.str.startswith("ANIMAL")],
        ["closure", "sigalert", "collision", "hazard"], "other")


def direction_on(fwy, location):
    """Travel direction of freeway `fwy` in a CHP location such as 'I10 E / Vineyard Ave' -> 'E'."""
    m = re.search(rf"\b(?:I|SR|US)-?{fwy}\s+([NSEW])\b", location, re.IGNORECASE)
    return m.group(1).upper() if m else None


def on_ramp(location):
    """True when the incident is on a ramp ('Tyler St Onr / Sr91 E'), not on the freeway itself."""
    return location.fillna("").str.split("/").str[0].str.contains(r"\b(?:Onr|Ofr)\b", case=False)


def match_incidents(incidents, sensors, start, num_steps, cfg):
    """One row per incident that falls inside our time window and within cfg.incident_match_miles of one
    of our sensors on the same freeway and direction."""
    inc = incidents.dropna(subset=["Fwy", "Abs PM", "dt"]).copy()
    inc["Fwy"] = inc.Fwy.astype(int)
    inc = inc[inc.Fwy.isin(sensors.Fwy.unique())]
    inc["kind"] = categorize(inc.DESCRIPTION)
    inc["Direction"] = [direction_on(f, loc) for f, loc in zip(inc.Fwy, inc.LOCATION.fillna(""))]
    inc["on_ramp"] = on_ramp(inc.LOCATION)
    inc["time"] = pd.to_datetime(inc.dt, format="mixed", errors="coerce")
    inc["start_step"] = np.floor((inc.time - start) / pd.Timedelta(minutes=cfg.freq_minutes))
    inc = inc[(inc.start_step >= 0) & (inc.start_step < num_steps) & inc.Direction.notna()]

    right = sensors[["Fwy", "Direction", "Abs PM", "station_id"]].copy()
    right["node"], right["sensor_pm"] = np.arange(len(sensors)), right["Abs PM"]
    ev = pd.merge_asof(inc.sort_values("Abs PM"), right.sort_values("Abs PM"), on="Abs PM",
                       by=["Fwy", "Direction"], direction="nearest", tolerance=cfg.incident_match_miles)
    # Only records with a recorded, positive duration: no default or capped values are filled in
    ev = ev.dropna(subset=["node"])
    ev = ev[ev.duration > 0]

    ev["start_step"] = ev.start_step.astype(int)
    ev["end_step"] = np.minimum(ev.start_step + np.ceil(ev.duration / cfg.freq_minutes).astype(int), num_steps)
    ev["node"] = ev.node.astype(int)
    ev["offset_miles"] = (ev["Abs PM"] - ev.sensor_pm).abs().round(2)
    cols = ["incident_id", "kind", "DESCRIPTION", "LOCATION", "time", "duration", "start_step", "end_step",
            "node", "station_id", "Fwy", "Direction", "offset_miles", "on_ramp"]
    return ev[cols].sort_values("start_step").reset_index(drop=True)


def event_mask(events, num_steps, num_nodes, kinds):
    """(T, N) True while a freeway incident of the given kinds is active at its sensor."""
    mask = np.zeros((num_steps, num_nodes), dtype=bool)
    for e in events[events.kind.isin(kinds) & ~events.on_ramp].itertuples():
        mask[e.start_step:e.end_step, e.node] = True
    return mask
