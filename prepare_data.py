"""Run Steps 1-7 and Phase 3 once; save everything training needs to data/processed/.

    python prepare_data.py                # needs the monthly traffic files (pMM_done.npy)
    python prepare_data.py --events-only  # sensors, road graph and incident matching only
"""
import argparse
import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from config import EVENTS, PROCESSED, DataConfig
from tgc import data as D
from tgc import incidents as I


def time_axis(cfg):
    start = pd.Timestamp(year=cfg.year, month=cfg.months[0], day=1)
    end = pd.Timestamp(year=cfg.year, month=cfg.months[-1], day=1) + pd.offsets.MonthBegin(1)
    return start, int((end - start) / pd.Timedelta(minutes=cfg.freq_minutes))


def main(cfg=DataConfig(), events_only=False):
    print(f"Step 1: District {cfg.district}{' / ' + cfg.county if cfg.county else ''}, mainline sensors, "
          f"{cfg.year} months {cfg.months}, target = {cfg.target}")
    sensors = D.load_sensors(cfg)
    start, T = time_axis(cfg)
    N = len(sensors)
    print(f"  {N} sensors on {sensors.groupby(['Fwy', 'Direction']).ngroups} freeway directions; "
          f"{T} time steps from {start}")

    print("Step 3: directed road graph from freeway, direction and postmile")
    dist = D.road_distances(sensors, cfg.max_link_km, cfg.interchange_km)
    adj = D.build_adjacency(dist, cfg.adj_threshold)
    print(f"  {int((adj > 0).sum()) - N} directed edges (plus self-loops)")

    print("Phase 3: match CHP incidents to sensors")
    events = I.match_incidents(I.load_incidents(cfg.year), sensors, start, T, cfg)
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(EVENTS, index=False)
    print(pd.crosstab(events.kind, events.on_ramp.map({False: "freeway", True: "ramp"})).to_string())
    incident = I.event_mask(events, T, N, cfg.incident_kinds)
    print(f"  saved {len(events)} events to {EVENTS}")
    if events_only:
        return

    print("Step 1 (traffic): read the monthly files")
    speed = D.load_traffic(sensors, cfg)
    assert speed.shape == (T, N), f"expected {(T, N)}, got {speed.shape}"

    print("Step 2: clean")
    speed, missing, report = D.clean_speeds(speed, cfg.bad_sensor_missing_rate)
    print(f"  {report['missing_share']:.1%} of readings missing; "
          f"{len(report['bad_sensors'])} sensors missing more than {cfg.bad_sensor_missing_rate:.0%}")

    print("Step 4: add time features")
    tfeat, stamps = D.time_features(T, start, cfg.freq_minutes, cfg.day_of_week)

    print("Steps 5-6: windows and time-ordered train/val/test split")
    train_t, val_t, test_t = D.split_samples(T, cfg.seq_len, cfg.horizon, cfg.train_frac, cfg.val_frac)
    print(f"  train {len(train_t)}, val {len(val_t)}, test {len(test_t)} windows; "
          f"test starts {stamps[test_t[0] + 1]}")

    print("Step 7: scale with training-period statistics")
    mean, std = D.fit_scaler(speed, train_t, cfg.seq_len)
    print(f"  mean {mean:.2f}, std {std:.2f}")
    features = np.concatenate([((speed - mean) / std)[..., None],
                               np.broadcast_to(tfeat[:, None, :], (T, N, tfeat.shape[1]))], -1)

    print("Unexplained slowdowns (30% rule), only to compare with recorded incidents")
    slots = D.slot_of_week(stamps, cfg.freq_minutes)
    typical = D.typical_speed(speed, slots, train_t[-1] + 1)
    slowdown = D.nonrecurring_slowdowns(speed, typical, slots, cfg.slowdown_drop, cfg.slowdown_min_steps)
    # Any recorded incident (not only closures) at the sensor itself or a road neighbour
    any_incident = I.event_mask(events, T, N, events.kind.unique())
    near = np.isfinite(dist) | np.isfinite(dist.T)
    incident_near = np.zeros_like(any_incident)
    for k in range(N):
        incident_near[:, near[k]] |= any_incident[:, [k]]
    print(f"  {slowdown.mean():.2%} of sensor-readings in a slowdown; "
          f"{(slowdown & incident_near).sum() / max(slowdown.sum(), 1):.1%} of those have a recorded incident nearby")

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PROCESSED, features=features.astype(np.float32), speed=speed, missing=missing,
        incident=incident, slowdown=slowdown, typical=typical, slots=slots, adj=adj, dist_km=dist,
        station_ids=sensors.station_id.to_numpy(), mean=mean, std=std, seq_len=cfg.seq_len,
        horizon=cfg.horizon, train_t=train_t, val_t=val_t, test_t=test_t)
    (PROCESSED.parent / "prepare_report.json").write_text(json.dumps({**report, **asdict(cfg)}, indent=2))
    print(f"saved {PROCESSED}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-only", action="store_true")
    ap.add_argument("--county", default=DataConfig.county, help='e.g. "San Bernardino" for a smaller network')
    args = ap.parse_args()
    main(replace(DataConfig(), county=args.county), events_only=args.events_only)
