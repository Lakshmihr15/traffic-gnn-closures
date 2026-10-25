"""Steps 1-7: load TraffiDent, clean, graph, time features, windows, splits, scaling.

Conventions used everywhere:
  speed      (T, N) target channel (speed in mph, or flow), 0 = missing reading
  adj[i, j]  > 0 means traffic flows from sensor i to sensor j (i is upstream of j)
  dist[i, j] driving distance in km from i to j, inf if not linked
  sample t   input = readings t-11..t, target = readings t+1..t+12
"""
import numpy as np
import pandas as pd
import torch

from config import CHANNELS, RAW

MILE_KM = 1.609344


# Step 1: load the sensors we study and their traffic
def load_sensors(cfg):
    """Mainline sensors of one district, with `col` = their column in the traffic files."""
    sensors = pd.read_csv(RAW / "sensor_meta_feature.csv", sep="\t")
    order = np.load(RAW / "node_order.npy")
    assert sensors.station_id.is_unique and set(order) == set(sensors.station_id)
    sensors = sensors.set_index("station_id").loc[order].reset_index()
    sensors["col"] = np.arange(len(sensors))
    keep = (sensors.Type == "Mainline") & (sensors.District == cfg.district)
    if cfg.county:
        keep &= sensors.County == cfg.county
    return sensors[keep].reset_index(drop=True)


def month_file(year, month):
    candidates = sorted(RAW.glob(f"**/*p{month:02d}*.npy"))   # Kaggle: year_2023/year_2023/2023_p01.npy
    in_year = [p for p in candidates if f"year_{year}" in str(p)]
    if not (in_year or candidates):
        raise FileNotFoundError(f"put {year}_p{month:02d}.npy (TraffiDent year_{year}) under {RAW}")
    return (in_year or candidates)[0]


def load_traffic(sensors, cfg):
    """(T, N) target channel for our sensors, months concatenated. Files are memory-mapped, so only our
    sensors end up in RAM. The 2023 files are (sensors, time, channels); the authors' notebook shows
    (time, sensors, channels), so the layout is detected from the sensor count."""
    cols, ch = sensors.col.to_numpy(), CHANNELS[cfg.target]
    n_all = len(np.load(RAW / "node_order.npy"))
    parts = []
    for m in cfg.months:
        month = np.load(month_file(cfg.year, m), mmap_mode="r")
        if month.shape[0] == n_all:
            parts.append(np.asarray(month[cols, :, ch], dtype=np.float32).T)
        else:
            parts.append(np.asarray(month[:, cols, ch], dtype=np.float32))
    return np.concatenate(parts)


# Step 2: clean the data
def clean_speeds(speed, bad_rate):
    """NaNs, zeros and negatives become 0 = missing. Sensors are kept but reported."""
    speed = np.nan_to_num(speed, nan=0.0)
    speed[speed < 0] = 0.0
    missing = speed == 0
    rate = missing.mean(0)
    report = {
        "missing_share": float(missing.mean()),
        "bad_sensors": np.flatnonzero(rate > bad_rate).tolist(),
        "max_speed": float(speed.max()),
    }
    return speed, missing, report


# Step 3: build the road network
def road_distances(sensors, max_link_km, interchange_km):
    """Directed driving distances (km) between sensors.

    Along a freeway: Caltrans postmiles grow northbound and eastbound, so on an N/E carriageway traffic
    flows from low to high postmile, and on S/W from high to low. dist[i, j] is set when j is downstream
    of i on the same freeway and direction, within max_link_km.
    Interchanges: sensors on different freeways within interchange_km are linked both ways. The data does
    not say which ramps connect which directions, so these links carry no up/downstream information.
    """
    n = len(sensors)
    dist = np.full((n, n), np.inf, dtype=np.float32)
    pm = sensors["Abs PM"].to_numpy() * MILE_KM
    for (_, direction), g in sensors.groupby(["Fwy", "Direction"]):
        idx = g.index.to_numpy()
        sign = 1 if direction in ("N", "E") else -1
        travel = sign * (pm[idx][None, :] - pm[idx][:, None])   # travel[a, b]: km driven from a to b
        ok = (travel >= 0) & (travel <= max_link_km)
        dist[np.ix_(idx, idx)] = np.where(ok, travel, np.inf)

    lat, lng = np.radians(sensors.Lat.to_numpy(float)), np.radians(sensors.Lng.to_numpy(float))
    a = (np.sin((lat[:, None] - lat[None]) / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin((lng[:, None] - lng[None]) / 2) ** 2)
    straight = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    fwy = sensors.Fwy.to_numpy()
    near = (straight <= interchange_km) & (fwy[:, None] != fwy[None])
    dist[near] = np.minimum(dist[near], straight[near])
    return dist


def build_adjacency(dist, threshold):
    """Gaussian kernel of driving distance, as in DCRNN's gen_adj_mx.py."""
    std = dist[~np.isinf(dist)].std()
    adj = np.exp(-np.square(dist / std))
    adj[adj < threshold] = 0
    return adj.astype(np.float32)


# Step 4: time-of-day (and optionally weekday) features
def time_features(num_steps, start, freq_minutes, day_of_week):
    stamps = pd.date_range(start, periods=num_steps, freq=f"{freq_minutes}min")
    tod = ((stamps.hour * 60 + stamps.minute) / 1440.0).to_numpy(np.float32)
    feats = [tod[:, None]]
    if day_of_week:
        feats.append(np.eye(7, dtype=np.float32)[stamps.dayofweek])
    return np.concatenate(feats, 1), stamps


# Steps 5 and 6: windows and time-ordered splits
def split_samples(num_steps, seq_len, horizon, train_frac, val_frac):
    """Sample t uses readings t-seq_len+1..t as input and t+1..t+horizon as target. Same split as DCRNN."""
    t = np.arange(seq_len - 1, num_steps - horizon)
    n_test = round(len(t) * (1 - train_frac - val_frac))
    n_train = round(len(t) * train_frac)
    n_val = len(t) - n_train - n_test
    return t[:n_train], t[n_train:n_train + n_val], t[n_train + n_val:]


# Step 7: common scale, fitted on the training period only
def fit_scaler(speed, train_t, seq_len):
    """Mean/std of every reading the training inputs see. Zeros are included, as in DCRNN."""
    train_speed = speed[train_t[0] - seq_len + 1: train_t[-1] + 1]
    return float(train_speed.mean()), float(train_speed.std())


# Typical speed and unexplained slowdowns (the old 30% rule, kept to compare with recorded incidents)
def slot_of_week(stamps, freq_minutes):
    return (stamps.dayofweek * 1440 + stamps.hour * 60 + stamps.minute).to_numpy() // freq_minutes


def typical_speed(speed, slots, train_end):
    """Median speed per sensor and 5-minute slot of the week, from the training period only."""
    df = pd.DataFrame(np.where(speed[:train_end] > 0, speed[:train_end], np.nan))
    typ = df.groupby(slots[:train_end]).median()
    typ = typ.reindex(range(slots.max() + 1))
    typ = typ.fillna(typ.median())
    return typ.to_numpy(np.float32)


def nonrecurring_slowdowns(speed, typical, slots, drop, min_steps):
    """True where speed is more than `drop` below typical for at least `min_steps` readings in a row.

    These are unusual slowdowns of any cause (crash, closure, event, weather, sensor issue); they are NOT
    closure labels. prepare_data.py reports how many of them have a recorded incident nearby.
    """
    low = (speed > 0) & (speed < (1 - drop) * typical[slots])
    c = np.concatenate([np.zeros((1, low.shape[1]), np.int32), np.cumsum(low, 0, dtype=np.int32)])
    full = (c[min_steps:] - c[:-min_steps]) == min_steps  # full[s]: low[s:s+min_steps] all True
    out = np.zeros_like(low)
    for k in range(min_steps):
        out[k:k + len(full)] |= full
    return out


# Loading processed data and making batches
class TrafficData:
    def __init__(self, path, device):
        z = np.load(path)
        self.device = device
        self.features = torch.from_numpy(z["features"]).to(device)       # (T, N, F) scaled
        self.speed = torch.from_numpy(z["speed"]).to(device)             # (T, N) raw mph
        self.incident = torch.from_numpy(z["incident"]).to(device)       # (T, N) bool, recorded incidents
        self.slowdown = torch.from_numpy(z["slowdown"]).to(device)       # (T, N) bool, unexplained drops
        self.typical = torch.from_numpy(z["typical"]).to(device)         # (slots of week, N) median speed
        self.slots = torch.from_numpy(z["slots"]).to(device)             # (T,) slot of week of each step
        self.adj = z["adj"]
        self.mean, self.std = float(z["mean"]), float(z["std"])
        self.seq_len, self.horizon = int(z["seq_len"]), int(z["horizon"])
        self.splits = {"train": z["train_t"], "val": z["val_t"], "test": z["test_t"]}
        self.x_off = torch.arange(-self.seq_len + 1, 1, device=device)
        self.y_off = torch.arange(1, self.horizon + 1, device=device)

    @property
    def num_nodes(self):
        return self.features.shape[1]

    @property
    def input_dim(self):
        return self.features.shape[2]

    def batches(self, split, batch_size, shuffle=False, rng=None):
        """Yields x (B, seq_len, N, F) scaled, y (B, horizon, N, 1) raw mph, and the target time indices."""
        t = self.splits[split]
        if shuffle:
            t = rng.permutation(t)
        for i in range(0, len(t), batch_size):
            tb = torch.as_tensor(t[i:i + batch_size], device=self.device)
            x = self.features[tb[:, None] + self.x_off]
            y_idx = tb[:, None] + self.y_off
            yield x, self.speed[y_idx].unsqueeze(-1), y_idx

    def scale(self, y):
        return (y - self.mean) / self.std

    def unscale(self, y):
        return y * self.std + self.mean
