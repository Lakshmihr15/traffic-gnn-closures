"""All paths and settings in one place. Model and training defaults follow the DCRNN paper (Li et al., ICLR 2018)."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "traffident"   # TraffiDent files from Kaggle (gpxlcj/xtraffic)
PROCESSED = ROOT / "data" / "processed" / "traffident.npz"
EVENTS = ROOT / "data" / "processed" / "events.csv"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"

# Monthly traffic files pMM_done.npy have shape (time, 16972 sensors, 3); sensors in node_order.npy order
CHANNELS = {"flow": 0, "occupancy": 1, "speed": 2}


@dataclass
class DataConfig:
    # Which slice of TraffiDent
    year: int = 2023
    months: tuple = (1, 2, 3, 4, 5, 6)   # consecutive; 6 months for enough held-out closures
    district: int = 8                    # Caltrans District 8: San Bernardino + Riverside
    county: str = ""                     # e.g. "San Bernardino" for a smaller network; "" = whole district
    target: str = "speed"                # "flow" reproduces the paper's DCRNN benchmark
    freq_minutes: int = 5
    # Windows and splits (the paper's 6:2:2)
    seq_len: int = 12                    # 1 hour in
    horizon: int = 12                    # 1 hour out
    train_frac: float = 0.6
    val_frac: float = 0.2
    # Road graph (Step 3)
    max_link_km: float = 10.0            # only link sensors this close along the road
    interchange_km: float = 0.5          # link sensors on different freeways this close (straight line)
    adj_threshold: float = 0.1           # drop graph edges weaker than this (DCRNN default)
    day_of_week: bool = False            # DCRNN uses time of day only; one-hot weekday is optional
    bad_sensor_missing_rate: float = 0.3 # report sensors missing more than this share of readings
    # Incidents (Phase 3)
    incident_match_miles: float = 0.5    # nearest sensor must be this close (the paper's dis_threshold)
    incident_kinds: tuple = ("closure",)  # CHP "CLOSURE of a Road" records are the labels we study
    # Unexplained slowdowns, kept only to compare with recorded incidents
    slowdown_drop: float = 0.3           # speed more than 30% below typical for that time of week
    slowdown_min_steps: int = 4          # sustained for at least 20 minutes


@dataclass
class ModelConfig:
    num_nodes: int = 207
    input_dim: int = 2                   # target channel + time of day (set from the data at run time)
    output_dim: int = 1
    rnn_units: int = 64
    num_layers: int = 2
    max_diffusion_step: int = 2          # K
    horizon: int = 12


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 100
    base_lr: float = 0.01
    epsilon: float = 1e-3
    lr_milestones: tuple = (20, 30, 40, 50)
    lr_decay: float = 0.1
    max_grad_norm: float = 5.0
    patience: int = 50
    cl_decay_steps: int = 2000           # scheduled sampling: how fast the decoder stops seeing ground truth
    seed: int = 1
