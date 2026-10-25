"""Quick checks that need no data download:  python -m pytest tests"""
import numpy as np
import torch

import pandas as pd

from config import ModelConfig
from tgc.closures import isolate, neighbours
from tgc.data import nonrecurring_slowdowns, road_distances
from tgc.incidents import categorize, direction_on, on_ramp
from tgc.model import DCRNN, build_supports


def chain_adj(n=5):
    """0 -> 1 -> 2 -> 3 -> 4, with self-loops."""
    adj = np.eye(n, dtype=np.float32)
    for i in range(n - 1):
        adj[i, i + 1] = 0.5
    return adj


def test_forward_supports_follow_traffic():
    S_fwd, S_bwd = build_supports(chain_adj())
    x = torch.zeros(5, 1)
    x[1] = 1.0
    assert (S_fwd @ x)[2, 0] > 0 and (S_fwd @ x)[0, 0] == 0   # forward: 1 reaches downstream 2
    assert (S_bwd @ x)[0, 0] > 0 and (S_bwd @ x)[2, 0] == 0   # backward: 1 reaches upstream 0


def test_output_shape():
    cfg = ModelConfig(num_nodes=5, input_dim=2, rnn_units=8, horizon=3)
    out = DCRNN(cfg, chain_adj()).eval()(torch.randn(4, 6, 5, 2))
    assert out.shape == (4, 3, 5, 1)


def test_isolated_node_does_not_affect_others():
    """Cut every edge of sensor 2: changing its inputs must leave all other forecasts unchanged."""
    torch.manual_seed(0)
    cfg = ModelConfig(num_nodes=5, input_dim=2, rnn_units=8, horizon=3)
    model = DCRNN(cfg, chain_adj()).eval()
    adj = chain_adj()
    adj[2, :] = adj[:, 2] = 0
    adj[2, 2] = 1
    S = build_supports(adj)
    x = torch.randn(1, 6, 5, 2)
    x2 = x.clone()
    x2[:, :, 2, 0] -= 3.0
    with torch.no_grad():
        diff = (model(x, supports=S) - model(x2, supports=S)).abs()[0, :, :, 0]
    assert diff[:, 2].max() > 0
    assert torch.allclose(diff[:, [0, 1, 3, 4]], torch.zeros(3, 4))


def test_slowdowns_need_to_be_sustained():
    typical = np.full((1, 1), 60, np.float32)
    speed = np.array([60, 30, 30, 30, 60, 30, 30, 30, 30, 60], np.float32)[:, None]
    slots = np.zeros(len(speed), int)
    out = nonrecurring_slowdowns(speed, typical, slots, drop=0.3, min_steps=4)[:, 0]
    assert out.tolist() == [False] * 5 + [True] * 4 + [False]


def test_road_direction_follows_postmiles():
    """Eastbound: low postmile is upstream. Westbound: high postmile is upstream. No link across directions."""
    sensors = pd.DataFrame({
        "Fwy": [10, 10, 10, 10], "Direction": ["E", "E", "W", "W"], "Abs PM": [1.0, 2.0, 1.0, 2.0],
        "Lat": [34.0, 34.0, 34.001, 34.001], "Lng": [-117.0, -116.98, -117.0, -116.98]})
    dist = road_distances(sensors, max_link_km=10, interchange_km=0.5)
    assert np.isclose(dist[0, 1], 1.609, atol=0.01) and np.isinf(dist[1, 0])   # E: 0 -> 1
    assert np.isclose(dist[3, 2], 1.609, atol=0.01) and np.isinf(dist[2, 3])   # W: 3 -> 2
    assert np.isinf(dist[0, 2]) and np.isinf(dist[2, 0])                        # opposite carriageways


def test_closure_isolate_and_neighbours():
    adj = chain_adj()                                   # 0 -> 1 -> 2 -> 3 -> 4
    cut = isolate(adj, 2)
    assert cut[2].sum() == cut[:, 2].sum() == 1 and cut[1, 2] == 0 and cut[0, 1] == adj[0, 1]
    dist = np.full((5, 5), np.inf)
    for i in range(5):
        for j in range(i, 5):
            dist[i, j] = j - i                          # km driven from i to j (downstream only)
    up, down = neighbours(dist, 2, max_km=1.5)
    assert up.tolist() == [1] and down.tolist() == [3]


def test_incident_parsing():
    desc = pd.Series(["CLOSURE of a Road", "SIG Alert", "1183-Trfc Collision-Unkn Inj", "1125-Traffic Hazard",
                      "FIRE-Report of Fire"])
    assert categorize(desc).tolist() == ["closure", "sigalert", "collision", "hazard", "other"]
    assert direction_on(10, "I10 E / Vineyard Ave") == "E"
    assert direction_on(91, "Tyler St Onr / Sr91 E") == "E"
    assert direction_on(10, "I105 W / Main St") is None
    assert on_ramp(pd.Series(["Tyler St Onr / Sr91 E", "I880 N / Davis St Onr"])).tolist() == [True, False]
