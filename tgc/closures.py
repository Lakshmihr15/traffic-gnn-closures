"""Closure simulator and spillover measures (E2).

For a recorded closure at sensor c starting at step s, the forecast is made at t0 = s + lead, so the input
hour already holds `lead` closure readings at c. The counterfactual "no closure" input replaces c's readings
from s onward with c's typical speed; every other sensor keeps its real readings. Each variant returns
(input, supports) for the run with the closure; the run without it always uses the full graph.

Slowdowns are in mph, positive = slower:
  predicted  forecast without closure - forecast with closure
  observed   typical speed - actual speed
"""
import numpy as np
import torch

from tgc.model import build_supports

VARIANTS = ("input", "input+graph", "graph")


def isolate(adj, node):
    """Road graph with every link into and out of `node` removed."""
    adj = adj.copy()
    adj[node, :] = adj[:, node] = 0
    adj[node, node] = 1
    return adj


def neighbours(dist_km, node, max_km):
    """Upstream (j -> node) and downstream (node -> j) sensors within max_km along the road. Pairs linked
    both ways are interchange links without a known direction and are left out."""
    up = np.isfinite(dist_km[:, node]) & (dist_km[:, node] <= max_km)
    down = np.isfinite(dist_km[node, :]) & (dist_km[node, :] <= max_km)
    up[node] = down[node] = False
    both = up & down
    return np.flatnonzero(up & ~both), np.flatnonzero(down & ~both)


def make_inputs(data, node, start, t0):
    """Real input window ending at t0 and the same window with `node` reset to typical from `start` on."""
    steps = torch.arange(t0 - data.seq_len + 1, t0 + 1, device=data.device)
    real = data.features[steps].clone()                          # (seq_len, N, F) scaled
    calm = real.clone()
    after = steps >= start
    calm[after, node, 0] = data.scale(data.typical[data.slots[steps[after]], node])
    return real, calm


@torch.no_grad()
def simulate(model, data, node, start, t0, variant):
    """Predicted slowdown (horizon, N) in mph caused by the closure at `node`."""
    real, calm = make_inputs(data, node, start, t0)
    cut = build_supports(isolate(data.adj, node)).to(data.device)
    closed_input = calm if variant == "graph" else real
    closed_supports = None if variant == "input" else cut
    with_closure = data.unscale(model(closed_input[None], supports=closed_supports))[0, ..., 0]
    without = data.unscale(model(calm[None]))[0, ..., 0]
    return (without - with_closure).cpu().numpy()


def observed(data, t0):
    """Observed slowdown (horizon, N) vs typical speed over the forecast hour; NaN where missing."""
    steps = torch.arange(t0 + 1, t0 + data.horizon + 1, device=data.device)
    speed = data.speed[steps]
    slow = data.typical[data.slots[steps]] - speed
    return torch.where(speed > 0, slow, torch.full_like(slow, float("nan"))).cpu().numpy()
