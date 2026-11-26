"""Graph gating: switch off what a sensor hears from its neighbours while an incident is active there.

Tests the explanation from the closure failure analysis: during a closure, nearby sensors that still look
normal pull the forecast toward recovery. If that is right, gating the closure sensor's incoming graph
messages should remove most of the DCRNN's "too fast" bias (and more so when the neighbours look normal).

The gate uses only what is known when the forecast is made: incidents active at the last input step.
"""
import torch

UPSTREAM, DOWNSTREAM = 0, 1          # index of the support that brings in upstream / downstream sensors


def incident_gate(active, y_idx, g=0.0, directions=(UPSTREAM, DOWNSTREAM), num_supports=2):
    """(B, num_supports, N) gate: g at sensors with an incident active at the forecast origin, 1 elsewhere."""
    on = active[y_idx[:, 0] - 1]                                  # (B, N)
    gate = torch.ones(on.shape[0], num_supports, on.shape[1], device=on.device)
    for s in directions:
        gate[:, s] = torch.where(on, torch.full_like(gate[:, s], g), gate[:, s])
    return gate


class Gated(torch.nn.Module):
    """A trained DCRNN that is run with incident_gate at inference. `active` (T, N) bool;
    pass active = all-True to gate every sensor (placebo / no-graph-at-inference check)."""

    def __init__(self, model, active, g=0.0, directions=(UPSTREAM, DOWNSTREAM)):
        super().__init__()
        self.model, self.active, self.g, self.directions = model, active, g, directions

    def predict(self, x, y_idx, data):
        gate = incident_gate(self.active, y_idx, self.g, self.directions, len(self.model.supports))
        return data.unscale(self.model(x, gate=gate))[..., 0]
