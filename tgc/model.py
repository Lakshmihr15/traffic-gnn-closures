"""Step 8: DCRNN (Li et al., ICLR 2018) in plain PyTorch.

The road graph enters only through `supports`, so a closure can later be applied by passing
modified supports to `forward` without retraining.

Graph gate (optional `gate` in `forward`, shape (B, num_supports, N), values in [0, 1]): each support S is
replaced, per sample and sensor, by  g * S + (1 - g) * I.  g = 1 is the normal graph; g = 0 means that sensor
"sees itself" in place of its neighbours, so no information from other sensors reaches it (in any layer or
diffusion step), while its inputs keep their usual range. Support 0 brings in upstream sensors, support 1
downstream ones, so the two directions can be gated separately.
"""
import random

import numpy as np
import torch
from torch import nn


def random_walk(adj):
    """Row-normalised adjacency D^-1 A. Rows with no edges stay zero."""
    deg = adj.sum(1)
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return inv[:, None] * adj


def build_supports(adj):
    """Dual random walk, as in DCRNN.

    (S_fwd @ x)[j] averages x over sensors upstream of j  -> forward diffusion (with traffic)
    (S_bwd @ x)[j] averages x over sensors downstream of j -> backward diffusion (against traffic)
    """
    adj = np.asarray(adj, dtype=np.float32)
    return torch.from_numpy(np.stack([random_walk(adj).T, random_walk(adj.T).T]))


class DiffusionConv(nn.Module):
    """Mixes each sensor's features with its K-hop upstream and downstream neighbours."""

    def __init__(self, in_dim, out_dim, num_supports, K, bias_start=0.0):
        super().__init__()
        self.K = K
        n_mats = 1 + num_supports * K
        self.weight = nn.Parameter(torch.empty(in_dim * n_mats, out_dim))
        self.bias = nn.Parameter(torch.full((out_dim,), bias_start))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x, supports, gate=None):  # x (B, N, F), supports (S, N, N), gate (B, S, N) or None
        terms = [x]
        for s, S in enumerate(supports):
            g = None if gate is None else gate[:, s, :, None]

            def diffuse(v):
                out = torch.einsum("nm,bmf->bnf", S, v)
                return out if g is None else g * out + (1 - g) * v

            prev, cur = x, diffuse(x)
            terms.append(cur)
            for _ in range(2, self.K + 1):
                prev, cur = cur, 2 * diffuse(cur) - prev
                terms.append(cur)
        return torch.cat(terms, -1) @ self.weight + self.bias


class DCGRUCell(nn.Module):
    """GRU cell whose matrix multiplications are replaced by diffusion convolutions."""

    def __init__(self, in_dim, units, num_supports, K):
        super().__init__()
        self.gates = DiffusionConv(in_dim + units, 2 * units, num_supports, K, bias_start=1.0)
        self.candidate = DiffusionConv(in_dim + units, units, num_supports, K)

    def forward(self, x, h, supports, gate=None):
        r, u = torch.sigmoid(self.gates(torch.cat([x, h], -1), supports, gate)).chunk(2, -1)
        c = torch.tanh(self.candidate(torch.cat([x, r * h], -1), supports, gate))
        return u * h + (1 - u) * c


class DCRNN(nn.Module):
    """Encoder-decoder of stacked DCGRU cells. Input (B, T_in, N, F) scaled, output (B, horizon, N, 1) scaled."""

    def __init__(self, cfg, adj):
        super().__init__()
        self.cfg = cfg
        if not cfg.use_graph:
            adj = np.zeros_like(adj)   # ablation: diffusion terms vanish, only each sensor's own term is left
        self.register_buffer("supports", build_supports(adj))
        S, K, U = len(self.supports), cfg.max_diffusion_step, cfg.rnn_units
        self.encoder = nn.ModuleList(
            DCGRUCell(cfg.input_dim if l == 0 else U, U, S, K) for l in range(cfg.num_layers))
        self.decoder = nn.ModuleList(
            DCGRUCell(cfg.output_dim if l == 0 else U, U, S, K) for l in range(cfg.num_layers))
        self.project = nn.Linear(U, cfg.output_dim)

    def forward(self, x, labels=None, teacher_prob=0.0, supports=None, gate=None):
        S = self.supports if supports is None else supports
        B, T_in, N, _ = x.shape
        h = [x.new_zeros(B, N, self.cfg.rnn_units) for _ in range(self.cfg.num_layers)]

        for t in range(T_in):
            inp = x[:, t]
            for l, cell in enumerate(self.encoder):
                h[l] = inp = cell(inp, h[l], S, gate)

        dec_in = x.new_zeros(B, N, self.cfg.output_dim)  # "GO" symbol
        outputs = []
        for t in range(self.cfg.horizon):
            inp = dec_in
            for l, cell in enumerate(self.decoder):
                h[l] = inp = cell(inp, h[l], S, gate)
            out = self.project(inp)
            outputs.append(out)
            # Scheduled sampling: early in training the decoder sometimes sees the true previous value
            use_truth = self.training and labels is not None and random.random() < teacher_prob
            dec_in = labels[:, t] if use_truth else out
        return torch.stack(outputs, 1)
