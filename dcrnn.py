"""Minimal PyTorch re-implementation of the original TensorFlow DCRNN (Li et al., ICLR 2018),
written to load the authors' pretrained METR-LA checkpoint exactly.

The road graph enters ONLY through `supports`, so the same weights can be run on a modified
graph (a simulated closure) by building a second model with a different adjacency matrix.
"""
import numpy as np
import torch

K_DIFF = 2          # max_diffusion_step
UNITS = 64
N = 207


def random_walk(adj):
    d = adj.sum(1)
    d_inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0.0)
    return d_inv[:, None] * adj                       # D^-1 A


def build_supports(adj):
    """dual_random_walk filter, as in the TF code: [(D^-1 A)^T, (D^-1 A^T)^T]."""
    s = [random_walk(adj).T, random_walk(adj.T).T]
    return [torch.tensor(m, dtype=torch.float32) for m in s]


class DCGRUCell:
    def __init__(self, w, prefix, supports, proj=False):
        g = lambda name: torch.tensor(w[prefix + name], dtype=torch.float32)
        self.gw, self.gb = g('gates/weights'), g('gates/biases')
        self.cw, self.cb = g('candidate/weights'), g('candidate/biases')
        self.pw = g('projection/w') if proj else None
        self.supports = supports

    def gconv(self, inp, state, W, b):
        B = inp.shape[0]
        x = torch.cat([inp.reshape(B, N, -1), state.reshape(B, N, -1)], dim=2)
        isz = x.shape[2]
        x0 = x.permute(1, 2, 0).reshape(N, isz * B)
        xs = [x0]
        for S in self.supports:                        # reproduces the TF loop exactly,
            x1 = S @ x0                                # including x0 carrying over between supports
            xs.append(x1)
            for _ in range(2, K_DIFF + 1):
                x2 = 2 * (S @ x1) - x0
                xs.append(x2)
                x1, x0 = x2, x1
        m = len(xs)
        x = torch.stack(xs, 0).reshape(m, N, isz, B).permute(3, 1, 2, 0).reshape(B * N, isz * m)
        return (x @ W + b).reshape(B, -1)

    def __call__(self, inp, state):
        B = inp.shape[0]
        v = torch.sigmoid(self.gconv(inp, state, self.gw, self.gb)).reshape(B, N, 2 * UNITS)
        r, u = v[..., :UNITS].reshape(B, -1), v[..., UNITS:].reshape(B, -1)
        c = torch.tanh(self.gconv(inp, r * state, self.cw, self.cb))
        new = u * state + (1 - u) * c
        if self.pw is not None:
            out = (new.reshape(-1, UNITS) @ self.pw).reshape(B, -1)
            return out, new
        return new, new


class DCRNN:
    def __init__(self, weights_npz, adj):
        w = np.load(weights_npz)
        sup = build_supports(adj)
        e, d = 'rnn/multi_rnn_cell/', 'rnn_decoder/multi_rnn_cell/'
        self.enc = [DCGRUCell(w, e + f'cell_{i}/dcgru_cell/', sup) for i in range(2)]
        self.dec = [DCGRUCell(w, d + 'cell_0/dcgru_cell/', sup),
                    DCGRUCell(w, d + 'cell_1/dcgru_cell/', sup, proj=True)]

    @torch.no_grad()
    def __call__(self, x):
        """x: (B, 12, N, 2) scaled speed + time of day. Returns (B, 12, N) scaled speed."""
        B = x.shape[0]
        h = [torch.zeros(B, N * UNITS) for _ in range(2)]
        for t in range(x.shape[1]):
            inp = x[:, t].reshape(B, -1)
            for l, cell in enumerate(self.enc):
                inp, h[l] = cell(inp, h[l])
        inp, outs = torch.zeros(B, N), []
        for t in range(12):
            o = inp
            for l, cell in enumerate(self.dec):
                o, h[l] = cell(o, h[l])
            outs.append(o)
            inp = o                                     # feed prediction back (test mode)
        return torch.stack(outs, 1)
