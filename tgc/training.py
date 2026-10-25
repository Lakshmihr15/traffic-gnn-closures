"""Shared pieces for Steps 9 and 10: masked metrics, one training epoch, evaluation, checkpoints."""
import math
from dataclasses import asdict

import numpy as np
import torch

from config import ModelConfig
from tgc.model import DCRNN


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def masked_mae(pred, true, weight=None):
    """MAE over readings that are not missing (true == 0 means missing), optionally weighted."""
    mask = (true != 0).float()
    if weight is not None:
        mask = mask * weight
    return (torch.abs(pred - true) * mask).sum() / mask.sum().clamp(min=1)


def teacher_prob(batches_seen, decay_steps):
    return decay_steps / (decay_steps + math.exp(min(batches_seen / decay_steps, 50)))


def train_epoch(model, data, opt, tcfg, rng, batches_seen, weight_fn=None, supports_fn=None, max_batches=None):
    """One pass over the training windows. weight_fn and supports_fn are the Step 10 hooks."""
    model.train()
    losses = []
    for b, (x, y, y_idx) in enumerate(data.batches("train", tcfg.batch_size, shuffle=True, rng=rng)):
        if max_batches and b >= max_batches:
            break
        p = teacher_prob(batches_seen, tcfg.cl_decay_steps)
        supports = supports_fn() if supports_fn else None
        pred = data.unscale(model(x, data.scale(y), p, supports))
        weight = weight_fn(y_idx).unsqueeze(-1) if weight_fn else None
        loss = masked_mae(pred, y, weight)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)
        opt.step()
        losses.append(loss.item())
        batches_seen += 1
    return float(np.mean(losses)), batches_seen


@torch.no_grad()
def evaluate(model, data, split, batch_size=64, horizons=(1, 3, 6, 12), max_batches=None):
    """MAE / RMSE / MAPE, missing readings masked, overall and at each horizon (1 = 5 min, 3 = 15 min).

    Also reports MAE during recorded incidents, during unexplained slowdowns, and in normal traffic (neither).
    """
    model.eval()
    sums = {}

    def add(key, err, mask):
        s = sums.setdefault(key, np.zeros(2))
        s += (float((err * mask).sum()), float(mask.sum()))

    for b, (x, y, y_idx) in enumerate(data.batches(split, batch_size)):
        if max_batches and b >= max_batches:
            break
        # Baselines (baselines.py) predict mph directly and may need the target times
        pred = model.predict(x, y_idx, data) if hasattr(model, "predict") else data.unscale(model(x))[..., 0]
        y = y[..., 0]
        valid = (y != 0).float()
        ae, se = (pred - y).abs(), (pred - y) ** 2
        ape = ae / y.abs().clamp(min=1e-6)
        inc, slow = data.incident[y_idx].float(), data.slowdown[y_idx].float()
        normal = (1 - inc) * (1 - slow)
        for name, m in [("all", valid), ("incident", valid * inc), ("slowdown", valid * slow),
                        ("normal", valid * normal)]:
            add(f"mae_{name}", ae, m)
        add("rmse", se, valid)
        add("mape", ape, valid)
        for h in horizons:
            v = valid[:, h - 1]
            add(f"mae@{h}", ae[:, h - 1], v)
            add(f"rmse@{h}", se[:, h - 1], v)
            add(f"mape@{h}", ape[:, h - 1], v)

    out = {k: s[0] / max(s[1], 1) for k, s in sums.items()}
    for k in list(out):
        if k.startswith("rmse"):
            out[k] = math.sqrt(out[k])
        elif k.startswith("mape"):
            out[k] *= 100
    return out


def save_checkpoint(path, model, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_cfg": asdict(model.cfg), **extra}, path)


def load_model(path, data, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = DCRNN(ModelConfig(**ckpt["model_cfg"]), data.adj).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def format_metrics(m):
    keys = ["mae_all", "mae@1", "mae@3", "mae@6", "mae@12", "rmse@12", "mape@12",
            "mae_normal", "mae_incident", "mae_slowdown"]
    return "  ".join(f"{k} {m[k]:.3f}" for k in keys if k in m)
