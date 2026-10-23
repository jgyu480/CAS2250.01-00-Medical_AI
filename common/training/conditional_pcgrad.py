"""Modulewise partial PCGrad with static or batch-conditioned meta gates.

Both task gradients are anchored to the ORIGINAL opposing task gradient.
This is modulewise PCGrad, not a claim to reproduce global-vector PCGrad.
Raw gradients/statistics are detached; only a one-step gate hypergradient is used.
"""
import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.func import functional_call


GROUPS = {
    "cell_encoder": "cell.encoder.",
    "tissue_encoder": "tissue.encoder.",
    "encoder_c2t": "encoder_c2t.",
    "tissue_aspp": "tissue.aspp.",
    "tissue_decoder": "tissue.decoder.",
}


def group_for(name):
    matches = [k for k, prefix in GROUPS.items() if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Shared tensor has no unique PCGrad group: {name}")
    return matches[0]


class PCGradGates(nn.Module):
    def __init__(self, mode="conditional", initial=0.1, ema_decay=0.95):
        super().__init__()
        if mode not in ("static", "conditional") or not 0 < initial < 1:
            raise ValueError("Invalid gate mode/initial ratio")
        if not 0 <= ema_decay < 1:
            raise ValueError("Invalid EMA decay")
        self.mode, self.ema_decay = mode, float(ema_decay)
        # Columns: intercept, current negative cosine, log norm ratio, previous EMA.
        columns = 1 if mode == "static" else 4
        coef = torch.zeros(len(GROUPS), 2, columns)
        coef[:, :, 0] = math.log(initial / (1 - initial))
        self.coefficients = nn.Parameter(coef)
        self.register_buffer("ema", torch.zeros(len(GROUPS)))
        self.register_buffer("ema_count", torch.zeros(len(GROUPS), dtype=torch.long))

    @torch.no_grad()
    def observe(self, raw):
        accum = {}
        for name, (c, t) in raw.items():
            if c is None or t is None:
                continue
            key = group_for(name)
            # FP64 only for small reductions is unnecessary; raw vectors stay FP32.
            cf, tf = c.detach().float(), t.detach().float()
            vec = torch.stack(((cf * tf).sum(), cf.square().sum(), tf.square().sum()))
            if key not in accum:
                accum[key] = [vec, c.numel(), 1]
            else:
                accum[key][0].add_(vec)
                accum[key][1] += c.numel()
                accum[key][2] += 1
        result = {}
        for i, key in enumerate(GROUPS):
            if key not in accum:
                continue
            v, count, tensors = accum[key]
            if not bool(torch.isfinite(v).all()):
                raise FloatingPointError(f"Non-finite gradients: {key}")
            dot, c2, t2 = v.unbind()
            valid = bool((c2 > 1e-24) & (t2 > 1e-24))
            cosine = (dot / (c2.sqrt() * t2.sqrt()).clamp_min(1e-30)).clamp(-1, 1) if valid else dot * 0
            severity = (-cosine).clamp_min(0)
            ratio = (0.5 * ((t2 + 1e-24).log() - (c2 + 1e-24).log())).clamp(-8, 8)
            one = torch.ones_like(dot)
            old_ema = self.ema[i].to(dot)
            features = torch.stack((
                torch.stack((one, severity, ratio, old_ema)),
                torch.stack((one, severity, -ratio, old_ema)),
            )).to(self.coefficients).detach()
            result[key] = dict(
                index=i, dot=dot.detach(), c2=c2.detach(), t2=t2.detach(),
                cosine=cosine.detach(), valid=valid,
                conflict=valid and bool(dot < 0), severity=severity.detach(),
                features=features, shared_parameters=count, shared_tensors=tensors,
            )
        return result

    def ratios(self, observation):
        return {
            key: (self.coefficients[v["index"]] *
                  v["features"][:, :self.coefficients.shape[-1]]).sum(-1).sigmoid()
            for key, v in observation.items()
        }

    def mix(self, raw, observation, ratios=None):
        ratios = self.ratios(observation) if ratios is None else ratios
        mixed = {}
        for name, (c, t) in raw.items():
            if c is None or t is None:
                mixed[name] = t if c is None else c
                continue
            key = group_for(name)
            o = observation[key]
            if not o["conflict"]:
                mixed[name] = c + t
                continue
            rc, rt = ratios[key].unbind()
            # These equivalent coefficients avoid allocating two projected copies.
            # c' = c - rc*dot/t2*t; t' = t - rt*dot/c2*c.
            mixed[name] = ((1 - rt * o["dot"] / o["c2"].clamp_min(1e-24)) * c
                           + (1 - rc * o["dot"] / o["t2"].clamp_min(1e-24)) * t)
        return mixed

    @torch.no_grad()
    def advance(self, observation):
        for o in observation.values():
            if not o["valid"]:
                continue
            i = o["index"]
            self.ema[i].mul_(self.ema_decay).add_(o["severity"].to(self.ema), alpha=1-self.ema_decay)
            self.ema_count[i].add_(1)


def clip_factor(mixed, max_norm):
    gradients = [g for g in mixed.values() if g is not None]
    if not gradients or not max_norm:
        return 1.0
    # Preserve FP64 in finite-difference tests; real model parameters are FP32.
    norm = torch.stack([g.norm(2) for g in gradients]).norm(2)
    return (float(max_norm) / (norm + 1e-6)).clamp(max=1.0)


@torch.no_grad()
def assign_mixed(model, mixed):
    for name, p in model.named_parameters():
        g = mixed.get(name)
        p.grad = None if not p.requires_grad or g is None else g.detach().to(p.dtype)


def _safe_sqrt(v):
    # sqrt'(0) would produce NaN in the hypergradient even for a zero gradient.
    positive = v > 0
    return torch.where(positive, torch.where(positive, v, torch.ones_like(v)).sqrt(),
                       torch.zeros_like(v))


def virtual_adamw_parameters(model, optimizer, mixed, max_norm=None):
    """Differentiable next AdamW parameters without mutation of real state.

    Matches ordinary AdamW (no AMSGrad, maximize, capturable, or fused variants).
    Momentum, variance, step counter, bias correction, decay and global clipping
    are included. Only the dependence of the present step on gates is retained.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("Expected AdamW")
    factor = clip_factor(mixed, max_norm)
    named = dict(model.named_parameters())
    names = {id(p): n for n, p in named.items()}
    virtual = {n: p.detach() for n, p in named.items()}
    for group in optimizer.param_groups:
        if any(group.get(flag, False) for flag in ("amsgrad", "maximize", "capturable", "fused")):
            raise ValueError("Unsupported AdamW variant")
        b1, b2 = group["betas"]
        lr, wd, eps = float(group["lr"]), float(group["weight_decay"]), group["eps"]
        for p in group["params"]:
            name = names[id(p)]
            grad = mixed.get(name)
            if grad is None or not p.requires_grad:
                continue
            g = grad * factor
            state = optimizer.state.get(p, {})
            step = int(state["step"].item()) + 1 if state else 1
            old_m = state["exp_avg"].detach() if state else torch.zeros_like(p)
            old_v = state["exp_avg_sq"].detach() if state else torch.zeros_like(p)
            m = b1 * old_m + (1-b1) * g
            v = b2 * old_v + (1-b2) * g.square()
            denom = _safe_sqrt(v) / math.sqrt(1-b2**step) + eps
            virtual[name] = p.detach() * (1-lr*wd) - (lr/(1-b1**step)) * m / denom
    return virtual


@contextmanager
def evaluating(model):
    modes = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, mode in modes:
            module.training = mode


def meta_update(model, optimizer, gates, gate_optimizer, raw, observation,
                args, loss_fn, scales, max_norm):
    mixed = gates.mix(raw, observation)
    # A non-conflicting batch has no gate-dependent update and no learning signal.
    if not any(g is not None and g.requires_grad for g in mixed.values()):
        return dict(meta_update=0, meta_objective="", meta_cell="", meta_tissue="",
                    gate_grad_norm=0.0)
    virtual = virtual_adamw_parameters(model, optimizer, mixed, max_norm)
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    with evaluating(model):
        output = functional_call(model, (virtual, buffers), args, strict=True)
        losses = loss_fn(output)
    objective = 0.5 * (losses["cell"] / scales[0] + losses["tissue"] / scales[1])
    if not bool(torch.isfinite(objective)):
        raise FloatingPointError("Non-finite meta objective")
    derivatives = torch.autograd.grad(objective, tuple(gates.parameters()), allow_unused=True)
    gate_optimizer.zero_grad(set_to_none=True)
    for p, g in zip(gates.parameters(), derivatives):
        p.grad = torch.zeros_like(p) if g is None else g.detach()
    norm = torch.nn.utils.clip_grad_norm_(gates.parameters(), 1.0, error_if_nonfinite=True)
    info = dict(meta_update=1, meta_objective=objective.item(),
                meta_cell=losses["cell"].item(), meta_tissue=losses["tissue"].item(),
                gate_grad_norm=float(norm))
    gate_optimizer.step()
    gate_optimizer.zero_grad(set_to_none=True)
    if not all(bool(torch.isfinite(p).all()) for p in gates.parameters()):
        raise FloatingPointError("Non-finite gate coefficients")
    return info
