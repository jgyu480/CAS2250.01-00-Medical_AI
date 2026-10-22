"""Two branch-specific gradient gates, inspired by Auto-Lambda (TMLR 2022).

Reference: https://shikun.io/projects/auto-lambda
This is an experimental adaptation, not an implementation of Auto-Lambda.
At a_cell=a_tissue=.5, shared gradients equal gc+gt exactly. Private
parameters retain their single-task gradient. gt includes the tissue weight.
Gate learning differentiates each task's held-out loss through a one-step
SGD lookahead; the real model optimizer can remain AdamW. This surrogate
does not differentiate Adam moments or the full training trajectory.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.func import functional_call


def owner(name):
    """Exchange parameters belong to the receiving branch; no overlapping scopes."""
    if name.startswith(("cell.", "aspp_t2c.", "decoder_t2c.", "context_t2c.")):
        return 0
    if name.startswith(("tissue.", "encoder_c2t.")):
        return 1
    raise ValueError(f"Unassigned parameter: {name}")


class BranchGates(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))

    def forward(self):
        return self.logits.sigmoid()

    def values(self):
        a, b = self().detach().cpu().tolist()
        return {"cell_other_ratio": a, "tissue_other_ratio": b}

    def mix(self, name, cell, tissue):
        if cell is None:
            return tissue
        if tissue is None:
            return cell
        branch = owner(name)
        a = self()[branch]
        own, other = (cell, tissue) if branch == 0 else (tissue, cell)
        return 2 * (1 - a) * own + 2 * a * other


def task_gradients(model, cell_loss, weighted_tissue_loss):
    """Return detached unscaled gradients, without writing any parameter .grad."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    cell = torch.autograd.grad(cell_loss, params, retain_graph=True, allow_unused=True)
    tissue = torch.autograd.grad(weighted_tissue_loss, params, allow_unused=True)
    return {n: (c.detach() if c is not None else None,
                t.detach() if t is not None else None)
            for (n, _), c, t in zip(named, cell, tissue)}


@torch.no_grad()
def assign_gradients(model, gates, gradients, scale=1.0):
    """Write mixed gradients. Caller handles GradScaler, clipping and AdamW."""
    for name, p in model.named_parameters():
        pair = gradients.get(name)
        if not p.requires_grad or pair is None:
            p.grad = None
            continue
        mixed = gates.mix(name, *pair)
        p.grad = None if mixed is None else (mixed * scale).to(p.dtype)


@contextmanager
def _evaluation(model):
    modes = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, mode in modes:
            module.training = mode


def lookahead_losses(model, gates, gradients, learning_rates, args, loss_fn,
                     grad_clip_norm=None):
    """loss_fn(output) -> {'cell': scalar, 'tissue': scalar} on meta data.

    No real parameter, .grad, optimizer state, buffer or train/eval mode changes.
    Independent detached base parameters avoid second-order model derivatives.
    Gates remain connected through virtual parameters to held-out losses.
    Meta evaluation disables dropout; buffers are cloned defensively.
    """
    mixed = {n: gates.mix(n, *pair) for n, pair in gradients.items()}
    factor = 1.0
    if grad_clip_norm is not None and grad_clip_norm > 0:
        norms = [g.float().norm() for g in mixed.values() if g is not None]
        if norms:
            norm = torch.stack(norms).norm()
            factor = (grad_clip_norm / (norm + 1e-6)).clamp(max=1.0)
    virtual = {}
    for name, p in model.named_parameters():
        g = mixed.get(name)
        virtual[name] = p.detach()
        if g is not None:
            lr = float(learning_rates[name])
            if not math.isfinite(lr) or lr < 0:
                raise ValueError("Invalid lookahead learning rate")
            virtual[name] = p.detach() - lr * factor * g
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    with _evaluation(model):
        result = functional_call(model, (virtual, buffers), args, strict=True)
        losses = loss_fn(result)
    return losses


def gate_derivatives(gates, losses):
    """Gate C minimizes cell loss; gate T minimizes tissue loss (diagonal only)."""
    partials = []
    for key in ("cell", "tissue"):
        loss = losses[key]
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError(f"Non-finite meta loss: {key}")
        derivative = (torch.autograd.grad(
            loss, gates.logits, retain_graph=True, allow_unused=True
        )[0] if loss.requires_grad else None)
        partials.append(torch.zeros_like(gates.logits) if derivative is None
                        else derivative.detach())
    grad = torch.stack((partials[0][0], partials[1][1]))
    if not bool(torch.isfinite(grad).all()):
        raise FloatingPointError("Non-finite gate derivative")
    return grad


def update_gates(gates, optimizer, losses, max_grad_norm=1.0):
    grad = gate_derivatives(gates, losses)
    optimizer.zero_grad(set_to_none=True)
    gates.logits.grad = grad.clone()
    torch.nn.utils.clip_grad_norm_(gates.parameters(), max_grad_norm,
                                   error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if not bool(torch.isfinite(gates.logits).all()):
        raise FloatingPointError("Non-finite gate parameters")
    return {**gates.values(), "cell_gate_derivative": grad[0].item(),
            "tissue_gate_derivative": grad[1].item()}
