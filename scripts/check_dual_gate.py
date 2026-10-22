"""CPU synthetic checks only: no dataset, SSL weights or GPU training."""
import io
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.training.dual_gate import (
    BranchGates, owner, task_gradients, assign_gradients,
    lookahead_losses, gate_derivatives, update_gates,
)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.Linear(1, 1, bias=False)
        self.tissue = nn.Linear(1, 1, bias=False)
        self.private = nn.Parameter(torch.tensor([0.3]))
        self.register_buffer("counter", torch.tensor(0.0))

    def forward(self, x):
        if self.training:
            self.counter.add_(1)
        c, t = self.cell(x), self.tissue(x)
        return c + 0.4 * t + self.private, t + 0.3 * c


def losses(output):
    c, t = output
    return {"cell": ((c - 1.3) ** 2).mean(),
            "tissue": ((t + 0.8) ** 2).mean()}


def main():
    torch.manual_seed(42)
    model, gates = Toy().double(), BranchGates().double()
    rates = {n: 0.03 for n, _ in model.named_parameters()}
    x = torch.tensor([[1.0]], dtype=torch.double)
    meta_x = torch.tensor([[1.4]], dtype=torch.double)

    a = losses(model(x))
    reference = torch.autograd.grad(
        a["cell"] + a["tissue"], tuple(model.parameters()), retain_graph=True
    )
    raw = task_gradients(model, a["cell"], a["tissue"])
    assign_gradients(model, gates, raw)
    for p, expected in zip(model.parameters(), reference):
        torch.testing.assert_close(p.grad, expected, rtol=0, atol=0)
    print("[PASS] initial gates exactly reproduce the gradient sum")

    with torch.no_grad():
        gates.logits.copy_(torch.tensor([-2.0, 1.0]))
    assign_gradients(model, gates, raw)
    torch.testing.assert_close(model.private.grad, raw["private"][0])

    before = {k: v.clone() for k, v in model.state_dict().items()}
    before_grads = [p.grad.clone() for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    meta = lookahead_losses(model, gates, raw, rates, (meta_x,), losses)
    analytical = gate_derivatives(gates, meta)

    eps = 1e-5
    for i, key in enumerate(("cell", "tissue")):
        original = gates.logits[i].item()
        values = []
        for delta in (eps, -eps):
            with torch.no_grad():
                gates.logits[i] = original + delta
            values.append(
                lookahead_losses(
                    model, gates, raw, rates, (meta_x,), losses
                )[key].item()
            )
        with torch.no_grad():
            gates.logits[i] = original
        numeric = (values[0] - values[1]) / (2 * eps)
        torch.testing.assert_close(
            analytical[i], analytical.new_tensor(numeric),
            rtol=1e-5, atol=1e-8
        )

    assert [m.training for m in model.modules()] == modes
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k
    for p, previous in zip(model.parameters(), before_grads):
        assert torch.equal(p.grad, previous)
    print("[PASS] both gate derivatives match finite differences; real model untouched")

    optimizer = torch.optim.Adam(gates.parameters(), lr=0.01)
    old = gates.logits.detach().clone()
    meta = lookahead_losses(model, gates, raw, rates, (meta_x,), losses)
    update_gates(gates, optimizer, meta)
    assert torch.all((gates.logits.detach() - old) * analytical < 0)
    print("[PASS] each gate moves against its own task meta derivative")

    saved = io.BytesIO()
    torch.save(
        {"gates": gates.state_dict(), "optimizer": optimizer.state_dict()},
        saved
    )
    saved.seek(0)
    restored = torch.load(saved, weights_only=False)
    other = BranchGates().double()
    other_optimizer = torch.optim.Adam(other.parameters(), lr=0.01)
    other.load_state_dict(restored["gates"])
    other_optimizer.load_state_dict(restored["optimizer"])

    for g, opt in ((gates, optimizer), (other, other_optimizer)):
        meta = lookahead_losses(model, g, raw, rates, (meta_x,), losses)
        update_gates(g, opt, meta)
    torch.testing.assert_close(gates.logits, other.logits, rtol=0, atol=0)
    print("[PASS] gate and optimizer checkpoint preserves the next gate update")

    model.cell.weight.requires_grad_(False)
    a = losses(model(x))
    frozen_raw = task_gradients(model, a["cell"], a["tissue"])
    assign_gradients(model, gates, frozen_raw)
    assert model.cell.weight.grad is None
    assert "cell.weight" not in frozen_raw

    assert owner("encoder_c2t.weight") == 1
    assert owner("decoder_t2c.weight") == 0
    assert owner("cell.encoder.weight") == 0
    assert owner("tissue.encoder.weight") == 1
    try:
        owner("unknown.weight")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown owner must fail")

    print("[PASS] private gradients, frozen parameters and branch ownership")
    print("[PASS] synthetic dual-gate checks complete; training not launched")


if __name__ == "__main__":
    main()
