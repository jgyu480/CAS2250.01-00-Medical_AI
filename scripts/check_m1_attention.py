"""Check M1 module shapes, normalized tissue coordinates, and gradient routes.

Does not need OCELOT data or SSL weights. Run at repository root:
    python scripts/check_m1_attention.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from common.modules.cross_attention import (
    TissueGuidedCrossAttention, cell_token_coordinates,
    tissue_token_coordinates
)
from models.M1.model import M1


def require(ok, why):
    if not bool(ok):
        raise AssertionError(why)


class FakeEncoder(nn.Module):
    """Return the same feature shapes as the actual ResNet-50 encoder."""
    def __init__(self):
        super().__init__()
        self.low = nn.Conv2d(3, 256, 1)
        self.high = nn.Conv2d(3, 2048, 1)

    def forward(self, image):
        low = self.low(F.avg_pool2d(image, kernel_size=4))
        high = self.high(F.avg_pool2d(image, kernel_size=32))
        return {"low": low, "high": high}


def main():
    torch.manual_seed(42)
    boxes = torch.tensor([
        [0.50, 0.50, 0.75, 0.75],
        [0.00, 0.00, 0.25, 0.25]
    ])
    xy = cell_token_coordinates(4, 4, boxes)
    require(xy.shape == (2, 16, 2), "cell coordinate shape")
    require(torch.allclose(xy[0, 0], torch.tensor([0.53125, 0.53125])),
            "cell_box mapping incorrect")
    require(torch.allclose(tissue_token_coordinates(16, 16, device=boxes.device)[0, 0],
                           torch.tensor([1/32, 1/32])), "tissue token centers incorrect")
    print("[PASS] cell/tissue coordinate alignment")

    # Test full-size attention sequence lengths without the heavy encoder/decoder.
    module = TissueGuidedCrossAttention()
    cell = torch.randn(2, 256, 32, 32, requires_grad=True)
    tissue = torch.randn(2, 256, 32, 32, requires_grad=True)
    out, attn = module(cell, tissue, boxes, return_attention=True)
    require(out.shape == cell.shape, "attention output shape")
    require(attn.shape == (2, 4, 1024, 256), "attention tensor shape")
    require(torch.isfinite(out).all(), "nonfinite attention result")
    require(torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-5),
            "attention weights not normalized")
    out.square().mean().backward()
    require(tissue.grad is not None and tissue.grad.abs().sum() > 0,
            "cell loss must reach tissue through attention")
    require(any(p.grad is not None and p.grad.abs().sum() > 0
                for p in module.attention.parameters()), "attention params need gradient")
    print("[PASS] attention [B,4,1024,256] + gradients")

    # Small end-to-end M1 using fake feature encoders (no data or weights needed).
    model = M1(FakeEncoder(), FakeEncoder()).eval()
    image = torch.randn(2, 3, 64, 64)
    tissue_image = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        prediction = model(image, tissue_image, boxes)
        # The tissue segmentation must not change when the cell image changes.
        changed_cell = model(image + 1.0, tissue_image, boxes)
    require(all(v.shape == (2, 3, 64, 64) for v in prediction.values()),
            "M1 logits shape")
    require(torch.equal(prediction["tissue_logits"], changed_cell["tissue_logits"]),
            "M1 tissue prediction depends on cell input")
    print("[PASS] M1 forward/output API and no cell→tissue information path")

    # Check that Cell loss teaches the tissue context encoder, but Tissue loss
    # never reaches Cell-only parameters (a unidirectional architecture).
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(image, tissue_image, boxes)
    prediction["cell_logits"].square().mean().backward()
    tissue_encoder_grad = model.tissue.encoder.high.weight.grad
    require(tissue_encoder_grad is not None and tissue_encoder_grad.abs().sum() > 0,
            "cell loss does not reach tissue encoder")
    require(model.tissue.head.weight.grad is None,
            "cell loss should not reach tissue head")
    require(any(p.grad is not None and p.grad.abs().sum() > 0
                for p in model.aspp_cross_attention.parameters()),
            "cell loss should train cross-attention")
    model.zero_grad(set_to_none=True)
    prediction = model(image, tissue_image, boxes)
    prediction["tissue_logits"].square().mean().backward()
    require(all(p.grad is None for p in model.cell.parameters()),
            "tissue loss must not train cell branch")
    require(all(p.grad is None for p in model.aspp_cross_attention.parameters()),
            "tissue loss must not train attention")
    print("[PASS] unidirectional gradient routes")
    print("[DONE] Synthetic M1 smoke test passed (NOT actual GPU training)")


if __name__ == "__main__":
    main()
