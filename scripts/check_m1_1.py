"""Dependency-light, data-free M1-1 unit tests (tiny synthetic M1-compatible teacher).

python scripts/check_m1_1.py
Optional: python scripts/check_m1_1.py --real-m1-checkpoint /path/to/M1/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.losses.m0_losses import focal_loss
from common.modules.gated_fusion import mixture_log_probs
from models.M1_1.model import M1_1, build_model


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.low = nn.Conv2d(3, 256, 3, stride=4, padding=1)
        self.high = nn.Conv2d(256, 32, 1)

    def forward(self, image):
        low = self.low(image)
        high = self.high(F.avg_pool2d(low, 4))
        return {"low": low, "high": high}


class TinyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.aspp = nn.Conv2d(32, 256, 1)
        self.decoder = nn.Sequential()  # replaced below with compatible custom decoder
        self.decoder = TinyDecoder()
        self.head = nn.Conv2d(256, 3, 1)


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(512, 256, 1)

    def forward(self, context, low):
        return self.conv(torch.cat([F.interpolate(context, size=low.shape[-2:],
                                                   mode="bilinear", align_corners=False), low], dim=1))


class TinyCrossAttention(nn.Module):
    def forward(self, cell, tissue, boxes):
        return cell + tissue.mean(dim=(-1, -2), keepdim=True)


class TinyT2C(nn.Module):
    def forward(self, cell, tissue, boxes):
        return cell + tissue.mean(dim=(-1, -2), keepdim=True)


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = TinyBranch()
        self.tissue = TinyBranch()
        self.aspp_cross_attention = TinyCrossAttention()
        self.decoder_t2c = TinyT2C()

    def forward(self, cell_image, tissue_image, boxes):
        cell = self.cell.encoder(cell_image)
        tissue = self.tissue.encoder(tissue_image)
        cell_context = self.cell.aspp(cell["high"])
        tissue_context = self.tissue.aspp(tissue["high"])
        cell_context = self.aspp_cross_attention(cell_context, tissue_context, boxes)
        cell_decoded = self.cell.decoder(cell_context, cell["low"])
        tissue_decoded = self.tissue.decoder(tissue_context, tissue["low"])
        cell_decoded = self.decoder_t2c(cell_decoded, tissue_decoded, boxes)
        return {"cell_logits": F.interpolate(self.cell.head(cell_decoded),
                                              size=cell_image.shape[-2:], mode="bilinear", align_corners=False),
                "tissue_logits": F.interpolate(self.tissue.head(tissue_decoded),
                                                size=tissue_image.shape[-2:], mode="bilinear", align_corners=False)}


def test_fake():
    torch.manual_seed(42)
    model = M1_1(TinyTeacher())
    model.initialize_local_from_teacher()
    cell, tissue = torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)
    box = torch.tensor([[0.3, 0.25, 0.55, 0.5]])
    model.set_stage("gate").eval()
    out = model(cell, tissue, box)
    assert out["cell_logits"].shape == (1, 3, 32, 32)
    assert out["tissue_logits"].shape == (1, 3, 32, 32)
    assert out["gate_map"].shape == (1, 1, 8, 8)
    assert torch.allclose(out["gate_map"], torch.full_like(out["gate_map"], 0.5))
    probs = out["cell_logits"].softmax(1)
    assert torch.allclose(probs.sum(1), torch.ones_like(probs[:, 0]), atol=1e-6)
    assert torch.isfinite(out["cell_logits"]).all()
    teacher_out = model.teacher(cell, tissue, box)
    assert torch.allclose(out["m1_logits"], teacher_out["cell_logits"], atol=1e-6)
    assert torch.allclose(out["tissue_logits"], teacher_out["tissue_logits"], atol=1e-6)
    zero = mixture_log_probs(out["local_logits"], out["m1_logits"],
                             out["gate_map"], fixed_weight=0)
    one = mixture_log_probs(out["local_logits"], out["m1_logits"],
                            out["gate_map"], fixed_weight=1)
    assert torch.allclose(zero.softmax(1), out["local_logits"].softmax(1), atol=1e-6)
    assert torch.allclose(one.softmax(1), out["m1_logits"].softmax(1), atol=1e-6)
    # Changing only tissue input must NOT alter the independent local path.
    alternate = model(cell, tissue + 5, box)
    assert torch.allclose(out["local_logits"], alternate["local_logits"], atol=1e-6)
    assert not torch.allclose(out["m1_logits"], alternate["m1_logits"])
    model.set_stage("local").train()
    assert not model.teacher.training
    assert not any(p.requires_grad for p in model.teacher.parameters())
    assert not any(p.requires_grad for p in model.gate.parameters())
    stage_a = model(cell, tissue, box)
    assert torch.allclose(stage_a["cell_logits"], stage_a["local_logits"])
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 12:16, 12:16] = 1
    focal_loss(stage_a["cell_logits"], target).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.local_decoder.parameters())
    assert all(p.grad is None for p in model.teacher.parameters())
    assert all(p.grad is None for p in model.gate.parameters())
    model.zero_grad(set_to_none=True)
    model.set_stage("gate").train()
    final = model(cell, tissue, box)
    (focal_loss(final["cell_logits"], target) +
     0.2 * focal_loss(final["local_logits"], target)).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.gate.parameters()), "Gate must get gradient"
    assert all(p.grad is None for p in model.teacher.parameters())
    # Full state round-trip (including teacher) is compatible with checkpoint loading.
    other = M1_1(TinyTeacher())
    other.load_state_dict(model.state_dict(), strict=True)
    other.set_stage("gate").eval()
    assert torch.allclose(other(cell, tissue, box)["cell_logits"],
                          model.eval()(cell, tissue, box)["cell_logits"], atol=1e-6)
    print("PASS: shapes, probability, gate endpoints, local independence, "
          "frozen teacher, original-M1 equivalence, stage gradients, checkpoint reload")


def test_real(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model()
    model.teacher.load_state_dict(ckpt["model"], strict=True)
    model.initialize_local_from_teacher()
    model.set_stage("gate").eval()
    cell, tissue = torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128)
    box = torch.tensor([[0.25, 0.25, 0.5, 0.5]])
    with torch.no_grad():
        o = model(cell, tissue, box)
        old = model.teacher(cell, tissue, box)
    assert torch.allclose(o["m1_logits"], old["cell_logits"], atol=1e-5)
    assert torch.allclose(o["tissue_logits"], old["tissue_logits"], atol=1e-5)
    print("PASS: real M1 checkpoint strict load and frozen output equivalence")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real-m1-checkpoint")
    args = p.parse_args()
    test_fake()
    if args.real_m1_checkpoint:
        test_real(args.real_m1_checkpoint)
