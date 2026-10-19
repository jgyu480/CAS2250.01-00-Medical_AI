"""Synthetic correctness checks; no dataset, pretrained weights, or training run."""
import argparse
import csv
import gc
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from common.training.gradient_conflict import GradientConflictMonitor, geometry
from common.training.experiment_models import build_model, initialize_from
from common.losses.m0_losses import m0_task_losses


def check_geometry():
    assert geometry(-1, 1, 1)["conflict"] == 1
    assert geometry(-1, 1, 1)["cancellation_fraction"] == 1
    assert geometry(1, 1, 1)["conflict"] == 0
    assert geometry(0, 1, 1)["cosine"] == 0
    assert geometry(0, 0, 1) is None
    assert math.isclose(geometry(-2, 1, 4)["magnitude_similarity"], .8)
    assert geometry(-2, 1, 4)["cell_first_order_adverse"] == 1


def check_monitor(folder):
    model = nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(1)
    # Autograd.grad must leave an existing .grad completely unchanged.
    model.weight.grad = torch.full_like(model.weight, 7)
    monitor = GradientConflictMonitor(folder)
    monitor.begin(1, model, False)
    x = torch.tensor([[1., 2.]])
    out = model(x).sum()
    monitor.observe(out, -out, 1, ["synthetic"])
    assert torch.equal(model.weight.grad, torch.full_like(model.weight, 7))
    summary = monitor.finish(1)
    assert summary["conflict_rate"] == 1
    out.backward()
    assert torch.equal(model.weight.grad, torch.tensor([[8., 9.]]))
    # Resume removes incomplete observations past the checkpointed epoch.
    monitor.begin(2, model, False)
    out = model(x).sum()
    monitor.observe(out, out, 1, ["interrupted"])
    monitor = GradientConflictMonitor(folder, start_epoch=2)
    with (Path(folder)/"gradient_conflict/batches.csv").open() as f:
        assert all(int(r["epoch"]) == 1 for r in csv.DictReader(f))


def check_models(folder):
    for family, config_path in (("M0", "models/M0/config.json"),
                                 ("M0_1", "models/M0_1/config.json")):
        torch.manual_seed(42)
        config = json.loads((ROOT/config_path).read_text())
        model = build_model(config, initialize_encoders=False)
        model.train()
        x, t = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
        boxes = torch.tensor([[.25, .25, .5, .5]])
        output = model(x, t, boxes)
        assert output["cell_logits"].shape == (1, 3, 64, 64)
        targets = torch.randint(0, 3, (1, 64, 64))
        losses = m0_task_losses(output["cell_logits"], output["tissue_logits"], targets, targets)
        monitor = GradientConflictMonitor(Path(folder)/family)
        monitor.begin(1, model, False)
        monitor.observe(losses["cell"], losses["tissue"], 1, ["synthetic"])
        summary = monitor.finish(1)
        assert summary["valid_batches"] == 1
        (losses["cell"] + losses["tissue"]).backward()
        for branch in (model.cell, model.tissue):
            assert branch.encoder.layer1[0].conv1.weight.grad is not None
            for module in branch.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    assert not module.training
        enc_ids = {id(p) for p in model.encoder_parameters()}
        new_ids = {id(p) for p in model.new_module_parameters()}
        assert not enc_ids & new_ids
        assert enc_ids | new_ids == {id(p) for p in model.parameters()}
        if family == "M0":
            # Small backbone-only fixture: avoids writing a full 300MB model.
            weights = {k: v for k, v in model.state_dict().items()
                       if k.startswith(("cell.encoder.", "tissue.encoder."))}
            checkpoint = Path(folder)/"backbones.pt"
            torch.save({"model": weights}, checkpoint)
            del weights
        else:
            initialize_from(model, checkpoint, "encoders")
            source = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
            for key, value in source.items():
                assert torch.equal(model.state_dict()[key], value)
            # Preserve output stride 16: stages 3 and 4 have the same spatial size.
            with torch.no_grad():
                features = model.cell.encoder(x)
            assert [f.shape[-1] for f in features] == [16, 8, 4, 4]
            del source, features
        print(f"[PASS] {family}: forward, both loss gradients, frozen BN, parameter groups")
        del model, monitor, output, losses
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    with tempfile.TemporaryDirectory() as folder:
        check_geometry()
        check_monitor(folder)
        print("[PASS] gradient geometry, non-mutating observation, resume cleanup")
        if not args.metrics_only:
            check_models(folder)
    print("[PASS] synthetic checks complete; no dataset training launched")


if __name__ == "__main__":
    main()
