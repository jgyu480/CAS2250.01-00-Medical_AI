#!/usr/bin/env python3
"""Compare OCELOT map mean, spatial TTA, and a per-pixel model gate."""

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


METHODS = ("map_mean", "softctm_tta", "pixel_gate")


class PixelGate(nn.Module):
    """Predict separate simplex weights over models for BC and TC."""

    def __init__(self, count):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2 * count, 16, 1), nn.ReLU(), nn.Conv2d(16, 2 * count, 1)
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, maps):
        batch, count, channels, height, width = maps.shape
        scores = self.layers(maps.reshape(batch, count * channels, height, width))
        weights = scores.reshape_as(maps).softmax(dim=1)
        return (weights * maps).sum(dim=1)


def transform(tensor, flip, turns):
    if flip:
        tensor = tensor.flip(-1)
    return tensor.rot90(turns, (-2, -1)) if turns else tensor


def undo(tensor, flip, turns):
    if turns:
        tensor = tensor.rot90(-turns, (-2, -1))
    return tensor.flip(-1) if flip else tensor


def transform_box(box, flip, turns):
    x0, y0, x1, y1 = box.unbind(-1)
    if flip:
        x0, x1 = 1 - x1, 1 - x0
    for _ in range(turns):
        x0, y0, x1, y1 = y0, 1 - x1, y1, 1 - x0
    return torch.stack((x0, y0, x1, y1), dim=-1)


@torch.no_grad()
def infer_maps(models, batch, device, transforms, prepare, autocast):
    by_model = []
    augmented_sum = None
    for model, size, activation, _, _, use_amp in models:
        images, _, box = prepare(batch, device, size)
        model_sum = None
        for flip, turns in transforms:
            view = [transform(image, flip, turns).contiguous() for image in images]
            cast, _ = autocast(device, use_amp)
            with cast:
                logits = model(*view, transform_box(box, flip, turns))["cell_logits"].float()
            scores = logits.softmax(dim=1)[:, 1:] if activation == "softmax" else logits.sigmoid()
            if scores.shape[1] != 2:
                raise ValueError(f"Expected two BC/TC score maps, got {scores.shape}")
            scores = undo(scores, flip, turns)
            if scores.shape[-2:] != (1024, 1024):
                scores = F.interpolate(scores, (1024, 1024), mode="bilinear", align_corners=False)
            if not flip and not turns:
                by_model.append(scores[0])
            model_sum = scores[0].clone() if model_sum is None else model_sum.add_(scores[0])
        augmented_sum = model_sum if augmented_sum is None else augmented_sum.add_(model_sum)
    native = torch.stack(by_model)
    return native, augmented_sum / (len(models) * len(transforms))


def atomic_torch_save(payload, path):
    tmp = path.with_suffix(".part")
    torch.save(payload, tmp)
    tmp.replace(path)


def fit_gate(args, gate, loader, infer, checkpoint, device, heatmaps, identity):
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Run settings changed. Choose a fresh --output directory.")
        gate.load_state_dict(saved["model"])
        print("[resume] Reusing pixel gate", flush=True)
        return gate.eval()

    patches, targets = [], []
    rng = np.random.default_rng(args.seed)
    crop = 128
    for i, batch in enumerate(loader, 1):
        maps, _ = infer(batch, ((False, 0),))
        maps = maps.cpu()
        points = batch["cell_points"][0].cpu().numpy().reshape(-1, 3)
        label = heatmaps(points, 6., 0., mode="gaussian")
        for patch in range(4):
            if patch < 2 and len(points):
                xy = points[rng.integers(len(points)), :2] - crop // 2
                x, y = np.clip(np.rint(xy), 0, 1024 - crop).astype(int)
            else:
                x, y = rng.integers(0, 1024 - crop + 1, size=2)
            patches.append(maps[:, :, y:y + crop, x:x + crop].to(torch.float16))
            targets.append(torch.from_numpy(label[:, y:y + crop, x:x + crop]).to(torch.float16))
        if i % 20 == 0 or i == len(loader):
            print(f"[fit data] {i}/{len(loader)}", flush=True)

    x, y = torch.stack(patches), torch.stack(targets)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=0.001)
    gate.train()
    for epoch in range(args.epochs):
        total = 0.
        for index in torch.randperm(len(x)).split(8):
            pred = gate(x[index].to(device=device, dtype=torch.float32))
            truth = y[index].to(device=device, dtype=torch.float32)
            weight = 1 + 9 * truth
            loss = (F.binary_cross_entropy(pred.clamp(1e-5, 1 - 1e-5), truth, reduction="none") * weight).sum() / weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        print(f"[pixel_gate] epoch={epoch + 1}/{args.epochs} BCE={total / len(x):.5f}", flush=True)
    atomic_torch_save({"identity": identity, "model": gate.state_dict()}, checkpoint)
    return gate.eval()


@torch.no_grad()
def evaluate(loader, infer, gate, transforms, grids, fusion, metric_type, test):
    from scripts.diagnose_m0 import localization_labels, nearest_distances

    metrics = {name: {key: metric_type() for key in grids[name]} for name in METHODS}
    near_count = 0
    near_misses = dict.fromkeys(METHODS, 0)
    for i, batch in enumerate(loader, 1):
        maps, tta = infer(batch, transforms)
        outputs = (maps.mean(0), tta, gate(maps.unsqueeze(0))[0])
        truth = np.rint(batch["cell_points"][0].cpu().numpy()).astype(np.int32).reshape(-1, 3)
        if test:
            distance = nearest_distances(truth[:, :2])
            close = (truth[:, 2] == 2) & (distance >= 14) & (distance < 30)
            near_count += int(close.sum())
        for name, heatmap in zip(METHODS, outputs):
            for setting, pred in fusion.candidate_points(heatmap, False, grids[name]):
                metrics[name][setting].update(pred, truth)
                if test and close.any():
                    labels, _ = localization_labels(pred, truth)
                    near_misses[name] += int((close & (labels == "missed")).sum())
        if i % 10 == 0 or i == len(loader):
            print(f"[{'test' if test else 'val_tune'}] {i}/{len(loader)}", flush=True)

    selected, rows = {}, []
    for name, by_setting in metrics.items():
        key = max(by_setting, key=lambda setting: by_setting[setting].compute()["mF1"])
        selected[name] = key
        metric = by_setting[key]
        rows.append({
            "split": "test" if test else "val_tune", "method": name,
            **metric.compute(), "FP_BC": metric.counts[1][1], "FP_TC": metric.counts[2][1],
            "near_TC_missed": near_misses[name] if test else None,
            "near_TC_count": near_count if test else None,
            "near_TC_miss": near_misses[name] / near_count if test and near_count else None,
            "sigma": key[0], "min_distance": key[1], "threshold": key[2],
        })
    return selected, rows


def save_summary(output, rows):
    tmp = output / "summary.part"
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(output / "summary.csv")
    for row in rows:
        print(f"[result] {row['split']:8s} {row['method']:12s} mF1={row['mF1']:.4f} "
              f"near_TC_miss={row['near_TC_miss']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/ocelot_ensemble"))
    parser.add_argument("--source-run", default="cell_tissue_aspp_seed42")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tta", choices=("flip", "d4"), default="d4")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    root = args.root.resolve()
    source = root / "experiments" / args.source_run
    config = json.loads((source / "run.json").read_text())
    output = args.output or root / "experiments" / f"tta_{args.tta}_pixel_gate_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    os.environ["OUTPUT_ROOT"] = str(root / "outputs")
    os.environ["OCELOT_DATA_ROOT"] = config["data_root"]
    sys.path[:0] = [str(root), str(root / "repo")]
    import ocelot_fusion_comparison as fusion
    from common.data.center_heatmaps import points_to_heatmaps
    from common.data.tensors import TorchOcelotDataset, collate_pairs
    from common.metrics.m0_eval import CellF1
    from common.training.m0_loop import autocast, pick_device, prepare, seed_everything

    seed_everything(args.seed)
    device = pick_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    torch.set_num_threads(2)
    transforms = (((False, 0), (True, 0)) if args.tta == "flip"
                  else tuple(itertools.product((False, True), range(4))))
    paths = [Path(config["checkpoints"][name]) for name in config["model_order"]]
    fit_ids, tune_ids = set(map(str, config["fit_ids"])), set(map(str, config["tune_ids"]))
    if not fit_ids or not tune_ids or fit_ids & tune_ids:
        raise ValueError("The source run must contain non-overlapping fit and tuning IDs.")
    identity = {
        "seed": args.seed, "epochs": args.epochs, "tta": args.tta,
        "source_run": str(source), "fit_ids": sorted(fit_ids), "tune_ids": sorted(tune_ids),
        "checkpoints": [(name, str(path), path.stat().st_size, path.stat().st_mtime_ns)
                        for name, path in zip(config["model_order"], paths)],
    }

    val = TorchOcelotDataset(split="val")
    lookup = {str(row["pair_id"]): i for i, row in enumerate(val.source.rows)}

    def loader(dataset):
        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_pairs)

    models = [fusion.load_base(path) for path in paths]
    for model, *_ in models:
        model.to(device).eval()

    def infer(batch, views):
        return infer_maps(models, batch, device, views, prepare, autocast)

    gate = PixelGate(len(models)).to(device)
    gate = fit_gate(args, gate, loader(Subset(val, [lookup[key] for key in sorted(fit_ids)])),
                    infer, output / "pixel_gate.pt", device, points_to_heatmaps, identity)
    val_state = output / "val_tune.pt"
    if val_state.is_file():
        saved = torch.load(val_state, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Validation settings changed. Choose a fresh --output directory.")
        selected, val_rows = saved["selected"], saved["rows"]
        print("[resume] Reusing validation thresholds", flush=True)
    else:
        grid = tuple(itertools.product(*fusion.GRID))
        selected, val_rows = evaluate(
            loader(Subset(val, [lookup[key] for key in sorted(tune_ids)])), infer, gate,
            transforms, dict.fromkeys(METHODS, grid), fusion, CellF1, test=False,
        )
        atomic_torch_save({"identity": identity, "selected": selected, "rows": val_rows}, val_state)
    save_summary(output, val_rows)
    _, test_rows = evaluate(
        loader(TorchOcelotDataset(split="test")), infer, gate, transforms,
        {name: (key,) for name, key in selected.items()}, fusion, CellF1, test=True,
    )
    save_summary(output, [*val_rows, *test_rows])
    print(f"[done] {output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
