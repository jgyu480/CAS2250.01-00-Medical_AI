#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'HELP'
Usage: bash run_ocelot_m3_retrain_val.sh [--publish-only]
Retrain G/GA with the selected M2 recipe, then run the same M3 D4 validation pipeline.
Options can also be set through environment variables:
  ENSEMBLE_ROOT       /root/ocelot_ensemble
  M3_SOURCE_CONFIG    path to the previous five-model ensemble run.json
  M3_RUN_ID           m3_context_d4_retrained_val_seed42
  M3_PYTHON           existing Python with torch/torchvision/numpy/scipy/Pillow
  M3_EPOCHS           12
  M3_DEVICE           cuda
  OCELOT_DATA_ROOT    existing OCELOT v1.0.1 folder, if moved
  M3_CACHE            disk location for the reusable validation D4 cache (~4 GiB)
  M3_PUBLISH          set to 0 to save locally and publish later
  M0_CHECKPOINT M1_CHECKPOINT M11_CHECKPOINT
                      existing M0/M1/M1-1 checkpoint overrides
  SSL_WEIGHTS_PATH    optional existing Lunit BT weights
HELP
}
mode=${1:-}
case "$mode" in
  -h|--help) usage; exit 0 ;;
  ''|--publish-only) ;;
  *) usage >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { usage >&2; exit 2; }

root=${ENSEMBLE_ROOT:-/root/ocelot_ensemble}
mkdir -p "$root"
root=$(cd "$root" && pwd -P)
run=${M3_RUN_ID:-m3_context_d4_retrained_val_seed42}
[[ "$run" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] || { echo 'Invalid M3_RUN_ID' >&2; exit 2; }

out="$root/experiments/$run"
mkdir -p "$out"
exec 9>"$out/run.lock"
flock -n 9 || { echo "Already running: $out" >&2; exit 1; }
self=$(realpath "$0")
target=jgyu480/CAS2250.01-00-Medical_AI
url="https://github.com/$target.git"
revision=14f930ac47d69bb9abc6d86c6abcb5bdaf35a8da
work="$root/m3_runtime"
branch="results/ensemble-$run"
destination="results/ensemble/$run"
repo="$work/repo"
export GIT_TERMINAL_PROMPT=0
publish_enabled=${M3_PUBLISH:-1}
case "$mode" in
  --publish-only) publish_enabled=1 ;;
esac

if [[ "$publish_enabled" == 1 ]]; then
  if ! command -v gh >/dev/null; then
    if [[ $(id -u) == 0 ]] && command -v apt-get >/dev/null; then
      apt-get update && apt-get install -y gh
    else
      echo 'Install GitHub CLI (gh), then rerun.' >&2
      exit 1
    fi
  fi
  if ! gh auth status --hostname github.com >/dev/null 2>&1; then
    if [[ ! -t 0 ]]; then
      echo 'First log in interactively: gh auth login --hostname github.com' >&2
      exit 1
    fi
    gh auth login --hostname github.com
  fi
  if gh auth setup-git --help >/dev/null 2>&1; then
    gh auth setup-git --hostname github.com
  fi
  can_push=$(gh api "repos/$target" --jq '.permissions.push')
  [[ "$can_push" == true ]] || { echo "The active GitHub account needs write access to $target." >&2; exit 1; }
fi

mkdir -p "$work"
exec 8>"$work/setup.lock"
flock 8
if [[ ! -d "$repo/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout --single-branch --branch main "$url" "$repo"
fi
if ! git -C "$repo" cat-file -e "$revision^{commit}" 2>/dev/null; then
  git -C "$repo" fetch --no-tags "$url" refs/heads/main
fi
if [[ ! -f "$repo/common/metrics/m0_eval.py" ]] || [[ $(git -C "$repo" rev-parse HEAD) != "$revision" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git -C "$repo" checkout --detach "$revision"
fi

publish() (
  [[ -s "$out/complete.ok" && -s "$out/summary.csv" ]] || { echo "Incomplete experiment: $out" >&2; exit 1; }
  index_dir=$(mktemp -d "$out/.index.XXXXXXXX")
  trap 'rm -rf "$index_dir"' EXIT
  export GIT_INDEX_FILE="$index_dir/index"
  cd "$repo"
  ref="refs/heads/$branch"
  login=$(gh api user --jq .login)
  user_id=$(gh api user --jq .id)
  for attempt in 1 2 3; do
    listing=$(git ls-remote "$url" "$ref")
    parent=$revision
    if [[ -n "$listing" ]]; then
      git fetch --no-tags "$url" "$ref"
      parent=${listing%%[[:space:]]*}
    fi
    git read-tree "$parent"
    for artifact in run.json postprocess.json summary.csv validation_grid.csv epoch_history.csv \
        per_pair.csv predictions.jsonl.gz console.log complete.ok pointwise.pt m3_1.pt m3_2.pt README.md M2_training.csv; do
      test -s "$out/$artifact"
      object=$(git hash-object -w -- "$out/$artifact")
      git update-index --add --cacheinfo "100644,$object,$destination/$artifact"
    done
    object=$(git hash-object -w -- "$out/run_ocelot_m3_retrain_val.sh")
    git update-index --add --cacheinfo "100755,$object,$destination/run_ocelot_m3_retrain_val.sh"
    tree=$(git write-tree)
    commit=$parent
    if [[ "$tree" != "$(git rev-parse "$parent^{tree}")" ]]; then
      commit=$(git -c "user.name=$login" -c "user.email=$user_id+$login@users.noreply.github.com" \
        commit-tree "$tree" -p "$parent" -m "Save OCELOT M3 D4 TTA experiment $run")
    fi
    if [[ "$commit" == "$parent" && -n "$listing" ]] || git push "$url" "$commit:$ref"; then
      link="https://github.com/$target/tree/$commit/$destination"
      printf '%s\n' "$link" > "$out/github_url.txt"
      echo "[GitHub] $link"
      exit 0
    fi
    sleep 3
  done
  echo "[GitHub] Upload failed. Use M3_RUN_ID=$run bash $self --publish-only" >&2
  exit 1
)

if [[ "$mode" == --publish-only ]]; then flock -u 8; publish; exit 0; fi
if [[ -s "$out/complete.ok" ]]; then
  flock -u 8
  if [[ "$publish_enabled" == 1 ]]; then publish; fi
  echo "[done] $out/summary.csv"
  exit 0
fi

if [[ -n ${M3_PYTHON:-} ]]; then
  python=$M3_PYTHON
elif [[ -x "$root/venv/bin/python" ]]; then
  python="$root/venv/bin/python"
elif [[ -x /opt/conda/envs/cas2250-medical-ai/bin/python ]]; then
  python=/opt/conda/envs/cas2250-medical-ai/bin/python
else
  python=python3
fi
python=$(realpath -s -- "$(command -v "$python")")
"$python" -c 'import torch, torchvision, scipy, PIL, numpy as np; torch.from_numpy(np.zeros(1))'

if [[ -f "$out/run_ocelot_m3_retrain_val.sh" ]]; then
  cmp -s "$self" "$out/run_ocelot_m3_retrain_val.sh" || {
    echo 'Launcher changed. Use a new M3_RUN_ID.' >&2
    exit 1
  }
else
  cp "$self" "$out/run_ocelot_m3_retrain_val.sh"
fi
# Pin the existing inference adapter and postprocessor to the recorded experiment.
git -C "$repo" fetch --no-tags "$url" refs/heads/results/ensemble-cell_tissue_aspp_seed42
legacy=f20e9c6e1b39aa9b0774456744f77b210fe705ab
prefix=results/ensemble/cell_tissue_aspp_seed42
legacy_dir="$out/legacy"
mkdir -p "$legacy_dir"
git -C "$repo" show "$legacy:$prefix/run_ocelot_fusion_comparison.sh" |
  awk '/^cat > .*<<.PYTHON_OCELOT_ENSEMBLE./ {body=1; next} /^PYTHON_OCELOT_ENSEMBLE$/ {exit} body' > "$legacy_dir/ocelot_fusion_comparison.py"
test -s "$legacy_dir/ocelot_fusion_comparison.py"
source_config=${M3_SOURCE_CONFIG:-$root/experiments/cell_tissue_aspp_seed42/run.json}
if [[ ! -f "$source_config" && -z ${M3_SOURCE_CONFIG:-} ]]; then
  source_config="$legacy_dir/source_run.json"
  git -C "$repo" show "$legacy:$prefix/run.json" > "$source_config"
fi
test -s "$source_config"
source_config=$(realpath -- "$source_config")
search_commit=7ff09e00fe7e296f11abad5e4e6b1e1841613e77
search_repo="$work/search-${search_commit:0:7}"
if ! git -C "$repo" cat-file -e "$search_commit^{commit}" 2>/dev/null; then
  git -C "$repo" fetch --no-tags "$url" refs/heads/feature/m2-sigma-search
fi
if [[ ! -f "$search_repo/scripts/train_m2.py" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git -C "$repo" worktree add --detach "$search_repo" "$search_commit"
fi
[[ $(git -C "$search_repo" rev-parse HEAD) == "$search_commit" ]]
export M3_SEARCH_REPO="$search_repo"
for override in OCELOT_DATA_ROOT M3_CACHE SSL_WEIGHTS_PATH M0_CHECKPOINT M1_CHECKPOINT M11_CHECKPOINT; do
  if [[ -n ${!override:-} && ${!override} != /* && ${!override} != '~/'* ]]; then
    export "$override=$(realpath -m -- "${!override}")"
  fi
done
flock -u 8

export M3_REPO="$repo" M3_OUTPUT="$out" M3_SOURCE_CONFIG="$source_config" M3_STATE_ROOT="$root"
export M3_SOURCE_SHA="$revision" M3_ADAPTER_SHA="$legacy"
export M3_MODE="$mode" PYTHONUNBUFFERED=1
cat > "$out/m3_context.py" <<'M3_PYTHON'
"""M3-1 fixed context and M3-2 learned context on frozen D4 model outputs."""

from contextlib import contextmanager
import csv
import fcntl
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

OUT = Path(os.environ["M3_OUTPUT"])
REPO = Path(os.environ["M3_REPO"])
ROOT = Path(os.environ["M3_STATE_ROOT"])
sys.path[:0] = [str(REPO), str(OUT / "legacy")]

import ocelot_fusion_comparison as legacy
from common.data.audit_metadata import load_metadata
from common.data.audit_split import inspect_split
from common.data.center_heatmaps import points_to_heatmaps
from common.data.geometry import cell_box
from common.data.tensors import TorchOcelotDataset, collate_pairs
from common.metrics.m0_eval import CellF1, gaussian_blur
from common.modules.exchange import crop_to_cell
from common.training.m0_loop import autocast, prepare, seed_everything
from scripts.diagnose_m0 import localization_labels, nearest_distances

MODEL_NAMES = ("G", "GA", "M0", "M1", "M1-1")
OVERRIDES = ("G_CHECKPOINT", "GA_CHECKPOINT", "M0_CHECKPOINT", "M1_CHECKPOINT", "M11_CHECKPOINT")
VARIANTS = ("pointwise", "m3_1", "m3_2")
METHODS = ("tta_mean", *VARIANTS)
VIEWS = tuple(itertools.product((False, True), range(4)))
CELL_SIZE = 1024
TISSUE_SIZE = 256
SEED = 42

# The same discretized 14–30 px annulus is used in both cell classes.
RING_OFFSETS = sorted({
    (round(radius * math.sin(angle)), round(radius * math.cos(angle)))
    for radius in (14, 22, 30)
    for angle in (k * math.pi / 8 for k in range(16))
    if 14 <= math.hypot(round(radius * math.sin(angle)), round(radius * math.cos(angle))) <= 30
})


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows):
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def transform(image, flip, turns):
    image = image.flip(-1) if flip else image
    return image.rot90(turns, (-2, -1)) if turns else image


def undo(image, flip, turns):
    image = image.rot90(-turns, (-2, -1)) if turns else image
    return image.flip(-1) if flip else image


def transform_box(box, flip, turns):
    x0, y0, x1, y1 = box.unbind(-1)
    if flip:
        x0, x1 = 1 - x1, 1 - x0
    for _ in range(turns):
        x0, y0, x1, y1 = y0, 1 - x1, y1, 1 - x0
    return torch.stack((x0, y0, x1, y1), dim=-1)


def resize(image, size):
    if image.shape[-2:] == (size, size):
        return image
    return F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)


@torch.inference_mode()
def infer(models, batch, device):
    cell_models, tissue_models, aligned_models = [], [], []
    baseline_sum = None
    for model, size, activation, _, _, amp in models:
        images, _, box = prepare(batch, device, size)
        cell_sum = tissue_sum = None
        for flip, turns in VIEWS:
            cast, _ = autocast(device, amp)
            with cast:
                result = model(
                    *(transform(image, flip, turns).contiguous() for image in images),
                    transform_box(box, flip, turns),
                )
            logits = result["cell_logits"].float()
            cell = logits.softmax(dim=1)[:, 1:] if activation == "softmax" else logits.sigmoid()
            if cell.shape[1] != 2:
                raise ValueError(f"Expected BC/TC maps, got {cell.shape}")
            cell = resize(undo(cell, flip, turns), CELL_SIZE)
            # BG=0, CA=1. Conditional CA probability over the two supervised classes.
            tissue = result["tissue_logits"].float()
            if tissue.shape[1] not in (2, 3):
                raise ValueError(f"Unexpected tissue channels: {tissue.shape}")
            ca = (tissue[:, 1:2] - tissue[:, :1]).sigmoid()
            ca = undo(ca, flip, turns)
            cell_sum = cell.clone() if cell_sum is None else cell_sum.add_(cell)
            tissue_sum = ca.clone() if tissue_sum is None else tissue_sum.add_(ca)
        cell_models.append((cell_sum / len(VIEWS))[0].half().cpu().numpy())
        tissue_mean = tissue_sum / len(VIEWS)
        context = F.interpolate(tissue_mean, (TISSUE_SIZE, TISSUE_SIZE), mode="area")
        tissue_models.append(context[0].half().cpu().numpy())
        aligned = crop_to_cell(tissue_mean, box, (TISSUE_SIZE, TISSUE_SIZE))
        aligned_models.append(aligned[0].half().cpu().numpy())
        baseline_sum = cell_sum if baseline_sum is None else baseline_sum.add_(cell_sum)
    return {
        "cells": np.concatenate(cell_models),
        "tissue": np.concatenate(tissue_models),
        "aligned": np.concatenate(aligned_models),
        "base": (baseline_sum[0] / (len(VIEWS) * len(models))).cpu().numpy(),
    }


def create_cache(cache, sets, paths, device):
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "cache.lock").open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        pending = [(name, dataset, [
            i for i in indices if not cache_file(cache, name, dataset, i).is_file()
        ]) for name, dataset, indices in sets]
        count = sum(len(indices) for _, _, indices in pending)
        if count == 0:
            print(f"[cache] Reusing {cache}", flush=True)
            return
        required = count * (10 * CELL_SIZE**2 * 2 + 10 * TISSUE_SIZE**2 * 2 + 2 * CELL_SIZE**2 * 4)
        if shutil.disk_usage(cache).free < required + (256 << 20):
            raise RuntimeError(f"Cache needs {required / 2**30:.2f} GiB free at {cache}. Set M3_CACHE to a larger disk.")
        models = [legacy.load_base(path) for path in paths]
        for name, (model, _, _, kind, mode, _) in zip(MODEL_NAMES, models):
            expected = {"G": ("M2", "gaussian"), "GA": ("M2", "ha")}
            if name in expected and (kind, mode) != expected[name]:
                raise ValueError(f"{name}: expected {expected[name]}, got {(kind, mode)}")
            if name in ("M0", "M1", "M1-1") and kind != name.replace("-", "_"):
                raise ValueError(f"{name}: unexpected checkpoint model {kind}")
            model.to(device).eval()

        for name, dataset, indices in pending:
            loader = DataLoader(Subset(dataset, indices), batch_size=1, collate_fn=collate_pairs)
            for number, (index, batch) in enumerate(zip(indices, loader), 1):
                path = cache_file(cache, name, dataset, index)
                arrays = infer(models, batch, device)
                if not all(np.isfinite(array).all() for array in arrays.values()):
                    raise ValueError(f"Non-finite cache values for {path.stem}")
                tmp = path.with_suffix(".part")
                with tmp.open("wb") as handle:
                    np.savez(handle, **arrays)
                tmp.replace(path)
                if number == 1 or number % 10 == 0 or number == len(indices):
                    print(f"[D4 cache] {name} {number}/{len(indices)}", flush=True)


def cache_file(cache, name, dataset, index):
    return cache / f"{name}-{dataset.source.rows[index]['pair_id']}.npz"


def box_mean(image, k_y, k_x):
    # Separable box mean, with the divisor reduced at the tissue/cell edge.
    for kernel, padding in (((1, k_x), (0, k_x // 2)), ((k_y, 1), (k_y // 2, 0))):
        image = F.avg_pool2d(image, kernel, stride=1, padding=padding, count_include_pad=False)
    return image


def ring_max(mean_cell):
    height, width = mean_cell.shape[-2:]
    pad = 30
    padded = F.pad(mean_cell, (pad,) * 4, value=0)
    maximum = torch.zeros_like(mean_cell)
    for dy, dx in RING_OFFSETS:
        y, x = pad + dy, pad + dx
        maximum = torch.maximum(maximum, padded[..., y:y + height, x:x + width])
    return maximum


def fixed_cell_context(cells):
    mean_cell = cells.reshape(cells.shape[0], 5, 2, *cells.shape[-2:]).mean(dim=1)
    contrast = mean_cell - gaussian_blur(mean_cell.flatten(0, 1), 6).reshape_as(mean_cell)
    density = box_mean(mean_cell.amax(dim=1, keepdim=True), 65, 65)
    return torch.cat((ring_max(mean_cell), contrast, density), dim=1)


def fixed_tissue_context(tissue, tissue_extent):
    mean = tissue.mean(dim=1, keepdim=True)
    features, variances = [], []
    span_x, span_y = tissue_extent
    for side in (65, 129, 257):
        # Convert a cell-pixel radius to the downsampled tissue grid for this pair.
        k_x = 2 * round((side // 2) * span_x * tissue.shape[-1] / CELL_SIZE) + 1
        k_y = 2 * round((side // 2) * span_y * tissue.shape[-2] / CELL_SIZE) + 1
        local_mean = box_mean(mean, k_y, k_x)
        features.append(local_mean)
        if side != 257:
            variance = box_mean(mean.square(), k_y, k_x) - local_mean.square()
            variances.append(variance.clamp_min(0).sqrt())
    return torch.cat((*features, *variances), dim=1)


class LearnedContext(nn.Module):
    def __init__(self, channels, dilations):
        super().__init__()
        # The first layer reads neighboring pixels before the dilated filters.
        self.local = nn.Sequential(nn.Conv2d(channels, 4, 3, padding=1), nn.ReLU())
        self.branches = nn.ModuleList(
            nn.Conv2d(4, width, 3, padding=dilation, dilation=dilation)
            for width, dilation in zip((2, 2, 1), dilations)
        )

    def forward(self, maps):
        local = self.local(maps)
        return torch.cat([branch(local) for branch in self.branches], dim=1)


class Ensemble(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        if variant == "m3_2":
            self.cell_context = LearnedContext(10, (1, 14, 29))
            self.tissue_context = LearnedContext(5, (1, 3, 7))
        self.head = nn.Sequential(
            nn.Conv2d(15 if variant == "pointwise" else 25, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, cells, tissue, base, aligned, box, extent):
        size = cells.shape[-2:]
        features = [cells, resize(aligned, size[0])]
        if self.variant == "m3_1":
            features.extend((fixed_cell_context(cells),
                             crop_to_cell(fixed_tissue_context(tissue, extent), box, size)))
        elif self.variant == "m3_2":
            features.extend((self.cell_context(cells),
                             crop_to_cell(self.tissue_context(tissue), box, size)))
        delta = self.head(torch.cat(features, dim=1)).float()
        return torch.logit(base, eps=1e-6) + delta


class Pairs:
    def __init__(self, dataset, indices, name, cache, device):
        self.dataset = dataset
        self.indices = indices
        self.name = name
        self.cache = cache
        self.device = device
        self.rows = [dataset.source.rows[i] for i in indices]
        self.boxes = [cell_box(dataset.source.metadata[row["pair_id"]]) for row in self.rows]
        self.existing = legacy.CachedPairs(dataset, indices, {})

    def read(self, position):
        index = self.indices[position]
        path = cache_file(self.cache, self.name, self.dataset, index)
        with np.load(path, allow_pickle=False) as data:
            maps = tuple(torch.from_numpy(data[key]).to(self.device, dtype=torch.float32)[None]
                         for key in ("cells", "tissue", "base", "aligned"))
        coords = self.boxes[position]
        box = torch.as_tensor(coords, device=self.device, dtype=torch.float32)[None]
        extent = tuple(coords[2:] - coords[:2])
        points = self.existing.records[position][2]
        return (*maps, box, extent), points

    def __len__(self):
        return len(self.indices)


def sampled_loss(logits, base, target, generator):
    scores, truth = logits.flatten(), target.flatten()
    positive = torch.nonzero(truth >= 0.05, as_tuple=True)[0]
    hard_scores = base.flatten().masked_fill(truth >= 0.05, -1)
    hard_values, hard = hard_scores.topk(4096, sorted=False)
    hard = hard[hard_values >= 0]
    random_pixels = torch.randint(truth.numel(), (4096,), generator=generator).to(truth.device)
    indices = torch.cat((positive, hard, random_pixels))
    weight = 1 + 9 * truth[indices]
    loss = F.binary_cross_entropy_with_logits(scores[indices], truth[indices], reduction="none")
    return (weight * loss).sum() / weight.sum()


def fit(variant, pairs, epochs, device, targets_dir):
    seed_everything(SEED)
    model = Ensemble(variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    path = OUT / f"{variant}.pt"
    history = []
    if path.is_file():
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        print(f"[resume] {variant}: {len(history)} epochs", flush=True)
    model.train()
    for epoch in range(len(history), epochs):
        generator = torch.Generator().manual_seed(SEED + epoch)
        order = torch.randperm(len(pairs), generator=generator).tolist()
        total = 0
        for position in order:
            inputs, points = pairs.read(position)
            pair_id = pairs.rows[position]["pair_id"]
            target_path = targets_dir / f"{pair_id}.npy"
            if target_path.is_file():
                target_np = np.load(target_path, allow_pickle=False)
            else:
                target_np = points_to_heatmaps(points, 6., 0., mode="gaussian").astype(np.float16)
                legacy.save_array(target_path, target_np)
            target = torch.from_numpy(target_np).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs)
            loss = sampled_loss(logits, inputs[2], target, generator)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in {variant}, epoch {epoch + 1}, pair {pair_id}")
            loss.backward()
            optimizer.step()
            total += loss.item()
        value = total / len(pairs)
        history.append({"method": variant, "epoch": epoch + 1, "weighted_BCE": value})
        legacy.atomic_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "history": history, "epoch": epoch + 1, "variant": variant,
        }, path)
        print(f"[{variant}] epoch {epoch + 1}/{epochs} weighted BCE={value:.6f}", flush=True)
    return model.eval(), history


@torch.inference_mode()
def output_maps(models, inputs):
    yield "tta_mean", inputs[2][0]
    for name, model in models.items():
        yield name, model(*inputs)[0].sigmoid()


def setting_fields(setting):
    return dict(zip(("sigma", "min_distance", "threshold"), setting))


@torch.inference_mode()
def tune(pairs, models):
    grid = tuple(itertools.product(*legacy.GRID))
    metrics = {name: {key: CellF1() for key in grid} for name in METHODS}
    for position in range(len(pairs)):
        inputs, points = pairs.read(position)
        truth = np.rint(points).astype(np.int32)
        for name, maps in output_maps(models, inputs):
            for key, pred in legacy.candidate_points(maps, False, grid):
                metrics[name][key].update(pred, truth)
        if (position + 1) % 10 == 0 or position + 1 == len(pairs):
            print(f"[thresholds] val_tune {position + 1}/{len(pairs)}", flush=True)
    chosen, rows = {}, []
    for name, options in metrics.items():
        scores = {key: metric.compute() for key, metric in options.items()}
        chosen[name] = max(grid, key=lambda key: scores[key]["mF1"])
        for key in grid:
            rows.append({"method": name, "selected": key == chosen[name],
                         **setting_fields(key), **scores[key]})
    return chosen, rows


def diagnostic_counts(labels, points, regions):
    distances = nearest_distances(points[:, :2])
    near = (distances >= 14) & (distances < 30)
    counts = {}
    for name, mask in {
        "near": near,
        "near_TC": near & (points[:, 2] == 2),
        "BC_in_CA": (points[:, 2] == 1) & (regions == 1),
        "TC_in_BG": (points[:, 2] == 2) & (regions == 0),
    }.items():
        counts[f"{name}_count"] = int(mask.sum())
        counts[f"{name}_missed"] = int((mask & (labels == "missed")).sum())
        counts[f"{name}_misclassified"] = int((mask & (labels == "misclassified")).sum())
    return counts


def diagnostic_rates(counts):
    result = {}
    for group in ("near", "near_TC", "BC_in_CA", "TC_in_BG"):
        n = counts[f"{group}_count"]
        missed = counts[f"{group}_missed"]
        classified = n - missed
        result[f"{group}_miss_rate"] = missed / n if n else None
        result[f"{group}_misclass_rate_among_matched"] = (
            counts[f"{group}_misclassified"] / classified if classified else None
        )
    return result


def metric_counts(metric):
    result = {}
    for cls, label in ((1, "BC"), (2, "TC")):
        tp, fp, gt = metric.counts[cls]
        result.update({f"TP_{label}": tp, f"FP_{label}": fp, f"FN_{label}": gt - tp})
    return result


def group_settings(chosen):
    common = chosen["tta_mean"]
    return {"shared": dict.fromkeys(METHODS, common), "per_method": chosen}


@torch.inference_mode()
def evaluate(split, pairs, models, chosen, predictions):
    policies = group_settings(chosen)
    totals = {policy: {name: CellF1() for name in METHODS} for policy in policies}
    diagnostics = {(policy, name): {} for policy in policies for name in METHODS}
    per_pair = []
    for position, row in enumerate(pairs.rows):
        inputs, points = pairs.read(position)
        truth = np.rint(points).astype(np.int32)
        regions = pairs.existing.regions(position, truth)
        for name, maps in output_maps(models, inputs):
            parameters = sorted({settings[name] for settings in policies.values()})
            result = dict(legacy.candidate_points(maps, False, parameters))
            for policy, settings in policies.items():
                key = settings[name]
                pred = result[key]
                totals[policy][name].update(pred, truth)
                pair_metric = CellF1()
                pair_metric.update(pred, truth)
                labels, _ = localization_labels(pred, truth)
                counts = diagnostic_counts(labels, truth, regions)
                aggregate = diagnostics[policy, name]
                for label, count in counts.items():
                    aggregate[label] = aggregate.get(label, 0) + count
                identity = {"split": split, "policy": policy, "method": name,
                            "pair_id": row["pair_id"]}
                per_pair.append({**identity, **metric_counts(pair_metric), **counts})
                predictions.write(json.dumps({**identity, "predictions": pred},
                                             separators=(",", ":"), allow_nan=False) + "\n")
        if (position + 1) % 10 == 0 or position + 1 == len(pairs):
            print(f"[metrics] {split} {position + 1}/{len(pairs)}", flush=True)
    rows = []
    for policy, settings in policies.items():
        for name, key in settings.items():
            metric = totals[policy][name]
            counts = diagnostics[policy, name]
            row = {"split": split, "policy": policy, "method": name, **metric.compute(),
                   **metric_counts(metric), **counts, **diagnostic_rates(counts),
                   **setting_fields(key)}
            row["delta_mF1_vs_tta"] = row["mF1"] - totals[policy]["tta_mean"].compute()["mF1"]
            row["delta_mF1_vs_pointwise"] = row["mF1"] - totals[policy]["pointwise"].compute()["mF1"]
            rows.append(row)
            print(f"[result] {split} {policy:10s} {name:10s} "
                  f"mF1={row['mF1']:.6f} FP={row['FP_BC'] + row['FP_TC']}", flush=True)
    return rows, per_pair


@contextmanager
def data_paths(source, splits=("val",)):
    override = os.environ.get("OCELOT_DATA_ROOT")
    candidates = [Path(override).expanduser()] if override else [
        Path(source["data_root"]).expanduser(),
        ROOT / "assets/ocelot2023_v1.0.1",
        Path("/root/m2_overnight/assets/ocelot2023_v1.0.1"),
        Path("/root/repo/data/raw/ocelot2023_v1.0.1"),
        Path("/mnt/data/ocelot2023_v1.0.1"),
    ]
    for candidate in candidates:
        if (candidate / "metadata.json").is_file():
            data_root = candidate.resolve()
            break
    else:
        raise SystemExit("OCELOT v1.0.1 was not found. Set OCELOT_DATA_ROOT to the folder with metadata.json.")

    original, _ = load_metadata(data_root / "metadata.json")
    metadata = {pair: row for pair, row in original.items() if row.get("subset") in splits}
    extensions = {("images", "cell"): ".jpg", ("images", "tissue"): ".jpg",
                  ("annotations", "cell"): ".csv", ("annotations", "tissue"): ".png"}
    with tempfile.TemporaryDirectory(prefix="scoped-data-", dir=OUT) as directory:
        scoped = Path(directory)
        output = scoped / "outputs"
        audit = output / "audit"
        audit.mkdir(parents=True)
        write_json(scoped / "metadata.json", {"sample_pairs": dict(sorted(metadata.items()))})
        report = {"errors": [], "warnings": [], "splits": {}}
        manifest_rows = []
        ids_by_split = {}
        for split in splits:
            pairs = {pair for pair, row in metadata.items() if row["subset"] == split}
            for (prefix, role), suffix in extensions.items():
                folder = Path(prefix) / split / role
                source_folder = (data_root / folder).resolve(strict=True)
                if source_folder != data_root / folder:
                    raise ValueError(f"Unexpected linked directory: {data_root / folder}")
                target = scoped / folder
                target.mkdir(parents=True)
                for pair in pairs:
                    file = source_folder / f"{pair}{suffix}"
                    if file.is_symlink() or not file.is_file():
                        raise ValueError(f"Expected a regular OCELOT file: {file}")
                    (target / file.name).symlink_to(file)
            info, ids, slides, _, rows = inspect_split(scoped, split, metadata, report)
            if report["errors"] or ids != pairs or len(rows) != {"train": 400, "val": 137}[split]:
                raise ValueError(f"{split} audit failed: {report['errors']}, pairs={len(rows)}.")
            ids_by_split[split] = slides
            report["splits"][split] = info
            manifest_rows.extend(rows)
        if ids_by_split.get("train", set()) & ids_by_split.get("val", set()):
            raise ValueError("Official train/val share slide IDs.")
        report["ok"] = True
        write_json(audit / "report.json", report)
        write_csv(audit / "manifest.csv", sorted(manifest_rows, key=lambda row: row["pair_id"]))
        previous = {key: os.environ.get(key) for key in ("OCELOT_DATA_ROOT", "OUTPUT_ROOT")}
        os.environ.update(OCELOT_DATA_ROOT=str(scoped), OUTPUT_ROOT=str(output))
        try:
            yield data_root, sha256(scoped / "metadata.json"), audit / "manifest.csv"
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


SEARCH_COMMIT = "7ff09e00fe7e296f11abad5e4e6b1e1841613e77"
M2_FINALS = {
    "G": ("gaussian", 5.0, 0.9, "final_gaussian_s5_l0.9_ft-from-s6-l0.9"),
    "GA": ("ha", 6.0, 0.5, "final_ha_s6_l0.5_ft-from-s6-l0.9"),
}


def verify_m2(name, path):
    mode, sigma, coefficient, run_name = M2_FINALS[name]
    saved = torch.load(path, map_location="cpu", weights_only=False)
    settings = saved.get("settings", {})
    target = settings.get("cell_target", {})
    base = Path(settings.get("init_from") or "")
    expected = (mode, sigma, coefficient, run_name)
    epoch = saved.get("epoch", -1) + 1
    observed = (target.get("mode"), target.get("sigma"), target.get("lambda"),
                settings.get("run_name"))
    commit = settings.get("git_commit") or ""
    if (observed != expected or not 1 <= epoch <= 10
            or len(commit) < 7 or not SEARCH_COMMIT.startswith(commit)
            or settings.get("epochs") != 10 or settings.get("train_pairs") != 400
            or settings.get("val_pairs") != 137 or settings.get("data_split") is not None
            or settings.get("seed") != 42 or settings.get("label_source") != "official"
            or settings.get("model_config") != "models/M2/config.json"
            or settings.get("limit_train") is not None or settings.get("limit_val") is not None
            or base.parent.name != f"final_{mode}_s6_l0.9_base" or base.name != "best.pt"
            or "model" not in saved or "val" not in saved):
        raise ValueError(f"{name}: invalid final checkpoint {path}; expected {expected}, got {observed}.")
    print(f"[checkpoint] {name} {run_name} epoch={epoch}: {path}", flush=True)
    return {"mode": mode, "sigma": sigma, "lambda": coefficient,
            "run_name": run_name, "best_epoch": epoch, "seed": 42}


def get_existing_checkpoints(source):
    if tuple(source["model_order"]) != MODEL_NAMES:
        raise ValueError(f"Expected source model order {MODEL_NAMES}, got {source['model_order']}")
    paths = []
    for name, env in zip(MODEL_NAMES[2:], OVERRIDES[2:]):
        original = Path(source["checkpoints"][name]).expanduser()
        override = os.environ.get(env)
        if override:
            candidates = [Path(override).expanduser()]
        else:
            candidates = [original, ROOT / "assets" / f"{name}.pt"]
            if original.is_relative_to("/root/ocelot_ensemble"):
                candidates.append(ROOT / original.relative_to("/root/ocelot_ensemble"))
        for path in candidates:
            if path.is_file() and path.stat().st_size > 0:
                with path.open("rb") as handle:
                    if not handle.read(64).startswith(b"version https://git-lfs.github.com/spec/"):
                        paths.append(path.resolve())
                        break
        else:
            tried = "\n  ".join(map(str, candidates))
            raise SystemExit(f"Missing {name} checkpoint. Set {env} to the full best.pt file. Tried:\n  {tried}")
    return paths


def split_val(val, source):
    rows = val.source.rows
    keys = []
    for row in rows:
        slide = row["slide_name"]
        match = re.match(r"^(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})(?:-|$)", slide)
        keys.append(match.group(1) if match else slide)
    src_fit, src_tune = (set(map(str, source.get(f"{part}_ids", []))) for part in ("fit", "tune"))
    all_ids = {row["pair_id"] for row in rows}
    if src_fit and src_tune and not src_fit & src_tune and src_fit | src_tune == all_ids:
        fit = [i for i, row in enumerate(rows) if row["pair_id"] in src_fit]
        tune = [i for i, row in enumerate(rows) if row["pair_id"] in src_tune]
        if not {keys[i] for i in fit} & {keys[i] for i in tune}:
            return fit, tune, "source_groups"
    groups = sorted(set(keys))
    if len(groups) < 2:
        raise ValueError("At least two validation case/slide groups are required.")
    random.Random(SEED).shuffle(groups)
    holdout = set(groups[:max(1, round(len(groups) * 0.25))])
    return ([i for i, key in enumerate(keys) if key not in holdout],
            [i for i, key in enumerate(keys) if key in holdout], "case_grouped_seed42")


def ssl_weights(search):
    expected = json.loads((search / "configs/components.json").read_text())["checkpoint_sha256"]
    override = os.environ.get("SSL_WEIGHTS_PATH")
    candidates = [Path(override).expanduser()] if override else [
        Path("/root/repo/weights/bt_rn50_ep200.torch"),
        Path("/root/m2_overnight/assets/bt_rn50_ep200.torch"),
        ROOT / "assets/bt_rn50_ep200.torch",
    ]
    for path in candidates:
        if path.is_file() and sha256(path) == expected:
            return path.resolve()
    if override:
        raise ValueError("SSL_WEIGHTS_PATH does not match the expected Lunit BT SHA-256.")
    path = OUT / "bt_rn50_ep200.torch"
    if not path.is_file() or sha256(path) != expected:
        tmp = path.with_suffix(".part")
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--output", str(tmp),
                        "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/"
                        "pretrained-weights/bt_rn50_ep200.torch"], check=True)
        if sha256(tmp) != expected:
            raise ValueError("Downloaded Lunit BT checkpoint SHA-256 mismatch.")
        tmp.replace(path)
    return path


def train_m2(source):
    search = Path(os.environ["M3_SEARCH_REPO"])
    train_root = OUT / "m2_training"
    train_root.mkdir(exist_ok=True)
    device = os.environ.get("M3_DEVICE", "cuda")
    weights = ssl_weights(search)
    outputs = {}
    rows = []
    with data_paths(source, ("train", "val")) as (_, _, manifest):
        audit = train_root / "audit"
        audit.mkdir(exist_ok=True)
        for name in ("manifest.csv", "report.json"):
            shutil.copyfile(manifest.parent / name, audit / name)
        env = dict(os.environ, OUTPUT_ROOT=str(train_root), SSL_WEIGHTS_PATH=str(weights))
        for name, (mode, sigma, lam, final) in M2_FINALS.items():
            base = f"final_{mode}_s6_l0.9_base"
            initial = train_root / "m2" / base / "best.pt"
            steps = ((base, 6.0, 0.9, 50, []),
                     (final, sigma, lam, 10,
                      ["--init-from", str(initial), "--epochs", "10",
                       "--freeze-epochs", "0", "--lr-scale", "0.1"]))
            for run, step_sigma, step_lam, epochs, extra in steps:
                folder = train_root / "m2" / run
                last = folder / "last.pt"
                best = folder / "best.pt"
                marker = folder / "training_complete.ok"
                if not (marker.is_file() and best.is_file()):
                    args = [sys.executable, "scripts/train_m2.py", "--config", "configs/m2_search_train.json",
                            "--seed", "42", "--target-mode", mode, "--sigma", str(step_sigma),
                            "--lambda", str(step_lam), "--device", device, "--run-name", run, *extra]
                    if last.is_file():
                        args.extend(("--resume", str(last)))
                    subprocess.run(args, cwd=search, env=env, check=True)
                    state = torch.load(last, map_location="cpu", weights_only=False)
                    if state["epoch"] + 1 != epochs or not best.is_file():
                        raise RuntimeError(f"Incomplete stage: {run}")
                    del state
                    marker.write_text("complete\n")
            outputs[name] = best
            verified = verify_m2(name, best)
            evaluation = folder / "eval_val.json"
            if not evaluation.is_file():
                subprocess.run([sys.executable, "scripts/eval_m2.py", "--checkpoint", str(best),
                                "--split", "val", "--tune", "--device", device],
                               cwd=search, env=env, check=True)
            result = json.loads(evaluation.read_text())
            rows.append({"model": name, "split": "val", "checkpoint": str(best),
                         **verified, "mF1": result["overall"]["mF1"]})
    write_csv(OUT / "M2_training.csv", rows)
    return [outputs[name] for name in M2_FINALS]


def main():
    source_file = Path(os.environ["M3_SOURCE_CONFIG"])
    source = json.loads(source_file.read_text())
    existing = get_existing_checkpoints(source)
    device = torch.device(os.environ.get("M3_DEVICE", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    paths = [*train_m2(source), *existing]
    selected_m2 = {name: verify_m2(name, path) for name, path in zip(MODEL_NAMES, paths)
                   if name in M2_FINALS}
    digests = dict(zip(MODEL_NAMES, map(sha256, paths)))
    with data_paths(source) as (data_root, metadata_hash, manifest):
        run_validation(source_file, source, paths, digests, selected_m2, data_root, metadata_hash, manifest)


def run_validation(source_file, source, paths, digests, selected_m2, data_root, metadata_hash, manifest):
    device = torch.device(os.environ.get("M3_DEVICE", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check nvidia-smi and the selected M3_PYTHON.")
    torch.set_num_threads(2)
    epochs = int(os.environ.get("M3_EPOCHS", "12"))
    if epochs < 1:
        raise ValueError("M3_EPOCHS must be positive.")
    val = TorchOcelotDataset(split="val")
    fit_idx, tune_idx, split_rule = split_val(val, source)
    val_idx = sorted(fit_idx + tune_idx)
    if len(val_idx) != 137 or not fit_idx or not tune_idx:
        raise ValueError("Expected all 137 OCELOT v1.0.1 validation pairs.")
    cache_identity = {
        "model_order": list(MODEL_NAMES),
        "checkpoint_sha256": digests,
        "metadata_sha256": metadata_hash,
        "audit_manifest_sha256": sha256(manifest),
        "repo_commit": os.environ["M3_SOURCE_SHA"],
        "adapter_commit": os.environ["M3_ADAPTER_SHA"],
        "implementation_sha256": sha256(Path(__file__)),
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "device": str(device), "gpu": torch.cuda.get_device_name(device)
                    if device.type == "cuda" else None},
        "tta": "D4: identity plus 7 spatial transforms for both cell and tissue",
        "cache_layout": "1024 BC/TC fp16 per model; 256 full and aligned CA fp16 per model; full cell TTA mean fp32",
    }
    cache_key = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()[:24]
    cache_parent = Path(os.environ.get("M3_CACHE", ROOT / "m3_d4_cache")).expanduser().resolve()
    cache = cache_parent / cache_key
    config = {
        **cache_identity, "data_root": str(data_root), "source_run_json_sha256": sha256(source_file),
        "checkpoint_paths": dict(zip(MODEL_NAMES, map(str, paths))),
        "cache": str(cache), "seed": SEED, "epochs": epochs,
        "fit_ids": [val.source.rows[i]["pair_id"] for i in fit_idx],
        "tune_ids": [val.source.rows[i]["pair_id"] for i in tune_idx],
        "m2_selection_commit": SEARCH_COMMIT,
        "m2_final_checkpoints": selected_m2,
        "evaluation_splits": ["val"],
        "split_rule": split_rule,
        "validation_note": "Combiner uses official val split; it may overlap base checkpoint selection.",
        "fixed_ring_offsets": [list(offset) for offset in RING_OFFSETS],
        "postprocess": "Shared baseline-tuned settings and separately val-tuned settings; never tuned on test.",
    }
    run_file = OUT / "run.json"
    if run_file.is_file() and json.loads(run_file.read_text()) != config:
        raise RuntimeError("Run provenance/config changed. Use a new M3_RUN_ID.")
    write_json(run_file, config)
    create_cache(cache, [("val", val, val_idx)], paths, device)

    targets = OUT / "fit_targets"
    targets.mkdir(exist_ok=True)
    fit_pairs = Pairs(val, fit_idx, "val", cache, device)
    models, history = {}, []
    for variant in VARIANTS:
        model, epochs_completed = fit(variant, fit_pairs, epochs, device, targets)
        models[variant] = model
        history.extend(epochs_completed)
    write_csv(OUT / "epoch_history.csv", history)

    tune_pairs = Pairs(val, tune_idx, "val", cache, device)
    chosen_file = OUT / "postprocess.json"
    if chosen_file.is_file() and (OUT / "validation_grid.csv").is_file():
        selected = json.loads(chosen_file.read_text())["per_method"]
        chosen = {name: tuple(value) for name, value in selected.items()}
        print("[resume] Reusing validation threshold grid", flush=True)
    else:
        chosen, grid = tune(tune_pairs, models)
        write_csv(OUT / "validation_grid.csv", grid)
        write_json(chosen_file, group_settings(chosen))

    pred_path = OUT / "predictions.jsonl.gz"
    temp_path = pred_path.with_suffix(".part")
    with gzip.open(temp_path, "wt", encoding="utf-8", compresslevel=4) as predictions:
        summary, records = evaluate("val_tune", tune_pairs, models, chosen, predictions)
    temp_path.replace(pred_path)
    write_csv(OUT / "per_pair.csv", records)
    write_csv(OUT / "summary.csv", summary)
    (OUT / "README.md").write_text(
        """# OCELOT M3 공식 validation 비교

- G는 공식 train 내부 holdout으로 σ 6 → 5 파인튜닝을 선택하였다. holdout mF1 0.7247, 공식 val mF1 0.7339.
- GA는 같은 절차로 σ 6과 λ 0.9 → 0.5 파인튜닝을 선택하였다. holdout mF1 0.7241, 공식 val mF1 0.7337.
- 위 공식 val 점수는 원 탐색 기록의 단일 모델 점수이다. 새로 학습한 G·GA의 별도 val 결과는 `M2_training.csv`에 있다. 이 실행의 모든 앙상블은 새 G·GA와 기존 M0·M1·M1-1의 D4 TTA를 사용한다.
- `summary.csv`는 TTA 평균, 15채널 결합기, M3-1, M3-2의 `val_tune` 지표이다. `shared`는 TTA 평균으로 고른 동일 후처리, `per_method`는 각 방법의 validation 최적 후처리이다.
- `per_pair.csv`에는 세포 매칭 오류 및 14–30px 누락 집계가 있고, `validation_grid.csv`에는 후처리 탐색 전체 결과가 있다.
- `val_tune`은 후처리 설정 선택에도 사용하였다. 독립 test 점수는 이 실행에 포함되지 않는다.
- `run.json`에서 모델 경로와 SHA-256, validation fit/tune ID, M2 탐색 커밋을 확인할 수 있다.
""", encoding="utf-8")



if __name__ == "__main__":
    main()
M3_PYTHON
cd "$repo"
"$python" "$out/m3_context.py" 2>&1 | tee -a "$out/console.log"
printf 'complete\n' > "$out/complete.ok"
echo "[done] $out/summary.csv"
if [[ "$publish_enabled" == 1 ]]; then publish; fi
exit 0
