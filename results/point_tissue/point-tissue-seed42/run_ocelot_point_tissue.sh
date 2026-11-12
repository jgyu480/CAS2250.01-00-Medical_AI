#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONUNBUFFERED=1 GIT_TERMINAL_PROMPT=0
src=${ENSEMBLE_ROOT:-/root/ocelot_ensemble}
repo=${OCELOT_REPO:-"$src/repo"}
previous=${OCELOT_PREVIOUS_RUN:-"$src/experiments/five_models_seed42"}
work=${POINT_TISSUE_ROOT:-/root/ocelot_point_tissue}
id=${RUN_ID:-point-tissue-seed42}
[[ $id =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$ ]] || exit 2
mode=${1:-}
[[ $# -le 1 && ( -z $mode || $mode == --publish-only ) ]] || exit 2
out="$work/$id"
mkdir -p "$out"
exec 9>"$out/run.lock"
flock -n 9 || { echo "[running] $out"; exit 1; }
trap 'echo "[failed] line $LINENO. Results and cache: $out" >&2' ERR
cd "$repo"
base=14f930ac47d69bb9abc6d86c6abcb5bdaf35a8da
if [[ $(git rev-parse HEAD) != "$base" ]]; then
  echo "[error] Set OCELOT_REPO to the original ensemble checkout at $base." >&2
  exit 1
fi
if ! git diff --quiet HEAD -- models common configs scripts; then
  echo '[error] Model, dataset or evaluation files in the checkout differ from the original commit.' >&2
  exit 1
fi

publish() {
  local ref="refs/heads/results/$id" parent listing name hash tree commit index
  test -f "$out/complete.ok"
  listing=$(git ls-remote origin "$ref")
  parent=$base
  if [[ -n $listing ]]; then
    git fetch --no-tags origin "$ref"
    parent=$(git rev-parse FETCH_HEAD)
  fi
  index=$(mktemp "$out/git-index.XXXXXX")
  rm -f "$index"
  (
    export GIT_INDEX_FILE="$index"
    trap 'rm -f "$GIT_INDEX_FILE"' EXIT
    git read-tree "$parent"
    for name in run_ocelot_point_tissue.sh summary_cell.csv summary_tissue.csv settings.json point_mlp.pt; do
      test -s "$out/$name"
      hash=$(git hash-object -w "$out/$name")
      git update-index --add --cacheinfo "100644,$hash,results/point_tissue/$id/$name"
    done
    tree=$(git write-tree)
    if [[ -n $listing && $tree == "$(git rev-parse "$parent^{tree}")" ]]; then
      echo "[GitHub] Already uploaded: results/$id"
      exit 0
    fi
    commit=$(git -c user.name='OCELOT Experiments' -c user.email='ocelot-experiments@users.noreply.github.com'       commit-tree "$tree" -p "$parent" -m "Save OCELOT point and tissue ensemble $id")
    if command -v gh >/dev/null && gh auth status --hostname github.com >/dev/null 2>&1; then
      git -c 'credential.https://github.com.helper='           -c 'credential.https://github.com.helper=!gh auth git-credential'           push origin "$commit:$ref"
    else
      git push origin "$commit:$ref"
    fi
    echo "[GitHub] Uploaded: results/$id ($commit)"
  )
}
if [[ $mode == --publish-only || -f "$out/complete.ok" ]]; then
  publish
  exit 0
fi

py=${ENSEMBLE_PYTHON:-}
if [[ -z $py ]]; then
  for candidate in "$src/venv/bin/python" /opt/conda/envs/cas2250-medical-ai/bin/python python3; do
    if "$candidate" -c 'import torch, torchvision, numpy, scipy, PIL; assert torch.cuda.is_available()' >/dev/null 2>&1; then
      py=$candidate
      break
    fi
  done
fi
: "${py:?Set ENSEMBLE_PYTHON to the Python used by the previous CUDA ensemble}"
test -s "$previous/run.json"
test -s "$previous/summary.csv"
export OUTPUT_ROOT=${OCELOT_AUDIT_ROOT:-"$src/outputs"}
export OCELOT_DATA_ROOT=${OCELOT_DATA_ROOT:-$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["data_root"])' "$previous/run.json")}
if [[ ! -s "$OUTPUT_ROOT/audit/manifest.csv" || ! -s "$OUTPUT_ROOT/audit/report.json" ]]; then
  "$py" -u scripts/check_data.py
fi
helper="$src/ocelot_ensemble.py"
if [[ ! -s $helper ]]; then
  revision=5fe93e322f235614aa7dc0054bff084b949d9eeb
  git fetch --no-tags origin "$revision"
  helper="$out/original_ensemble.py"
  git show "$revision:results/ensemble/five_models_seed42/run_ocelot_ensemble.sh" |
    sed -n "/^#!\/usr\/bin\/env python3$/,/^PYTHON_OCELOT_ENSEMBLE$/{ /^PYTHON_OCELOT_ENSEMBLE$/d; p; }" > "$helper"
  test -s "$helper"
fi
if ! cmp -s "$(realpath "$0")" "$out/run_ocelot_point_tissue.sh"; then
  cp "$(realpath "$0")" "$out/run_ocelot_point_tissue.sh"
fi
"$py" -u - "$repo" "$previous" "$out" "$helper" <<'PYTHON'
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

repo, previous, out, helper = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo))
spec = importlib.util.spec_from_file_location("original_ensemble", helper)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

from common.data.tensors import TorchOcelotDataset, collate_pairs
from common.metrics.m0_eval import CellF1, TissueIoU, gaussian_blur
from common.training.m0_loop import autocast, prepare, seed_everything
from scripts.diagnose_m0 import localization_labels, nearest_distances
from models.M0_1.model import M0_1

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_csv(name, rows):
    path = out / name
    with path.with_suffix(".part").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".part").replace(path)

def checkpoint(path):
    if not path.is_file() or path.stat().st_size < 1000:
        raise SystemExit(f"Missing full checkpoint: {path}")
    return path.resolve()

def indices(dataset, ids):
    by_id = {row["pair_id"]: i for i, row in enumerate(dataset.source.rows)}
    return [by_id[key] for key in ids]

def proposals(maps):
    # A model proposes one spatial peak while preserving both class scores.
    strength = gaussian_blur(maps, 1.).amax(dim=0)
    peaks = F.max_pool2d(strength[None, None], 7, stride=1, padding=3)[0, 0]
    y, x = ((strength == peaks) & (strength > .05)).nonzero(as_tuple=True)
    return torch.stack((x, y, strength[y, x]), dim=1).cpu().numpy()

def groups(proposed, radius):
    """Merge shortest edges first, with a diameter bound and one peak/model/group."""
    points = np.concatenate([p[:, :2] for p in proposed])
    models = np.repeat(np.arange(len(proposed)), [len(p) for p in proposed])
    scores = np.concatenate([p[:, 2] for p in proposed])
    if not len(points):
        return []
    edges = cKDTree(points).query_pairs(radius, output_type="ndarray")
    edges = edges[models[edges[:, 0]] != models[edges[:, 1]]]
    lengths = ((points[edges[:, 0]] - points[edges[:, 1]]) ** 2).sum(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0], lengths))
    parent = list(range(len(points)))
    members = {i: [i] for i in parent}
    voters = {i: {int(models[i])} for i in parent}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in edges[order]:
        a, b = root(int(i)), root(int(j))
        if a == b or voters[a] & voters[b]:
            continue
        delta = points[members[a], None] - points[None, members[b]]
        if ((delta ** 2).sum(axis=2) > radius ** 2).any():
            continue
        parent[b] = a
        members[a].extend(members.pop(b))
        voters[a].update(voters.pop(b))
    return [(np.average(points[m], axis=0, weights=scores[m]), m, models, points)
            for m in members.values()]

def grouped_features(proposed, maps, radius):
    merged = groups(proposed, radius)
    n = len(proposed)
    xy = np.empty((len(merged), 2), dtype=np.float32)
    features = np.zeros((len(merged), 3 * n + 1), dtype=np.float32)
    votes = np.zeros((len(merged), 2), dtype=np.float32)
    for i, (center, members, models, points) in enumerate(merged):
        xy[i] = center
        x, y = np.rint(center).astype(int)
        for m, score_map in enumerate(maps):
            features[i, 2*m:2*m+2] = score_map[:, y, x]
        for j in members:
            m = models[j]
            px, py = points[j].astype(int)
            scores = maps[m][:, py, px]
            features[i, 2*m:2*m+2] = scores
            features[i, 2*n + m] = 1
            votes[i] += scores / n
        features[i, -1] = np.linalg.norm(points[members] - center, axis=1).mean() / radius
    return xy, features, votes

def labels_for(xy, gt):
    # Assign one positive candidate per annotated cell. Remaining candidates are background.
    labels = np.zeros(len(xy), dtype=np.int64)
    if not len(xy) or not len(gt):
        return labels
    edges = cKDTree(xy).sparse_distance_matrix(cKDTree(gt[:, :2]), 15, output_type="ndarray")
    occupied = set()
    for edge in np.sort(edges, order=["v", "i", "j"]):
        i, j = int(edge["i"]), int(edge["j"])
        if not labels[i] and j not in occupied:
            labels[i] = int(gt[j, 2])
            occupied.add(j)
    return labels

def predictions(xy, scores, background=None):
    label = scores.argmax(axis=1) + 1
    conf = scores.max(axis=1)
    keep = np.ones(len(xy), dtype=bool) if background is None else conf > background
    return np.column_stack((xy, label, conf))[keep]

def filtered(points, thresholds):
    return points[points[:, 3] >= np.asarray(thresholds)[points[:, 2].astype(int) - 1]]

def tune(records):
    grid = (.01, .02, .03, .04, .05, .075, .1, .15, .2, .3, .4, .5, .65, .8, .9)
    # Each class has independent counts and a fixed class assignment.
    from common.metrics.m0_eval import match_counts
    picked = []
    for cls in (1, 2):
        totals = np.zeros((len(grid), 3), dtype=np.int64)
        for gt, points in records:
            pred = points[points[:, 2] == cls]
            truth = gt[gt[:, 2] == cls]
            for j, cutoff in enumerate(grid):
                totals[j] += match_counts(pred[pred[:, 3] >= cutoff], truth, cls)
        tp, fp, truth = totals.T
        scores = 2 * tp / np.maximum(tp + fp + truth, 1)
        picked.append(grid[int(scores.argmax())])
    return picked

def localization_counts(records, thresholds):
    close_miss = close_total = position_fp = 0
    for gt, all_points in records:
        points = filtered(all_points, thresholds)
        distances = nearest_distances(gt[:, :2])
        close = (gt[:, 2] == 2) & (distances >= 14) & (distances < 30)
        matched, extra = localization_labels(points, gt)
        close_miss += ((matched == "missed") & close).sum().item()
        close_total += close.sum().item()
        position_fp += extra["duplicate"] + extra["false_positive"]
    return close_miss, close_total, position_fp

def tune_budget(records, thresholds, cap):
    choices = []
    for scale in (.50, .65, .8, .9, 1., 1.1, 1.25, 1.5, 2., 3., 6.):
        adjusted = tuple(min(1.01, value * scale) for value in thresholds)
        missed, _, fp = localization_counts(records, adjusted)
        if fp <= cap:
            choices.append((missed, -fp, adjusted))
    return min(choices)[2] if choices else (1.01, 1.01)

def cell_row(split, method, records, thresholds, radius=""):
    metric = CellF1()
    for gt, points in records:
        metric.update(filtered(points, thresholds), gt)
    close_miss, close_total, position_fp = localization_counts(records, thresholds)
    row = dict(split=split, method=method, **metric.compute(),
               FP_BC=metric.counts[1][1], FP_TC=metric.counts[2][1],
               location_FP=position_fp, near_TC_missed=close_miss, near_TC_count=close_total,
               near_TC_miss=close_miss / close_total if close_total else "",
               threshold_BC=thresholds[0] if method != "map_mean" else "",
               threshold_TC=thresholds[1] if method != "map_mean" else "", radius=radius)
    print(f"[cell] {split} {method} mF1={row['mF1']:.4f} near TC missed={close_miss}/{close_total} loc FP={position_fp}")
    return row

def make_records(pairs, device, mean_settings, radii=(4, 6, 8)):
    result = []
    for i, (_, _, gt) in enumerate(pairs.records):
        gt = np.rint(gt).astype(np.int32)
        maps = torch.from_numpy(pairs.maps(i, (slice(None), slice(None)))).to(device)
        maps = maps.reshape(len(pairs.names), 2, 1024, 1024)
        _, mean = next(old.candidate_points(maps.mean(dim=0), False, [mean_settings]))
        proposed = [proposals(m) for m in maps]
        raw = maps.cpu().numpy()
        options = {r: grouped_features(proposed, raw, r) for r in radii}
        result.append((gt, np.asarray(mean).reshape(-1, 4), options))
        if (i + 1) % 25 == 0 or i + 1 == len(pairs):
            print(f"[cell candidates] {i+1}/{len(pairs)}")
    return result

def point_records(records, radius, model=None):
    result = []
    for gt, _, options in records:
        xy, features, votes = options[radius]
        if model is None:
            points = predictions(xy, votes)
        else:
            with torch.no_grad():
                p = model(torch.from_numpy(features)).softmax(dim=1).numpy()
            points = predictions(xy, p[:, 1:], p[:, 0])
        result.append((gt, points))
    return result

def fit_mlp(records, radius, seed, n_models):
    xy_features, targets = [], []
    for gt, _, options in records:
        xy, features, _ = options[radius]
        xy_features.append(features)
        targets.append(labels_for(xy, gt))
    x = torch.from_numpy(np.concatenate(xy_features))
    y = torch.from_numpy(np.concatenate(targets))
    if not len(y) or torch.unique(y).numel() != 3:
        raise ValueError("MLP fitting requires background, BC and TC candidates")
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], 32), nn.ReLU(), nn.Linear(32, 3))
    opt = torch.optim.AdamW(model.parameters(), lr=.002)
    rng = torch.Generator().manual_seed(seed)
    for epoch in range(12):
        for idx in torch.randperm(len(y), generator=rng).split(1024):
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if (epoch + 1) % 4 == 0:
            print(f"[point MLP] epoch {epoch+1}/12 loss={loss.item():.4f}")
    return model.eval()

def tissue_model(name, path):
    if name != "M0-1":
        model, size, _, _, _, amp = old.load_base(path)
        return model, size, amp
    saved = torch.load(path, map_location="cpu", weights_only=False)
    settings = saved["settings"]
    config = read_json(repo / settings["model_config"])
    if config["model"] != "M0_1":
        raise ValueError(f"Expected M0_1 checkpoint: {path}")
    model = M0_1(config)
    model.load_state_dict(saved["model"], strict=True)
    return model.eval(), settings["size"], settings.get("amp", True)

@torch.inference_mode()
def tissue_eval(split, dataset, idx, paths, device, methods, batch_size=1):
    # Store just CA as fp16. BG = 1 - CA conditional on the two supervised classes.
    cache = out / "tissue_cache" / split
    cache.mkdir(parents=True, exist_ok=True)
    folders = {}
    for name, path in paths.items():
        key = old.fingerprint([path, dataset.source.root / "metadata.json", out / "run_ocelot_point_tissue.sh"],
                              (str(device), str(repo), name))
        folder = cache / (name + "-" + key)
        folder.mkdir(exist_ok=True)
        folders[name] = folder
        pending = [i for i in idx if not (folder / (dataset.source.rows[i]["pair_id"] + ".npy")).is_file()]
        if not pending:
            continue
        model, size, amp = tissue_model(name, path)
        model.to(device)
        loader = DataLoader(Subset(dataset, pending), batch_size=batch_size, collate_fn=collate_pairs)
        for n, batch in enumerate(loader, 1):
            images, _, boxes = prepare(batch, device, size)
            cast, _ = autocast(device, amp)
            with cast:
                logits = model(*images, boxes)["tissue_logits"][:, :2].float()
            prob = logits.softmax(dim=1)
            prob = F.interpolate(prob, (1024, 1024), mode="bilinear", align_corners=False)
            for pair, ca in zip(batch["pair_id"], prob[:, 1]):
                old.save_array(folder / (pair + ".npy"), ca.cpu().numpy().astype(np.float16))
            if n % 25 == 0 or n == len(loader):
                print(f"[tissue maps] {split} {name} {n}/{len(loader)}")
        del model
    from PIL import Image
    from common.data.labels import tissue_to_target
    metrics = {name: TissueIoU() for name in methods}
    for i in idx:
        row = dataset.source.rows[i]
        with Image.open(dataset.source.root / row["tissue_mask"]) as image:
            target = torch.from_numpy(tissue_to_target(np.asarray(image))).to(device)[None]
        arrays = {name: torch.from_numpy(np.load(folder / (row["pair_id"] + ".npy"), allow_pickle=False)).to(device).float()
                  for name, folder in folders.items()}
        for name, members in methods.items():
            ca = torch.stack([arrays[m] for m in members]).mean(dim=0)
            metrics[name].update(torch.stack((1 - ca, ca))[None], target)
    result = []
    for name, metric in metrics.items():
        row = dict(split=split, method=name, **metric.compute())
        print(f"[tissue] {split} {name} mIoU={row['mIoU']:.4f}")
        result.append(row)
    return result

def main():
    previous_config = read_json(previous / "run.json")
    if previous_config["smoke"]:
        raise ValueError("Previous run must be the full ensemble")
    paths = {name: checkpoint(Path(previous_config["checkpoints"][name])) for name in previous_config["model_order"]}
    if any(old.fingerprint([path], "") != previous_config["checkpoint_identity"][name] for name, path in paths.items()):
        raise ValueError("Previous checkpoint identity changed. Restore the original checkpoints.")
    default_m01 = Path("/root/m0_1_runs/m0-1-seed42-20261111-160937/outputs/m0_1/m0-1-seed42-20261111-160937/best.pt")
    tissue_paths = {name: paths[name] for name in ("M0", "M1", "M1-1")}
    tissue_paths["M0-1"] = checkpoint(Path(os.environ.get("M01_CHECKPOINT", str(default_m01))))
    if Path(previous_config["data_root"]).resolve() != Path(os.environ["OCELOT_DATA_ROOT"]).resolve():
        raise ValueError("Use the previous ensemble's data root to reproduce the baseline")
    seed = previous_config["seed"]
    seed_everything(seed)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    with (previous / "summary.csv").open(newline="", encoding="utf-8") as handle:
        mean = next(row for row in csv.DictReader(handle) if row["split"] == "val_tune" and row["method"] == "mean")
    mean_settings = float(mean["sigma"]), int(mean["min_distance"]), float(mean["threshold"])
    val = TorchOcelotDataset(split="val")
    fit_idx = indices(val, previous_config["fit_ids"])
    tune_idx = indices(val, previous_config["tune_ids"])
    if ({val.source.rows[i]["slide_name"] for i in fit_idx} &
        {val.source.rows[i]["slide_name"] for i in tune_idx}):
        raise ValueError("Overlapping fit and tune slides")
    args = SimpleNamespace(repo=repo, workers=0)
    cell_cache = previous.parent / "cache"
    folders, _ = old.cache_maps(args, val, sorted(fit_idx + tune_idx), paths, cell_cache, device)
    tune_maps = make_records(old.CachedPairs(val, tune_idx, folders), device, mean_settings)
    rows = [cell_row("val_tune", "map_mean", [(gt, p) for gt, p, _ in tune_maps], (0, 0))]
    trials = []
    for radius in (4, 6, 8):
        records = point_records(tune_maps, radius)
        thresholds = tune(records)
        row = cell_row("val_tune", f"point_vote_r{radius}", records, thresholds, radius)
        rows.append(row)
        trials.append((row["mF1"], radius, thresholds))
    _, radius, vote_thresholds = max(trials, key=lambda trial: trial[0])
    fit_maps = make_records(old.CachedPairs(val, fit_idx, folders), device, mean_settings, (radius,))
    model = fit_mlp(fit_maps, radius, seed, len(paths))
    del fit_maps
    mlp_records = point_records(tune_maps, radius, model)
    mlp_thresholds = tune(mlp_records)
    rows.append(cell_row("val_tune", "point_mlp", mlp_records, mlp_thresholds, radius))
    budget = rows[0]["location_FP"]
    vote_records = point_records(tune_maps, radius)
    vote_budget = tune_budget(vote_records, vote_thresholds, budget)
    mlp_budget = tune_budget(mlp_records, mlp_thresholds, budget)
    rows.append(cell_row("val_tune", "point_vote_val_budget", vote_records, vote_budget, radius))
    rows.append(cell_row("val_tune", "point_mlp_val_budget", mlp_records, mlp_budget, radius))
    del tune_maps
    torch.save({"model": model.state_dict(), "model_order": list(paths), "radius": radius,
                "thresholds": mlp_thresholds}, out / "point_mlp.pt")
    tissue_methods = {f"base:{m}": [m] for m in tissue_paths}
    tissue_methods["m0_family_mean"] = ["M0", "M1", "M1-1"]
    tissue_methods["m0_family_plus_m0_1"] = list(tissue_paths)
    tissue_rows = tissue_eval("val_tune", val, tune_idx, tissue_paths, device, tissue_methods)
    settings = dict(previous_run=str(previous), seed=seed, fit_ids=previous_config["fit_ids"],
                    tune_ids=previous_config["tune_ids"], cell_checkpoints={k:str(v) for k,v in paths.items()},
                    tissue_checkpoints={k:str(v) for k,v in tissue_paths.items()},
                    baseline_postprocess=mean_settings, vote_radius=radius, vote_thresholds=vote_thresholds,
                    mlp_thresholds=mlp_thresholds, val_location_fp_budget=budget,
                    vote_budget_thresholds=vote_budget, mlp_budget_thresholds=mlp_budget,
                    tissue_methods=tissue_methods,
                    preferred_tissue_method=max(tissue_rows, key=lambda row: row["mIoU"])["method"])
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    write_csv("summary_cell.csv", rows)
    write_csv("summary_tissue.csv", tissue_rows)
    test = TorchOcelotDataset(split="test")
    test_idx = list(range(len(test)))
    folders, _ = old.cache_maps(args, test, test_idx, paths, cell_cache, device)
    records = make_records(old.CachedPairs(test, test_idx, folders), device, mean_settings, (radius,))
    rows.append(cell_row("test", "map_mean", [(gt,p) for gt,p,_ in records], (0, 0)))
    test_vote = point_records(records, radius)
    test_mlp = point_records(records, radius, model)
    rows.append(cell_row("test", "point_vote", test_vote, vote_thresholds, radius))
    rows.append(cell_row("test", "point_mlp", test_mlp, mlp_thresholds, radius))
    rows.append(cell_row("test", "point_vote_val_budget", test_vote, vote_budget, radius))
    rows.append(cell_row("test", "point_mlp_val_budget", test_mlp, mlp_budget, radius))
    write_csv("summary_cell.csv", rows)
    tissue_rows.extend(tissue_eval("test", test, test_idx, tissue_paths, device, tissue_methods))
    write_csv("summary_tissue.csv", tissue_rows)
    (out / "complete.ok").touch()
    print(f"[done] {out}")

if __name__ == "__main__":
    main()
PYTHON
publish
