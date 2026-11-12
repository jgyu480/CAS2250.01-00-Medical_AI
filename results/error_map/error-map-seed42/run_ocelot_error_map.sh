#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONUNBUFFERED=1 GIT_TERMINAL_PROMPT=0
src=${ENSEMBLE_ROOT:-/root/ocelot_ensemble}
repo=${OCELOT_REPO:-"$src/repo"}
previous=${OCELOT_PREVIOUS_RUN:-"$src/experiments/five_models_seed42"}
work=${ERROR_MAP_ROOT:-/root/ocelot_error_map}
id=${RUN_ID:-error-map-seed42}
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
    for name in run_ocelot_error_map.sh summary.csv settings.json error_cnn.pt previews.zip; do
      test -s "$out/$name"
      hash=$(git hash-object -w "$out/$name")
      git update-index --add --cacheinfo "100644,$hash,results/error_map/$id/$name"
    done
    tree=$(git write-tree)
    if [[ -n $listing && $tree == "$(git rev-parse "$parent^{tree}")" ]]; then
      echo "[GitHub] Already uploaded: results/$id"
      exit 0
    fi
    commit=$(git -c user.name='OCELOT Experiments' -c user.email='ocelot-experiments@users.noreply.github.com'       commit-tree "$tree" -p "$parent" -m "Save OCELOT error map experiment $id")
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
if ! cmp -s "$(realpath "$0")" "$out/run_ocelot_error_map.sh"; then
  cp "$(realpath "$0")" "$out/run_ocelot_error_map.sh"
fi
"$py" -u - "$repo" "$previous" "$out" "$helper" <<'PYTHON'
import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

repo, previous, out, helper = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo))
spec = importlib.util.spec_from_file_location("original_ensemble", helper)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
from common.data.tensors import TorchOcelotDataset
from common.paths import resolve_paths
from common.training.m0_loop import seed_everything

ERROR_NAMES = ("missed", "wrong_class", "false_positive")
TILE = 64
SIDE = 16
BUDGETS = (.10, .20)


def mark_counts(xy):
    tile = np.floor(np.asarray(xy).reshape(-1, 2) / TILE).astype(int).clip(0, SIDE - 1)
    return np.bincount(tile[:, 1] * SIDE + tile[:, 0], minlength=SIDE**2).reshape(SIDE, SIDE)


def make_targets(pred, gt):
    """Match correct classes first using the project's 15px score-first rule."""
    used_gt = np.zeros(len(gt), dtype=bool)
    used_pred = np.zeros(len(pred), dtype=bool)
    wrong = []
    if len(pred) and len(gt):
        neighbors = cKDTree(gt[:, :2]).query_ball_point(pred[:, :2], 15, return_sorted=True)
        for cls in (1, 2):
            idx = np.flatnonzero(pred[:, 2] == cls)
            for i in idx[np.argsort(-pred[idx, 3], kind="quicksort")]:
                options = np.asarray(neighbors[i], dtype=int)
                options = options[(~used_gt[options]) & (gt[options, 2] == cls)]
                if len(options):
                    j = options[np.argmin(((gt[options, :2] - pred[i, :2])**2).sum(axis=1))]
                    used_pred[i] = used_gt[j] = True
        # Attribute a wrong-class pair once, in the same score-first order.
        for i in np.argsort(-pred[:, 3], kind="stable"):
            if used_pred[i]:
                continue
            options = np.asarray(neighbors[i], dtype=int)
            options = options[(~used_gt[options]) & (gt[options, 2] != pred[i, 2])]
            if len(options):
                j = options[np.argmin(((gt[options, :2] - pred[i, :2])**2).sum(axis=1))]
                used_pred[i] = used_gt[j] = True
                wrong.append(j)
    return np.stack((mark_counts(gt[~used_gt, :2]), mark_counts(gt[np.asarray(wrong, dtype=int), :2]),
                     mark_counts(pred[~used_pred, :2]))).astype(np.int32)


@torch.inference_mode()
def prepare_inputs(pairs, root, folder, device, setting):
    folder.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, (row, _, gt) in enumerate(pairs.records):
        pair = row["pair_id"]
        item = folder / f"{pair}.npz"
        image_path = root / row["cell_image"]
        samples.append((pair, item, image_path))
        if item.is_file():
            continue
        maps = torch.from_numpy(pairs.maps(i, (slice(None), slice(None)))).to(device)
        maps = maps.reshape(len(pairs.names), 2, 1024, 1024)
        _, points = next(old.candidate_points(maps.mean(0), False, [setting]))
        pred = np.asarray(points, dtype=np.float32).reshape(-1, 4)
        gt = np.rint(gt).astype(np.int32)
        density = mark_counts(pred[:, :2])
        target = make_targets(pred, gt)
        with Image.open(image_path) as image:
            rgb = np.array(image.convert("RGB").resize((256, 256), Image.Resampling.BOX))
        reduced = F.max_pool2d(maps.flatten(0, 1)[None], 4)[0].cpu().numpy()
        # Retain final prediction locations as two channels.
        markers = np.zeros((2, 256, 256), dtype=np.float32)
        if len(pred):
            x, y = (pred[:, :2] // 4).astype(int).T
            np.maximum.at(markers, (pred[:, 2].astype(int)-1, y, x), pred[:, 3])
        features = np.concatenate((rgb.transpose(2, 0, 1).astype(np.float32) / 255, reduced, markers))
        # Disagreement among model scores, averaged over both classes and within each tile.
        disagreement = F.avg_pool2d(maps.std(dim=0, unbiased=False).mean(0)[None, None], TILE)[0, 0]
        temp = item.with_suffix(".part")
        with temp.open("wb") as handle:
            np.savez(handle, features=features.astype(np.float16), target=target,
                     density=density, disagreement=disagreement.cpu().numpy())
        temp.replace(item)
        if (i + 1) % 25 == 0 or i + 1 == len(pairs):
            print(f"[prepare] {folder.name} {i+1}/{len(pairs)}")
    return samples


class Tiles(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        with np.load(self.samples[i][1], allow_pickle=False) as sample:
            return (sample["features"].astype(np.float32), sample["target"].astype(np.float32),
                    sample["density"], sample["disagreement"])


class ErrorCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        blocks = []
        for width in (24, 32, 48):
            blocks += [nn.Conv2d(channels, width, 3, stride=2, padding=1),
                       nn.GroupNorm(8, width), nn.ReLU()]
            channels = width
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Conv2d(channels, len(ERROR_NAMES), 3, padding=1)

    def forward(self, x):
        features = self.encoder(x)
        return F.softplus(self.head(F.avg_pool2d(features, 2))), features


def review_order(score, fraction):
    n = int(np.ceil(SIDE**2 * fraction))
    return np.argsort(-score.ravel(), kind="stable")[:n]


@torch.inference_mode()
def evaluate(model, loader, device, split):
    counts = {}
    total = np.zeros(len(ERROR_NAMES), dtype=np.float64)
    for x, truth, density, disagreement in loader:
        risk = model(x.to(device))[0].cpu().numpy()
        for a, y, d, u in zip(risk, truth.numpy(), density.numpy(), disagreement.numpy()):
            total += y.sum(axis=(1, 2))
            flat = y.reshape(len(ERROR_NAMES), -1)
            for budget in BUDGETS:
                random_rate = np.ceil(SIDE**2 * budget) / SIDE**2
                for name, score in (("error_cnn", a.sum(0)), ("density", d),
                                    ("disagreement", u), ("random_expected", None)):
                    key = (budget, name)
                    if score is None:
                        caught = flat.sum(axis=1) * random_rate
                    else:
                        caught = flat[:, review_order(score, budget)].sum(axis=1)
                    counts[key] = counts.get(key, np.zeros_like(total)) + caught
    rows = []
    for (budget, name), caught in counts.items():
        values = np.r_[caught, caught.sum()]
        denominators = np.r_[total, total.sum()]
        for error, count, denominator in zip((*ERROR_NAMES, "all"), values, denominators):
            rows.append(dict(split=split, method=name, area_target=budget,
                             actual_area=np.ceil(SIDE**2 * budget) / SIDE**2,
                             error=error, caught=float(count), total=int(denominator),
                             recall=float(count / denominator) if denominator else ""))
    return rows


def fit(model, train, tune, device, seed):
    loader = DataLoader(train, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(seed))
    validation = DataLoader(tune, batch_size=4)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    best = -1
    for epoch in range(1, 21):
        model.train()
        for x, y, _, _ in loader:
            opt.zero_grad(set_to_none=True)
            rate, _ = model(x.to(device))
            loss = F.poisson_nll_loss(rate, y.to(device), log_input=False)
            loss.backward()
            opt.step()
        model.eval()
        rows = evaluate(model, validation, device, "val_tune")
        score = next(row["recall"] for row in rows if row["method"] == "error_cnn"
                     and row["error"] == "all" and row["area_target"] == .20)
        if not isinstance(score, float):
            raise ValueError("No error annotations in the tuning split")
        print(f"[train] epoch={epoch:02d} tune_review20_recall={score:.4f}")
        if score > best:
            best = score
            old.atomic_save({"model": model.state_dict(), "epoch": epoch,
                             "channels": train[0][0].shape[0], "tune_recall20": best}, out / "error_cnn.pt")
    saved = torch.load(out / "error_cnn.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    return saved["epoch"]


def overlay(rgb, score, color, pixelated=False):
    max_score = float(np.max(score))
    scaled = np.clip(score / max(max_score, 1e-9), 0, 1)
    mask = Image.fromarray(np.uint8(scaled * 173)).resize(rgb.size,
                     Image.Resampling.NEAREST if pixelated else Image.Resampling.BILINEAR)
    return Image.composite(Image.new("RGB", rgb.size, color), rgb, mask)


def render(model, samples, device):
    with zipfile.ZipFile(out / "previews.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for i, (pair, _, path) in enumerate(samples[:6]):
            x = Tiles(samples)[i][0]
            x = torch.from_numpy(x[None]).to(device).requires_grad_(True)
            risk, feature = model(x)
            total = risk[0].sum(0)
            index = int(total.argmax().item())
            y, xx = divmod(index, SIDE)
            error = int(risk[0, :, y, xx].argmax().item())
            gradient, = torch.autograd.grad(risk[0, error, y, xx], feature)
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (weights * feature).sum(dim=1).relu()[0].detach().cpu().numpy()
            score = total.detach().cpu().numpy()
            chosen = review_order(score, .20)
            risk_top = np.zeros_like(score)
            risk_top.flat[chosen] = score.flat[chosen]
            with Image.open(path) as image:
                rgb = image.convert("RGB").resize((512, 512))
            panels = [rgb, overlay(rgb, risk_top, "#e53122", True), overlay(rgb, cam, "#ef9b23")]
            for panel in panels[1:]:
                ImageDraw.Draw(panel).rectangle((xx * 32, y * 32, xx * 32 + 31, y * 32 + 31),
                                               outline="white", width=2)
            canvas = Image.new("RGB", (1536, 546), "#20242b")
            captions = [f"Pair {pair} | H&E", "Predicted errors | top 20% tiles",
                        f"Grad-CAM | {ERROR_NAMES[error]} | outlined tile"]
            draw = ImageDraw.Draw(canvas)
            for j, (panel, caption) in enumerate(zip(panels, captions)):
                canvas.paste(panel, (j * 512, 34))
                draw.text((j * 512 + 12, 10), caption, fill="white")
            preview_dir = out / "previews"
            preview_dir.mkdir(exist_ok=True)
            filename = preview_dir / f"{pair}.png"
            canvas.save(filename)
            archive.write(filename, filename.name)


def main():
    config = json.loads((previous / "run.json").read_text())
    seed = config["seed"]
    seed_everything(seed)
    device = torch.device("cuda")
    model_paths = {name: Path(config["checkpoints"][name]).resolve() for name in config["model_order"]}
    if len(model_paths) != 5 or any(old.fingerprint([path], "") != config["checkpoint_identity"][name]
                                   for name, path in model_paths.items()):
        raise ValueError("Use the original five frozen checkpoints")
    root = resolve_paths()["data_root"]
    if root != Path(config["data_root"]).resolve():
        raise ValueError("Use the previous ensemble's data root")
    with (previous / "summary.csv").open(newline="") as handle:
        row = next(row for row in csv.DictReader(handle) if row["method"] == "mean" and row["split"] == "val_tune")
    setting = (float(row["sigma"]), int(row["min_distance"]), float(row["threshold"]))
    val = TorchOcelotDataset(split="val")
    lookup = {row["pair_id"]: i for i, row in enumerate(val.source.rows)}
    fit_idx = [lookup[i] for i in config["fit_ids"]]
    tune_idx = [lookup[i] for i in config["tune_ids"]]
    if {val.source.rows[i]["slide_name"] for i in fit_idx} & {val.source.rows[i]["slide_name"] for i in tune_idx}:
        raise ValueError("Fit and tuning WSIs overlap")
    args = SimpleNamespace(repo=repo, workers=0)
    manifest = resolve_paths()["outputs"] / "audit/manifest.csv"
    cache_key = old.fingerprint([previous / "run.json", previous / "summary.csv", manifest,
                                 root / "metadata.json", out / "run_ocelot_error_map.sh"], "error-input-v1")
    cache = out / "cache" / cache_key
    folders, _ = old.cache_maps(args, val, sorted(fit_idx + tune_idx), model_paths, previous.parent / "cache", device)
    train = Tiles(prepare_inputs(old.CachedPairs(val, fit_idx, folders), root, cache / "val_fit", device, setting))
    tune = Tiles(prepare_inputs(old.CachedPairs(val, tune_idx, folders), root, cache / "val_tune", device, setting))
    model = ErrorCNN(5 * 2 + 3 + 2).to(device)
    epoch = fit(model, train, tune, device, seed)
    rows = evaluate(model, DataLoader(tune, batch_size=4), device, "val_tune")
    test = TorchOcelotDataset(split="test")
    idx = list(range(len(test)))
    folders, _ = old.cache_maps(args, test, idx, model_paths, previous.parent / "cache", device)
    samples = prepare_inputs(old.CachedPairs(test, idx, folders), root, cache / "test", device, setting)
    rows += evaluate(model, DataLoader(Tiles(samples), batch_size=4), device, "test")
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    settings = dict(tile=TILE, error_types=ERROR_NAMES, model_order=list(model_paths), fit_ids=config["fit_ids"],
                    tune_ids=config["tune_ids"], seed=seed, epoch=epoch, baseline_postprocess=setting,
                    preview_ids=[pair for pair, _, _ in samples[:6]])
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    render(model, samples, device)
    for row in rows:
        if row["split"] == "test" and row["error"] == "all":
            print(f"[test] {row['method']} top={row['area_target']:.0%} caught={row['caught']:.1f}/{row['total']}")
    (out / "complete.ok").touch()
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
PYTHON
publish
