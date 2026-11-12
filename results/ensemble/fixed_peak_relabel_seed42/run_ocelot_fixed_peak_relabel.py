import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

root = Path("/root/ocelot_ensemble")
run = root / "experiments/cell_tissue_aspp_seed42"
output = root / "experiments/fixed_peak_relabel_seed42"
config = json.loads((run / "run.json").read_text())
saved = torch.load(run / "fusion.pt", map_location="cpu", weights_only=False)
peak_setting = tuple(saved["postprocess"]["map_mean"])
paths = {name: Path(config["checkpoints"][name]) for name in config["model_order"]}

os.environ["OUTPUT_ROOT"] = str(root / "outputs")
os.environ["OCELOT_DATA_ROOT"] = config["data_root"]
sys.path.insert(0, str(root / "repo"))

# 기존 파이프라인이 같은 위치의 세포·조직 확률맵을 제공한다.
import ocelot_fusion_comparison as fusion
from common.data.tensors import TorchOcelotDataset
from common.metrics.m0_eval import CellF1

device = torch.device("cuda")
torch.manual_seed(config["seed"])
torch.set_num_threads(2)
args = SimpleNamespace(repo=root / "repo", workers=0)
cache = root / "experiments/cache"
cell_channels = 2 * len(paths)
output.mkdir(parents=True, exist_ok=True)


def match_candidates(pred, truth):
    """클래스를 보지 않고 점수 순서대로 15px 내 정답에 일대일 매칭."""
    matched = np.full(len(pred), -1, dtype=int)
    if len(pred) and len(truth):
        neighbors = cKDTree(truth[:, :2]).query_ball_point(pred[:, :2], r=15)
        used = set()
        for i in np.argsort(-pred[:, 3], kind="stable"):
            available = [j for j in neighbors[i] if j not in used]
            if available:
                j = min(available, key=lambda j: np.square(pred[i, :2] - truth[j, :2]).sum())
                matched[i] = j
                used.add(j)
    return matched


def prepare(split, indices=None):
    dataset = TorchOcelotDataset(split=split)
    if indices is None:
        indices = list(range(len(dataset)))
    folders, _ = fusion.cache_maps(args, dataset, indices, paths, cache, device)
    pairs = fusion.CachedPairs(dataset, indices, folders)
    records = []
    for i, (row, _, points) in enumerate(pairs.records):
        truth = np.rint(points).astype(np.int32)
        maps = torch.from_numpy(pairs.maps(i, (slice(None), slice(None)))).to(device)
        cell = maps[:cell_channels].reshape(len(paths), 2, *maps.shape[-2:])
        mean = cell.mean(0)
        _, detections = next(fusion.candidate_points(mean, False, [peak_setting]))
        pred = np.asarray(detections, dtype=np.float32).reshape(-1, 4)
        x, y = pred[:, :2].astype(int).T
        features = maps[:, y, x].T.cpu().numpy()
        margin = (mean[0, y, x] - mean[1, y, x]).cpu().numpy()
        votes = (features[:, :cell_channels:2] >= features[:, 1:cell_channels:2]).sum(1)
        matched = match_candidates(pred, truth)
        records.append({
            "id": str(row["pair_id"]), "pred": pred, "truth": truth,
            "features": features, "margin": margin,
            "disagree": (votes > 0) & (votes < len(paths)),
            "matched": matched, "regions": pairs.regions(i, truth),
        })
    return records


def fit(records, channels):
    features, labels = [], []
    for r in records:
        keep = r["matched"] >= 0
        features.append(r["features"][keep, :channels])
        labels.append(r["truth"][r["matched"][keep], 2] == 1)
    x = torch.from_numpy(np.concatenate(features)).float()
    y = torch.from_numpy(np.concatenate(labels)).float()
    if not len(y) or y.min() == y.max():
        raise RuntimeError("학습 분할에서 BC·TC 매칭 표본을 모두 찾지 못했어.")
    center = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-3)
    x = (x - center) / scale
    torch.manual_seed(config["seed"])
    model = nn.Sequential(nn.Linear(channels, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(15):
        for idx in torch.randperm(len(y)).split(256):
            loss = loss_fn(model(x[idx]).flatten(), y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval(), center, scale


@torch.no_grad()
def predict(records, trained):
    model, center, scale = trained
    return [
        model((torch.from_numpy(r["features"][:, :len(center)]) - center) / scale)
        .flatten().sigmoid().numpy()
        for r in records
    ]


def evaluate(records, probabilities=None, margin_limit=0., confidence=1.):
    metric = CellF1()
    stats = dict(flips=0, wrong_to_right=0, right_to_wrong=0,
                 BC_in_CA_wrong=0, TC_in_BG_wrong=0,
                 oracle_rescuable=0, baseline_wrong_matched=0)

    for i, r in enumerate(records):
        pred = r["pred"].copy()
        baseline = pred[:, 2].astype(int)
        if probabilities is not None:
            prob = probabilities[i]
            eligible = r["disagree"] | (np.abs(r["margin"]) <= margin_limit)
            confident = np.maximum(prob, 1 - prob) >= confidence
            change = eligible & confident
            pred[change, 2] = np.where(prob[change] >= 0.5, 1, 2)
        labels = pred[:, 2].astype(int)
        metric.update(pred.tolist(), r["truth"])

        keep = r["matched"] >= 0
        gt_indices = r["matched"][keep]
        target = r["truth"][gt_indices, 2]
        regions = r["regions"][gt_indices]
        before = baseline[keep] == target
        after = labels[keep] == target
        stats["flips"] += int((labels != baseline).sum())
        stats["wrong_to_right"] += int((~before & after).sum())
        stats["right_to_wrong"] += int((before & ~after).sum())
        stats["BC_in_CA_wrong"] += int(((target == 1) & (regions == 1) & ~after).sum())
        stats["TC_in_BG_wrong"] += int(((target == 2) & (regions == 0) & ~after).sum())
        per_model = r["features"][keep, :cell_channels].reshape(-1, len(paths), 2)
        agrees = (per_model.argmax(axis=2) + 1 == target[:, None]).any(axis=1)
        stats["oracle_rescuable"] += int((~before & agrees).sum())
        stats["baseline_wrong_matched"] += int((~before).sum())

    return {**metric.compute(), **stats}


val_dataset = TorchOcelotDataset(split="val")
row_index = {str(row["pair_id"]): i for i, row in enumerate(val_dataset.source.rows)}
fit_ids = set(map(str, config["fit_ids"]))
tune_ids = set(map(str, config["tune_ids"]))
assert fit_ids and tune_ids and not (fit_ids & tune_ids)
val = prepare("val", [row_index[k] for k in sorted(fit_ids | tune_ids)])
train = [r for r in val if r["id"] in fit_ids]
tune = [r for r in val if r["id"] in tune_ids]
val_baseline = evaluate(tune)
rows = [{"split": "val_tune", "method": "map_mean", **val_baseline}]
models, selected = {}, {}

for name, channels in (("relabel_cell", cell_channels),
                       ("relabel_cell_tissue", cell_channels * 2)):
    models[name] = fit(train, channels)
    probabilities = predict(tune, models[name])
    best, settings = val_baseline, None
    for margin in (0.02, 0.05, 0.10):
        for confidence in (0.55, 0.65, 0.75, 0.85, 0.95):
            score = evaluate(tune, probabilities, margin, confidence)
            if score["mF1"] > best["mF1"]:
                best, settings = score, (margin, confidence)
    selected[name] = settings
    rows.append({"split": "val_tune", "method": name, **best})
    print(f"[Validation] {name} mF1={best['mF1']:.4f} settings={settings}", flush=True)

torch.save({
    "models": {name: {"state_dict": model.state_dict(), "center": center, "scale": scale}
               for name, (model, center, scale) in models.items()},
    "selected": selected, "peak_setting": peak_setting, "model_order": list(paths),
}, output / "relabel.pt")


def save_summary():
    path = output / "summary.csv"
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


save_summary()
test = prepare("test")
test_baseline = evaluate(test)
rows.append({"split": "test", "method": "map_mean", **test_baseline})
for name, trained in models.items():
    settings = selected[name]
    result = (evaluate(test, predict(test, trained), *settings)
              if settings is not None else test_baseline)
    rows.append({"split": "test", "method": name, **result})

save_summary()
for row in rows:
    print(f"{row['split']:8s} {row['method']:22s} mF1={row['mF1']:.4f} "
          f"오답→정답={row['wrong_to_right']} 정답→오답={row['right_to_wrong']} "
          f"CA내 BC 오분류={row['BC_in_CA_wrong']} BG내 TC 오분류={row['TC_in_BG_wrong']}")
print(f"[결과] {output / 'summary.csv'}")
