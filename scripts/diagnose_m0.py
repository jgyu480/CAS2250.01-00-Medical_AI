"""M0 checkpoint diagnostics. Run from the repository root; no optimizer updates.

python scripts/diagnose_m0.py audit --run outputs/m0/M0_official_default_seed42
python scripts/diagnose_m0.py cohorts --run outputs/m0/M0_official_default_seed42
python scripts/diagnose_m0.py replay --run outputs/m0/M0_official_default_seed42
python scripts/diagnose_m0.py gradients --run outputs/m0/M0_official_default_seed42

CSV values are ratios on 0..1, coordinates and distances are in cell-image px.
Outputs go to a new timestamped analysis directory, preserving received files.
Designed against repository commit 9f9b93c. Validation only for design work.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def init(args):
    root = Path(args.project).expanduser().resolve()
    if not (root / "models/M0/model.py").is_file():
        raise SystemExit("[FAIL] 프로젝트 폴더에서 실행하거나 --project 경로를 지정하세요.")
    sys.path.insert(0, str(root))
    run = Path(args.run).expanduser()
    if not run.is_absolute():
        run = root / run
    if not run.is_dir():
        raise SystemExit(f"[FAIL] 결과 폴더 없음: {run}")
    tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = run / "analysis" / f"diagnostics_{args.command}_{tag}"
    out.mkdir(parents=True)
    args.root, args.run, args.out = root, run, out
    save_json(out / "request.json", {"command": args.command,
              "run": str(run), "project": str(root), "split": "val",
              "reference_commit": "9f9b93c", "note": "원본 결과와 가중치는 수정하지 않음"})
    print(f"[시작] {args.command} → {out}", flush=True)


def load_state(args):
    import torch
    # Uses the teammate's trusted checkpoint, as the repository training code does.
    state = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    if not {"model", "settings", "epoch", "val"} <= state.keys():
        raise ValueError("best.pt 필수 항목 누락")
    return state


def model_config(args, state):
    path = args.root / state["settings"]["model_config"]
    return read_json(path)


def load_model(args):
    import torch
    from common.backbones.ssl_resnet import SSLResNet50
    from common.training.m0_loop import pick_device
    from models.M0.model import M0
    state = load_state(args)
    # Complete trained weights are in best.pt; no original SSL weight file is needed.
    model = M0(SSLResNet50(), SSLResNet50(), model_config(args, state))
    model.load_state_dict(state["model"], strict=True)
    device = pick_device(args.device)
    return model.to(device).eval(), state, device


def normalize_records(records):
    ids = []
    for r in records:
        r["pair_id"] = str(r["pair_id"]).zfill(3)
        ids.append(r["pair_id"])
        for name, width in [("truths", 3), ("predictions", 4)]:
            a = np.asarray(r[name], dtype=float).reshape(-1, width)
            if (not np.isfinite(a).all() or (a[:, :2] < 0).any()
                    or (a[:, :2] >= 1024).any() or not np.isin(a[:, 2], [1, 2]).all()):
                raise ValueError(f"{r['pair_id']}: {name} 좌표/클래스 오류")
            if name == "predictions" and ((a[:, 3] < 0) | (a[:, 3] > 1)).any():
                raise ValueError("예측 점수가 확률 범위를 벗어남")
        if len(r["truths"]) != r["num_gt"] or len(r["predictions"]) != r["num_pred"]:
            raise ValueError(f"{r['pair_id']}: 세포 개수 기록 불일치")
    if len(ids) != len(set(ids)):
        raise ValueError("평가 ID 중복")
    return records


def cached_records(args):
    path = Path(args.records) if args.records else args.run / "eval_val_per_image.json"
    if not path.is_file():
        raise SystemExit("[FAIL] eval_val_per_image.json 없음. 먼저 replay를 실행하세요.")
    return normalize_records(read_json(path))


def official_counts(predictions, truths):
    """Numpy-only equivalent of common.metrics.m0_eval.match_counts (15px)."""
    result = []
    for cls in (1, 2):
        p = np.array([p for p in predictions if p[2] == cls], dtype=np.float32).reshape(-1, 4)
        g = np.array([g[:2] for g in truths if g[2] == cls], dtype=np.float32).reshape(-1, 2)
        tp, fp = 0, 0
        if len(p) and len(g):
            d = np.linalg.norm(p[:, None, :2] - g[None], axis=2)
            available = d <= 15
            for i in np.argsort(-p[:, 3], kind="quicksort"):
                near = np.flatnonzero(available[i])
                if len(near):
                    j = near[np.argmin(d[i, near])]
                    tp += 1
                    available[:, j] = False
                else:
                    fp += 1
        else:
            fp = len(p)
        result.append([tp, fp, len(g)])
    return np.asarray(result, dtype=int)


def score_counts(counts):
    out = {}
    for i, name in enumerate(("BC", "TC")):
        tp, fp, gt = counts[i]
        p = float(tp / (tp + fp + 1e-7))
        r = float(tp / (gt + 1e-7))
        out.update({f"precision_{name}": p, f"recall_{name}": r,
                    f"f1_{name}": 2 * p * r / (p + r + 1e-7)})
    out["mF1"] = (out["f1_BC"] + out["f1_TC"]) / 2
    return out


def audit(args):
    print("[1/4] 전달 파일 확인", flush=True)
    required = ["best.pt", "config.json", "history.csv", "postprocess.json",
                "eval_val.json", "eval_val_per_image.json"]
    missing = [p for p in required if not (args.run / p).is_file()]
    for p in required + ["last.pt", "eval_test.json", "log.txt"]:
        print(f"  {'OK' if (args.run / p).is_file() else '없음'}: {p}")
    if missing:
        raise SystemExit(f"[FAIL] 팀원에게 추가 요청: {', '.join(missing)}")
    state = load_state(args)
    s, cfg = state["settings"], read_json(args.run / "config.json")
    if s.get("size") != 1024 or s.get("label_source") != "official":
        raise ValueError("예상한 1024px 공식 라벨 M0 실험이 아닙니다.")
    if s.get("limit_val") is not None or s.get("limit_train") is not None:
        raise ValueError("일부 데이터만 사용하는 점검용 학습 설정입니다.")
    if s.get("train_pairs") != 400 or s.get("val_pairs") != 137:
        raise ValueError("학습/검증 쌍 수가 예상 v1.0.1 실험과 다릅니다.")
    keys = ["seed", "size", "model_config", "label_source", "augmentation",
            "epochs", "cell_loss", "tissue_loss_weight", "git_commit",
            "batch_size", "encoder_lr", "new_modules_lr"]
    differences = {k: [s.get(k), cfg.get(k)] for k in keys if s.get(k) != cfg.get(k)}
    if differences:
        raise ValueError(f"checkpoint와 config.json 불일치: {differences}")
    print("[2/4] checkpoint·학습 이력·후처리 연결", flush=True)
    with (args.run / "history.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    if len(epochs) != len(set(epochs)) or epochs != list(range(1, s["epochs"] + 1)):
        raise ValueError("학습 이력의 epoch 중복/누락/미완료")
    best = max(rows, key=lambda r: float(r["val_mF1"]))
    if int(best["epoch"]) != int(state["epoch"]) + 1:
        raise ValueError("best.pt epoch와 history 최고 mF1 epoch 불일치")
    if abs(float(best["val_mF1"]) - state["val"]["mF1"]) > 2e-6:
        raise ValueError("checkpoint 검증 점수와 이력 불일치")
    if any(not math.isfinite(float(r[k])) for r in rows for k in
           ("train_cell", "train_tissue", "val_cell", "val_tissue", "val_mF1")):
        raise ValueError("학습 로그에 NaN/Inf")
    pp = read_json(args.run / "postprocess.json")
    ev = read_json(args.run / "eval_val.json")
    if pp.get("selected_on") != "val":
        raise ValueError("후처리 선택 데이터가 validation으로 기록되지 않음")
    for k in ("sigma", "min_distance", "threshold"):
        if pp["chosen"][k] != ev["postprocess"][k]:
            raise ValueError("후처리 설정과 평가 설정 불일치")
    records = cached_records(args)
    if len(records) != 137 or ev.get("pairs") != 137:
        raise ValueError("validation 평가 결과가 137쌍이 아닙니다.")
    print("[3/4] 가중치 유한성·파일 SHA256", flush=True)
    import torch
    bad = [k for k, v in state["model"].items()
           if v.is_floating_point() and not bool(torch.isfinite(v).all())]
    if bad:
        raise ValueError(f"가중치 NaN/Inf: {bad[:5]}")
    sha = hashlib.sha256()
    with (args.run / "best.pt").open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    total = sum((official_counts(r["predictions"], r["truths"]) for r in records),
                np.zeros((2, 3), dtype=int))
    computed = score_counts(total)
    with (args.out / "checkpoint_settings.json").open("w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    report = {"best_epoch": int(state["epoch"]) + 1, "checkpoint_mF1": state["val"]["mF1"],
              "seed": s["seed"], "training_commit": s.get("git_commit"),
              "best_sha256": sha.hexdigest(), "postprocess": pp["chosen"],
              "cached_counts_metrics": computed,
              "note": "저장 예측 점수는 소수 4자리로 반올림되어 재집계는 근사 비교입니다. replay로 추론 재현을 확인합니다."}
    print("[4/4] 完了", flush=True)
    save_json(args.out / "audit.json", report)
    print(f"[PASS] best epoch={report['best_epoch']} seed={s['seed']}")
    print(f"[SHA256] {sha.hexdigest()} (팀원 값과 비교해야 전달 일치 확인)")


def localization_labels(predictions, truths):
    """Class-agnostic location-first matching: diagnostic, NOT official F1."""
    gt = np.asarray(truths, dtype=float).reshape(-1, 3)
    used = np.zeros(len(gt), dtype=bool)
    labels = np.full(len(gt), "missed", dtype=object)
    extra = {"duplicate": 0, "false_positive": 0}
    for p in sorted(predictions, key=lambda p: -p[3]):
        d = np.linalg.norm(gt[:, :2] - p[:2], axis=1)
        near = d <= 15
        free = near & ~used
        if free.any():
            candidates = np.flatnonzero(free)
            j = candidates[np.argmin(d[free])]
            used[j] = True
            labels[j] = "correct" if p[2] == gt[j, 2] else "misclassified"
        else:
            extra["duplicate" if near.any() else "false_positive"] += 1
    return labels, extra


def nearest_distances(points):
    from scipy.spatial import cKDTree
    if len(points) < 2:
        return np.full(len(points), np.inf)
    return cKDTree(points).query(points, k=2)[0][:, 1]


def boundary_distance(mask):
    """BG/CA boundary only, excluding UNK; distances in tissue-image px."""
    from scipy.ndimage import distance_transform_edt
    boundary = np.zeros(mask.shape, dtype=bool)
    for axis in (0, 1):
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        change = (mask[a] != mask[b]) & (mask[a] != 2) & (mask[b] != 2)
        boundary[a] |= change
        boundary[b] |= change
    return (distance_transform_edt(~boundary) if boundary.any()
            else np.full(mask.shape, np.inf))


def patient_id(slide):
    m = re.match(r"^(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})(?:-|$)", slide)
    return m.group(1) if m else "slide:" + slide


def pooled_table(rows, keys, count_map):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(str(r[k]) for k in keys)].append(r)
    output = []
    for key, group in sorted(groups.items()):
        counts = sum((count_map[r["pair_id"]] for r in group), np.zeros((2, 3), dtype=int))
        output.append({**dict(zip(keys, key)), "images": len(group),
                       "patients_or_slides": len({r['cluster'] for r in group}),
                       "gt_BC": int(counts[0, 2]), "gt_TC": int(counts[1, 2]),
                       **score_counts(counts),
                       "missed_location_rate": sum(r["missed"] for r in group) / max(1, int(counts[:, 2].sum()))})
    return output


def clustered_bootstrap(rows, count_map, repeats=1000):
    groups = defaultdict(lambda: np.zeros((2, 3), dtype=int))
    for r in rows:
        groups[r["cluster"]] += count_map[r["pair_id"]]
    blocks = np.stack(list(groups.values()))
    rng = np.random.default_rng(42)
    scores = [score_counts(blocks[rng.integers(len(blocks), size=len(blocks))].sum(0))["mF1"]
              for _ in range(repeats)]
    return {"unit": "TCGA patient when available, otherwise slide",
            "clusters": len(blocks), "replicates": repeats,
            "mF1": score_counts(blocks.sum(0))["mF1"],
            "CI95": np.quantile(scores, [0.025, 0.975]).tolist(),
            "note": "고정 checkpoint의 표본 불확실성입니다. seed에 따른 학습 편차를 나타내지 않습니다."}


def cohorts(args):
    from scipy.ndimage import laplace
    from common.data.dataset import OcelotDataset
    from common.data.geometry import points_in_tissue
    from common.data.photometric import lab_stats
    records = cached_records(args)
    ds = OcelotDataset(split="val", label_source="official", augment=False)
    index = {r["pair_id"]: i for i, r in enumerate(ds.rows)}
    if set(index) != {r["pair_id"] for r in records}:
        raise ValueError("평가 ID와 로컬 validation 데이터가 다릅니다.")
    cells, images, count_map = [], [], {}
    for n, r in enumerate(records, 1):
        raw = ds[index[r["pair_id"]]]
        gt = np.asarray(r["truths"], dtype=float).reshape(-1, 3)
        local = np.asarray([(round(float(x)), round(float(y)), int(c))
                            for x, y, c in raw["cell_points"]], dtype=float).reshape(-1, 3)
        if not np.array_equal(gt, local):
            raise ValueError(f"{r['pair_id']}: 저장된 정답과 로컬 정답이 다릅니다.")
        meta = ds.metadata[r["pair_id"]]
        if r["organ"] != meta["organ"]:
            raise ValueError("장기 metadata가 다릅니다.")
        labels, extra = localization_labels(r["predictions"], r["truths"])
        nn = nearest_distances(gt[:, :2])
        tissue = raw["tissue_target"]
        box = raw["cell_box"]
        xy = points_in_tissue(gt, box)
        x = np.clip(np.rint(xy[:, 0]).astype(int), 0, 1023)
        y = np.clip(np.rint(xy[:, 1]).astype(int), 0, 1023)
        regions = tissue[y, x]
        dist_map = boundary_distance(tissue)
        sx, sy = box[2] - box[0], box[3] - box[1]
        for j, (gx, gy, cls) in enumerate(gt):
            nearest = float(nn[j])
            bd = float(dist_map[y[j], x[j]])
            # The dataset uses approximately equal x/y scale. Avoid pretending to
            # provide exact cell-pixel boundary distance for anisotropic boxes.
            bd_cell = bd / sx if abs(sx - sy) < 1e-5 else np.inf
            cells.append({"pair_id": r["pair_id"], "organ": r["organ"],
                          "gt_index": j, "x": gx, "y": gy, "class": "BC" if cls == 1 else "TC",
                          "region": {0: "BG", 1: "CA", 2: "UNK"}[int(regions[j])],
                          "error": str(labels[j]), "nearest_px": nearest if math.isfinite(nearest) else "",
                          "neighbor_bin": "0-14" if nearest < 14 else "14-30" if nearest < 30 else "30+",
                          "boundary_cell_px": bd_cell if math.isfinite(bd_cell) else "",
                          "boundary_bin": "0-30" if bd_cell < 30 else "30-100" if bd_cell < 100 else "100+/no_boundary"})
        rgb = raw["cell_image"].astype(np.float32) / 255
        gray = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        lab_mean, lab_std = lab_stats(raw["cell_image"])
        counts = official_counts(r["predictions"], r["truths"])
        count_map[r["pair_id"]] = counts
        images.append({"pair_id": r["pair_id"], "organ": r["organ"],
                       "slide": meta["slide_name"], "cluster": patient_id(meta["slide_name"]),
                       "num_gt": len(gt), "num_pred": len(r["predictions"]),
                       "count_bin": "0" if len(gt) == 0 else "1-49" if len(gt) < 50 else "50-149" if len(gt) < 150 else "150-299" if len(gt) < 300 else "300+",
                       "missed": int(sum(labels == "missed")),
                       "misclassified": int(sum(labels == "misclassified")), **extra,
                       "brightness": float(gray.mean()), "contrast": float(gray.std()),
                       "sharpness_proxy": float(laplace(gray).var()),
                       "mean_R": float(rgb[:, :, 0].mean()), "mean_G": float(rgb[:, :, 1].mean()),
                       "mean_B": float(rgb[:, :, 2].mean()),
                       **{f"LAB_{name}_mean": float(lab_mean[k]) for k, name in enumerate(("L", "a", "b"))},
                       **{f"LAB_{name}_std": float(lab_std[k]) for k, name in enumerate(("L", "a", "b"))},
                       "tissue_mIoU_recorded": r["mIoU"], **score_counts(counts)})
        if n % 20 == 0 or n == len(records):
            print(f"[조건 분석] {n}/{len(records)}", flush=True)
    metrics = ("brightness", "contrast", "sharpness_proxy", "tissue_mIoU_recorded", "LAB_a_mean", "LAB_b_mean")
    for metric in metrics:
        edges = np.quantile([r[metric] for r in images], [0.25, 0.5, 0.75])
        for r in images:
            r[metric + "_quartile"] = f"Q{np.searchsorted(edges, r[metric], side='right') + 1}"
    tables = []
    for keys in (["organ"], ["count_bin"], ["organ", "count_bin"],
                 *[[m + "_quartile"] for m in metrics],
                 *[["organ", m + "_quartile"] for m in metrics]):
        for row in pooled_table(images, keys, count_map):
            tables.append({"grouping": "+".join(keys), **row})
    cell_groups = []
    for keys in (["class", "neighbor_bin"], ["class", "region"],
                 ["class", "region", "boundary_bin"], ["organ", "class", "region"]):
        groups = defaultdict(list)
        for c in cells:
            groups[tuple(c[k] for k in keys)].append(c)
        for key, group in sorted(groups.items()):
            matched = sum(c["error"] != "missed" for c in group)
            wrong = sum(c["error"] == "misclassified" for c in group)
            cell_groups.append({"grouping": "+".join(keys), **dict(zip(keys, key)),
                                "cells": len(group), "images": len({c['pair_id'] for c in group}),
                                "location_miss_rate": 1 - matched / len(group),
                                "wrong_class_rate_all_gt": wrong / len(group),
                                "wrong_class_rate_localized": wrong / matched if matched else ""})
    write_csv(args.out / "images.csv", images)
    write_csv(args.out / "cells.csv", cells)
    write_csv(args.out / "image_groups.csv", tables)
    write_csv(args.out / "cell_groups.csv", cell_groups)
    save_json(args.out / "bootstrap.json", clustered_bootstrap(images, count_map))
    save_json(args.out / "case_ids.json", {
        "worst_nontrivial": [r["pair_id"] for r in sorted(images, key=lambda r: r["mF1"]) if r["num_gt"] >= 10][:10],
        "most_missed": [r["pair_id"] for r in sorted(images, key=lambda r: -r["missed"])][:10],
        "most_misclassified": [r["pair_id"] for r in sorted(images, key=lambda r: -r["misclassified"])][:10]})
    save_json(args.out / "interpretation.json", {
        "official_metrics": "TP/FP/GT를 전체 합산한 클래스별 F1입니다. 저장 점수 반올림에 따른 미세한 차이가 가능합니다.",
        "error_metrics": "위치 우선 매칭으로 분류한 원인 탐색용 지표입니다. 공식 F1의 FN과 같지 않습니다.",
        "boundary": "조직 정답의 BG/CA 경계이며 UNK 경계는 제외합니다. 세포 영상 좌표로 환산합니다.",
        "confounding": "밝기·밀도·장기·클래스 구성은 서로 관련될 수 있습니다. 단일 집계에서 인과관계를 확정하지 않습니다.",
        "bootstrap": "환자/slide 단위 표본 CI입니다. seed 편차를 알려면 반복 학습이 필요합니다."})
    print("[PASS] images.csv / cells.csv / image_groups.csv / cell_groups.csv / bootstrap.json")


def replay(args):
    import torch
    from common.metrics.m0_eval import CellF1, TissueIoU, detect_cells
    from common.training.m0_loop import make_loader, prepare, truth_points
    model, state, device = load_model(args)
    s = state["settings"]
    if s["size"] != 1024:
        raise ValueError("정식 재평가는 1024px 설정이어야 합니다.")
    post = read_json(args.run / "postprocess.json")
    if post.get("selected_on") != "val":
        raise ValueError("validation에서 선택한 후처리 설정이 필요합니다.")
    pp = {k: post["chosen"][k] for k in ("sigma", "min_distance", "threshold")}
    loader, ds = make_loader("val", s["seed"], 1, 0, augment=False)
    overall, tissue_total = CellF1(), TissueIoU()
    records, probes = [], []
    with torch.no_grad():
        for n, batch in enumerate(loader, 1):
            images, targets, box = prepare(batch, device, 1024)
            out = model(*images, box)
            preds = detect_cells(out["cell_logits"], **pp)[0]
            gt = truth_points(batch["cell_points"][0])
            overall.update(preds, gt)
            tissue_total.update(out["tissue_logits"], targets[1])
            single, ti = CellF1(), TissueIoU()
            single.update(preds, gt)
            ti.update(out["tissue_logits"], targets[1])
            pid = batch["pair_id"][0]
            records.append({"pair_id": pid, "organ": ds.source.metadata[pid]["organ"],
                            "num_gt": len(gt), "num_pred": len(preds),
                            **single.compute(), **ti.compute(),
                            "predictions": preds, "truths": gt})
            prob = out["cell_logits"][0].float().softmax(0).cpu().numpy()
            labels, _ = localization_labels(preds, gt)
            for j, (x, y, cls) in enumerate(gt):
                x0, x1 = max(0, x - 7), min(1024, x + 8)
                y0, y1 = max(0, y - 7), min(1024, y + 8)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                disk = (xx - x) ** 2 + (yy - y) ** 2 <= 49
                max_fg = float((1 - prob[0, y0:y1, x0:x1])[disk].max())
                probes.append({"pair_id": pid, "gt_index": j, "class": "BC" if cls == 1 else "TC",
                               "error": str(labels[j]), "p_empty_at_gt": float(prob[0, y, x]),
                               "p_BC_at_gt": float(prob[1, y, x]), "p_TC_at_gt": float(prob[2, y, x]),
                               "max_foreground_within_7px": max_fg})
            if n % 10 == 0 or n == len(loader):
                print(f"[재평가] val {n}/{len(loader)} | device={device}", flush=True)
    metrics = {**overall.compute(), **tissue_total.compute()}
    baseline = read_json(args.run / "eval_val.json")
    # Repository summary nests its metrics under overall.
    old = baseline.get("overall", baseline.get("metrics", baseline))
    delta = {k: metrics[k] - old[k] for k in metrics if isinstance(old.get(k), (int, float))}
    save_json(args.out / "eval_val.json", {"pairs": len(records), "split": "val", "overall": metrics,
              "postprocess": pp, "difference_from_received": delta, "device": str(device),
              "note": "batch=1, fp32로 확인합니다. 기존 실험의 batch/device와 달라 작은 수치 차이가 가능합니다."})
    save_json(args.out / "eval_val_per_image.json", records)
    write_csv(args.out / "cell_probability_probes.csv", probes)
    print(f"[완료] mF1={metrics['mF1']:.6f}, mIoU={metrics['mIoU']:.6f}, 기존 결과와 차이={delta}")


def gradient_stats(gc, gt):
    dot = cc = tt = 0.0
    common = 0
    for c, t in zip(gc, gt):
        # MPS does not support float64; compute diagnostic reductions on CPU.
        c = c.detach().cpu().double() if c is not None else None
        t = t.detach().cpu().double() if t is not None else None
        if c is not None:
            cc += float((c ** 2).sum())
        if t is not None:
            tt += float((t ** 2).sum())
        if c is not None and t is not None:
            dot += float((c * t).sum())
            common += c.numel()
    nc, nt = math.sqrt(cc), math.sqrt(tt)
    return {"cell_norm": nc, "tissue_norm": nt,
            "cosine": dot / (nc * nt) if nc and nt else None,
            "tissue_to_cell_norm_ratio": nt / nc if nc else None,
            "common_gradient_elements": common}


def gradients(args):
    import torch
    from common.data.tensors import TorchOcelotDataset, collate_pairs
    from common.training.m0_loop import prepare, seed_everything
    from common.losses.m0_losses import m0_task_losses
    model, state, device = load_model(args)
    s = state["settings"]
    seed_everything(42)
    ds = TorchOcelotDataset(split="train", label_source=s["label_source"], augment=False)
    rng = np.random.default_rng(42)
    choices = rng.choice(len(ds), size=min(args.samples, len(ds)), replace=False)
    if args.full_parameters:
        selected = list(model.named_parameters())
    else:
        # Same parameter tensor for both losses. Limit to encoder final blocks
        # to reduce diagnostic memory; do not treat this as all-model conflict.
        selected = [(n, p) for n, p in model.named_parameters()
                    if n.startswith(("cell.encoder.layer4.2.", "tissue.encoder.layer4.2."))]
    selected = [(n, p) for n, p in selected if p.requires_grad]
    if not selected:
        raise ValueError("gradient를 측정할 파라미터가 없습니다.")
    write_csv(args.out / "parameter_scope.csv", [{"name": n, "elements": p.numel()} for n, p in selected])
    params = [p for _, p in selected]
    rows = []
    for j, i in enumerate(choices, 1):
        batch = collate_pairs([ds[int(i)]])
        images, targets, box = prepare(batch, device, s["size"])
        output = model(*images, box)
        loss = m0_task_losses(output["cell_logits"], output["tissue_logits"], *targets,
            gamma=s["cell_loss"]["gamma"], normalize=s["cell_loss"]["normalize"],
            min_foreground_fraction=s["cell_loss"].get("min_foreground_fraction", 0.0))
        gc = torch.autograd.grad(loss["cell"], params, retain_graph=True, allow_unused=True)
        gt = torch.autograd.grad(loss["tissue"] * s["tissue_loss_weight"], params, allow_unused=True)
        row = {"pair_id": batch["pair_id"][0], "cell_loss": float(loss["cell"].detach().cpu()),
               "tissue_loss": float(loss["tissue"].detach().cpu()), **gradient_stats(gc, gt)}
        rows.append(row)
        print(f"[gradient] {j}/{len(choices)} | {row['pair_id']} | cosine={row['cosine']}", flush=True)
        del gc, gt, loss, output, images, targets, batch
    valid = [r for r in rows if r["cosine"] is not None]
    write_csv(args.out / "gradients.csv", rows)
    save_json(args.out / "gradient_summary.json", {
        "samples": len(rows), "nonzero_both": len(valid),
        "negative_cosine_fraction": sum(r["cosine"] < 0 for r in valid) / len(valid) if valid else None,
        "median_cosine": float(np.median([r["cosine"] for r in valid])) if valid else None,
        "scope": "full model" if args.full_parameters else "last block of both encoders",
        "mode": "trained best checkpoint, eval mode, unaugmented training samples, no optimizer updates",
        "note": "best checkpoint의 탐색용 점검입니다. M3의 판단에는 M2 학습 도중의 gradient 기록도 필요합니다."})
    print("[완료] gradient_summary.json / gradients.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["audit", "cohorts", "replay", "gradients"])
    parser.add_argument("--run", default="outputs/m0/M0_official_default_seed42")
    parser.add_argument("--project", default=".")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--records", help="cohorts 분석에 사용할 다른 결과 JSON 경로")
    parser.add_argument("--samples", type=int, default=12, help="gradient 진단의 train 표본 수")
    parser.add_argument("--full-parameters", action="store_true", help="전체 파라미터 gradient 측정 (메모리 증가)")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    init(args)
    globals()[args.command](args)
    print(f"[결과] {args.out}")


if __name__ == "__main__":
    main()
