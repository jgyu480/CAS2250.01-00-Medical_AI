"""M2 정답 조건(G · H_A · H_AO)과 M0를 같은 기준으로 비교한다 (validation 탐색용).

보고서 6.2의 가까운 세포 진단과 8.4의 평가 항목을 run 결과 파일만으로 다시 계산한다.
모델을 다시 돌리지 않으므로 GPU·checkpoint가 필요 없다.

    python scripts/compare_m2.py \
        --m0 results/m0_dicefix_seed42 \
        --run G=results/m2_s6_l0.9_seed42/gaussian \
        --run H_A=results/m2_s6_l0.9_seed42/ha \
        --run H_AO=results/m2_s6_l0.9_seed42/hao \
        --metadata data/raw/ocelot2023_v1.0.1/metadata.json \
        --out results/m2_s6_l0.9_seed42/comparison.json

각 run 폴더에는 eval_val.json, eval_val_per_image.json이 있어야 하고,
postprocess.json, history.csv, close_cells_val.json은 있으면 함께 정리한다.

출력 항목
- overall: 공식 P/R/F1·mF1·mIoU (per-image 파일로 다시 계산해 eval_val.json과 대조)
- close_cells: 보고서 6.2 정의 (diagnose_m0.localization_labels, 15px, 0-14/14-30/30+).
  누락률·오분류율·정상 검출률, 가까움-멂 누락률 차이와 비율, M2의 누락 원인(합침/약함)
- common_postprocess: 각 run이 고른 후처리 값을 다른 run의 격자 결과에서 찾아 같은 값으로 비교
  (postprocess.json의 격자 탐색 결과라 재추론 없이 정확하다. 공식 지표만 있고 거리 구간 진단은 없다)
- paired_bootstrap: 환자(TCGA)/slide 단위로 같은 표본을 모든 run에 함께 뽑아 run 간 차이의 95% 구간
  (diagnose_m0.clustered_bootstrap과 같은 단위·seed. metadata가 없으면 이미지 단위)
- pairs_14_30: 이웃이 14-30px인 세포 중 놓친 세포와 검출된 세포 각각에서 그 이웃이 검출된 비율
  (놓친 쪽이 더 높으면 "둘 중 하나만 잡는" 형태, 더 낮으면 "주변이 함께 약한" 형태)
- false_positives: 15px 안에 정답이 없는 예측의 가장 가까운 정답까지 거리 분포
- training: history.csv의 best epoch, 학습 중 val 지표, val 세포 loss 최소 epoch

주의: checkpoint와 후처리를 모두 validation으로 골랐으므로 낙관적인 탐색용 수치다.
seed 1개라 학습 편차는 bootstrap 구간에 들어 있지 않다. 위치 우선 매칭 진단은 공식 F1이 아니다.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_m0 import (localization_labels, nearest_distances, official_counts,
                                 patient_id, read_json, save_json, score_counts)

CLASSES = ("BC", "TC")
BINS = ("0-14", "14-30", "30+")
FP_BINS = ((15, 20, "15-20"), (20, 30, "20-30"), (30, np.inf, "30+"))
METRICS = ("precision_BC", "recall_BC", "f1_BC", "precision_TC", "recall_TC", "f1_TC", "mF1")


def neighbor_bin(distance):
    return 0 if distance < 14 else 1 if distance < 30 else 2


def load_run(folder):
    folder = Path(folder)
    run = {"folder": str(folder),
           "eval": read_json(folder / "eval_val.json"),
           "records": read_json(folder / "eval_val_per_image.json")}
    for name, key in (("postprocess.json", "postprocess"), ("close_cells_val.json", "close_cells")):
        if (folder / name).is_file():
            run[key] = read_json(folder / name)
    if (folder / "history.csv").is_file():
        with (folder / "history.csv").open(encoding="utf-8") as f:
            run["history"] = list(csv.DictReader(f))
    return run


def image_stats(record):
    """한 이미지: 공식 TP/FP/GT (2,3), 구간별 [정답 수, 누락, 오분류] (2,3,3), FP 거리."""
    gt = np.asarray(record["truths"], dtype=float).reshape(-1, 3)
    counts = np.asarray(official_counts(record["predictions"], record["truths"]), dtype=int)
    bins = np.zeros((2, 3, 3), dtype=int)
    # 14-30px 이웃이 있는 정답: [클래스, 0=놓침/1=검출, (세포 수, 가장 가까운 이웃도 검출된 수)]
    pairs = np.zeros((2, 2, 2), dtype=int)
    labels, extra = localization_labels(record["predictions"], record["truths"])
    nearest = nearest_distances(gt[:, :2])
    partner = (cKDTree(gt[:, :2]).query(gt[:, :2], k=2)[1][:, 1] if len(gt) >= 2
               else np.zeros(len(gt), dtype=int))
    found = labels != "missed"
    for j, (_, _, cls) in enumerate(gt):
        b = neighbor_bin(nearest[j])
        row = bins[int(cls) - 1, b]
        row[0] += 1
        row[1] += labels[j] == "missed"
        row[2] += labels[j] == "misclassified"
        if b == 1:
            cell = pairs[int(cls) - 1, int(found[j])]
            cell[0] += 1
            cell[1] += found[partner[j]]
    preds = np.asarray(record["predictions"], dtype=float).reshape(-1, 4)
    if len(gt) and len(preds):
        d = np.linalg.norm(preds[:, None, :2] - gt[None, :, :2], axis=2).min(axis=1)
    else:
        d = np.full(len(preds), np.inf)
    fp_distance = d[d > 15]
    if len(fp_distance) != extra["false_positive"]:
        raise ValueError(f"{record['pair_id']}: FP 수가 위치 우선 매칭 결과와 다릅니다.")
    return {"counts": counts, "bins": bins, "pairs": pairs, "extra": extra,
            "fp_distance": fp_distance, "predictions": len(preds)}


def pair_summary(pairs):
    """놓친 가까운 세포의 이웃이 검출되었는가: 둘 다 놓침(영역 전체가 약함) vs 하나만 검출."""
    out = {}
    for c, cls in enumerate(CLASSES):
        missed, found = pairs[c, 0], pairs[c, 1]
        out[cls] = {"missed_cells": int(missed[0]),
                    "missed_with_partner_detected": float(missed[1] / max(missed[0], 1)),
                    "detected_cells": int(found[0]),
                    "detected_with_partner_detected": float(found[1] / max(found[0], 1))}
    return out


def bin_rates(bins):
    """bins (2,3,3) 합계 → 클래스·구간별 비율과 가까움(14-30)-멂(30+) 차이."""
    out = {}
    for c, cls in enumerate(CLASSES):
        for b, name in enumerate(BINS):
            cells, missed, wrong = (int(v) for v in bins[c, b])
            n = max(cells, 1)
            out[f"{cls}_{name}"] = {"cells": cells, "missed": missed, "misclassified": wrong,
                                    "miss_rate": missed / n, "misclassified_rate": wrong / n,
                                    "correct_rate": (cells - missed - wrong) / n}
        close, far = out[f"{cls}_14-30"]["miss_rate"], out[f"{cls}_30+"]["miss_rate"]
        out[f"{cls}_close_minus_far"] = close - far
        out[f"{cls}_close_over_far"] = close / far if far > 0 else None
    return out


def summary_values(counts, bins):
    """bootstrap에서 비교할 값들."""
    s = score_counts(counts)
    r = bin_rates(bins)
    values = {k: s[k] for k in ("mF1", "f1_BC", "f1_TC")}
    for cls in CLASSES:
        values[f"{cls}_miss_14-30"] = r[f"{cls}_14-30"]["miss_rate"]
        values[f"{cls}_miss_30+"] = r[f"{cls}_30+"]["miss_rate"]
        values[f"{cls}_close_minus_far"] = r[f"{cls}_close_minus_far"]
    return values


def clusters_for(pair_ids, metadata_path):
    if metadata_path is None:
        return {p: p for p in pair_ids}, "image (metadata 없음)"
    pairs = read_json(metadata_path)["sample_pairs"]
    missing = [p for p in pair_ids if p not in pairs]
    if missing:
        raise ValueError(f"metadata에 없는 pair: {missing[:5]}")
    return {p: patient_id(pairs[p]["slide_name"]) for p in pair_ids}, \
        "TCGA patient when available, otherwise slide"


def paired_bootstrap(stats, cluster_of, comparisons, repeats, seed=42):
    names = list(stats)
    ids = sorted(cluster_of)
    groups = sorted(set(cluster_of.values()))
    index = {g: i for i, g in enumerate(groups)}
    counts = {n: np.zeros((len(groups), 2, 3), dtype=int) for n in names}
    bins = {n: np.zeros((len(groups), 2, 3, 3), dtype=int) for n in names}
    for p in ids:
        g = index[cluster_of[p]]
        for n in names:
            counts[n][g] += stats[n][p]["counts"]
            bins[n][g] += stats[n][p]["bins"]
    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    for _ in range(repeats):
        pick = rng.integers(len(groups), size=len(groups))
        values = {n: summary_values(counts[n][pick].sum(0), bins[n][pick].sum(0)) for n in names}
        for a, b in comparisons:
            for k in values[a]:
                draws[(a, b, k)].append(values[a][k] - values[b][k])
        for n in names:
            draws[(n, None, "mF1")].append(values[n]["mF1"])
    point = {n: summary_values(counts[n].sum(0), bins[n].sum(0)) for n in names}
    result = {"clusters": len(groups), "replicates": repeats, "seed": seed, "differences": [],
              "mF1_CI95": {n: np.quantile(draws[(n, None, "mF1")], [0.025, 0.975]).tolist()
                           for n in names}}
    for a, b in comparisons:
        for k in point[a]:
            d = np.asarray(draws[(a, b, k)])
            low, high = np.quantile(d, [0.025, 0.975])
            result["differences"].append({
                "comparison": f"{a} - {b}", "value": k, "difference": point[a][k] - point[b][k],
                "CI95": [float(low), float(high)],
                "share_above_0": float((d > 0).mean()),
                "excludes_0": bool(low > 0 or high < 0)})
    return result


def common_postprocess(runs):
    """각 run이 고른 후처리 값을, 격자를 가진 모든 run에서 찾아 공식 지표를 나란히 둔다."""
    grids = {n: r["postprocess"] for n, r in runs.items() if "postprocess" in r and "search" in r["postprocess"]}
    if not grids:
        return []
    keys = ("sigma", "min_distance", "threshold")
    settings = []
    for g in grids.values():
        s = tuple(g["chosen"][k] for k in keys)
        if s not in settings:
            settings.append(s)
    rows = []
    for s in settings:
        for n, g in grids.items():
            hit = next((e for e in g["search"] if tuple(e[k] for k in keys) == s), None)
            if hit is not None:
                rows.append({"setting": dict(zip(keys, s)), "run": n,
                             "chosen_by_this_run": tuple(g["chosen"][k] for k in keys) == s,
                             **{k: hit[k] for k in METRICS}})
    return rows


def grid_edges(run):
    """고른 값이 격자 끝에 있는지 (격자를 넓혀 볼 필요가 있는지)."""
    g = run.get("postprocess")
    if not g or "search" not in g:
        return None
    out = {}
    for k in ("sigma", "min_distance", "threshold"):
        values = sorted({e[k] for e in g["search"]})
        out[k] = {"chosen": g["chosen"][k], "grid": values,
                  "at_edge": g["chosen"][k] in (values[0], values[-1])}
    return out


def training_summary(history):
    if not history:
        return None
    rows = [{k: float(v) if k != "encoder_frozen" else v for k, v in r.items()} for r in history]
    best = max(rows, key=lambda r: r["val_mF1"])
    low = min(rows, key=lambda r: r["val_cell"])
    return {"epochs": len(rows), "best_epoch": int(best["epoch"]),
            "best_val_mF1_training_postprocess": best["val_mF1"],
            "best_epoch_val_mIoU": best["val_mIoU"],
            "last_val_mF1": rows[-1]["val_mF1"], "max_val_mIoU": max(r["val_mIoU"] for r in rows),
            "min_val_cell_loss_epoch": int(low["epoch"]), "min_val_cell_loss": low["val_cell"],
            "last_val_cell_loss": rows[-1]["val_cell"],
            "last_train_cell_loss": rows[-1]["train_cell"]}


def fp_summary(stats):
    distance = np.concatenate([s["fp_distance"] for s in stats.values()]) if stats else np.zeros(0)
    extra = defaultdict(int)
    for s in stats.values():
        for k, v in s["extra"].items():
            extra[k] += v
    total = sum(s["predictions"] for s in stats.values())
    out = {"predictions": total, "duplicate": extra["duplicate"],
           "false_positive": extra["false_positive"],
           "false_positive_share_of_predictions": extra["false_positive"] / max(total, 1)}
    for low, high, name in FP_BINS:
        out[f"fp_nearest_gt_{name}px"] = int(((distance > low) & (distance <= high)).sum())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m0", required=True, help="M0 run 폴더 (eval_val*.json)")
    parser.add_argument("--run", action="append", required=True, metavar="NAME=FOLDER",
                        help="비교할 M2 run. 여러 번 지정")
    parser.add_argument("--metadata", help="OCELOT metadata.json (환자/slide 단위 bootstrap)")
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runs = {"M0": load_run(args.m0)}
    for item in args.run:
        name, _, folder = item.partition("=")
        if not folder or name in runs:
            raise SystemExit(f"[중단] --run은 NAME=FOLDER 형식이고 이름이 겹치면 안 됩니다: {item}")
        runs[name] = load_run(folder)

    pair_ids = {n: [r["pair_id"] for r in run["records"]] for n, run in runs.items()}
    reference = sorted(pair_ids["M0"])
    for n, ids in pair_ids.items():
        if sorted(ids) != reference or len(set(ids)) != len(ids):
            raise SystemExit(f"[중단] {n}의 validation 이미지 목록이 M0와 다릅니다.")
        truths = {r["pair_id"]: r["truths"] for r in runs[n]["records"]}
        if truths != {r["pair_id"]: r["truths"] for r in runs["M0"]["records"]}:
            raise SystemExit(f"[중단] {n}의 정답 점이 M0와 다릅니다.")

    stats = {n: {r["pair_id"]: image_stats(r) for r in run["records"]} for n, run in runs.items()}

    overall, close = {}, {}
    for n, run in runs.items():
        counts = sum((s["counts"] for s in stats[n].values()), np.zeros((2, 3), dtype=int))
        bins = sum((s["bins"] for s in stats[n].values()), np.zeros((2, 3, 3), dtype=int))
        scores = score_counts(counts)
        recorded = run["eval"]["overall"]
        gap = max(abs(round(scores[k], 4) - recorded[k]) for k in METRICS)
        if gap > 1e-4:
            raise SystemExit(f"[중단] {n}: per-image로 다시 계산한 F1이 eval_val.json과 다릅니다 ({gap}).")
        overall[n] = {**{k: recorded[k] for k in (*METRICS, "mIoU", "iou_BG", "iou_CA")},
                      "postprocess": run["eval"].get("postprocess"),
                      "epoch": run["eval"].get("epoch"),
                      "cell_target": run["eval"].get("cell_target", {}).get("mode", "M0 (focal)"),
                      "postprocess_grid_edges": grid_edges(run),
                      "training": training_summary(run.get("history"))}
        pairs = sum((s["pairs"] for s in stats[n].values()), np.zeros((2, 2, 2), dtype=int))
        close[n] = {"rates": bin_rates(bins), "pairs_14_30": pair_summary(pairs),
                    "localization": fp_summary(stats[n])}
        cc = run.get("close_cells", {}).get("M2")
        if cc and cc.get("cause_split"):
            close[n]["miss_causes"] = {f"{t['class']}_{t['neighbor_bin']}":
                                       {"missed": t["missed"], "peak_merged": t["peak_merged"],
                                        "weak_response": t["weak_response"]}
                                       for t in cc["by_class_and_neighbor"]}

    names = [n for n in runs if n != "M0"]
    comparisons = [(n, "M0") for n in names]
    if "G" in names:
        comparisons += [(n, "G") for n in names if n != "G"]
    cluster_of, unit = clusters_for(reference, args.metadata)
    boot = paired_bootstrap(stats, cluster_of, comparisons, args.repeats)
    boot["unit"] = unit

    result = {
        "split": "val", "pairs": len(reference),
        "gt_cells": int(sum(s["counts"][:, 2].sum() for s in stats["M0"].values())),
        "runs": {n: run["folder"] for n, run in runs.items()},
        "overall": overall, "close_cells": close,
        "common_postprocess": common_postprocess({n: runs[n] for n in names}),
        "paired_bootstrap": boot,
        "notes": {
            "selection": "checkpoint와 후처리를 모두 validation으로 골랐다. test 평가 전 탐색용 수치다.",
            "seed": "seed 1개. bootstrap 구간은 표본(환자/slide) 불확실성이며 학습 seed 편차는 포함하지 않는다.",
            "close_cells": "보고서 6.2 정의: 클래스 무관 위치 우선 매칭(15px). 공식 F1의 FN과 같지 않다.",
            "common_postprocess": "격자 탐색 결과에서 같은 후처리 값을 찾아 비교한다. 거리 구간 진단은 각 run이 고른 후처리 값 기준이다.",
            "miss_causes": "M2 점수 지도에서 정답 위치 점수가 threshold보다 높으면 peak_merged, 아니면 weak_response.",
            "pairs_14_30": "가장 가까운 정답이 14-30px인 세포를, 놓친 것과 검출된 것으로 나눠 그 이웃이 검출된 비율을 비교한다.",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, result)

    print(f"[비교] val {len(reference)}쌍, 정답 세포 {result['gt_cells']}개, bootstrap 단위 {unit} "
          f"({boot['clusters']}개 × {boot['replicates']}회)")
    for n in runs:
        o, r = overall[n], close[n]["rates"]
        print(f"  {n:5s} mF1 {o['mF1']:.4f} (BC {o['f1_BC']:.4f}, TC {o['f1_TC']:.4f}) mIoU {o['mIoU']:.4f} | "
              f"14-30 누락 BC {r['BC_14-30']['miss_rate']:.1%} TC {r['TC_14-30']['miss_rate']:.1%} | "
              f"30+ 누락 BC {r['BC_30+']['miss_rate']:.1%} TC {r['TC_30+']['miss_rate']:.1%} | "
              f"FP {close[n]['localization']['false_positive']}")
    for d in boot["differences"]:
        if d["value"] in ("mF1", "BC_miss_14-30", "TC_miss_14-30", "BC_close_minus_far", "TC_close_minus_far"):
            print(f"  {d['comparison']:10s} {d['value']:18s} {d['difference']:+.4f} "
                  f"[{d['CI95'][0]:+.4f}, {d['CI95'][1]:+.4f}]{' *' if d['excludes_0'] else ''}")
    print(f"[PASS] {out}")


if __name__ == "__main__":
    main()
