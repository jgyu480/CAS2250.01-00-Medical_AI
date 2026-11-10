"""M2 평가와 가까운 세포 진단 (보고서 8.4의 평가 항목).

1) validation에서 후처리 값 고르기 (best.pt로 한 번 추론 후 격자 탐색):
    python scripts/eval_m2.py --checkpoint outputs/m2/<run>/best.pt --split val --tune
   → <run>/postprocess.json
2) test 평가 (설정을 모두 고정한 뒤 마지막에 한 번):
    python scripts/eval_m2.py --checkpoint outputs/m2/<run>/best.pt --split test
3) M0와 같은 기준으로 비교 (M0 결과 파일은 저장소에 있다):
    --compare results/m0_dicefix_seed42/eval_val_per_image.json

결과
- eval_<split>.json, eval_<split>_per_image.json: M0와 같은 형식 (공식 F1·mIoU·장기별).
- close_cells_<split>.json: 보고서 6.2와 같은 정의의 진단.
  · 정답 세포마다 가장 가까운 다른 정답 세포(클래스 무관)까지 거리 → 0-14 / 14-30 / 30+ px
  · 위치 우선 매칭(클래스 무관, 15px, 점수 높은 예측부터): missed / misclassified / correct,
    남는 예측은 duplicate(근처에 이미 매칭된 정답) / false_positive
  · 놓친 세포의 원인 구분 (M2 점수 지도가 있을 때만):
    peak_merged = 정답 위치의 점수가 threshold보다 높은데 봉우리가 없다 (이웃 봉우리에 흡수),
    weak_response = 정답 위치의 점수가 threshold 이하.
  위치 우선 매칭과 원인 구분은 진단용이며 공식 F1이 아니다.
"""

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.metrics.heatmap_eval import detect_cells_heatmap, heatmap_scores
from common.metrics.m0_eval import CellF1, gaussian_blur
from common.training.m0_loop import pick_device, to_full_size
from common.training.m2_loop import make_loader, evaluate, cell_loss_kwargs
from models.M2.model import build_model
from scripts.diagnose_m0 import localization_labels, nearest_distances
from scripts.eval_m0 import organ_scores

BINS = ("0-14", "14-30", "30+")
KEYS = ("sigma", "min_distance", "threshold")


def neighbor_bin(distance):
    return "0-14" if distance < 14 else "14-30" if distance < 30 else "30+"


def tune(cached, grid, size):
    """캐시된 세포 출력으로 후처리 격자 탐색. 기준: validation 세포 mF1."""
    results = []
    for sigma, distance, threshold in itertools.product(
        grid["sigma"], grid["min_distance"], grid["threshold"]
    ):
        metric = CellF1()
        for _, logits, truths in cached:
            preds = detect_cells_heatmap(logits[None].float(), sigma=sigma,
                                         min_distance=distance, threshold=threshold)[0]
            metric.update(to_full_size(preds, size), truths)
        score = metric.compute()
        results.append({"sigma": sigma, "min_distance": distance, "threshold": threshold,
                        **{k: round(v, 4) for k, v in score.items()}})
        print(f"  σ={sigma} d={distance} thr={threshold}: mF1 {score['mF1']:.4f}", flush=True)
    return max(results, key=lambda r: r["mF1"]), results


def close_cell_table(records, score_map=None, postprocess=None):
    """records: eval_*_per_image.json 형식.
    score_map: pair_id → (1024, 1024) 부드럽게 한 점수 지도를 돌려주는 함수 (없으면 원인 구분 생략)."""
    cells = defaultdict(lambda: {"cells": 0, "missed": 0, "misclassified": 0,
                                 "peak_merged": 0, "weak_response": 0})
    extra = {"duplicate": 0, "false_positive": 0, "predictions": 0}
    for r in records:
        gt = np.asarray(r["truths"], dtype=float).reshape(-1, 3)
        labels, more = localization_labels(r["predictions"], r["truths"])
        extra["duplicate"] += more["duplicate"]
        extra["false_positive"] += more["false_positive"]
        extra["predictions"] += len(r["predictions"])
        nearest = nearest_distances(gt[:, :2])
        missed = any(label == "missed" for label in labels)
        score = score_map(r["pair_id"]) if score_map is not None and missed else None
        for j, (x, y, cls) in enumerate(gt):
            row = cells[("BC" if cls == 1 else "TC", neighbor_bin(nearest[j]))]
            row["cells"] += 1
            row["missed"] += labels[j] == "missed"
            row["misclassified"] += labels[j] == "misclassified"
            if labels[j] == "missed" and score is not None:
                value = float(score[int(round(y)), int(round(x))])
                key = "peak_merged" if value > postprocess["threshold"] else "weak_response"
                row[key] += 1
    table = []
    for cls in ("BC", "TC"):
        for b in BINS:
            row = dict(cells[(cls, b)])
            n = max(row["cells"], 1)
            table.append({"class": cls, "neighbor_bin": b, **row,
                          "miss_rate": round(row["missed"] / n, 4),
                          "misclassified_rate": round(row["misclassified"] / n, 4)})
    for cls in ("BC", "TC"):
        close, far = (next(t for t in table if t["class"] == cls and t["neighbor_bin"] == b)
                      for b in ("14-30", "30+"))
        extra[f"{cls}_close_minus_far_miss_rate"] = round(close["miss_rate"] - far["miss_rate"], 4)
    return {"by_class_and_neighbor": table, "totals": extra,
            "cause_split": score_map is not None}


def print_table(title, result):
    print(f"[진단] {title}")
    for t in result["by_class_and_neighbor"]:
        cause = (f" | 합침 {t['peak_merged']} / 약함 {t['weak_response']}"
                 if result["cause_split"] else "")
        print(f"  {t['class']} {t['neighbor_bin']:>5}: {t['cells']:6d}개, "
              f"누락 {t['miss_rate']:.1%}, 오분류 {t['misclassified_rate']:.1%}{cause}")
    e = result["totals"]
    print(f"  중복 {e['duplicate']} / 오검출 {e['false_positive']} / 예측 {e['predictions']} | "
          f"가까움-멂 누락률 차이 BC {e['BC_close_minus_far_miss_rate']:+.1%}, "
          f"TC {e['TC_close_minus_far_miss_rate']:+.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--compare", help="같은 split의 M0 eval_*_per_image.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, help="점검용 일부만 평가")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    folder = checkpoint.parent
    postprocess_path = folder / "postprocess.json"
    if args.split == "test" and args.tune:
        raise SystemExit("[중단] test로 후처리를 고르지 않습니다. val에서 --tune.")
    if args.split == "test" and not postprocess_path.exists():
        raise SystemExit("[중단] postprocess.json이 없습니다. 먼저 --split val --tune을 실행하세요.")

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    s = state["settings"]
    device = pick_device(args.device)
    model_config = json.loads((ROOT / s["model_config"]).read_text(encoding="utf-8"))
    model = build_model(model_config)
    model.load_state_dict(state["model"])
    model.to(device)

    loader, dataset = make_loader(args.split, s["seed"], args.batch_size,
                                  args.num_workers, s["cell_target"], limit=args.limit)
    organs = {k: v["organ"] for k, v in dataset.source.metadata.items()}
    if args.split == "test" or not args.tune:
        chosen = (json.loads(postprocess_path.read_text())["chosen"]
                  if postprocess_path.exists() else s["postprocess_during_training"])
        postprocess = {k: chosen[k] for k in KEYS}
    else:
        postprocess = s["postprocess_during_training"]

    print(f"[평가] {checkpoint} (epoch {state['epoch'] + 1}) | {args.split} "
          f"{len(loader.dataset)}쌍 | 정답 {s['cell_target']} | 후처리 {postprocess}")
    metrics, records, cached = evaluate(
        model, loader, device, postprocess, size=s["size"], amp=s["amp"],
        tissue_weight=s["tissue_loss_weight"], loss_kwargs=cell_loss_kwargs(s),
        keep_cell_logits=True, organs=organs
    )

    if args.tune:
        print("[탐색] 후처리 격자 (validation 세포 mF1 기준)")
        best, table = tune(cached, s["postprocess_search"], s["size"])
        postprocess_path.write_text(json.dumps({
            "chosen": best, "selected_on": "val", "checkpoint": str(checkpoint),
            "search": table}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[선택] {best} → {postprocess_path}")
        postprocess = {k: best[k] for k in KEYS}
        by_id = {pid: logits for pid, logits, _ in cached}
        for r in records:
            preds = to_full_size(detect_cells_heatmap(
                by_id[r["pair_id"]][None].float(), **postprocess)[0], s["size"])
            r["predictions"] = [[x, y, c, round(p, 4)] for x, y, c, p in preds]
            single = CellF1()
            single.update(preds, r["truths"])
            r.update({k: round(v, 4) for k, v in single.compute().items()})
            r["num_pred"] = len(preds)
        metrics.update({k: best[k] for k in best if k not in KEYS})

    # 진단은 1024 해상도 점수 지도에서만 원인 구분을 한다 (한 장씩 계산).
    score_map = None
    if s["size"] == 1024:
        by_id = {pid: logits for pid, logits, _ in cached}

        def score_map(pair_id):
            _, best_map, _ = heatmap_scores(by_id[pair_id][None].float())
            return gaussian_blur(best_map, postprocess["sigma"])[0].numpy()
    diagnosis = {"postprocess": postprocess,
                 "M2": close_cell_table(records, score_map, postprocess)}
    print_table("M2", diagnosis["M2"])
    if args.compare:
        m0_records = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        if {r["pair_id"] for r in m0_records} != {r["pair_id"] for r in records}:
            # 평가 결과는 그대로 저장하고 비교만 건너뛴다.
            diagnosis["compare_error"] = f"{args.compare}: 이미지 목록이 다름"
            print("[경고] 비교 파일의 이미지 목록이 달라 M0 비교를 건너뜁니다 "
                  "(같은 split인지 확인).")
        else:
            diagnosis["compare_file"] = args.compare
            diagnosis["M0"] = close_cell_table(m0_records)
            print_table("비교 (M0, 원인 구분 없음)", diagnosis["M0"])

    summary = {
        "checkpoint": str(checkpoint), "epoch": state["epoch"] + 1,
        "split": args.split, "pairs": len(records),
        "cell_target": s["cell_target"], "postprocess": postprocess,
        "overall": {k: round(v, 4) for k, v in metrics.items()},
        "by_organ": organ_scores(records)
    }
    (folder / f"eval_{args.split}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (folder / f"eval_{args.split}_per_image.json").write_text(
        json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8")
    (folder / f"close_cells_{args.split}.json").write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    o = summary["overall"]
    print(f"[결과] {args.split}: 세포 mF1 {o['mF1']:.4f} (BC {o['f1_BC']:.4f}, "
          f"TC {o['f1_TC']:.4f}) | 조직 mIoU {o['mIoU']:.4f} "
          f"(BG {o['iou_BG']:.4f}, CA {o['iou_CA']:.4f})")
    for organ, v in summary["by_organ"].items():
        print(f"  {organ:15s} {v['pairs']:3d}쌍  mF1 {v['mF1']:.4f}")


if __name__ == "__main__":
    main()
