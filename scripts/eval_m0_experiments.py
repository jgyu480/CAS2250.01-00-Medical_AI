"""M0 평가.

1) validation에서 후처리 값 고르기 (best.pt로 한 번 추론 후 격자 탐색):
    python scripts/eval_m0.py --checkpoint outputs/m0/<run>/best.pt --split val --tune
   → <run>/postprocess.json 저장

2) test 평가 (설정을 모두 고정한 뒤 마지막에 한 번):
    python scripts/eval_m0.py --checkpoint outputs/m0/<run>/best.pt --split test
   → postprocess.json이 있어야 실행된다. test 결과로 설정을 바꾸지 않는다.

결과: <run>/eval_<split>.json (전체·장기별 지표), <run>/eval_<split>_per_image.json
"""

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.metrics.m0_eval import CellF1, detect_cells
from common.training.m0_loop import (
    pick_device, make_loader, evaluate, to_full_size
)
from common.training.experiment_models import build_model


def organ_scores(records):
    """장기별 세포 mF1 (논문 Table 6처럼). 이미지 결과를 장기별로 합산한다."""
    groups = defaultdict(list)
    for r in records:
        groups[r["organ"]].append(r)
    result = {}
    for organ, items in sorted(groups.items()):
        metric = CellF1()
        for r in items:
            metric.update([tuple(p) for p in r["predictions"]], r["truths"])
        result[organ] = {"pairs": len(items),
                         **{k: round(v, 4) for k, v in metric.compute().items()}}
    return result


def tune(cached, grid, size):
    """캐시된 세포 출력으로 후처리 격자 탐색. 기준: validation 세포 mF1."""
    results = []
    for sigma, distance, threshold in itertools.product(
        grid["sigma"], grid["min_distance"], grid["threshold"]
    ):
        metric = CellF1()
        for _, logits, truths in cached:
            preds = detect_cells(logits[None].float(), sigma=sigma,
                                 min_distance=distance,
                                 threshold=threshold)[0]
            metric.update(to_full_size(preds, size), truths)
        score = metric.compute()
        results.append({"sigma": sigma, "min_distance": distance,
                        "threshold": threshold,
                        **{k: round(v, 4) for k, v in score.items()}})
        print(f"  σ={sigma} d={distance} thr={threshold}: "
              f"mF1 {score['mF1']:.4f}", flush=True)
    return max(results, key=lambda r: r["mF1"]), results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--tune", action="store_true")
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
        raise SystemExit("[중단] postprocess.json이 없습니다. "
                         "먼저 --split val --tune을 실행하세요.")

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    s = state["settings"]
    device = pick_device(args.device)
    model_config = s.get("model_config_snapshot")
    if model_config is None:
        model_config = json.loads((ROOT / s["model_config"]).read_text(encoding="utf-8"))
    model = build_model(model_config, initialize_encoders=False)
    model.load_state_dict(state["model"])
    model.to(device)

    loader, dataset = make_loader(args.split, s["seed"], args.batch_size,
                                  args.num_workers, limit=args.limit)
    organs = {k: v["organ"] for k, v in dataset.source.metadata.items()}
    if args.split == "test" or not args.tune:
        postprocess = (json.loads(postprocess_path.read_text())["chosen"]
                       if postprocess_path.exists()
                       else s["postprocess_during_training"])
        postprocess = {k: postprocess[k]
                       for k in ("sigma", "min_distance", "threshold")}
    else:
        postprocess = s["postprocess_during_training"]

    print(f"[평가] {checkpoint} (epoch {state['epoch'] + 1}) | "
          f"{args.split} {len(loader.dataset)}쌍 | 후처리 {postprocess}")
    metrics, records, cached = evaluate(
        model, loader, device, postprocess, size=s["size"], amp=s["amp"],
        tissue_weight=s["tissue_loss_weight"],
        loss_kwargs={"gamma": s["cell_loss"]["gamma"],
                     "normalize": s["cell_loss"]["normalize"],
                     "min_foreground_fraction":
                         s["cell_loss"].get("min_foreground_fraction", 0.0)},
        keep_cell_logits=args.tune, organs=organs
    )

    if args.tune:
        print("[탐색] 후처리 격자 (validation 세포 mF1 기준)")
        best, table = tune(cached, s["postprocess_search"], s["size"])
        postprocess_path.write_text(json.dumps({
            "chosen": best, "selected_on": "val", "checkpoint": str(checkpoint),
            "search": table
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[선택] {best} → {postprocess_path}")
        # 고른 값으로 다시 계산한 validation 결과를 함께 남긴다.
        postprocess = {k: best[k] for k in ("sigma", "min_distance",
                                            "threshold")}
        for r in records:
            logits = next(l for pid, l, _ in cached if pid == r["pair_id"])
            preds = to_full_size(detect_cells(logits[None].float(),
                                              **postprocess)[0], s["size"])
            r["predictions"] = [[x, y, c, round(p, 4)] for x, y, c, p in preds]
            single = CellF1()
            single.update(preds, r["truths"])
            r.update({k: round(v, 4) for k, v in single.compute().items()})
            r["num_pred"] = len(preds)
        metrics.update({k: best[k] for k in best
                        if k not in ("sigma", "min_distance", "threshold")})

    summary = {
        "checkpoint": str(checkpoint), "epoch": state["epoch"] + 1,
        "split": args.split, "pairs": len(records),
        "postprocess": postprocess,
        "overall": {k: round(v, 4) for k, v in metrics.items()},
        "by_organ": organ_scores(records)
    }
    (folder / f"eval_{args.split}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (folder / f"eval_{args.split}_per_image.json").write_text(
        json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8")

    o = summary["overall"]
    print(f"[결과] {args.split}: 세포 mF1 {o['mF1']:.4f} "
          f"(BC {o['f1_BC']:.4f}, TC {o['f1_TC']:.4f}) | "
          f"조직 mIoU {o['mIoU']:.4f} (BG {o['iou_BG']:.4f}, "
          f"CA {o['iou_CA']:.4f})")
    for organ, v in summary["by_organ"].items():
        print(f"  {organ:15s} {v['pairs']:3d}쌍  mF1 {v['mF1']:.4f}")


if __name__ == "__main__":
    main()
