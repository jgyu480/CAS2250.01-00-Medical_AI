"""Post-hoc diagnosis of M1 -> M1-1 classification changes by tissue region.

Run after `eval_m1_1.py --split val --tune --ablation`:
  python scripts/analyze_m1_1_errors.py --checkpoint outputs/m1_1/M1_1_seed42/best.pt

GT tissue masks are used ONLY for offline diagnosis, never as a gate input.
Class-agnostic 15px diagnostic matching differs from official per-class mF1.
"""
import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.training.m0_loop import make_loader, pick_device, prepare
from models.M1_1.model import build_model


def match_diagnostics(predictions, truths, cutoff=15):
    """Highest-score-first class-agnostic matching, return status per GT cell."""
    result = ["missed"] * len(truths)
    available = set(range(len(truths)))
    for pred in sorted(predictions, key=lambda p: -float(p[3])):
        x, y, klass = pred[:3]
        candidates = [(math.hypot(float(x)-float(truths[i][0]),
                                  float(y)-float(truths[i][1])), i)
                      for i in available]
        if not candidates:
            break
        distance, i = min(candidates)
        if distance > cutoff:
            continue
        available.remove(i)
        result[i] = "correct" if int(klass) == int(truths[i][2]) else "wrong_class"
    return result


def tissue_region(target, box, x, y):
    # GT centers are expressed in the original 1024px cell image frame.
    left, top, right, bottom = map(float, box)
    u = left + ((float(x) + 0.5) / 1024.0) * (right - left)
    v = top + ((float(y) + 0.5) / 1024.0) * (bottom - top)
    h, w = target.shape
    xx = min(w - 1, max(0, int(u * w)))
    yy = min(h - 1, max(0, int(v * h)))
    klass = int(target[yy, xx])
    return {0: "BG", 1: "CA", 2: "UNK"}.get(klass, "INVALID")


def summarize(values):
    return {"count": len(values), "mean": round(statistics.fmean(values), 4) if values else None,
            "std": round(statistics.pstdev(values), 4) if len(values) > 1 else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int)
    p.add_argument("--no-gate-map", action="store_true", help="Only analyse saved predictions")
    args = p.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    folder = checkpoint.parent
    records_path = folder / f"eval_{args.split}_ablation_per_image.json"
    if not records_path.exists():
        raise SystemExit(f"[STOP] Run eval_m1_1.py --split {args.split} --ablation first")
    variants = json.loads(records_path.read_text(encoding="utf-8"))
    indexed = {name: {str(r["pair_id"]): r for r in records}
               for name, records in variants.items()}
    if "m1" not in indexed or "dynamic" not in indexed:
        raise ValueError("Need both m1 and dynamic ablation records")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    s = state["settings"]
    device = pick_device(args.device)
    model = None
    if not args.no_gate_map:
        config = json.loads((ROOT / s["model_config"]).read_text(encoding="utf-8"))
        model = build_model(config)
        model.load_state_dict(state["model"], strict=True)
        model.set_stage("gate")
        model.to(device).eval()
    loader, _ = make_loader(args.split, s["seed"], args.batch_size,
                            args.num_workers, limit=args.limit)
    groups = defaultdict(lambda: {"GT": 0, "m1_correct": 0, "m1_wrong": 0,
                                  "m1_missed": 0, "gate_correct": 0, "gate_wrong": 0,
                                  "gate_missed": 0, "rescued": 0, "harmed": 0})
    gate_all = []
    gate_by_cell = defaultdict(list)
    gate_by_region = defaultdict(list)
    examples = []
    for batch in loader:
        if model is not None:
            images, _, boxes = prepare(batch, device, s["size"])
            with torch.no_grad():
                out = model(*images, boxes)
                gate = out["gate_map"].float().cpu()
        else:
            gate = None
        for i, pid_raw in enumerate(batch["pair_id"]):
            pid = str(pid_raw)
            if pid not in indexed["m1"] or pid not in indexed["dynamic"]:
                raise ValueError(f"Missing paired ablation prediction for {pid}")
            m1_rec, gate_rec = indexed["m1"][pid], indexed["dynamic"][pid]
            truth = gate_rec["truths"]
            if truth != m1_rec["truths"]:
                raise ValueError(f"GT mismatch for {pid}")
            m1_status = match_diagnostics(m1_rec["predictions"], truth)
            new_status = match_diagnostics(gate_rec["predictions"], truth)
            tissue = batch["tissue_target"][i]
            box = batch["cell_box"][i].tolist()
            for j, ((x, y, cls), before, after) in enumerate(zip(truth, m1_status, new_status)):
                region = tissue_region(tissue, box, x, y)
                typ = {1: "BC", 2: "TC"}.get(int(cls), "OTHER")
                bucket = groups[f"{typ}_in_{region}"]
                bucket["GT"] += 1
                bucket[f"m1_{'wrong' if before == 'wrong_class' else before}"] += 1
                bucket[f"gate_{'wrong' if after == 'wrong_class' else after}"] += 1
                if before != "correct" and after == "correct":
                    bucket["rescued"] += 1
                    examples.append({"pair_id": pid, "x": x, "y": y, "cell": typ,
                                     "region": region, "change": "rescue"})
                if before == "correct" and after != "correct":
                    bucket["harmed"] += 1
                    examples.append({"pair_id": pid, "x": x, "y": y, "cell": typ,
                                     "region": region, "change": "harm"})
                if gate is not None:
                    hh, ww = gate.shape[-2:]
                    gx = min(ww - 1, max(0, int((float(x)+0.5) / 1024.0 * ww)))
                    gy = min(hh - 1, max(0, int((float(y)+0.5) / 1024.0 * hh)))
                    value = float(gate[i, 0, gy, gx])
                    gate_all.append(value)
                    gate_by_cell[typ].append(value)
                    gate_by_region[region].append(value)
    for k, v in groups.items():
        for prefix in ("m1", "gate"):
            denominator = v[f"{prefix}_correct"] + v[f"{prefix}_wrong"]
            v[f"{prefix}_misclassification_rate_among_detected"] = (
                round(v[f"{prefix}_wrong"] / denominator, 5) if denominator else None
            )
            v[f"{prefix}_miss_rate"] = round(v[f"{prefix}_missed"] / v["GT"], 5) if v["GT"] else None
    result = {"checkpoint": str(checkpoint), "split": args.split,
              "matching": "diagnostic, class-agnostic, greedy score-first within 15px",
              "groups": dict(sorted(groups.items())),
              "gate": {"all_GT": summarize(gate_all),
                       "by_cell": {k: summarize(v) for k, v in gate_by_cell.items()},
                       "by_tissue_region": {k: summarize(v) for k, v in gate_by_region.items()}},
              "changed_examples": examples}
    path = folder / f"error_analysis_{args.split}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {path}")
    for key, row in result["groups"].items():
        print(f"  {key:12s} n={row['GT']:4d} rescued={row['rescued']:3d} harmed={row['harmed']:3d}")
    if gate_all:
        print(f"[gate at GT] mean={result['gate']['all_GT']['mean']} "
              f"std={result['gate']['all_GT']['std']}")


if __name__ == "__main__":
    main()
