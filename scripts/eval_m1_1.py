"""Evaluate M1-1 with the existing OCELOT official-style metrics/postprocessing.

Validation tuning (only dynamic gate):
  python scripts/eval_m1_1.py --checkpoint outputs/m1_1/M1_1_seed42/best.pt --split val --tune --ablation
Test (requires frozen validation-selected postprocess):
  python scripts/eval_m1_1.py --checkpoint outputs/m1_1/M1_1_seed42/best.pt --split test --ablation

Ablation scores use the SAME dynamic-gate postprocessing, so effects of changing
only the prediction mixture can be assessed without retuning each variant.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.modules.gated_fusion import mixture_log_probs
from common.training.m0_loop import pick_device, make_loader, evaluate, to_full_size
from common.metrics.m0_eval import CellF1, detect_cells
from models.M1_1.model import build_model
from scripts.eval_m1 import organ_scores, tune


class BranchView(nn.Module):
    """Present an ablation as ordinary M0/M1-style model to common evaluator."""
    def __init__(self, model, branch):
        super().__init__()
        self.model = model
        self.branch = branch

    def forward(self, cell_image, tissue_image, cell_box):
        out = self.model(cell_image, tissue_image, cell_box)
        if self.branch == "dynamic":
            return out
        if self.branch == "local":
            logits = out["local_logits"]
        elif self.branch == "m1":
            logits = out["m1_logits"]
        elif self.branch == "fixed_0_5":
            logits = mixture_log_probs(out["local_logits"], out["m1_logits"],
                                       out["gate_map"], fixed_weight=0.5)
        else:
            raise ValueError(f"Unknown ablation {self.branch}")
        return {"cell_logits": logits, "tissue_logits": out["tissue_logits"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    if args.split == "test" and args.tune:
        raise SystemExit("[STOP] Cannot tune on test")
    checkpoint = Path(args.checkpoint).resolve()
    folder = checkpoint.parent
    post_path = folder / "postprocess.json"
    if args.split == "test" and not post_path.exists():
        raise SystemExit("[STOP] Tune validation first (postprocess.json required)")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("stage") != "gate":
        raise SystemExit("[STOP] Use stage-B checkpoint (best.pt), not warmup.pt")
    s = saved["settings"]
    config = json.loads((ROOT / s["model_config"]).read_text(encoding="utf-8"))
    model = build_model(config)
    model.load_state_dict(saved["model"], strict=True)
    model.set_stage("gate")
    device = pick_device(args.device)
    model.to(device).eval()
    loader, dataset = make_loader(args.split, s["seed"], args.batch_size,
                                  args.num_workers, limit=args.limit)
    organs = {k: v["organ"] for k, v in dataset.source.metadata.items()}
    loss_kwargs = {"gamma": s["cell_loss"]["gamma"],
                   "normalize": s["cell_loss"]["normalize"],
                   "min_foreground_fraction": s["cell_loss"].get("min_foreground_fraction", 0.0)}
    if post_path.exists() and not args.tune:
        chosen = json.loads(post_path.read_text(encoding="utf-8"))["chosen"]
        post = {k: chosen[k] for k in ("sigma", "min_distance", "threshold")}
    else:
        post = s["postprocess_during_training"]
    metrics, records, cached = evaluate(
        model, loader, device, post, size=s["size"], amp=s["amp"],
        tissue_weight=s["tissue_loss_weight"], loss_kwargs=loss_kwargs,
        keep_cell_logits=args.tune, organs=organs
    )
    if args.tune:
        best, search = tune(cached, s["postprocess_search"], s["size"])
        post = {k: best[k] for k in ("sigma", "min_distance", "threshold")}
        post_path.write_text(json.dumps({"chosen": best, "selected_on": "val",
                                         "checkpoint": str(checkpoint), "search": search},
                                        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        totals = CellF1()
        cache_map = {pid: (logit, gt) for pid, logit, gt in cached}
        for r in records:
            logits, gt = cache_map[r["pair_id"]]
            preds = to_full_size(detect_cells(logits[None].float(), **post)[0], s["size"])
            r["predictions"] = [[x, y, c, round(conf, 4)] for x, y, c, conf in preds]
            r["num_pred"] = len(preds)
            single = CellF1()
            single.update(preds, gt)
            r.update({k: round(v, 4) for k, v in single.compute().items()})
            totals.update(preds, gt)
        metrics.update(totals.compute())
    summary = {"checkpoint": str(checkpoint), "epoch": saved["epoch"] + 1,
               "split": args.split, "pairs": len(records), "postprocess": post,
               "overall": {k: round(v, 4) for k, v in metrics.items()},
               "by_organ": organ_scores(records)}
    (folder / f"eval_{args.split}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (folder / f"eval_{args.split}_per_image.json").write_text(
        json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[M1-1 {args.split}] dynamic mF1={metrics['mF1']:.4f} "
          f"BC={metrics['f1_BC']:.4f} TC={metrics['f1_TC']:.4f} "
          f"tissue mIoU={metrics['mIoU']:.4f}; post={post}", flush=True)
    if args.ablation:
        branches = {"dynamic": {"overall": summary["overall"],
                                "by_organ": summary["by_organ"]}}
        all_records = {"dynamic": records}
        for branch in ("local", "m1", "fixed_0_5"):
            test_metrics, rows, _ = evaluate(
                BranchView(model, branch), loader, device, post,
                size=s["size"], amp=s["amp"],
                tissue_weight=s["tissue_loss_weight"],
                loss_kwargs=loss_kwargs, organs=organs
            )
            branches[branch] = {"overall": {k: round(v, 4) for k, v in test_metrics.items()},
                                "by_organ": organ_scores(rows)}
            all_records[branch] = rows
            print(f"  {branch:12s} mF1={test_metrics['mF1']:.4f} "
                  f"BC={test_metrics['f1_BC']:.4f} TC={test_metrics['f1_TC']:.4f}", flush=True)
        (folder / f"eval_{args.split}_ablation.json").write_text(
            json.dumps({"split": args.split, "shared_postprocess": post,
                        "variants": branches}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        (folder / f"eval_{args.split}_ablation_per_image.json").write_text(
            json.dumps(all_records, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
