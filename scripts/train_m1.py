"""Train M1 with exactly the same data/loss/optimizer/evaluator as M0.

Run at repository root:
  python scripts/train_m1.py --seed 42 --run-name M1_seed42
  python scripts/train_m1.py --smoke
  python scripts/train_m1.py --resume outputs/m1/M1_seed42/last.pt

By design this imports M0's shared training helpers; does not edit M0 files.
Outputs: outputs/m1/<run>/config.json, history.csv, best.pt, last.pt, log.txt
"""

import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import resolve_paths
from common.losses.m0_losses import m0_task_losses
from common.training.m0_loop import (
    seed_everything, pick_device, autocast, make_loader, prepare, evaluate
)
from models.M1.model import build_model
from scripts.train_m0 import (
    parse_args, build_settings, set_encoder_trainable,
    cosine_factor, git_commit
)


def main():
    args = parse_args()
    # Keep the entire M0 CLI and settings parser; switch only the default config.
    if args.config == "configs/m0_train.json":
        args.config = "configs/m1_train.json"
    s = build_settings(args)
    if s["model_config"] != "models/M1/config.json" and not args.model_config:
        raise ValueError("M1 runner requires models/M1/config.json")
    seed_everything(s["seed"])
    device = pick_device(args.device)
    paths = resolve_paths()
    name = args.run_name or (
        f"M1_{s['label_source'].replace(':', '-')}_{s['augmentation']}"
        f"_seed{s['seed']}" + ("_smoke" if args.smoke else "")
    )
    folder = paths["outputs"] / "m1" / name
    folder.mkdir(parents=True, exist_ok=True)
    log_file = (folder / "log.txt").open("a", encoding="utf-8")

    def log(*parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    train_loader, train_set = make_loader(
        "train", s["seed"], s["batch_size"], s["num_workers"],
        label_source=s["label_source"], augment=True,
        augmentation=s["augmentation"], limit=s["limit_train"], shuffle=True
    )
    val_loader, val_set = make_loader(
        "val", s["seed"], s["batch_size"], s["num_workers"],
        limit=s["limit_val"]
    )
    model_config = json.loads((ROOT / s["model_config"]).read_text(encoding="utf-8"))
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder_parameters(), "lr": s["encoder_lr"]},
        {"params": model.new_module_parameters(), "lr": s["new_modules_lr"]}
    ], weight_decay=s["weight_decay"])
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    steps_per_epoch = math.ceil(len(train_loader) / s["grad_accum_steps"])
    total_steps = steps_per_epoch * s["epochs"]
    cast, amp_dtype = autocast(device, s["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    loss_kwargs = {
        "gamma": s["cell_loss"]["gamma"],
        "normalize": s["cell_loss"]["normalize"],
        "min_foreground_fraction": s["cell_loss"].get("min_foreground_fraction", 0.0)
    }
    organs = {key: value["organ"] for key, value in val_set.source.metadata.items()}

    if args.init_from and not args.resume:
        source = torch.load(args.init_from, map_location="cpu", weights_only=False)["model"]
        # Unlike M0->M0 fine-tuning, M0->M1 includes new attention params.
        # Require an M1-compatible checkpoint rather than silently dropping keys.
        model.load_state_dict(source, strict=True)
        log(f"[시작 가중치] M1-compatible checkpoint: {args.init_from}")
    start_epoch, step, best = 0, 0, -1.0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, step, best = state["epoch"] + 1, state["step"], state["best"]
        log(f"[이어서] {args.resume} → epoch {start_epoch + 1}부터")

    s.update(device=str(device), amp_dtype=str(amp_dtype),
             git_commit=git_commit(), torch=torch.__version__,
             python=platform.python_version(),
             train_pairs=len(train_loader.dataset),
             val_pairs=len(val_loader.dataset),
             steps_per_epoch=steps_per_epoch)
    (folder / "config.json").write_text(
        json.dumps(s, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"[시작] {name} | device {device} | amp {amp_dtype} | "
        f"train {s['train_pairs']} / val {s['val_pairs']} | "
        f"batch {s['batch_size']}×{s['grad_accum_steps']} | {s['epochs']} epoch")

    history_path = folder / "history.csv"
    fields = ["epoch", "encoder_frozen", "lr_encoder", "lr_new",
              "train_cell", "train_tissue", "train_total",
              "val_cell", "val_tissue", "val_total",
              "val_mF1", "val_f1_BC", "val_f1_TC",
              "val_mIoU", "val_iou_BG", "val_iou_CA", "seconds"]
    if not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fields).writeheader()

    if args.eval_init and start_epoch == 0:
        began = time.time()
        val, _, _ = evaluate(
            model, val_loader, device, s["postprocess_during_training"],
            size=s["size"], amp=s["amp"],
            tissue_weight=s["tissue_loss_weight"], loss_kwargs=loss_kwargs,
            organs=organs
        )
        row = {"epoch": 0, "encoder_frozen": "",
               "val_cell": val["loss_cell"], "val_tissue": val["loss_tissue"],
               "val_total": val["loss_total"], "val_mF1": val["mF1"],
               "val_f1_BC": val["f1_BC"], "val_f1_TC": val["f1_TC"],
               "val_mIoU": val["mIoU"], "val_iou_BG": val["iou_BG"],
               "val_iou_CA": val["iou_CA"],
               "seconds": round(time.time() - began, 1)}
        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fields).writerow(
                {k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()}
            )
        log(f"[epoch 0 (학습 전)] val {val['loss_total']:.4f} | mF1 {val['mF1']:.4f}")

    for epoch in range(start_epoch, s["epochs"]):
        began = time.time()
        frozen = epoch < s["encoder_freeze_epochs"]
        set_encoder_trainable(model, not frozen)
        train_set.set_epoch(epoch)
        model.train()
        sums = {"cell": 0.0, "tissue": 0.0, "total": 0.0}
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(train_loader):
            images, targets, boxes = prepare(batch, device, s["size"])
            with cast:
                output = model(*images, boxes)
            losses = m0_task_losses(
                output["cell_logits"], output["tissue_logits"],
                *targets, **loss_kwargs
            )
            total = losses["cell"] + s["tissue_loss_weight"] * losses["tissue"]
            if not torch.isfinite(total):
                raise SystemExit(f"[중단] epoch {epoch + 1} batch {index}: NaN/Inf")
            scaler.scale(total / s["grad_accum_steps"]).backward()
            last = index + 1 == len(train_loader)
            if (index + 1) % s["grad_accum_steps"] == 0 or last:
                factor = cosine_factor(step, total_steps)
                for group, lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = lr * factor
                if s.get("grad_clip_norm"):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), s["grad_clip_norm"]
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            n = images[0].shape[0]
            for key, value in (("cell", losses["cell"]),
                               ("tissue", losses["tissue"]),
                               ("total", total)):
                sums[key] += value.item() * n
            seen += n

        val, records, _ = evaluate(
            model, val_loader, device, s["postprocess_during_training"],
            size=s["size"], amp=s["amp"],
            tissue_weight=s["tissue_loss_weight"], loss_kwargs=loss_kwargs,
            organs=organs
        )
        row = {
            "epoch": epoch + 1, "encoder_frozen": frozen,
            "lr_encoder": optimizer.param_groups[0]["lr"],
            "lr_new": optimizer.param_groups[1]["lr"],
            **{f"train_{k}": value / max(seen, 1) for k, value in sums.items()},
            "val_cell": val["loss_cell"], "val_tissue": val["loss_tissue"],
            "val_total": val["loss_total"], "val_mF1": val["mF1"],
            "val_f1_BC": val["f1_BC"], "val_f1_TC": val["f1_TC"],
            "val_mIoU": val["mIoU"], "val_iou_BG": val["iou_BG"],
            "val_iou_CA": val["iou_CA"],
            "seconds": round(time.time() - began, 1)
        }
        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fields).writerow(
                {k: round(v, 6) if isinstance(v, float) else v
                 for k, v in row.items()}
            )
        log(f"[epoch {epoch + 1}/{s['epochs']}]"
            f"{' (백본 고정)' if frozen else ''} "
            f"train {row['train_total']:.4f} "
            f"(cell {row['train_cell']:.4f}, tissue {row['train_tissue']:.4f})"
            f" | val {val['loss_total']:.4f} | mF1 {val['mF1']:.4f} "
            f"(BC {val['f1_BC']:.4f}, TC {val['f1_TC']:.4f}) "
            f"| mIoU {val['mIoU']:.4f} | {row['seconds']}s")
        if val["mF1"] > best:
            best = val["mF1"]
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val": val, "settings": s}, folder / "best.pt")
            log(f"[저장] best.pt (val mF1 {best:.4f})")
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch,
                    "step": step, "best": best, "settings": s}, folder / "last.pt")
    log(f"[완료] best val mF1 {best:.4f}. 결과: {folder}")
    log_file.close()


if __name__ == "__main__":
    main()
