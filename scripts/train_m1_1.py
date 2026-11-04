"""Two-stage M1-1 training: frozen M1 checkpoint is mandatory.

python scripts/train_m1_1.py --m1-checkpoint outputs/m1/M1_seed42/best.pt --seed 42
python scripts/train_m1_1.py --resume outputs/m1_1/M1_1_seed42/last.pt
python scripts/train_m1_1.py --m1-checkpoint .../best.pt --smoke --device cpu
"""

import argparse
import csv
import hashlib
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
from common.losses.m0_losses import focal_loss
from common.training.m0_loop import (
    seed_everything, pick_device, autocast, make_loader, prepare, evaluate
)
from models.M1_1.model import build_model
from scripts.train_m0 import git_commit


FIELDS = ["epoch", "stage", "lr", "train_final", "train_local", "train_total",
          "val_cell", "val_tissue", "val_mF1", "val_f1_BC", "val_f1_TC",
          "val_mIoU", "val_iou_BG", "val_iou_CA", "seconds"]


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m1_1_train.json")
    parser.add_argument("--m1-checkpoint", help="Trained M1 best.pt (strict load), required unless resuming")
    parser.add_argument("--resume", help="Resume M1-1 last.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name")
    parser.add_argument("--local-epochs", type=int)
    parser.add_argument("--gate-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--label-source")
    parser.add_argument("--augmentation")
    parser.add_argument("--size", type=int, choices=(128, 256, 512, 1024), default=1024)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Short 128px, 2+2 images, 1+1 epochs")
    return parser.parse_args()


def build_settings(args):
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        s = saved["settings"].copy()
        # No silent change of data, training stages or hyperparameters upon resume.
        if args.m1_checkpoint and str(args.m1_checkpoint) != s["m1_checkpoint"]:
            raise ValueError("Resume checkpoint has a different M1 reference")
        return s
    s = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    for key, value in (("batch_size", args.batch_size),
                       ("grad_accum_steps", args.grad_accum),
                       ("num_workers", args.num_workers),
                       ("label_source", args.label_source),
                       ("augmentation", args.augmentation),
                       ("local_warmup_epochs", args.local_epochs),
                       ("gate_epochs", args.gate_epochs)):
        if value is not None:
            s[key] = value
    s.update(seed=args.seed, size=args.size,
             limit_train=args.limit_train, limit_val=args.limit_val,
             amp=s["amp"] and not args.no_amp)
    if args.smoke:
        s.update(size=128, limit_train=2, limit_val=2, batch_size=1,
                 num_workers=0, grad_accum_steps=1,
                 local_warmup_epochs=1, gate_epochs=1)
    s["epochs"] = s["local_warmup_epochs"] + s["gate_epochs"]
    if s["local_warmup_epochs"] < 1 or s["gate_epochs"] < 1:
        raise ValueError("Both stages need >=1 epoch")
    if not args.m1_checkpoint:
        raise ValueError("--m1-checkpoint trained M1 best.pt is required")
    source = Path(args.m1_checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"M1 checkpoint not found: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    s["m1_checkpoint"] = str(source)
    s["m1_checkpoint_sha256"] = digest.hexdigest()
    return s


def cosine(stage_step, stage_total):
    return 0.5 * (1.0 + math.cos(math.pi * min(stage_step, stage_total) /
                                  max(1, stage_total)))


def main():
    args = args_parser()
    s = build_settings(args)
    seed_everything(s["seed"])
    device = pick_device(args.device)
    paths = resolve_paths()
    run_name = args.run_name or f"M1_1_seed{s['seed']}" + ("_smoke" if args.smoke else "")
    if args.resume:
        outdir = Path(args.resume).resolve().parent
    else:
        outdir = paths["outputs"] / "m1_1" / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    train_loader, train_set = make_loader(
        "train", s["seed"], s["batch_size"], s["num_workers"],
        label_source=s["label_source"], augment=True, augmentation=s["augmentation"],
        limit=s["limit_train"], shuffle=True
    )
    val_loader, val_set = make_loader(
        "val", s["seed"], s["batch_size"], s["num_workers"],
        limit=s["limit_val"]
    )
    config = json.loads((ROOT / s["model_config"]).read_text(encoding="utf-8"))
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        list(model.local_decoder.parameters()) + list(model.local_head.parameters()) +
        list(model.gate.parameters()),
        lr=s["new_modules_lr"], weight_decay=s["weight_decay"]
    )
    cast, amp_dtype = autocast(device, s["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    start_epoch, global_step, best = 0, 0, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        global_step, best = ckpt["step"], ckpt["best"]
        print(f"[M1-1 resume] epoch {start_epoch + 1}/{s['epochs']}", flush=True)
    else:
        state = torch.load(s["m1_checkpoint"], map_location="cpu", weights_only=False)
        if "model" not in state:
            raise KeyError("M1 checkpoint must have a 'model' state dictionary")
        model.teacher.load_state_dict(state["model"], strict=True)
        model.initialize_local_from_teacher()
        model.freeze_teacher()
        print(f"[M1 checkpoint] strict load succeeded: {s['m1_checkpoint']}", flush=True)
    s.update(git_commit=git_commit(), torch=torch.__version__,
             python=platform.python_version(),
             train_pairs=len(train_loader.dataset), val_pairs=len(val_loader.dataset),
             device=str(device), amp_dtype=str(amp_dtype))
    (outdir / "config.json").write_text(
        json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    history_path = outdir / "history.csv"
    if not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, FIELDS).writeheader()
    loss_kwargs = {"gamma": s["cell_loss"]["gamma"],
                   "normalize": s["cell_loss"]["normalize"],
                   "min_foreground_fraction": s["cell_loss"].get("min_foreground_fraction", 0.0)}
    organs = {k: v["organ"] for k, v in val_set.source.metadata.items()}
    steps_per_epoch = math.ceil(len(train_loader) / s["grad_accum_steps"])
    warmup_steps = steps_per_epoch * s["local_warmup_epochs"]
    gate_steps = steps_per_epoch * s["gate_epochs"]
    print(f"[M1-1] {len(train_loader.dataset)}/{len(val_loader.dataset)} pairs | "
          f"local={s['local_warmup_epochs']} gate={s['gate_epochs']} | {device}", flush=True)
    for epoch in range(start_epoch, s["epochs"]):
        begin = time.time()
        stage = "local" if epoch < s["local_warmup_epochs"] else "gate"
        model.set_stage(stage)
        model.train()
        train_set.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        totals = {"final": 0.0, "local": 0.0, "total": 0.0}
        seen = 0
        for index, batch in enumerate(train_loader):
            images, targets, boxes = prepare(batch, device, s["size"])
            with cast:
                output = model(*images, boxes)
                loss_local = focal_loss(output["local_logits"], targets[0], **loss_kwargs)
                if stage == "gate":
                    loss_final = focal_loss(output["cell_logits"], targets[0], **loss_kwargs)
                    total = loss_final + s["local_aux_weight"] * loss_local
                else:
                    loss_final = loss_local
                    total = loss_local
            if not torch.isfinite(total):
                raise RuntimeError(f"NaN/Inf at epoch {epoch + 1} batch {index}")
            scaler.scale(total / s["grad_accum_steps"]).backward()
            is_last = index + 1 == len(train_loader)
            if (index + 1) % s["grad_accum_steps"] == 0 or is_last:
                stage_step = global_step if stage == "local" else global_step - warmup_steps
                stage_total = warmup_steps if stage == "local" else gate_steps
                optimizer.param_groups[0]["lr"] = s["new_modules_lr"] * cosine(stage_step, stage_total)
                if s.get("grad_clip_norm"):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),
                                                   s["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            n = images[0].size(0)
            totals["final"] += loss_final.detach().item() * n
            totals["local"] += loss_local.detach().item() * n
            totals["total"] += total.detach().item() * n
            seen += n
        val, _, _ = evaluate(model, val_loader, device,
                             s["postprocess_during_training"], size=s["size"],
                             amp=s["amp"], tissue_weight=s["tissue_loss_weight"],
                             loss_kwargs=loss_kwargs, organs=organs)
        row = {"epoch": epoch + 1, "stage": stage,
               "lr": optimizer.param_groups[0]["lr"],
               **{f"train_{key}": x / max(1, seen) for key, x in totals.items()},
               "val_cell": val["loss_cell"], "val_tissue": val["loss_tissue"],
               "val_mF1": val["mF1"], "val_f1_BC": val["f1_BC"],
               "val_f1_TC": val["f1_TC"], "val_mIoU": val["mIoU"],
               "val_iou_BG": val["iou_BG"], "val_iou_CA": val["iou_CA"],
               "seconds": time.time() - begin}
        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, FIELDS).writerow({
                k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()
            })
        # Full state makes eval/restart independent of the original M1 weights file.
        # The original file hash/path remain recorded for provenance.
        saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "settings": s,
                 "epoch": epoch, "step": global_step, "best": best,
                 "stage": stage}
        if stage == "gate" and val["mF1"] > best:
            best = val["mF1"]
            saved["best"] = best
            torch.save(saved, outdir / "best.pt")
        if epoch + 1 == s["local_warmup_epochs"]:
            torch.save(saved, outdir / "warmup.pt")
        saved["best"] = best
        torch.save(saved, outdir / "last.pt")
        print(f"[epoch {epoch+1}/{s['epochs']}] {stage} "
              f"loss={row['train_total']:.4f} val_mF1={val['mF1']:.4f} "
              f"(BC={val['f1_BC']:.4f} TC={val['f1_TC']:.4f}) "
              f"tissue_mIoU={val['mIoU']:.4f} best_B={best:.4f}", flush=True)
    print(f"[saved] {outdir}", flush=True)


if __name__ == "__main__":
    main()
