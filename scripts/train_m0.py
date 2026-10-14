"""M0 학습.

본 학습 (VESSL GPU):
    python scripts/train_m0.py --seed 42
맥에서 짧게 동작 확인 (작은 크기·일부 데이터, 성능 의미 없음):
    python scripts/train_m0.py --smoke

결과 폴더: outputs/m0/<run 이름>/
- config.json   실제 사용한 설정 (재현용)
- history.csv   epoch별 학습·검증 손실, 검증 mF1·mIoU, 학습률
- last.pt       마지막 epoch (이어서 학습: --resume)
- best.pt       검증 세포 mF1이 가장 높은 epoch
- log.txt       실행 기록
"""

import argparse
import csv
import json
import math
import platform
import subprocess
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
from models.M0.model import build_model


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m0_train.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-source", help="예: official, annotator:jinho")
    parser.add_argument("--augmentation", help="configs/augmentation.json preset")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", help="last.pt 경로")
    parser.add_argument("--no-amp", action="store_true")
    # 꺼보기(ablation)·fine-tuning용
    parser.add_argument("--model-config",
                        help="모델 설정 파일 (기본: configs/m0_train.json의 model_config)")
    parser.add_argument("--init-from",
                        help="이 checkpoint의 모델 가중치로 시작 (fine-tuning)")
    parser.add_argument("--lr-scale", type=float, default=1.0,
                        help="encoder·새 부품 학습률에 곱할 값 (fine-tuning은 0.1 권장)")
    parser.add_argument("--freeze-epochs", type=int,
                        help="처음 몇 epoch 백본을 고정할지 (기본: 설정 파일 값)")
    parser.add_argument("--eval-init", action="store_true",
                        help="학습 전 상태를 validation으로 평가해 epoch 0으로 기록")
    # 점검용
    parser.add_argument("--size", type=int, default=1024,
                        choices=[128, 256, 512, 1024])
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--smoke", action="store_true",
                        help="256px, train 4쌍, val 2쌍, 2 epoch (백본 고정 1)")
    return parser.parse_args()


def build_settings(args):
    settings = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    overrides = {
        "label_source": args.label_source, "augmentation": args.augmentation,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum, "num_workers": args.num_workers
    }
    settings.update({k: v for k, v in overrides.items() if v is not None})
    if args.model_config:
        settings["model_config"] = args.model_config
    if args.freeze_epochs is not None:
        settings["encoder_freeze_epochs"] = args.freeze_epochs
    if args.lr_scale != 1.0:
        settings["encoder_lr"] *= args.lr_scale
        settings["new_modules_lr"] *= args.lr_scale
    settings.update(init_from=args.init_from, lr_scale=args.lr_scale)
    settings.update(seed=args.seed, size=args.size,
                    limit_train=args.limit_train, limit_val=args.limit_val,
                    amp=settings["amp"] and not args.no_amp)
    if args.smoke:
        settings.update(size=256, limit_train=4, limit_val=2, epochs=2,
                        encoder_freeze_epochs=1, batch_size=2,
                        grad_accum_steps=1, num_workers=0)
    return settings


def set_encoder_trainable(model, trainable):
    for p in model.encoder_parameters():
        p.requires_grad_(trainable)


def cosine_factor(step, total):
    return 0.5 * (1 + math.cos(math.pi * min(step, total) / max(total, 1)))


def main():
    args = parse_args()
    s = build_settings(args)
    seed_everything(s["seed"])
    device = pick_device(args.device)
    paths = resolve_paths()

    name = args.run_name or (
        f"M0_{s['label_source'].replace(':', '-')}_{s['augmentation']}"
        f"_seed{s['seed']}" + ("_smoke" if args.smoke else "")
    )
    folder = paths["outputs"] / "m0" / name
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

    model_config = json.loads((ROOT / s["model_config"]).read_text(
        encoding="utf-8"))
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder_parameters(), "lr": s["encoder_lr"]},
        {"params": model.new_module_parameters(), "lr": s["new_modules_lr"]}
    ], weight_decay=s["weight_decay"])
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    steps_per_epoch = math.ceil(len(train_loader) / s["grad_accum_steps"])
    total_steps = steps_per_epoch * s["epochs"]
    cast, amp_dtype = autocast(device, s["amp"])
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_dtype == torch.float16
    )
    loss_kwargs = {"gamma": s["cell_loss"]["gamma"],
                   "normalize": s["cell_loss"]["normalize"],
                   "min_foreground_fraction":
                       s["cell_loss"].get("min_foreground_fraction", 0.0)}
    organs = {k: v["organ"] for k, v in val_set.source.metadata.items()}

    if args.init_from and not args.resume:
        # 꺼진 연결의 가중치는 모델에 자리가 없으므로 건너뛴다.
        # 모델에 있는 가중치가 checkpoint에 없으면 중단한다 (잘못된 파일 방지).
        source = torch.load(args.init_from, map_location="cpu",
                            weights_only=False)["model"]
        own = model.state_dict()
        missing = [k for k in own if k not in source]
        if missing:
            raise SystemExit(f"[중단] checkpoint에 없는 가중치: {missing[:5]}")
        skipped = sorted({k.split(".")[0] for k in source if k not in own})
        model.load_state_dict({k: v for k, v in source.items() if k in own})
        log(f"[시작 가중치] {args.init_from} | 건너뛴 부품: "
            f"{', '.join(skipped) if skipped else '없음'}")

    start_epoch, step, best = 0, 0, -1.0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu",
                           weights_only=False)
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
        json.dumps(s, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"[시작] {name} | device {device} | amp {amp_dtype} | "
        f"train {s['train_pairs']} / val {s['val_pairs']} | "
        f"batch {s['batch_size']}×{s['grad_accum_steps']} | "
        f"{s['epochs']} epoch")

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
        # 학습 없이 지금 상태를 평가한다 (꺼보기: 끄자마자 얼마나 떨어지는지).
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
                {k: round(v, 6) if isinstance(v, float) else v
                 for k, v in row.items()})
        log(f"[epoch 0 (학습 전)] val {val['loss_total']:.4f} | "
            f"mF1 {val['mF1']:.4f} (BC {val['f1_BC']:.4f}, "
            f"TC {val['f1_TC']:.4f}) | mIoU {val['mIoU']:.4f}")

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
                out = model(*images, boxes)
            losses = m0_task_losses(out["cell_logits"], out["tissue_logits"],
                                    *targets, **loss_kwargs)
            total = losses["cell"] + s["tissue_loss_weight"] * losses["tissue"]
            if not torch.isfinite(total):
                raise SystemExit(f"[중단] epoch {epoch + 1} batch {index}: "
                                 "손실 NaN/Inf")
            scaler.scale(total / s["grad_accum_steps"]).backward()

            last = index + 1 == len(train_loader)
            if (index + 1) % s["grad_accum_steps"] == 0 or last:
                factor = cosine_factor(step, total_steps)
                for group, lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = lr * factor
                if s.get("grad_clip_norm"):
                    # 한 batch의 큰 gradient가 모델을 망가뜨리지 않도록 제한한다.
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), s["grad_clip_norm"])
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
            **{f"train_{k}": v / max(seen, 1) for k, v in sums.items()},
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
                 for k, v in row.items()})
        log(f"[epoch {epoch + 1}/{s['epochs']}]"
            f"{' (백본 고정)' if frozen else ''} "
            f"train {row['train_total']:.4f} "
            f"(cell {row['train_cell']:.4f}, tissue {row['train_tissue']:.4f})"
            f" | val {row['val_total']:.4f} | mF1 {val['mF1']:.4f} "
            f"(BC {val['f1_BC']:.4f}, TC {val['f1_TC']:.4f}) | "
            f"mIoU {val['mIoU']:.4f} | {row['seconds']}s")

        if val["mF1"] > best:
            best = val["mF1"]
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val": val, "settings": s}, folder / "best.pt")
            log(f"[저장] best.pt (val mF1 {best:.4f})")
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch,
                    "step": step, "best": best, "settings": s},
                   folder / "last.pt")

    log(f"[완료] best val mF1 {best:.4f}. 결과: {folder}")
    log_file.close()


if __name__ == "__main__":
    main()
