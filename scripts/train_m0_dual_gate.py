"""Independent dual-gate experiment. Generated from the baseline trainer.
See common/training/dual_gate.py for the gate protocol.
"""
"""M0 / M0-1 experiments with optional full-epoch online gradient diagnostics.

Derived from scripts/train_m0.py; original entry point remains unchanged.

M0 학습.

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

from common.training.dual_gate import (
    BranchGates, task_gradients, assign_gradients, lookahead_losses, update_gates
)
from common.training.dual_gate_run import (
    gate_loaders, atomic_save, capture_rng, restore_rng, GateJournal
)

from common.paths import resolve_paths
from common.losses.m0_losses import m0_task_losses
from common.training.m0_loop import (
    seed_everything, pick_device, autocast, make_loader, prepare, evaluate
)
from common.training.experiment_models import build_model, initialize_from
from common.training.gradient_conflict import GradientConflictMonitor


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
    parser.add_argument("--gradient-conflict", action="store_true",
                        help="record every training minibatch; additional backward cost")
    parser.add_argument("--init-scope", choices=["full", "encoders"], default="full")
    # 점검용
    parser.add_argument("--size", type=int, default=1024,
                        choices=[128, 256, 512, 1024])
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--smoke", action="store_true",
                        help="256px, train 4쌍, val 2쌍, 2 epoch (백본 고정 1)")
    parser.add_argument("--gate-mode", choices=["learned", "fixed"], default="learned")
    parser.add_argument("--meta-count", type=int, default=40)
    parser.add_argument("--meta-seed", type=int, default=2042)
    parser.add_argument("--gate-every", type=int, default=5)
    parser.add_argument("--gate-lr", type=float, default=0.01)
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
    args.gradient_conflict = True
    if args.smoke:
        args.meta_count = 2
    if args.gate_every < 1 or args.gate_lr <= 0:
        raise SystemExit("gate-every and gate-lr must be positive")
    s = build_settings(args)
    s.update(gate_mode=args.gate_mode, meta_count=args.meta_count,
             meta_seed=args.meta_seed, gate_every=args.gate_every, gate_lr=args.gate_lr,
             gate_protocol="two independent gates; diagonal task meta losses; SGD lookahead")
    s.update(gradient_conflict=args.gradient_conflict, init_scope=args.init_scope)
    model_config = json.loads((ROOT / s["model_config"]).read_text(encoding="utf-8"))
    s["model_config_snapshot"] = model_config
    family = model_config.get("model", "M0")
    if family != "M0" or s["label_source"] != "official":
        raise SystemExit("This experiment supports M0 with official labels only")
    if args.resume and args.init_from:
        raise SystemExit("Use --resume OR --init-from, not both.")
    if family == "M0_1" and not args.init_from and not args.resume and not args.smoke:
        raise SystemExit("M0-1 requires --init-from M0_CHECKPOINT --init-scope encoders")
    if family == "M0_1" and args.init_from and args.init_scope != "encoders":
        raise SystemExit("M0-1 transfer requires --init-scope encoders")
    if s["grad_accum_steps"] != 1:
        raise SystemExit("This entry point requires --grad-accum 1 for minibatch/update correspondence.")
    if s["epochs"] < 1 or s["batch_size"] < 1 or s["tissue_loss_weight"] <= 0:
        raise SystemExit("Invalid training settings.")
    seed_everything(s["seed"])
    device = pick_device(args.device)
    paths = resolve_paths()

    name = args.run_name or f"dual_gate_{args.gate_mode}_seed{args.seed}"
    baseline_default_name = (
        f"{family}_{s['label_source'].replace(':', '-')}_{s['augmentation']}"
        f"_seed{s['seed']}" + ("_smoke" if args.smoke else "")
    )
    family_folder = "m0_1" if family == "M0_1" else "m0"
    folder = paths["outputs"] / family_folder / name
    if args.resume:
        if Path(args.resume).resolve().parent != folder.resolve():
            raise SystemExit("--resume must belong to this --run-name output directory")
    elif folder.exists() and any(folder.iterdir()):
        raise SystemExit(f"Output already exists: {folder}; use --resume or another --run-name")
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

    train_loader, meta_loader, split_report = gate_loaders(
        train_loader, train_set, s, args.meta_count, args.meta_seed
    )
    s["gate_split"] = split_report
    (folder / "gate_split.json").write_text(
        json.dumps(split_report, indent=2) + "\n", encoding="utf-8")
    log("[split] model train", len(train_loader.dataset), "| gate meta",
        len(meta_loader.dataset), "| official val", len(val_loader.dataset))
    model_config = json.loads((ROOT / s["model_config"]).read_text(
        encoding="utf-8"))
    model = build_model(model_config, initialize_encoders=not (args.init_from or args.resume)).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder_parameters(), "lr": s["encoder_lr"]},
        {"params": model.new_module_parameters(), "lr": s["new_modules_lr"]}
    ], weight_decay=s["weight_decay"])
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    gates = BranchGates().to(device)
    gate_optimizer = torch.optim.Adam(gates.parameters(), lr=args.gate_lr)
    log("[gate]", args.gate_mode, gates.values(), "| update every", args.gate_every,
        "batches | lookahead: SGD surrogate for real AdamW")

    steps_per_epoch = math.ceil(len(train_loader) / s["grad_accum_steps"])
    total_steps = steps_per_epoch * s["epochs"]
    cast, amp_dtype = autocast(device, s["amp"])
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_dtype == torch.float16
    )
    if amp_dtype == torch.float16:
        raise SystemExit("Dual gate requires BF16 or FP32; use --no-amp on this GPU")
    loss_kwargs = {"gamma": s["cell_loss"]["gamma"],
                   "normalize": s["cell_loss"]["normalize"],
                   "min_foreground_fraction":
                       s["cell_loss"].get("min_foreground_fraction", 0.0)}
    organs = {k: v["organ"] for k, v in val_set.source.metadata.items()}

    if args.init_from and not args.resume:
        report = initialize_from(model, args.init_from, args.init_scope)
        (folder / "initialization.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log("[initialization]", report)

    start_epoch, step, best = 0, 0, -1.0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu",
                           weights_only=False)
        old = state["settings"]
        strict_keys = ("seed", "size", "batch_size", "grad_accum_steps", "augmentation",
                       "label_source", "encoder_freeze_epochs", "cell_loss", "tissue_loss_weight",
                       "encoder_lr", "new_modules_lr", "weight_decay", "grad_clip_norm",
                       "epochs", "amp", "gradient_conflict", "model_config_snapshot",
                       "gate_mode", "meta_count", "meta_seed", "gate_every", "gate_lr",
                       "gate_protocol", "gate_split")
        different = [k for k in strict_keys if old.get(k) != s.get(k)]
        if different:
            raise SystemExit(f"Resume settings differ: {different}; reuse original arguments")
        gates.load_state_dict(state["gates"])
        gate_optimizer.load_state_dict(state["gate_optimizer"])
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

    # Trim history from any interrupted epoch, retaining checkpointed epochs only.
    if args.resume and history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as f:
            past = [r for r in csv.DictReader(f) if int(r["epoch"]) <= start_epoch]
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(past)
    monitor = (GradientConflictMonitor(folder, start_epoch + 1)
               if args.gradient_conflict else None)
    journal = GateJournal(folder, start_epoch + 1)
    if args.resume:
        restore_rng(state["rng"])
    for epoch in range(start_epoch, s["epochs"]):
        began = time.time()
        frozen = epoch < s["encoder_freeze_epochs"]
        set_encoder_trainable(model, not frozen)
        train_set.set_epoch(epoch)
        train_loader.generator.manual_seed(s["seed"] + epoch)
        meta_loader.generator.manual_seed(args.meta_seed + epoch)
        meta_iterator = iter(meta_loader) if args.gate_mode == "learned" else None
        model.train()
        if monitor:
            monitor.begin(epoch + 1, model, frozen)

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
            raw = task_gradients(model, losses["cell"],
                                 s["tissue_loss_weight"] * losses["tissue"])
            if monitor:
                monitor.observe(None, None, index + 1, batch["pair_id"],
                                precomputed_gradients=raw)
            n = images[0].shape[0]
            for key, value in (("cell", losses["cell"]),
                               ("tissue", losses["tissue"]), ("total", total)):
                sums[key] += value.item() * n
            seen += n
            del out, losses, total, value

            factor = cosine_factor(step, total_steps)
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = lr * factor
            before = gates.values()
            info, meta_ids = {}, []
            if args.gate_mode == "learned" and step % args.gate_every == 0:
                try:
                    meta_batch = next(meta_iterator)
                except StopIteration:
                    meta_iterator = iter(meta_loader)
                    meta_batch = next(meta_iterator)
                mi, mt, mb = prepare(meta_batch, device, s["size"])
                meta_ids = meta_batch["pair_id"]
                if set(meta_ids) & set(batch["pair_id"]):
                    raise RuntimeError("Training/meta batch overlap")
                rates = {id(p): group["lr"] for group in optimizer.param_groups
                         for p in group["params"]}
                rates = {name: rates[id(p)] for name, p in model.named_parameters()}
                def meta_loss(output):
                    return m0_task_losses(output["cell_logits"], output["tissue_logits"],
                                          *mt, **loss_kwargs)
                with cast:
                    meta_losses = lookahead_losses(
                        model, gates, raw, rates, (*mi, mb), meta_loss,
                        grad_clip_norm=s.get("grad_clip_norm")
                    )
                meta_values = {"meta_cell": meta_losses["cell"].item(),
                               "meta_tissue": meta_losses["tissue"].item()}
                info = update_gates(gates, gate_optimizer, meta_losses)
                info.update(meta_values)
                del meta_losses, mi, mt, mb, meta_batch
            assign_gradients(model, gates, raw)
            del raw
            if s.get("grad_clip_norm"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), s["grad_clip_norm"],
                                               error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            journal.record(epoch + 1, index + 1, step, frozen, args.gate_mode,
                           before, gates, info, batch["pair_id"], meta_ids)
            step += 1

        if monitor:
            summary = monitor.finish(len(train_loader))
            log("[gradient]", summary)
        log("[gate]", journal.finish(epoch + 1, len(train_loader)))
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

        if device.type == "cuda":
            log("[gpu] peak allocated GiB:", round(torch.cuda.max_memory_allocated(device)/1024**3, 2),
                "| peak reserved GiB:", round(torch.cuda.max_memory_reserved(device)/1024**3, 2))
        gate_state = dict(gates=gates.state_dict(),
                          gate_optimizer=gate_optimizer.state_dict(), rng=capture_rng())
        if val["mF1"] > best:
            best = val["mF1"]
            atomic_save({"model": model.state_dict(), "epoch": epoch,
                        "val": val, "settings": s, **gate_state}, folder / "best.pt")
            log(f"[저장] best.pt (val mF1 {best:.4f})")
        atomic_save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch,
                    "step": step, "best": best, "settings": s, **gate_state},
                   folder / "last.pt")

    log(f"[완료] best val mF1 {best:.4f}. 결과: {folder}")
    log_file.close()


if __name__ == "__main__":
    main()
