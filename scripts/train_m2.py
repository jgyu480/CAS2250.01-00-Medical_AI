"""M2 학습 (보고서 8.4: 세포 중심 정답 지도 H_AO + 독립 sigmoid 제곱오차).

σ·λ는 scripts/select_gaussian_params.py --scope classwise 결과를
configs/m2_train.json의 cell_target에 적거나 --sigma/--lambda로 준다.

본 학습 (VESSL GPU):
    python scripts/train_m2.py --seed 42
정답 지도 조건 바꾸기 (세 조건 비교 시, σ·λ는 같게):
    python scripts/train_m2.py --target-mode gaussian --seed 42
맥/CPU에서 짧게 동작 확인 (256px, 일부 데이터, 성능 의미 없음):
    python scripts/train_m2.py --smoke
이어서 학습:
    python scripts/train_m2.py --resume outputs/m2/<run>/last.pt
공식 train 안의 분할로 학습 (계수 선택용, 공식 val·test는 쓰지 않는다):
    python scripts/train_m2.py --split-file configs/train_split_seed42.json \
        --train-part fit --val-part select --sigma 4 --target-mode gaussian
저장된 가중치에서 시작 (파인튜닝, optimizer는 새로 만든다):
    python scripts/train_m2.py --init-from outputs/m2/<run>/best.pt --epochs 10 \
        --freeze-epochs 0 --lr-scale 0.1 --sigma 5 ...

결과 폴더: outputs/m2/<run 이름>/  (OUTPUT_ROOT를 따른다)
- config.json   실제 사용한 설정 (재현용)
- history.csv   epoch별 학습·검증 손실, 검증 mF1·mIoU, 학습률
- last.pt       마지막 epoch (이어서 학습: --resume)
- best.pt       검증 세포 mF1이 가장 높은 epoch
- log.txt       실행 기록
학습 loop·optimizer·scheduler·checkpoint 규칙은 scripts/train_m0.py와 같다.
"""

import argparse
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

from common.data.center_heatmaps import check_parameters
from common.paths import resolve_paths
from common.training.m0_loop import seed_everything, pick_device, autocast
from common.training.m2_loop import (
    make_loader, prepare, evaluate, task_losses, cell_loss_kwargs, load_split
)
from models.M2.model import build_model
from scripts.train_m0 import git_commit, set_encoder_trainable, cosine_factor

FIELDS = ["epoch", "encoder_frozen", "lr_encoder", "lr_new",
          "train_cell", "train_tissue", "train_total",
          "val_cell", "val_tissue", "val_total",
          "val_mF1", "val_f1_BC", "val_f1_TC",
          "val_mIoU", "val_iou_BG", "val_iou_CA", "seconds"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/m2_train.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, help="정답 지도 σ (px, 1024 기준)")
    parser.add_argument("--lambda", dest="lam", type=float, help="억제 강도 λ (0 이상 1 미만)")
    parser.add_argument("--target-mode", choices=["gaussian", "ha", "hao"])
    parser.add_argument("--label-source", help="예: official, annotator:jinho")
    parser.add_argument("--augmentation", help="configs/augmentation.json preset")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", help="last.pt 경로 (저장된 설정을 그대로 사용)")
    # 공식 train 안의 분할 (scripts/make_train_split.py)
    parser.add_argument("--split-file", help="분할 파일. 주면 학습·검증 모두 이 파일의 부분을 쓴다")
    parser.add_argument("--train-part", default="fit", help="학습에 쓸 부분 이름")
    parser.add_argument("--val-part", default="select", help="checkpoint 선택에 쓸 부분 이름")
    # 파인튜닝 (scripts/train_m0.py와 같은 옵션)
    parser.add_argument("--init-from", help="이 checkpoint의 모델 가중치로 시작 (optimizer는 새로)")
    parser.add_argument("--lr-scale", type=float, default=1.0,
                        help="encoder·새 부품 학습률에 곱할 값 (파인튜닝은 0.1)")
    parser.add_argument("--freeze-epochs", type=int, help="처음 몇 epoch 백본을 고정할지")
    parser.add_argument("--no-amp", action="store_true")
    # 점검용
    parser.add_argument("--size", type=int, default=1024, choices=[128, 256, 512, 1024])
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--smoke", action="store_true",
                        help="256px, train 4쌍, val 2쌍, 2 epoch (백본 고정 1)")
    return parser.parse_args()


def build_settings(args):
    if args.resume:
        # 이어서 학습할 때 정답·학습 설정이 조용히 바뀌지 않게 저장된 값을 쓴다.
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        s = dict(state["settings"])
        for key, value in (("num_workers", args.num_workers),):
            if value is not None:
                s[key] = value
        return s

    s = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    target = dict(s["cell_target"])
    if args.sigma is not None:
        target["sigma"] = args.sigma
    if args.lam is not None:
        target["lambda"] = args.lam
    if args.target_mode:
        target["mode"] = args.target_mode
    if target.get("sigma") is None or target.get("lambda") is None:
        raise SystemExit("[중단] σ·λ가 비어 있습니다. select_gaussian_params.py "
                         "--scope classwise 결과를 configs/m2_train.json에 적거나 "
                         "--sigma/--lambda로 주세요.")
    check_parameters(float(target["sigma"]), float(target["lambda"]), target["mode"])
    target.update(sigma=float(target["sigma"]), **{"lambda": float(target["lambda"])})
    s["cell_target"] = target

    overrides = {
        "label_source": args.label_source, "augmentation": args.augmentation,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum, "num_workers": args.num_workers
    }
    s.update({k: v for k, v in overrides.items() if v is not None})
    if args.freeze_epochs is not None:
        s["encoder_freeze_epochs"] = args.freeze_epochs
    if args.lr_scale != 1.0:
        s["encoder_lr"] *= args.lr_scale
        s["new_modules_lr"] *= args.lr_scale
    s.update(init_from=args.init_from, lr_scale=args.lr_scale)
    s["data_split"] = None
    if args.split_file:
        if args.train_part == args.val_part:
            raise SystemExit("[중단] 학습 부분과 checkpoint 선택 부분이 같습니다.")
        train_part = load_split(args.split_file, args.train_part)
        val_part = load_split(args.split_file, args.val_part)
        s["data_split"] = {"file": train_part["file"], "sha256": train_part["sha256"],
                           "source_split": train_part["source_split"],
                           "train_part": args.train_part, "val_part": args.val_part}
    s.update(seed=args.seed, size=args.size,
             limit_train=args.limit_train, limit_val=args.limit_val,
             amp=s["amp"] and not args.no_amp)
    if args.smoke:
        s.update(size=256, limit_train=4, limit_val=2, epochs=2,
                 encoder_freeze_epochs=1, batch_size=2,
                 grad_accum_steps=1, num_workers=0)
    return s


def run_name(args, s):
    if args.run_name:
        return args.run_name
    t = s["cell_target"]
    split = s.get("data_split")
    return (f"M2_{t['mode']}_s{t['sigma']:g}_l{t['lambda']:g}"
            f"_{s['label_source'].replace(':', '-')}_seed{s['seed']}"
            + (f"_{Path(split['file']).stem}-{split['train_part']}" if split else "")
            + ("_ft" if s.get("init_from") else "")
            + ("_smoke" if args.smoke else ""))


def main():
    args = parse_args()
    s = build_settings(args)
    seed_everything(s["seed"])
    device = pick_device(args.device)
    paths = resolve_paths()

    name = s.get("run_name") if args.resume else run_name(args, s)
    s["run_name"] = name
    folder = paths["outputs"] / "m2" / name
    folder.mkdir(parents=True, exist_ok=True)
    log_file = (folder / "log.txt").open("a", encoding="utf-8")

    def log(*parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    target = s["cell_target"]
    split = s.get("data_split")
    train_ids = val_ids = None
    if split:
        train_part = load_split(split["file"], split["train_part"])
        val_part = load_split(split["file"], split["val_part"])
        if train_part["sha256"] != split["sha256"]:
            raise SystemExit(f"[중단] 분할 파일이 바뀌었습니다: {split['file']}")
        train_ids, val_ids = train_part["pair_ids"], val_part["pair_ids"]
        if set(train_ids) & set(val_ids):
            raise SystemExit("[중단] 학습 부분과 선택 부분이 겹칩니다.")
    train_loader, train_set = make_loader(
        split["source_split"] if split else "train", s["seed"], s["batch_size"],
        s["num_workers"], target, label_source=s["label_source"], augment=True,
        augmentation=s["augmentation"], limit=s["limit_train"], shuffle=True,
        pair_ids=train_ids
    )
    val_loader, val_set = make_loader(
        split["source_split"] if split else "val", s["seed"], s["batch_size"],
        s["num_workers"], target, limit=s["limit_val"], pair_ids=val_ids
    )

    model_config = json.loads((ROOT / s["model_config"]).read_text(encoding="utf-8"))
    model = build_model(model_config).to(device)
    if s.get("init_from") and not args.resume:
        source = torch.load(s["init_from"], map_location="cpu", weights_only=False)
        model.load_state_dict(source["model"])
        log(f"[시작 가중치] {s['init_from']} (epoch {source['epoch'] + 1}, "
            f"정답 {source['settings']['cell_target']['mode']} "
            f"σ={source['settings']['cell_target']['sigma']:g})")
        del source
    optimizer = torch.optim.AdamW([
        {"params": model.encoder_parameters(), "lr": s["encoder_lr"]},
        {"params": model.new_module_parameters(), "lr": s["new_modules_lr"]}
    ], weight_decay=s["weight_decay"])
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    steps_per_epoch = math.ceil(len(train_loader) / s["grad_accum_steps"])
    total_steps = steps_per_epoch * s["epochs"]
    cast, amp_dtype = autocast(device, s["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    loss_kwargs = cell_loss_kwargs(s)
    organs = {k: v["organ"] for k, v in val_set.source.metadata.items()}

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
        json.dumps(s, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"[시작] {name} | device {device} | amp {amp_dtype} | "
        f"정답 {target['mode']} σ={target['sigma']:g} λ={target['lambda']:g} | "
        f"{'분할 ' + split['file'] + ' ' + split['train_part'] + '/' + split['val_part'] + ' | ' if split else ''}"
        f"train {s['train_pairs']} / val {s['val_pairs']} | "
        f"batch {s['batch_size']}×{s['grad_accum_steps']} | {s['epochs']} epoch")

    history_path = folder / "history.csv"
    if not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, FIELDS).writeheader()

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
            losses = task_losses(out["cell_logits"], out["tissue_logits"],
                                 *targets, **loss_kwargs)
            total = losses["cell"] + s["tissue_loss_weight"] * losses["tissue"]
            if not torch.isfinite(total):
                raise SystemExit(f"[중단] epoch {epoch + 1} batch {index}: 손실 NaN/Inf")
            scaler.scale(total / s["grad_accum_steps"]).backward()

            last = index + 1 == len(train_loader)
            if (index + 1) % s["grad_accum_steps"] == 0 or last:
                factor = cosine_factor(step, total_steps)
                for group, lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = lr * factor
                if s.get("grad_clip_norm"):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), s["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

            n = images[0].shape[0]
            for key, value in (("cell", losses["cell"]), ("tissue", losses["tissue"]),
                               ("total", total)):
                sums[key] += value.item() * n
            seen += n

        val, _, _ = evaluate(
            model, val_loader, device, s["postprocess_during_training"],
            size=s["size"], amp=s["amp"], tissue_weight=s["tissue_loss_weight"],
            loss_kwargs=loss_kwargs, organs=organs
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
            "val_iou_CA": val["iou_CA"], "seconds": round(time.time() - began, 1)
        }
        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, FIELDS).writerow(
                {k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()})
        log(f"[epoch {epoch + 1}/{s['epochs']}]{' (백본 고정)' if frozen else ''} "
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
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "step": step,
                    "best": best, "settings": s}, folder / "last.pt")

    log(f"[완료] best val mF1 {best:.4f}. 결과: {folder}")
    log_file.close()


if __name__ == "__main__":
    main()
