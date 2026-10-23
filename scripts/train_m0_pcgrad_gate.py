"""M0 with modulewise static/conditional PCGrad gates. Independent entry point."""
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
from common.training.dual_gate import task_gradients
from common.training.dual_gate_run import gate_loaders, atomic_save, capture_rng, restore_rng
from common.training.conditional_pcgrad import PCGradGates, assign_mixed, meta_update
from common.training.conditional_pcgrad_run import Journal, calibrate, fingerprint, write_protocol, shared_calibration
from common.training.experiment_models import build_model
from common.training.gradient_conflict import GradientConflictMonitor
from common.training.m0_loop import seed_everything, pick_device, autocast, make_loader, prepare, evaluate
from scripts.check_pcgrad_gate_gpu import source_hashes


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/m0_train.json")
    p.add_argument("--gate-mode", choices=("static", "conditional"), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--size", type=int, default=1024, choices=(128,256,512,1024))
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--freeze-epochs", type=int, default=3)
    p.add_argument("--meta-count", type=int, default=40)
    p.add_argument("--meta-seed", type=int, default=2042)
    p.add_argument("--gate-every", type=int, default=5)
    p.add_argument("--gate-lr", type=float, default=0.01)
    p.add_argument("--gate-initial", type=float, default=0.1)
    p.add_argument("--ema-decay", type=float, default=0.95)
    p.add_argument("--limit-train", type=int)
    p.add_argument("--limit-val", type=int)
    p.add_argument("--run-name")
    p.add_argument("--resume")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--stop-after-epoch", type=int, help="interrupt/resume verification only")
    return p.parse_args()


def settings(args):
    s = json.loads((ROOT/args.config).read_text())
    s.update(seed=args.seed, size=args.size, batch_size=args.batch_size,
             num_workers=args.num_workers, epochs=args.epochs,
             encoder_freeze_epochs=args.freeze_epochs, limit_train=args.limit_train,
             limit_val=args.limit_val, gate_mode=args.gate_mode,
             meta_count=args.meta_count, meta_seed=args.meta_seed,
             gate_every=args.gate_every, gate_lr=args.gate_lr,
             gate_initial=args.gate_initial, ema_decay=args.ema_decay,
             amp=s["amp"] and not args.no_amp,
             gate_protocol="module_pcgrad_v1_normalized_joint_meta_actual_adamw")
    s["model_config_snapshot"] = json.loads((ROOT/s["model_config"]).read_text())
    if s["model_config_snapshot"].get("model", "M0") != "M0":
        raise ValueError("This experiment requires M0")
    if s["label_source"] != "official" or s["grad_accum_steps"] != 1:
        raise ValueError("Requires official labels and grad accumulation 1")
    if min(args.epochs, args.batch_size, args.gate_every, args.meta_count) < 1:
        raise ValueError("epochs/batch-size/gate-every/meta-count must be positive")
    if args.num_workers < 0 or args.freeze_epochs < 0 or args.gate_lr <= 0:
        raise ValueError("Invalid workers/freeze-epochs/gate-lr")
    return s


def trim_history(path, last_epoch):
    if not path.exists():
        return
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames
        rows = [x for x in r if int(x["epoch"]) <= last_epoch]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    s = settings(args)
    strict_settings = dict(s)
    seed_everything(s["seed"])
    paths, device = resolve_paths(), pick_device(args.device)
    name = args.run_name or f"pcgrad_{args.gate_mode}_seed{args.seed}"
    folder = paths["outputs"] / "m0" / name
    if args.resume:
        if Path(args.resume).resolve().parent != folder.resolve():
            raise ValueError("Resume checkpoint must be in this run directory")
    elif folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Existing output: {folder}; use --resume")
    folder.mkdir(parents=True, exist_ok=True)
    log_file = (folder / "log.txt").open("a")
    def log(*parts):
        line = " ".join(map(str, parts))
        print(line, flush=True)
        log_file.write(line+"\n")
        log_file.flush()

    train_loader, train_set = make_loader(
        "train", s["seed"], s["batch_size"], s["num_workers"],
        label_source=s["label_source"], augment=True, augmentation=s["augmentation"],
        limit=s["limit_train"], shuffle=True)
    val_loader, val_set = make_loader(
        "val", s["seed"], s["batch_size"], s["num_workers"], limit=s["limit_val"])
    train_loader, meta_loader, split = gate_loaders(
        train_loader, train_set, s, args.meta_count, args.meta_seed)
    s["gate_split"] = split
    (folder/"gate_split.json").write_text(json.dumps(split, indent=2)+"\n")
    # Existing completed fixed run is an optional matched-data reference.
    reference = paths["outputs"] / "m0" / f"dual_gate_fixed_seed{args.seed}"
    if not args.limit_train and (reference/"gate_split.json").exists():
        if json.loads((reference/"gate_split.json").read_text()) != split:
            raise ValueError("Split differs from completed fixed-gate control")
        old = json.loads((reference/"config.json").read_text())
        keys = ("seed", "size", "batch_size", "epochs", "augmentation", "cell_loss",
                "encoder_freeze_epochs", "encoder_lr", "new_modules_lr", "weight_decay",
                "grad_clip_norm", "tissue_loss_weight", "postprocess_during_training",
                "amp", "label_source", "model_config_snapshot")
        mismatch = [k for k in keys if old.get(k) != s.get(k)]
        if mismatch:
            raise ValueError(f"Completed control settings differ: {mismatch}")
        log("[reference] completed fixed-gate control has identical split/settings")

    model = build_model(s["model_config_snapshot"], initialize_encoders=not bool(args.resume)).to(device)
    optimizer = torch.optim.AdamW([
        dict(params=model.encoder_parameters(), lr=s["encoder_lr"]),
        dict(params=model.new_module_parameters(), lr=s["new_modules_lr"]),
    ], weight_decay=s["weight_decay"], foreach=False)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    gates = PCGradGates(args.gate_mode, args.gate_initial, args.ema_decay).to(device)
    gate_optimizer = torch.optim.Adam(gates.parameters(), lr=args.gate_lr)
    cast, dtype = autocast(device, s["amp"])
    if dtype == torch.float16:
        raise ValueError("Use BF16 or --no-amp; FP16 is not supported")
    kwargs = dict(gamma=s["cell_loss"]["gamma"], normalize=s["cell_loss"]["normalize"],
                  min_foreground_fraction=s["cell_loss"].get("min_foreground_fraction",0.))
    def loss_fn(out, targets):
        return m0_task_losses(out["cell_logits"], out["tissue_logits"], *targets, **kwargs)

    start, step, best = 0, 0, -1.
    state = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["strict_settings"] != strict_settings or state["settings"]["gate_split"] != split:
            raise ValueError("Resume configuration/split changed; use original arguments")
        if state["settings"].get("source_sha256") != source_hashes():
            raise ValueError("Source changed since checkpoint; inspect before resuming")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        gates.load_state_dict(state["gates"])
        gate_optimizer.load_state_dict(state["gate_optimizer"])
        start, step, best = state["epoch"]+1, state["step"], state["best"]
        scales = state["meta_scales"]
        initial_hash = state["settings"]["initial_model_sha256"]
    else:
        initial_hash = fingerprint(model)
        identity = dict(
            version="shared_initial_meta_scales_v1", model_sha256=initial_hash,
            meta_ids=split["gate_meta_ids"], seed=s["seed"], size=s["size"],
            batch_size=s["batch_size"], cell_loss=s["cell_loss"],
            torch_version=str(torch.__version__), amp_dtype=str(dtype),
            source_sha256=source_hashes())
        scales = shared_calibration(
            paths["outputs"], identity,
            lambda: calibrate(model, meta_loader,
                              lambda b: prepare(b,device,s["size"]), loss_fn, cast))
    try:
        commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = None
    s.update(meta_scales=scales, initial_model_sha256=initial_hash, git_commit=commit,
             source_sha256=source_hashes(),
             device=str(device), amp_dtype=str(dtype), torch=torch.__version__,
             python=platform.python_version(), train_pairs=len(train_loader.dataset),
             meta_pairs=len(meta_loader.dataset), val_pairs=len(val_loader.dataset),
             steps_per_epoch=len(train_loader))
    (folder/"config.json").write_text(json.dumps(s,indent=2)+"\n")
    history = folder/"history.csv"
    fields = ["epoch","encoder_frozen","lr_encoder","lr_new","train_cell","train_tissue",
              "train_total","val_cell","val_tissue","val_total","val_mF1","val_f1_BC",
              "val_f1_TC","val_mIoU","val_iou_BG","val_iou_CA","seconds"]
    if args.resume:
        trim_history(history,start)
    if not history.exists():
        with history.open("w",newline="") as f:
            csv.DictWriter(f,fields).writeheader()
    journal = Journal(folder,start+1)
    write_protocol(folder)
    monitor = GradientConflictMonitor(folder,start+1)
    if state is not None:
        restore_rng(state["rng"])
        del state
    total_steps = len(train_loader)*s["epochs"]
    organs = {k:v["organ"] for k,v in val_set.source.metadata.items()}
    log("[split] model train",s["train_pairs"],"| gate meta",s["meta_pairs"],"| official val",s["val_pairs"])
    log("[gate]",args.gate_mode,"| parameters",sum(p.numel() for p in gates.parameters()),
        "| actual-state AdamW lookahead | fixed meta scales",scales)
    log(f"[시작] {name} | device {device} | amp {dtype} | {s['epochs']} epoch")
    for epoch in range(start,s["epochs"]):
        began = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        frozen = epoch < s["encoder_freeze_epochs"]
        for p in model.encoder_parameters():
            p.requires_grad_(not frozen)
        train_set.set_epoch(epoch)
        train_loader.generator.manual_seed(s["seed"]+epoch)
        meta_loader.generator.manual_seed(args.meta_seed+epoch)
        meta_iterator = iter(meta_loader)
        model.train()
        monitor.begin(epoch+1,model,frozen)
        sums = dict(cell=0.,tissue=0.,total=0.)
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for index,batch in enumerate(train_loader):
            images,targets,boxes = prepare(batch,device,s["size"])
            with cast:
                out = model(*images,boxes)
            losses = loss_fn(out,targets)
            total = losses["cell"]+s["tissue_loss_weight"]*losses["tissue"]
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"Loss NaN/Inf at epoch {epoch+1}, batch {index+1}")
            raw = task_gradients(model,losses["cell"],s["tissue_loss_weight"]*losses["tissue"])
            monitor.observe(None,None,index+1,batch["pair_id"],precomputed_gradients=raw)
            n = images[0].shape[0]
            sums["cell"] += float(losses["cell"].detach())*n
            sums["tissue"] += float(losses["tissue"].detach())*n
            sums["total"] += float(total.detach())*n
            seen += n
            del out,losses,total,images,targets,boxes
            factor = .5*(1+math.cos(math.pi*min(step,total_steps)/max(total_steps,1)))
            for group,lr in zip(optimizer.param_groups,base_lrs):
                group["lr"] = lr*factor
            observation = gates.observe(raw)
            with torch.no_grad():
                before = gates.ratios(observation)
            info,meta_ids = dict(meta_update=0),[]
            if step % args.gate_every == 0:
                try:
                    meta_batch = next(meta_iterator)
                except StopIteration:
                    meta_iterator = iter(meta_loader)
                    meta_batch = next(meta_iterator)
                mi,mt,mb = prepare(meta_batch,device,s["size"])
                meta_ids = meta_batch["pair_id"]
                if set(meta_ids)&set(batch["pair_id"]):
                    raise RuntimeError("Fit/meta batch overlap")
                # Autocast context only surrounds the virtual meta forward, just as
                # real gradients/AdamW arithmetic are FP32 outside autocast.
                def meta_loss(output):
                    return loss_fn(output,mt)
                # Parameters are calculated outside autocast inside meta_update;
                # elementwise AdamW ops preserve FP32 even inside this context.
                with cast:
                    info = meta_update(model,optimizer,gates,gate_optimizer,raw,observation,
                                       (*mi,mb),meta_loss,scales,s.get("grad_clip_norm"))
                del mi,mt,mb,meta_batch,meta_loss
            with torch.no_grad():
                after = gates.ratios(observation)
                mixed = gates.mix(raw,observation,after)
            assign_mixed(model,mixed)
            del mixed,raw
            if s.get("grad_clip_norm"):
                torch.nn.utils.clip_grad_norm_(model.parameters(),s["grad_clip_norm"],
                                               error_if_nonfinite=True,foreach=False)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            journal.record(epoch+1,index+1,step,frozen,gates,observation,before,after,
                           info,batch["pair_id"],meta_ids)
            gates.advance(observation)
            step += 1
            if (index+1)%50 == 0:
                log(f"[batch] epoch {epoch+1} {index+1}/{len(train_loader)} | meta_update {info['meta_update']}")
        log("[gradient]",monitor.finish(len(train_loader)))
        log("[gate]",journal.finish(epoch+1,len(train_loader),gates))
        val,_,_ = evaluate(model,val_loader,device,s["postprocess_during_training"],
                           size=s["size"],amp=s["amp"],tissue_weight=s["tissue_loss_weight"],
                           loss_kwargs=kwargs,organs=organs)
        row = dict(epoch=epoch+1,encoder_frozen=int(frozen),
                   lr_encoder=optimizer.param_groups[0]["lr"],lr_new=optimizer.param_groups[1]["lr"],
                   **{f"train_{k}":v/seen for k,v in sums.items()},
                   val_cell=val["loss_cell"],val_tissue=val["loss_tissue"],val_total=val["loss_total"],
                   val_mF1=val["mF1"],val_f1_BC=val["f1_BC"],val_f1_TC=val["f1_TC"],
                   val_mIoU=val["mIoU"],val_iou_BG=val["iou_BG"],val_iou_CA=val["iou_CA"],
                   seconds=round(time.time()-began,1))
        with history.open("a",newline="") as f:
            csv.DictWriter(f,fields).writerow({k:round(v,6) if isinstance(v,float) else v for k,v in row.items()})
        peak = torch.cuda.max_memory_allocated(device)/1024**3 if device.type=="cuda" else 0
        log(f"[epoch {epoch+1}/{s['epochs']}] train {row['train_total']:.4f} | val {row['val_total']:.4f} | mF1 {val['mF1']:.4f} (BC {val['f1_BC']:.4f}, TC {val['f1_TC']:.4f}) | mIoU {val['mIoU']:.4f} | {row['seconds']}s | peak {peak:.2f} GiB")
        improved = val["mF1"] > best
        best = max(best,val["mF1"])
        checkpoint = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                          gates=gates.state_dict(),gate_optimizer=gate_optimizer.state_dict(),
                          epoch=epoch,step=step,best=best,val=val,settings=s,
                          strict_settings=strict_settings,meta_scales=scales,rng=capture_rng())
        atomic_save(checkpoint,folder/"last.pt")
        if improved:
            atomic_save(checkpoint,folder/"best.pt")
            log("[저장] best.pt | val mF1",best)
        del checkpoint
        if args.stop_after_epoch and epoch+1 >= args.stop_after_epoch:
            log("[verification stop] checkpoint saved; original total epochs unchanged")
            break
    log("[완료]",name,"| best val mF1",best,"|",folder)
    log_file.close()


if __name__ == "__main__":
    main()
