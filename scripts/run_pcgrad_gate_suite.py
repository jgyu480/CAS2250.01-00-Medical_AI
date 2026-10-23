"""Verify -> static 50 epochs -> conditional 50 epochs. Dry-run by default."""
import argparse
import csv
import fcntl
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common.paths import resolve_paths
from scripts.check_pcgrad_gate_gpu import source_hashes, verify_run, read_csv


def write_json(path,value):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2)+"\n")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--resume",action="store_true")
    args = p.parse_args()
    jobs = []
    for mode in ("static","conditional"):
        name = f"pcgrad_{mode}_seed{args.seed}"
        command = [sys.executable,"-u","scripts/train_m0_pcgrad_gate.py",
                   "--gate-mode",mode,"--seed",str(args.seed),"--num-workers",str(args.num_workers),
                   "--run-name",name]
        jobs.append((mode,name,command))
        print("[command]",shlex.join(command),flush=True)
    if not args.execute:
        print("[dry-run] checks -> GPU preflight -> static 50 -> conditional 50; no training launched")
        return
    paths = resolve_paths()
    logs = paths["outputs"]/"logs"
    logs.mkdir(parents=True,exist_ok=True)
    lock = (logs/"pcgrad_gate_suite.lock").open("a+")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another PCGrad suite is running")
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid())+"\n")
    lock.flush()
    for _,name,_ in jobs:
        folder = paths["outputs"]/"m0"/name
        if not args.resume and ((folder.exists() and any(folder.iterdir())) or
                                (logs/f"{name}.console.log").exists()):
            raise SystemExit(f"Existing run/log: {name}; use --resume")
    status_path = logs/f"pcgrad_gate_suite_seed{args.seed}.status.json"
    status = dict(pid=os.getpid(),phase="checks",seed=args.seed,runs={})
    write_json(status_path,status)
    try:
        for script in ("check_pcgrad_gate.py","check_pcgrad_gate_integration.py"):
            subprocess.run([sys.executable,"-u",f"scripts/{script}"],cwd=ROOT,check=True)
        report_path = paths["outputs"]/"pcgrad_gate_preflight_latest.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if report.get("status")!="pass" or report.get("source_sha256")!=source_hashes():
            status.update(phase="gpu_preflight")
            write_json(status_path,status)
            subprocess.run([sys.executable,"-u","scripts/check_pcgrad_gate_gpu.py"],cwd=ROOT,check=True)
        report = json.loads(report_path.read_text())
        if report.get("status")!="pass" or report.get("source_sha256")!=source_hashes():
            raise RuntimeError("GPU preflight/source verification failed")
        results,reference = [],None
        for mode,name,original in jobs:
            folder = paths["outputs"]/"m0"/name
            last = folder/"last.pt"
            command = list(original)
            complete = False
            if args.resume and folder.exists() and any(folder.iterdir()):
                if not last.exists():
                    raise RuntimeError(f"No completed checkpoint to resume: {folder}; inspect this run")
                import torch
                state = torch.load(last,map_location="cpu",weights_only=False)
                complete = state["epoch"]==49
                del state
                command += ["--resume",str(last)]
            if not complete:
                logfile = logs/f"{name}.console.log"
                status.update(phase="running",current=name)
                status["runs"][name] = dict(status="running",log=str(logfile))
                write_json(status_path,status)
                print("[start]",name,"|",logfile,flush=True)
                with logfile.open("a" if args.resume else "x") as f:
                    subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
            result = verify_run(folder,mode,50)
            if reference is None:
                reference = result
            else:
                for key in ("split","initial"):
                    if result[key]!=reference[key]:
                        raise RuntimeError(f"Experiments differ in {key}")
                if not all(math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-8)
                           for a,b in zip(result["scales"],reference["scales"])):
                    raise RuntimeError("Initial meta loss scales differ")
                left = dict(reference["settings"])
                right = dict(result["settings"])
                left.pop("gate_mode")
                right.pop("gate_mode")
                left.pop("meta_scales")
                right.pop("meta_scales")
                if left!=right:
                    raise RuntimeError("Matched experiment settings differ")
            best = result["best"]
            results.append(dict(mode=mode,run=name,seed=args.seed,best_epoch=best["epoch"],
                                val_mF1=best["val_mF1"],val_f1_BC=best["val_f1_BC"],
                                val_f1_TC=best["val_f1_TC"],val_mIoU=best["val_mIoU"],
                                sum_epoch_seconds=round(result["sum_epoch_seconds"],1)))
            status["runs"][name] = dict(status="completed",folder=str(folder))
            write_json(status_path,status)
            print("[completed]",name,"| best val mF1",best["val_mF1"],flush=True)
        # Include existing controls when available, without modifying their files.
        for mode in ("fixed","learned"):
            name = f"dual_gate_{mode}_seed{args.seed}"
            folder = paths["outputs"]/"m0"/name
            if (folder/"history.csv").exists() and (folder/"gate_split.json").exists():
                if json.loads((folder/"gate_split.json").read_text())!=reference["split"]:
                    raise RuntimeError(f"Existing control has a different split: {name}")
                history = read_csv(folder/"history.csv")
                best = max(history,key=lambda r:float(r["val_mF1"]))
                results.append(dict(mode="previous_"+mode,run=name,seed=args.seed,
                                    best_epoch=best["epoch"],val_mF1=best["val_mF1"],
                                    val_f1_BC=best["val_f1_BC"],val_f1_TC=best["val_f1_TC"],
                                    val_mIoU=best["val_mIoU"],
                                    sum_epoch_seconds=round(sum(float(r["seconds"]) for r in history),1)))
        comparison = paths["outputs"]/f"pcgrad_gate_comparison_seed{args.seed}.csv"
        with comparison.open("w",newline="") as f:
            w = csv.DictWriter(f,list(results[0]))
            w.writeheader()
            w.writerows(results)
        status.update(phase="completed",comparison=str(comparison))
        write_json(status_path,status)
        print("[completed] Both 50-epoch experiments finished:",comparison,flush=True)
    except BaseException as exc:
        status.update(phase="failed",error=f"{type(exc).__name__}: {exc}")
        write_json(status_path,status)
        raise
    finally:
        lock.close()


if __name__=="__main__":
    main()
