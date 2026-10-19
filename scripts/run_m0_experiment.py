"""Prepare reproducible VESSL commands; execution requires explicit --execute.

gradient: M0 from SSL, full training trajectory, online diagnostics.
control/m0_1: matched warm starts, identical M0 encoders, fresh non-encoder modules.
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, choices=["gradient", "control", "m0_1"])
    parser.add_argument("--checkpoint", default=os.environ.get("M0_CHECKPOINT"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-name")
    parser.add_argument("--resume")
    parser.add_argument("--gradient-conflict", action="store_true",
                        help="also monitor a warm-start comparison run")
    parser.add_argument("--fp32", action="store_true", help="disable mixed precision")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Invalid batch size / workers")
    config = {"gradient": "configs/m0_train.json", "control": "configs/m0_control_train.json",
              "m0_1": "configs/m0_1_train.json"}[args.job]
    name = args.run_name or f"{args.job}_seed{args.seed}"
    cmd = [sys.executable, "scripts/train_m0_experiments.py", "--config", config,
           "--seed", str(args.seed), "--batch-size", str(args.batch_size),
           "--grad-accum", "1", "--num-workers", str(args.num_workers),
           "--run-name", name]
    if args.job != "gradient":
        cmd += ["--init-scope", "encoders"]
        if not args.resume:
            cmd += ["--init-from", args.checkpoint or "/SET/M0_CHECKPOINT/best.pt"]
    if args.job == "gradient" or args.gradient_conflict:
        cmd += ["--gradient-conflict"]
    if args.fp32:
        cmd += ["--no-amp"]
    if args.resume:
        cmd += ["--resume", args.resume]
    print("[command]", shlex.join(cmd), flush=True)
    if not args.execute:
        print("[dry-run] command prepared; training was not started")
        return
    import torch
    from common.paths import resolve_paths
    paths = resolve_paths()
    if not torch.cuda.is_available():
        raise SystemExit("This launcher requires a CUDA GPU for --execute")
    if not paths["data_root"].is_dir():
        raise SystemExit(f"Dataset missing: {paths['data_root']}; set OCELOT_DATA_ROOT")
    if not all((paths["outputs"] / "audit" / p).is_file()
               for p in ("report.json", "manifest.csv")):
        raise SystemExit("Dataset audit missing under OUTPUT_ROOT; run python scripts/check_data.py first")
    if args.resume:
        if not Path(args.resume).is_file():
            raise SystemExit(f"Resume checkpoint missing: {args.resume}")
    elif args.job == "gradient":
        if not paths["ssl_weights"].is_file():
            raise SystemExit(f"SSL weights missing: {paths['ssl_weights']}; set SSL_WEIGHTS_PATH")
    elif not args.checkpoint or not Path(args.checkpoint).is_file():
        raise SystemExit("Set M0_CHECKPOINT or --checkpoint to the trained M0 best.pt")
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
