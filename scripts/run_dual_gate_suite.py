"""Run learned then fixed gates with matched data and initialization. Default: dry-run."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.paths import resolve_paths


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")

    paths = resolve_paths()
    jobs = []
    for mode in ("learned", "fixed"):
        name = f"dual_gate_{mode}_seed{args.seed}"
        command = [
            sys.executable, "scripts/train_m0_dual_gate.py",
            "--config", "configs/m0_train.json", "--device", "cuda",
            "--seed", str(args.seed), "--size", "1024", "--batch-size", "2",
            "--grad-accum", "1", "--num-workers", str(args.num_workers),
            "--epochs", "50", "--freeze-epochs", "3", "--gate-mode", mode,
            "--meta-count", "40", "--meta-seed", "2042",
            "--gate-every", "5", "--gate-lr", "0.01", "--run-name", name,
        ]
        jobs.append((mode, name, command))
        print("[command]", shlex.join(command), flush=True)

    if not args.execute:
        print("[dry-run] learned 50 epochs -> fixed 50 epochs; training not launched")
        return

    preflight = paths["outputs"] / "dual_gate_preflight_latest.json"
    report = json.loads(preflight.read_text())
    if report.get("status") != "pass" or not report.get("source_sha256"):
        raise SystemExit("GPU preflight has not passed")

    for relative, expected in report["source_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"Code changed since preflight: {relative}")

    if not paths["data_root"].is_dir() or not paths["ssl_weights"].is_file():
        raise SystemExit("Dataset or SSL weights missing; check environment variables")

    logs = paths["outputs"] / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    lock = (logs / "dual_gate_suite.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another dual-gate suite is already running")

    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()) + "\n")
    lock.flush()

    if not args.resume:
        for _, name, _ in jobs:
            folder = paths["outputs"] / "m0" / name
            if (
                (folder.exists() and any(folder.iterdir()))
                or (logs / f"{name}.console.log").exists()
            ):
                raise SystemExit(f"Existing run: {name}; use --resume to continue")

    status_path = logs / f"dual_gate_suite_seed{args.seed}.status.json"
    status = dict(pid=os.getpid(), phase="starting", seed=args.seed, runs={})
    write_json(status_path, status)

    import torch
    summaries, reference_split = [], None

    def inspect(checkpoint, mode):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        settings = state["settings"]
        expected = dict(
            seed=args.seed, size=1024, epochs=50, batch_size=2,
            grad_accum_steps=1, encoder_freeze_epochs=3,
            gate_mode=mode, meta_count=40, meta_seed=2042,
            gate_every=5, gate_lr=0.01
        )
        if any(settings.get(k) != v for k, v in expected.items()):
            raise RuntimeError(f"Checkpoint settings differ: {checkpoint}")
        result = dict(
            epoch=state["epoch"], split=settings["gate_split"],
            ratios=state["gates"]["logits"].sigmoid().tolist()
        )
        del state
        return result

    try:
        for mode, name, original in jobs:
            folder = paths["outputs"] / "m0" / name
            checkpoint = folder / "last.pt"
            command = list(original)
            completed = False

            if args.resume and folder.exists() and any(folder.iterdir()):
                if not checkpoint.is_file():
                    raise RuntimeError(
                        f"No completed epoch checkpoint: {folder}; use a new run name"
                    )
                existing = inspect(checkpoint, mode)
                completed = existing["epoch"] == 49
                command += ["--resume", str(checkpoint)]

            if not completed:
                logfile = logs / f"{name}.console.log"
                status.update(phase="running", current=name)
                status["runs"][name] = dict(status="running", log=str(logfile))
                write_json(status_path, status)
                print("[start]", name, "| log:", logfile, flush=True)
                with logfile.open("a" if args.resume else "x") as handle:
                    subprocess.run(
                        command, cwd=ROOT, stdout=handle,
                        stderr=subprocess.STDOUT, check=True
                    )

            final = inspect(checkpoint, mode)
            if final["epoch"] != 49:
                raise RuntimeError(f"Training did not complete 50 epochs: {name}")

            if reference_split is None:
                reference_split = final["split"]
            elif final["split"] != reference_split:
                raise RuntimeError("Learned and fixed experiments used different splits")

            with (folder / "history.csv").open(newline="") as handle:
                history = list(csv.DictReader(handle))
            if [int(r["epoch"]) for r in history] != list(range(1, 51)):
                raise RuntimeError(f"History epochs missing or duplicated: {name}")

            best = max(history, key=lambda r: float(r["val_mF1"]))
            summaries.append(dict(
                mode=mode, run=name, seed=args.seed, best_epoch=best["epoch"],
                val_mF1=best["val_mF1"], val_f1_BC=best["val_f1_BC"],
                val_f1_TC=best["val_f1_TC"], val_mIoU=best["val_mIoU"],
                sum_epoch_seconds=round(
                    sum(float(r["seconds"]) for r in history), 1
                ),
                model_train_pairs=len(final["split"]["model_train_ids"]),
                gate_meta_pairs=len(final["split"]["gate_meta_ids"]),
                last_cell_other_ratio=final["ratios"][0],
                last_tissue_other_ratio=final["ratios"][1]
            ))
            status["runs"][name] = dict(status="completed", folder=str(folder))
            write_json(status_path, status)
            print(
                "[completed]", name, "| best val mF1:",
                best["val_mF1"], flush=True
            )

        comparison = paths["outputs"] / f"dual_gate_comparison_seed{args.seed}.csv"
        with comparison.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)

        status.update(phase="completed", comparison=str(comparison))
        write_json(status_path, status)
        print(
            "[completed] Both experiments finished. Comparison:",
            comparison, flush=True
        )
    except BaseException as exc:
        status.update(phase="failed", error=str(exc))
        write_json(status_path, status)
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
