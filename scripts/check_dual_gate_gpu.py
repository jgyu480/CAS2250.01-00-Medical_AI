"""Two-epoch real-data CUDA preflight; separate output, not a research result."""
import csv
import hashlib
import json
import math
import random
import re
import shlex
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.paths import resolve_paths


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def choose_pool(rows):
    for count in range(6, min(len(rows), 64) + 1):
        groups = {}
        for row in rows[:count]:
            slide = row["slide_name"]
            match = re.search(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", slide.upper())
            case = match.group(0) if match else slide
            groups[case] = groups.get(case, 0) + 1
        if len(groups) < 3:
            continue
        keys = sorted(groups)
        random.Random(2042).shuffle(keys)
        held = 0
        for case in keys:
            if held >= 2:
                break
            if held + groups[case] < count:
                held += groups[case]
        if held >= 2 and count - held >= 4:
            return count
    raise SystemExit("Cannot select a small preflight pool with >=3 cases")


def main():
    paths = resolve_paths()
    for path in (
        paths["data_root"], paths["ssl_weights"],
        paths["outputs"] / "audit/report.json",
        paths["outputs"] / "audit/manifest.csv",
    ):
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")

    audit = json.loads((paths["outputs"] / "audit/report.json").read_text())
    if not audit.get("ok"):
        raise SystemExit("Dataset audit did not pass")

    rows = sorted(
        (r for r in read_csv(paths["outputs"] / "audit/manifest.csv")
         if r["split"] == "train"),
        key=lambda r: r["pair_id"]
    )
    count = choose_pool(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"dual_gate_preflight_{stamp}_{uuid.uuid4().hex[:6]}"
    folder = paths["outputs"] / "m0" / name
    report_path = paths["outputs"] / "dual_gate_preflight_latest.json"
    report_path.write_text(
        json.dumps(dict(status="running", run=name)) + "\n"
    )

    command = [
        sys.executable, "scripts/train_m0_dual_gate.py",
        "--config", "configs/m0_train.json", "--device", "cuda",
        "--seed", "42", "--size", "1024", "--batch-size", "2",
        "--grad-accum", "1", "--num-workers", "0",
        "--epochs", "2", "--freeze-epochs", "1",
        "--gate-mode", "learned", "--gate-every", "1",
        "--gate-lr", "0.01", "--meta-count", "2", "--meta-seed", "2042",
        "--limit-train", str(count), "--limit-val", "2", "--run-name", name,
    ]
    print("[preflight] pool:", count, "| 1024px, batch 2, 2 epochs", flush=True)
    print("[command]", shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

    print("[check] completed checkpoint reload on CUDA", flush=True)
    subprocess.run(
        command + ["--resume", str(folder / "last.pt")],
        cwd=ROOT, check=True
    )

    import torch
    state = torch.load(
        folder / "last.pt", map_location="cpu", weights_only=False
    )
    settings = state["settings"]
    assert state["epoch"] == 1
    assert all(
        key in state for key in ("gates", "gate_optimizer", "optimizer", "rng")
    )
    steps = settings["steps_per_epoch"]
    assert state["step"] == 2 * steps

    split = settings["gate_split"]
    fit, meta = set(split["model_train_ids"]), set(split["gate_meta_ids"])
    assert not fit & meta
    assert fit | meta == {r["pair_id"] for r in rows[:count]}
    print("[PASS] split, checkpoint and optimizer states", flush=True)

    history = read_csv(folder / "history.csv")
    assert len(history) == 2
    for row in history:
        for key in ("train_total", "val_total", "val_mF1", "val_mIoU"):
            assert math.isfinite(float(row[key])), (key, row)

    batches = read_csv(folder / "dual_gate/batches.csv")
    assert len(batches) == 2 * steps
    assert len({(r["epoch"], r["batch"]) for r in batches}) == len(batches)
    assert all(r["meta_update"] == "1" for r in batches)

    for row in batches:
        assert set(row["pair_ids"].split("|")) <= fit
        assert set(row["meta_pair_ids"].split("|")) <= meta
        for key in ("cell_other_ratio", "tissue_other_ratio"):
            assert 0 < float(row[key]) < 1

    for branch in ("cell", "tissue"):
        derivatives = [
            float(r[branch + "_gate_derivative"])
            for r in batches if r["epoch"] == "2"
        ]
        assert all(math.isfinite(x) for x in derivatives)
        assert any(x != 0 for x in derivatives), (
            f"No {branch} gate learning signal"
        )

    ratios = state["gates"]["logits"].sigmoid().tolist()
    assert all(math.isfinite(x) for x in ratios)
    for key, value in zip(
        ("cell_other_ratio", "tissue_other_ratio"), ratios
    ):
        assert abs(float(batches[-1][key]) - value) < 1e-6
        assert any(abs(float(r[key]) - 0.5) > 1e-7 for r in batches)

    print("[PASS] both gates changed; every batch recorded", ratios, flush=True)

    gradients = [
        r for r in read_csv(folder / "gradient_conflict/batches.csv")
        if r["group"] == "all_shared"
    ]
    assert len(gradients) == len(batches)
    assert all(r["valid"] == "1" for r in gradients if r["epoch"] == "2")
    assert len(read_csv(folder / "dual_gate/epochs.csv")) == 2
    assert (folder / "best.pt").stat().st_size > 0

    critical = (
        "common/training/dual_gate.py",
        "common/training/dual_gate_run.py",
        "common/training/gradient_conflict.py",
        "scripts/train_m0_dual_gate.py",
        "configs/m0_train.json",
        "models/M0/config.json",
    )
    report = dict(
        status="pass", run=name, folder=str(folder), size=1024, batch_size=2,
        epochs=2, amp_dtype=settings["amp_dtype"], last_gate_ratios=ratios,
        source_sha256={
            p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
            for p in critical
        }
    )
    temp = report_path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(report_path)

    print("[PASS] raw gradient logs and epoch summaries", flush=True)
    print("[saved]", folder, flush=True)
    print("[PASS] 4/6단계 완료 — GPU 검사만 완료, 본학습 미시작", flush=True)


if __name__ == "__main__":
    main()
