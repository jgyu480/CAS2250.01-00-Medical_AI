"""Real-data, full-resolution CUDA preflight for BOTH new gate modes."""
import csv
import hashlib
import json
import math
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common.paths import resolve_paths
from scripts.check_dual_gate_gpu import choose_pool


def source_hashes():
    files = list((ROOT/"common").rglob("*.py"))+list((ROOT/"models").rglob("*.py"))
    files += list((ROOT/"configs").glob("*.json"))+list((ROOT/"models").rglob("*.json"))
    files += [ROOT/"scripts"/name for name in (
        "train_m0_pcgrad_gate.py","check_pcgrad_gate.py","check_pcgrad_gate_gpu.py",
        "check_pcgrad_gate_integration.py","run_pcgrad_gate_suite.py")]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(files))}


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def verify_run(folder, mode, epochs):
    import torch
    from common.training.conditional_pcgrad import GROUPS
    state = torch.load(folder/"last.pt",map_location="cpu",weights_only=False)
    assert state["epoch"]==epochs-1, "Incomplete training"
    cfg = state["settings"]
    assert cfg["gate_mode"]==mode
    assert state["gates"]["coefficients"].shape==(5,2,1 if mode=="static" else 4)
    assert all(torch.isfinite(v).all() for v in state["gates"].values())
    assert all(torch.isfinite(v).all() for v in state["model"].values())
    assert "optimizer" in state and "gate_optimizer" in state and "rng" in state
    rows = read_csv(folder/"pcgrad_gate/batches.csv")
    steps = cfg["steps_per_epoch"]
    assert len(rows)==epochs*steps*len(GROUPS)
    assert len({(r["epoch"],r["batch"],r["group"]) for r in rows})==len(rows)
    split = cfg["gate_split"]
    fit,meta = set(split["model_train_ids"]),set(split["gate_meta_ids"])
    assert not fit&meta
    for row in rows:
        assert set(row["pair_ids"].split("|")) <= fit
        if row["meta_pair_ids"]:
            assert set(row["meta_pair_ids"].split("|")) <= meta
        if row["active"]=="1":
            assert all(0 <= float(row[k]) <= 1 for k in ("rho_cell","rho_tissue"))
        for k,v in row.items():
            if k in ("meta_objective","gate_grad_norm","corrected_cell_dot","corrected_tissue_dot") and v:
                assert math.isfinite(float(v)), (k,row)
    assert len(read_csv(folder/"pcgrad_gate/epochs.csv"))==epochs*len(GROUPS)
    assert len(read_csv(folder/"pcgrad_gate/coefficients.csv"))==epochs*len(GROUPS)*2
    history = read_csv(folder/"history.csv")
    assert [int(r["epoch"]) for r in history]==list(range(1,epochs+1))
    assert all(math.isfinite(float(r[k])) for r in history for k in ("train_total","val_total","val_mF1","val_mIoU"))
    raw = [r for r in read_csv(folder/"gradient_conflict/batches.csv") if r["group"]=="all_shared"]
    assert len(raw)==epochs*steps
    # The five groups must cover the ENTIRE shared parameter set without overlap.
    counts = {}
    for row in rows:
        key = (row["epoch"],row["batch"])
        counts[key] = counts.get(key,0)+int(row["shared_parameters"])
    for row in raw:
        assert counts[(row["epoch"],row["batch"])]==int(row["shared_parameters"])
    best = max(history,key=lambda r:float(r["val_mF1"]))
    result = dict(settings=cfg,split=split,steps=state["step"],best=best,
                  initial=cfg["initial_model_sha256"],scales=state["meta_scales"],
                  updated=any(float(r["gate_grad_norm"] or 0)>0 for r in rows),
                  sum_epoch_seconds=sum(float(r["seconds"]) for r in history))
    del state
    return result


def main():
    import torch
    paths = resolve_paths()
    assert torch.cuda.is_available(), "CUDA required for real-data preflight"
    audit = json.loads((paths["outputs"]/"audit/report.json").read_text())
    assert audit.get("ok"), "Dataset audit must pass"
    assert paths["data_root"].is_dir() and paths["ssl_weights"].is_file()
    rows = sorted((r for r in read_csv(paths["outputs"]/"audit/manifest.csv") if r["split"]=="train"),key=lambda r:r["pair_id"])
    count = max(16,choose_pool(rows))
    report_path = paths["outputs"]/"pcgrad_gate_preflight_latest.json"
    report_path.write_text(json.dumps(dict(status="running"))+"\n")
    token = uuid.uuid4().hex[:10]
    results = []
    for mode in ("static","conditional"):
        name = f"pcgrad_preflight_{mode}_{token}"
        folder = paths["outputs"]/"m0"/name
        command = [sys.executable,"-u","scripts/train_m0_pcgrad_gate.py",
                   "--gate-mode",mode,"--device","cuda","--size","1024",
                   "--batch-size","2","--num-workers","0","--epochs","2",
                   "--freeze-epochs","1","--gate-every","1","--meta-count","2",
                   "--limit-train",str(count),"--limit-val","2","--run-name",name]
        print("[GPU preflight]",mode,"| full 1024px, batch 2; separate output",flush=True)
        if mode=="conditional":
            subprocess.run(command+["--stop-after-epoch","1"],cwd=ROOT,check=True)
            subprocess.run(command+["--resume",str(folder/"last.pt")],cwd=ROOT,check=True)
        else:
            subprocess.run(command,cwd=ROOT,check=True)
        result = verify_run(folder,mode,2)
        assert result["updated"], "No gate meta signal in preflight; inspect before full training"
        results.append(result)
        print("[PASS]",mode,"full-resolution training, gradients, gate learning, logs, checkpoints",flush=True)
    assert results[0]["split"]==results[1]["split"]
    assert results[0]["initial"]==results[1]["initial"]
    assert results[0]["scales"]==results[1]["scales"]
    report_path.write_text(json.dumps(dict(status="pass",source_sha256=source_hashes(),
                                          initial_sha256=results[0]["initial"],
                                          runs=[f"pcgrad_preflight_{m}_{token}" for m in ("static","conditional")]),indent=2)+"\n")
    print("[PASS] CUDA preflight complete; matched initialization/split/scales and actual resume",flush=True)


if __name__=="__main__":
    main()
