#!/usr/bin/env bash
set -euo pipefail

root=${ENSEMBLE_ROOT:-/root/ocelot_ensemble}
val=${M3_VAL_RUN:-"$root/experiments/m3_context_d4_retrained_val_seed42"}
repo="$root/m3_runtime/repo"
mode=${1:---m3}
if [[ $# -gt 1 || ( "$mode" != --m3 && "$mode" != --legacy && "$mode" != --historical ) ]]; then
  echo 'Usage: bash run_ocelot_retrained_test.sh [--m3 | --legacy | --historical]' >&2
  exit 2
fi
for name in complete.ok run.json summary.csv postprocess.json m3_context.py; do
  [[ -s "$val/$name" ]] || { echo "Finish the retraining/validation run first: missing $val/$name" >&2; exit 1; }
done
test -s "$val/legacy/ocelot_fusion_comparison.py"
test -d "$repo/.git"
if [[ -x "$root/venv/bin/python" ]]; then
  default_python="$root/venv/bin/python"
elif [[ -x /opt/conda/envs/cas2250-medical-ai/bin/python ]]; then
  default_python=/opt/conda/envs/cas2250-medical-ai/bin/python
else
  default_python=python3
fi

export M3_REPO="$repo" M3_OUTPUT="$val" M3_STATE_ROOT="$root" M3_TEST_MODE="$mode"
out="$val/official_test_${mode#--}"
mkdir -p "$out"
exec 9>"$out/run.lock"
flock -n 9 || { echo "Evaluation already running: $out" >&2; exit 1; }
"${M3_PYTHON:-$default_python}" -u - <<'PY' 2>&1 | tee -a "$out/console.log"
import csv
import fcntl
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

run = Path(os.environ["M3_OUTPUT"]).resolve()
mode = os.environ["M3_TEST_MODE"].removeprefix("--")
out = run / f"official_test_{mode}"
repo = Path(os.environ["M3_REPO"]).resolve()

def read(path):
    return json.loads(path.read_text())

config = read(run / "run.json")
spec = importlib.util.spec_from_file_location("m3_frozen", run / "m3_context.py")
m3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m3)
if config.get("evaluation_splits") != ["val"] or tuple(config["model_order"]) != m3.MODEL_NAMES:
    raise SystemExit("Expected completed validation-only five-model run.")
if config.get("implementation_sha256") != m3.sha256(run / "m3_context.py"):
    raise SystemExit("Stored validation implementation changed.")
head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
if head != config["repo_commit"] or config["adapter_commit"] != "f20e9c6e1b39aa9b0774456744f77b210fe705ab":
    raise SystemExit("Repository or legacy adapter revision does not match validation.")
adapter = subprocess.check_output(
    ["git", "-C", str(repo), "show", config["adapter_commit"] +
     ":results/ensemble/cell_tissue_aspp_seed42/run_ocelot_fusion_comparison.sh"],
    text=True).splitlines(keepends=True)
start = next(i for i, line in enumerate(adapter) if line.startswith("cat > ") and "<<'PYTHON_OCELOT_ENSEMBLE'" in line)
end = next(i for i in range(start + 1, len(adapter)) if adapter[i].strip() == "PYTHON_OCELOT_ENSEMBLE")
if "".join(adapter[start + 1:end]) != Path(m3.legacy.__file__).read_text():
    raise SystemExit("The archived ensemble adapter has changed.")

paths = {name: Path(config["checkpoint_paths"][name]) for name in m3.MODEL_NAMES}
for name, path in paths.items():
    if not path.is_file() or m3.sha256(path) != config["checkpoint_sha256"][name]:
        raise SystemExit(f"Checkpoint missing or changed: {name} {path}")
    if name in m3.M2_FINALS:
        m3.verify_m2(name, path)

device = torch.device(os.environ.get("M3_DEVICE", config["runtime"]["device"]))
if device.type == "cuda" and not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable.")
torch.set_num_threads(2)
data = Path(os.environ.get("OCELOT_DATA_ROOT", config["data_root"])).resolve()

def scoped_audit(split):
    # Read only the requested split's images and annotations.
    audit_root = out / f"{split}_audit"
    audit = audit_root / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    metadata, _ = m3.load_metadata(data / "metadata.json")
    expected = {pair: row for pair, row in metadata.items() if row.get("subset") == split}
    report = {"errors": [], "warnings": []}
    info, ids, _, _, rows = m3.inspect_split(data, split, expected, report)
    expected_count = {"val":137,"test":126}[split]
    if report["errors"] or set(expected) != ids or len(rows) != expected_count:
        raise ValueError(f"{split} audit failed: {report['errors']}; pairs={len(rows)}")
    m3.write_csv(audit / "manifest.csv", sorted(rows, key=lambda row: row["pair_id"]))
    m3.write_json(audit / "report.json", {**report, "ok": True, "splits": {split: info}})
    os.environ.update(OCELOT_DATA_ROOT=str(data), OUTPUT_ROOT=str(audit_root))

def freeze(settings):
    path = out / "frozen_before_test.json"
    settings = json.loads(json.dumps(settings))
    if path.is_file() and read(path) != settings:
        raise SystemExit("Frozen validation selection changed. Keep the original test evaluation.")
    m3.write_json(path, settings)
    print(f"[frozen] {settings['selected_method']}", flush=True)

def best_by_validation(rows, names):
    scores = {row["method"]: float(row["mF1"]) for row in rows if row["method"] in names}
    if set(scores) != set(names):
        raise ValueError("Missing validation results for one or more selection candidates.")
    # Resolve ties by the recorded candidate order, independently of the test split.
    return max(names, key=scores.__getitem__), scores

identity = {"validation_run_sha256": m3.sha256(run / "run.json"),
            "base_checkpoint_sha256": config["checkpoint_sha256"],
            "test_data_root": str(data), "device": str(device)}

if mode == "m3":
    with (run / "summary.csv").open(newline="") as handle:
        val_rows = [row for row in csv.DictReader(handle)
                    if row["split"] == "val_tune" and row["policy"] == "per_method"]
    chosen = {name: tuple(key) for name, key in read(run / "postprocess.json")["per_method"].items()}
    models = {}
    for name in m3.VARIANTS:
        file = run / f"{name}.pt"
        state = torch.load(file, map_location="cpu", weights_only=True)
        if state["variant"] != name or state["epoch"] != config["epochs"]:
            raise ValueError(f"Incomplete ensemble checkpoint: {file}")
        model = m3.Ensemble(name).to(device)
        model.load_state_dict(state["model"])
        models[name] = model.eval()
    selection, scores = best_by_validation(val_rows, m3.METHODS)
    freeze({**identity, "selected_method": selection, "validation_scores": scores,
            "selection_policy": "per_method", "postprocess": m3.group_settings(chosen),
            "combiner_sha256": {n: m3.sha256(run / f"{n}.pt") for n in models}})
    if not (out / "complete.ok").is_file():
        scoped_audit("test")
        test = m3.TorchOcelotDataset(split="test")
        indices = list(range(len(test)))
        cache = out / "cache"
        m3.create_cache(cache, [("test", test, indices)], list(paths.values()), device)
        pairs = m3.Pairs(test, indices, "test", cache, device)
        temporary = out / "predictions.jsonl.gz.part"
        with gzip.open(temporary, "wt", encoding="utf-8") as predictions:
            summary, records = m3.evaluate("test", pairs, models, chosen, predictions)
        temporary.replace(out / "predictions.jsonl.gz")
        m3.write_csv(out / "per_pair.csv", records)
        m3.write_csv(out / "summary.csv", summary)

    # Reuse the per-model D4 maps already materialized by the test evaluator.
    if not (out / "tta_leave_one_out.csv").is_file() or not (out / "tta_error_overlap.csv").is_file():
        from itertools import combinations
        import numpy as np
        if "test" not in locals():
            scoped_audit("test")
            test = m3.TorchOcelotDataset(split="test")
            pairs = m3.Pairs(test, list(range(len(test))), "test", out / "cache", device)
        historical = run / "official_test_historical/frozen_before_test.json"
        if historical.is_file():
            key = tuple(read(historical)["postprocess"]["softctm_tta"])
        else:
            key = tuple(chosen["tta_mean"])
        variants = ("all_five", *(f"without_{name}" for name in m3.MODEL_NAMES))
        totals = {name: m3.CellF1() for name in (*variants, *m3.MODEL_NAMES)}
        diagnostics = {name: {} for name in totals}
        comparisons = tuple(combinations(m3.MODEL_NAMES, 2))
        overlaps = {pair: [0, 0, 0, 0] for pair in comparisons}
        for idx in range(len(pairs)):
            inputs, points = pairs.read(idx)
            truth = np.rint(points).astype(np.int32)
            regions = pairs.existing.regions(idx, truth)
            cell = inputs[0][0].reshape(len(m3.MODEL_NAMES), 2, 1024, 1024)
            summed = cell.sum(dim=0)
            maps = {"all_five": summed / len(cell)}
            maps.update({f"without_{name}": (summed - cell[i]) / (len(cell) - 1)
                         for i, name in enumerate(m3.MODEL_NAMES)})
            maps.update(zip(m3.MODEL_NAMES, cell))
            misses = {}
            for name, score in maps.items():
                _, pred = next(m3.legacy.candidate_points(score, False, [key]))
                totals[name].update(pred, truth)
                labels, _ = m3.localization_labels(pred, truth)
                for label, count in m3.diagnostic_counts(labels, truth, regions).items():
                    diagnostics[name][label] = diagnostics[name].get(label, 0) + count
                if name in m3.MODEL_NAMES:
                    misses[name] = labels == "missed"
            for (left, right), tally in overlaps.items():
                left_miss, right_miss = misses[left], misses[right]
                tally[0] += int(left_miss.sum())
                tally[1] += int(right_miss.sum())
                tally[2] += int((left_miss & right_miss).sum())
                tally[3] += int((left_miss | right_miss).sum())
        baseline = totals["all_five"].compute()["mF1"]
        m3.write_csv(out / "tta_leave_one_out.csv", [
            {"split": "test", "method": name, "n_models": 5 if name == "all_five" else 4 if name in variants else 1,
             **m3.setting_fields(key), **metric.compute(), **m3.metric_counts(metric),
             **diagnostics[name], **m3.diagnostic_rates(diagnostics[name]),
             "delta_vs_all_five": metric.compute()["mF1"] - baseline}
            for name, metric in totals.items()
        ])
        m3.write_csv(out / "tta_error_overlap.csv", [
            {"split": "test", "left": left, "right": right, "left_missed": a,
             "right_missed": b, "both_missed": both, "either_missed": either,
             "miss_jaccard": both / either if either else None}
            for (left, right), (a, b, both, either) in overlaps.items()
        ])

elif mode == "historical":
    import itertools
    from torch.utils.data import DataLoader, Subset
    from common.data.tensors import collate_pairs
    from common.training.m0_loop import autocast, prepare, seed_everything
    from common.data.center_heatmaps import points_to_heatmaps

    old_commit = "7c51ab6d4e026f34e3b66097b285faa21d162add"
    if subprocess.call(["git", "-C", str(repo), "cat-file", "-e", old_commit + "^{commit}"],
                       stderr=subprocess.DEVNULL):
        subprocess.run(["git", "-C", str(repo), "fetch", "--no-tags", "origin",
                        "refs/heads/results/ensemble-comparisons-seed42"], check=True)
    prefix = "results/ensemble/tta_d4_pixel_gate_seed42"
    original = out / "archived_tta_pixel_gate.py"
    original.write_bytes(subprocess.check_output(["git", "-C", str(repo), "show",
                         f"{old_commit}:{prefix}/run_ocelot_tta_pixel_gate.py"]))
    old_spec = importlib.util.spec_from_file_location("archived_pixel_gate", original)
    old = importlib.util.module_from_spec(old_spec)
    old_spec.loader.exec_module(old)
    # Reuse the actual validation cases and order from the archived comparison.
    fusion_prefix = "results/ensemble/cell_tissue_aspp_seed42"
    archived_config = json.loads(subprocess.check_output(["git", "-C", str(repo), "show",
                          f"{config['adapter_commit']}:{fusion_prefix}/run.json"]))
    fit_ids, tune_ids = (sorted(map(str, archived_config[key])) for key in ("fit_ids", "tune_ids"))
    if not fit_ids or not tune_ids or set(fit_ids) & set(tune_ids):
        raise ValueError("Archived validation split is invalid.")
    views = tuple(itertools.product((False, True), range(4)))
    old_identity = {**identity, "archived_commit": old_commit, "seed": 42, "epochs": 12,
                    "fit_ids": fit_ids, "tune_ids": tune_ids, "tta": "D4"}
    seed_everything(42)
    base = [m3.legacy.load_base(p) for p in paths.values()]
    for model, *_ in base:
        model.to(device).eval()

    def infer(batch, transforms):
        return old.infer_maps(base, batch, device, transforms, prepare, autocast)

    def loader(dataset):
        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_pairs)

    val_state = out / "val_tune.pt"
    if val_state.is_file():
        state = torch.load(val_state, map_location="cpu", weights_only=False)
        if state["identity"] != old_identity:
            raise ValueError("Historical selection or inputs changed.")
        chosen, val_rows = state["selected"], state["rows"]
        gate = old.PixelGate(len(paths)).to(device)
        saved = torch.load(out / "pixel_gate.pt", map_location="cpu", weights_only=False)
        if saved["identity"] != old_identity:
            raise ValueError("Historical gate checkpoint does not match the selection.")
        gate.load_state_dict(saved["model"])
        gate.eval()
    else:
        scoped_audit("val")
        val = m3.TorchOcelotDataset(split="val")
        lookup = {str(row["pair_id"]): i for i, row in enumerate(val.source.rows)}
        if set(lookup) != set(fit_ids) | set(tune_ids):
            raise ValueError("Official validation IDs do not match the archived comparison.")
        gate = old.fit_gate(SimpleNamespace(seed=42, epochs=12), old.PixelGate(len(paths)).to(device),
                            loader(Subset(val, [lookup[i] for i in fit_ids])), infer,
                            out / "pixel_gate.pt", device, points_to_heatmaps, old_identity)
        grid = tuple(itertools.product(*m3.legacy.GRID))
        chosen, val_rows = old.evaluate(loader(Subset(val, [lookup[i] for i in tune_ids])),
                            infer, gate, views, dict.fromkeys(old.METHODS, grid),
                            m3.legacy, m3.CellF1, test=False)
        old.atomic_torch_save({"identity": old_identity, "selected": chosen, "rows": val_rows}, val_state)
    selection, scores = best_by_validation(val_rows, old.METHODS)
    freeze({**old_identity, "selected_method": selection, "validation_scores": scores,
            "postprocess": chosen, "gate_sha256": m3.sha256(out / "pixel_gate.pt")})
    if not (out / "complete.ok").is_file():
        scoped_audit("test")
        test = m3.TorchOcelotDataset(split="test")
        _, rows = old.evaluate(loader(test), infer, gate, views,
                               {name: (key,) for name, key in chosen.items()},
                               m3.legacy, m3.CellF1, test=True)
        m3.write_csv(out / "summary.csv", val_rows + rows)


else:
    # Run the archived Python CLI on validation first, passing all five checkpoint overrides.
    # Keep its validation directory stable so its original resume checks still work.
    val_run = out / "legacy_val"
    fusion = val_run / "fusion.pt"
    if not (val_run / "complete.ok").is_file():
        scoped_audit("val")
        args = [
            sys.executable, "-u", m3.legacy.__file__, "--repo", str(repo),
            "--device", str(device), "--output", str(out), "--run-id", "legacy_val",
            "--seed", "42", "--epochs", "12",
        ]
        for name, path in paths.items():
            args.extend(("--model", f"{name}={path}"))
        subprocess.run(args, check=True)
    state = torch.load(fusion, map_location="cpu", weights_only=False)
    if state["config"]["checkpoints"] != {name: str(path) for name, path in paths.items()}:
        raise SystemExit("Archived ensemble validation checkpoints do not match this retrained run.")
    chosen = state["postprocess"]
    names = ("map_mean", *m3.legacy.VARIANTS)
    selection, scores = best_by_validation(state["val_rows"], names)
    freeze({**identity, "selected_method": selection, "validation_scores": scores,
            "postprocess": chosen, "fusion_sha256": m3.sha256(fusion)})
    if not (out / "complete.ok").is_file():
        models = {}
        for name in m3.legacy.VARIANTS:
            model = m3.legacy.FusionCNN(len(paths), name).to(device)
            model.load_state_dict(state["models"][name])
            models[name] = model.eval()
        scoped_audit("test")
        test = m3.TorchOcelotDataset(split="test")
        indices = list(range(len(test)))
        cfg = SimpleNamespace(repo=repo, workers=0)
        folders, activations = m3.legacy.cache_maps(cfg, test, indices, paths, out / "cache", device)
        _, rows = m3.legacy.evaluate(models, m3.legacy.CachedPairs(test, indices, folders),
                                     activations, device, selected=chosen)
        m3.write_csv(out / "summary.csv", [{"split": "test", **row} for row in rows])

with (out / "summary.csv").open(newline="") as handle:
    report = next(row for row in csv.DictReader(handle)
                  if row["split"] == "test" and row["method"] == selection and row.get("policy", "per_method") == "per_method")
m3.write_csv(out / "selected_test_score.csv", [report])
(out / "complete.ok").write_text("complete\n")
print(f"[selected on validation] {selection} | official test Cell mF1={float(report['mF1']):.6f}")
print(f"[results] {out}")
PY
