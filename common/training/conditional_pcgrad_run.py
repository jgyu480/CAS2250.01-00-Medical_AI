"""Durable journals and fixed meta-loss scales for conditional PCGrad."""
import csv
import hashlib
import json
from pathlib import Path

import torch

from common.training.conditional_pcgrad import GROUPS, evaluating
from common.training.dual_gate_run import capture_rng, restore_rng


def fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def calibrate(model, loader, prepare_fn, loss_fn, cast):
    """Use ALL held-out training images, no val/test; don't consume training RNG."""
    rng, generator = capture_rng(), loader.generator.get_state()
    sums, seen = [0., 0.], 0
    try:
        with evaluating(model), torch.no_grad():
            for batch in loader:
                images, targets, boxes = prepare_fn(batch)
                with cast:
                    output = model(*images, boxes)
                losses = loss_fn(output, targets)
                n = images[0].shape[0]
                sums[0] += float(losses["cell"]) * n
                sums[1] += float(losses["tissue"]) * n
                seen += n
    finally:
        restore_rng(rng)
        loader.generator.set_state(generator)
    result = [max(v / seen, 1e-8) for v in sums]
    if not all(torch.isfinite(torch.tensor(result))):
        raise FloatingPointError("Invalid meta scales")
    return result


class Journal:
    def __init__(self, folder, start_epoch=1):
        self.folder = Path(folder) / "pcgrad_gate"
        self.folder.mkdir(parents=True, exist_ok=True)
        for file in ("batches.csv", "epochs.csv", "coefficients.csv"):
            path = self.folder / file
            if path.exists():
                with path.open(newline="") as f:
                    reader = csv.DictReader(f)
                    fields = reader.fieldnames
                    rows = [r for r in reader if int(r["epoch"]) < start_epoch]
                with path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fields)
                    writer.writeheader()
                    writer.writerows(rows)
        self.rows = []

    def append(self, name, rows):
        if not rows:
            return
        path = self.folder / name
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, list(rows[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    @torch.no_grad()
    def record(self, epoch, batch, step, frozen, gates, observation, before, after,
               info, pair_ids, meta_ids):
        rows = []
        for key in GROUPS:
            o = observation.get(key)
            active = o is not None
            dot, c2, t2 = ([float(o[n]) for n in ("dot", "c2", "t2")]
                           if active else (0., 0., 0.))
            rc, rt = after[key].detach().cpu().tolist() if active else ("", "")
            old = before[key].detach().cpu().tolist() if active else ("", "")
            conflict = active and o["conflict"]
            wc = 1 - rt * dot / max(c2, 1e-24) if conflict else 1.
            wt = 1 - rc * dot / max(t2, 1e-24) if conflict else 1.
            valid = active and o["valid"]
            row = dict(
                epoch=epoch, batch=batch, step=step, group=key, mode=gates.mode,
                encoder_frozen=int(frozen), active=int(active), valid=int(valid),
                conflict=int(conflict), shared_parameters=o["shared_parameters"] if active else 0,
                shared_tensors=o["shared_tensors"] if active else 0,
                dot=dot, cell_sq=c2, tissue_sq=t2,
                cosine=float(o["cosine"]) if valid else "",
                severity=float(o["severity"]) if valid else "",
                log_tissue_to_cell_norm=float(o["features"][0, 2]) if active else "",
                previous_severity_ema=float(o["features"][0, 3]) if active else "",
                before_rho_cell=old[0], before_rho_tissue=old[1],
                rho_cell=rc, rho_tissue=rt,
                cell_coefficient=wc if active else "", tissue_coefficient=wt if active else "",
                raw_cell_dot=c2+dot if active else "", raw_tissue_dot=t2+dot if active else "",
                corrected_cell_dot=wc*c2+wt*dot if active else "",
                corrected_tissue_dot=wt*t2+wc*dot if active else "",
                raw_cell_adverse=int(c2+dot < -1e-7*(c2+t2)) if valid else "",
                raw_tissue_adverse=int(t2+dot < -1e-7*(c2+t2)) if valid else "",
                corrected_cell_adverse=int(wc*c2+wt*dot < -1e-7*(c2+t2)) if valid else "",
                corrected_tissue_adverse=int(wt*t2+wc*dot < -1e-7*(c2+t2)) if valid else "",
                meta_update=info.get("meta_update", 0),
                meta_cell=info.get("meta_cell", ""), meta_tissue=info.get("meta_tissue", ""),
                meta_objective=info.get("meta_objective", ""),
                gate_grad_norm=info.get("gate_grad_norm", ""),
                pair_ids="|".join(map(str, pair_ids)), meta_pair_ids="|".join(map(str, meta_ids)),
            )
            rows.append(row)
        self.append("batches.csv", rows)
        self.rows.extend(rows)

    def finish(self, epoch, expected_batches, gates):
        summaries = []
        for key in GROUPS:
            all_rows = [r for r in self.rows if r["group"] == key]
            if len(all_rows) != expected_batches:
                raise RuntimeError(f"Missing gate journal batches: {key}")
            valid = [r for r in all_rows if r["valid"]]
            row = dict(epoch=epoch, group=key, mode=gates.mode,
                       observed_batches=len(all_rows), valid_batches=len(valid),
                       meta_updates=sum(r["meta_update"] for r in all_rows))
            for field in ("conflict", "rho_cell", "rho_tissue", "raw_cell_adverse",
                          "raw_tissue_adverse", "corrected_cell_adverse", "corrected_tissue_adverse"):
                row["mean_"+field] = sum(r[field] for r in valid)/len(valid) if valid else ""
            for field in ("rho_cell", "rho_tissue"):
                values = [r[field] for r in valid]
                row["min_"+field] = min(values) if values else ""
                row["max_"+field] = max(values) if values else ""
            summaries.append(row)
        self.append("epochs.csv", summaries)
        coefficients = []
        for i, key in enumerate(GROUPS):
            for j, task in enumerate(("cell", "tissue")):
                values = gates.coefficients[i, j].detach().cpu().tolist()
                coefficients.append(dict(
                    epoch=epoch, group=key, corrected_gradient=task,
                    beta=values[0], gamma=values[1] if len(values)>1 else "",
                    delta=values[2] if len(values)>1 else "",
                    eta=values[3] if len(values)>1 else "",
                    severity_ema=float(gates.ema[i]), ema_count=int(gates.ema_count[i])))
        self.append("coefficients.csv", coefficients)
        self.rows.clear()
        return summaries


def write_protocol(folder):
    (Path(folder) / "pcgrad_gate/protocol.json").write_text(json.dumps(dict(
        method="modulewise partial PCGrad with original opposing gradient anchors",
        groups=GROUPS, gate_heads="correct cell gradient; correct tissue gradient",
        features=["intercept", "negative cosine", "log other/own norm clipped [-8,8]", "previous severity EMA"],
        static="intercept only; 10 parameters", conditional="4 coefficients per head; 40 parameters",
        meta="equal mean of cell/a_cell and tissue/a_tissue; fixed scales from initial meta split",
        lookahead="present-step differentiable AdamW, including state/decay/bias correction/global clipping",
        limitations="first-order local gate meta step; no differentiation through past training; journal adverse rates describe shared-gradient SGD geometry, NOT actual AdamW loss change",
        timing="raw gradients before gate; applied ratios after meta update; EMA advances once after application",
        adverse_tolerance="same cutoff before/after: dot < -1e-7*(cell_sq+tissue_sq)",
    ), indent=2)+"\n")



def shared_calibration(output_root, identity, compute):
    """Compute initial meta scales ONCE; both modes use the same exact values."""
    import fcntl
    import math
    import os
    folder = Path(output_root) / "pcgrad_gate_calibration"
    folder.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(encoded.encode()).hexdigest()
    path = folder / (key + ".json")
    with (folder / (key + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["identity"] != identity:
                raise ValueError("Shared meta calibration identity mismatch")
            values = saved["scales"]
        else:
            values = list(map(float, compute()))
            if len(values) != 2 or not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError("Invalid initial meta scales")
            temp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
            temp.write_text(json.dumps(dict(identity=identity, scales=values), indent=2) + "\n")
            os.replace(temp, path)
        if len(values) != 2 or not all(math.isfinite(float(v)) and float(v) > 0 for v in values):
            raise ValueError("Invalid stored initial meta scales")
    return list(map(float, values))
