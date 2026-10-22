"""Online task-gradient diagnostics; never writes parameter .grad or changes updates.

PCGrad (Yu et al., NeurIPS 2020), Definitions 1 and 2:
https://papers.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
Epoch averages and cancellation_fraction are descriptive aggregations, not a
published universal severity score or evidence that Adam updates increase loss.
"""
import csv
import json
import math
from pathlib import Path

import torch

GROUPS = {
    "all_shared": "",
    "cell_encoder": "cell.encoder.",
    "tissue_encoder": "tissue.encoder.",
    "encoder_c2t": "encoder_c2t.",
    "context_t2c": "context_t2c.",
    "aspp_t2c": "aspp_t2c.",
    "decoder_t2c": "decoder_t2c.",
    "cell_aspp": "cell.aspp.",
    "tissue_aspp": "tissue.aspp.",
    "cell_decoder": "cell.decoder.",
    "tissue_decoder": "tissue.decoder.",
}


def geometry(dot, c2, t2):
    """None: at least one zero-norm task gradient (undefined angle)."""
    if not all(math.isfinite(v) for v in (dot, c2, t2)):
        raise RuntimeError("Non-finite task gradients; retry --no-amp.")
    if c2 <= 0 or t2 <= 0:
        return None
    cosine = max(-1.0, min(1.0, dot / math.sqrt(c2 * t2)))
    phi = 2 * math.sqrt(c2 * t2) / (c2 + t2)
    return dict(
        cosine=cosine, conflict=int(cosine < 0),
        negative_cosine=max(0.0, -cosine),
        magnitude_similarity=phi,
        cell_norm=math.sqrt(c2), tissue_norm=math.sqrt(t2),
        cancellation_fraction=2 * max(0.0, -dot) / (c2 + t2),
        cell_first_order_adverse=int(c2 + dot < 0),
        tissue_first_order_adverse=int(t2 + dot < 0),
    )


class GradientConflictMonitor:
    def __init__(self, folder, start_epoch=1):
        self.folder = Path(folder) / "gradient_conflict"
        self.folder.mkdir(parents=True, exist_ok=True)
        # Resume from last checkpoint: discard uncheckpointed/duplicate epochs.
        for name in ("batches.csv", "epochs.csv"):
            path = self.folder / name
            if path.exists():
                with path.open(newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    fields = reader.fieldnames
                    rows = [r for r in reader if int(r["epoch"]) < start_epoch]
                with path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
        (self.folder / "protocol.json").write_text(json.dumps({
            "reference": "Yu et al. 2020, PCGrad, Definitions 1 and 2",
            "when": "every training minibatch, same forward, before backward/clip/step",
            "losses": "cell and lambda*tissue; unscaled by GradScaler",
            "scope": "trainable tensors reached by BOTH losses; excludes private tensors",
            "primary": "conflict_rate = mean(cosine < 0) over valid minibatches",
            "aggregation": "equal weight per minibatch; batch size recorded; no sampling",
            "cancellation_fraction": "2*max(0,-dot)/(cell_norm^2+tissue_norm^2); derived descriptive statistic",
            "epoch_cancellation_fraction": "sum(2*max(0,-dot))/sum(norms^2); derived statistic",
            "zero_norm": "undefined angle excluded; coverage and invalid batches reported",
            "limitations": "not curvature, causal harm, per-image conflict, or actual Adam update analysis",
            "freeze": "encoder-frozen and unfrozen epochs have different trainable scope",
        }, indent=2) + "\n", encoding="utf-8")

    def begin(self, epoch, model, frozen):
        self.epoch, self.frozen = epoch, int(frozen)
        self.named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.rows = []

    @staticmethod
    def _append(path, rows):
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    def observe(self, cell_loss, tissue_loss, batch_index, pair_ids,
                precomputed_gradients=None):
        if precomputed_gradients is None:
            parameters = [p for _, p in self.named]
            gc = torch.autograd.grad(cell_loss, parameters, retain_graph=True,
                                     allow_unused=True)
            gt = torch.autograd.grad(tissue_loss, parameters, retain_graph=True,
                                     allow_unused=True)
        else:
            pairs = [precomputed_gradients[n] for n, _ in self.named]
            gc, gt = zip(*pairs)
        stats = {}
        # Three reductions per tensor; no giant flattened gradient concatenation.
        for (name, _), c, t in zip(self.named, gc, gt):
            if c is None or t is None:
                continue
            c, t = c.detach().float(), t.detach().float()
            value = torch.stack(((c * t).sum(), (c * c).sum(), (t * t).sum()))
            for group, prefix in GROUPS.items():
                if not prefix or name.startswith(prefix):
                    if group not in stats:
                        stats[group] = [value.clone(), 1, c.numel()]
                    else:
                        stats[group][0].add_(value)
                        stats[group][1] += 1
                        stats[group][2] += c.numel()
        del gc, gt
        batch_rows = []
        # Emit absent scopes too (frozen encoders must not look like zero conflict).
        for group in GROUPS:
            values, tensors, elements = stats.get(group, (None, 0, 0))
            dot, c2, t2 = values.cpu().tolist() if values is not None else (0., 0., 0.)
            result = geometry(dot, c2, t2)
            row = dict(epoch=self.epoch, batch=batch_index, group=group,
                       encoder_frozen=self.frozen, batch_size=len(pair_ids),
                       pair_ids="|".join(map(str, pair_ids)), shared_tensors=tensors,
                       shared_parameters=elements, valid=int(result is not None),
                       dot=dot, cell_sq=c2, tissue_sq=t2)
            row.update(result or {k: "" for k in (
                "cosine", "conflict", "negative_cosine", "magnitude_similarity",
                "cell_norm", "tissue_norm", "cancellation_fraction",
                "cell_first_order_adverse", "tissue_first_order_adverse")})
            batch_rows.append(row)
        self._append(self.folder / "batches.csv", batch_rows)
        self.rows.extend(batch_rows)

    def finish(self, expected_batches):
        summaries = []
        for group in GROUPS:
            all_rows = [r for r in self.rows if r["group"] == group]
            if len(all_rows) != expected_batches:
                raise RuntimeError("Gradient diagnostic missed training minibatches.")
            rows = [r for r in all_rows if r["valid"]]
            def avg(key):
                return sum(r[key] for r in rows) / len(rows) if rows else ""
            denom = sum(r["cell_sq"] + r["tissue_sq"] for r in rows)
            summaries.append(dict(
                epoch=self.epoch, group=group, encoder_frozen=self.frozen,
                observed_batches=len(all_rows), valid_batches=len(rows),
                invalid_batches=len(all_rows)-len(rows),
                conflict_rate=avg("conflict"), mean_cosine=avg("cosine"),
                mean_negative_cosine=avg("negative_cosine"),
                mean_magnitude_similarity=avg("magnitude_similarity"),
                mean_cell_norm=avg("cell_norm"), mean_tissue_norm=avg("tissue_norm"),
                epoch_cancellation_fraction=(
                    sum(2*max(0., -r["dot"]) for r in rows)/denom if denom else ""),
                cell_first_order_adverse_rate=avg("cell_first_order_adverse"),
                tissue_first_order_adverse_rate=avg("tissue_first_order_adverse")))
        self._append(self.folder / "epochs.csv", summaries)
        self.rows.clear()
        return summaries[0]
