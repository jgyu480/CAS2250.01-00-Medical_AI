"""Data split, deterministic loaders and logs for the dual-gate experiment."""
import csv
import os
import random
import re
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from common.data.tensors import TorchOcelotDataset, collate_pairs
from common.training.m0_loop import _init_worker


def gate_loaders(original, dataset, settings, count, seed):
    indices = (list(original.dataset.indices) if isinstance(original.dataset, Subset)
               else list(range(len(dataset))))
    groups = {}
    for i in indices:
        row = dataset.source.rows[i]
        slide = row["slide_name"]
        match = re.search(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", slide.upper())
        case = match.group(0) if match else slide
        groups.setdefault(case, []).append(i)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if count < 1 or count >= len(indices) or len(keys) < 2:
        raise ValueError("Need >=2 cases and 0 < meta-count < training pool size")

    held, held_cases = [], []
    for case in keys:
        if len(held) >= count:
            break
        if len(held) + len(groups[case]) < len(indices):
            held.extend(groups[case])
            held_cases.append(case)
    if len(held) < count:
        raise ValueError("Cannot satisfy meta-count without exhausting train cases")

    held_set = set(held)
    fit = sorted(i for i in indices if i not in held_set)
    meta_set = TorchOcelotDataset(
        split="train", label_source="official", augment=False,
        seed=settings["seed"]
    )
    if [r["pair_id"] for r in meta_set.source.rows] != [
            r["pair_id"] for r in dataset.source.rows]:
        raise ValueError("Training and meta dataset order differs")

    def loader(base, selected, workers, loader_seed):
        return DataLoader(
            Subset(base, selected), batch_size=settings["batch_size"],
            shuffle=True, num_workers=workers, collate_fn=collate_pairs,
            generator=torch.Generator().manual_seed(loader_seed),
            worker_init_fn=partial(_init_worker, seed=loader_seed),
            persistent_workers=False, drop_last=False
        )

    train = loader(dataset, fit, settings["num_workers"], settings["seed"])
    meta = loader(meta_set, sorted(held), 0, seed)
    report = dict(
        seed=seed, requested_meta_pairs=count,
        grouping="TCGA case when identifiable, otherwise full slide name",
        model_train_ids=[dataset.source.rows[i]["pair_id"] for i in fit],
        gate_meta_ids=[dataset.source.rows[i]["pair_id"] for i in sorted(held)],
        gate_meta_cases=held_cases,
        source="official train only; official val/test not used to learn gates"
    )
    assert not set(report["model_train_ids"]) & set(report["gate_meta_ids"])
    return train, meta, report


def atomic_save(state, path):
    path = Path(path)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        torch.save(state, temp)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def capture_rng():
    return dict(
        python=random.getstate(), numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class GateJournal:
    def __init__(self, folder, start_epoch):
        self.folder = Path(folder) / "dual_gate"
        self.folder.mkdir(parents=True, exist_ok=True)
        for filename in ("batches.csv", "epochs.csv"):
            path = self.folder / filename
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

    def append(self, filename, row):
        path = self.folder / filename
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def record(self, epoch, batch, step, frozen, mode, before, gates, info,
               pair_ids, meta_ids):
        values = gates.values()
        c, t = values["cell_other_ratio"], values["tissue_other_ratio"]
        row = dict(
            epoch=epoch, batch=batch, step=step, encoder_frozen=int(frozen),
            mode=mode, meta_update=int(bool(info)),
            before_cell_other_ratio=before["cell_other_ratio"],
            before_tissue_other_ratio=before["tissue_other_ratio"],
            **values, cell_own_weight=2*(1-c), cell_other_weight=2*c,
            tissue_own_weight=2*(1-t), tissue_other_weight=2*t,
            meta_cell=info.get("meta_cell", ""),
            meta_tissue=info.get("meta_tissue", ""),
            cell_gate_derivative=info.get("cell_gate_derivative", ""),
            tissue_gate_derivative=info.get("tissue_gate_derivative", ""),
            pair_ids="|".join(map(str, pair_ids)),
            meta_pair_ids="|".join(map(str, meta_ids))
        )
        self.append("batches.csv", row)
        self.rows.append(row)

    def finish(self, epoch, expected_batches):
        if len(self.rows) != expected_batches:
            raise RuntimeError("Gate journal missed batches")
        row = dict(
            epoch=epoch, observed_batches=len(self.rows),
            meta_updates=sum(r["meta_update"] for r in self.rows)
        )
        for key in ("cell_other_ratio", "tissue_other_ratio"):
            values = [r[key] for r in self.rows]
            row.update({
                "mean_"+key: sum(values)/len(values),
                "min_"+key: min(values), "max_"+key: max(values),
                "last_"+key: values[-1]
            })
        self.append("epochs.csv", row)
        self.rows.clear()
        return row
