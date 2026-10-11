"""M0 학습·평가 스크립트가 함께 쓰는 도구 (데이터 로더, 검증 loop)."""

import random
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from common.data.tensors import TorchOcelotDataset, collate_pairs
from common.losses.m0_losses import m0_task_losses
from common.metrics.m0_eval import CellF1, TissueIoU, detect_cells

FULL_SIZE = 1024


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _init_worker(worker, seed):
    seed_everything(seed + worker)


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device, enabled):
    """CUDA에서만 혼합 정밀도를 쓴다. bf16 지원 GPU면 bf16, 아니면 fp16."""
    if not enabled or device.type != "cuda":
        return nullcontext(), None
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
             else torch.float16)
    return torch.autocast("cuda", dtype=dtype), dtype


def make_loader(split, seed, batch_size, num_workers, label_source="official",
                augment=False, augmentation="default", limit=None,
                shuffle=False):
    dataset = TorchOcelotDataset(
        split=split, label_source=label_source, augment=augment, seed=seed,
        augmentation=augmentation
    )
    data = Subset(dataset, range(min(limit, len(dataset)))) if limit else dataset
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_pairs,
        generator=generator, drop_last=False,
        # 증강은 Dataset.set_epoch 값으로 바뀐다. worker를 epoch마다 새로 만들어야
        # 바뀐 epoch 값이 전달되므로 persistent_workers는 끈다.
        persistent_workers=False,
        worker_init_fn=partial(_init_worker, seed=seed)
    )
    return loader, dataset


def prepare(batch, device, size=FULL_SIZE):
    """batch를 장치로 옮긴다. size<1024이면 점검용으로 줄인다."""
    images = [batch[k].to(device, non_blocking=True)
              for k in ("cell_image", "tissue_image")]
    targets = [batch[k].to(device, non_blocking=True)
               for k in ("cell_target", "tissue_target")]
    if size != FULL_SIZE:
        images = [F.interpolate(x, size=(size, size), mode="bilinear",
                                align_corners=False) for x in images]
        targets = [F.interpolate(t[:, None].float(), size=(size, size),
                                 mode="nearest")[:, 0].long()
                   for t in targets]
    return images, targets, batch["cell_box"].to(device)


def truth_points(points):
    """Dataset의 (N, 3) [x, y, 클래스] → [(x, y, 클래스), ...]."""
    return [(int(round(x)), int(round(y)), int(c))
            for x, y, c in points.tolist()]


def to_full_size(predictions, size):
    """줄인 크기에서 찾은 점을 1024 좌표로 되돌린다 (픽셀 중심 기준)."""
    if size == FULL_SIZE:
        return predictions
    scale = FULL_SIZE / size
    return [(int(min(FULL_SIZE - 1, (x + 0.5) * scale - 0.5)),
             int(min(FULL_SIZE - 1, (y + 0.5) * scale - 0.5)), c, s)
            for x, y, c, s in predictions]


@torch.no_grad()
def evaluate(model, loader, device, postprocess, size=FULL_SIZE, amp=True,
             tissue_weight=1.0, loss_kwargs=None, keep_cell_logits=False,
             organs=None):
    """검증·평가 loop. 손실, 세포 mF1, 조직 mIoU, 이미지별 기록을 반환한다."""
    model.eval()
    cast, _ = autocast(device, amp)
    cell_f1, tissue_iou = CellF1(), TissueIoU()
    totals = {"cell": 0.0, "tissue": 0.0, "total": 0.0}
    count = 0
    records, cached = [], []

    for batch in loader:
        images, targets, boxes = prepare(batch, device, size)
        with cast:
            out = model(*images, boxes)
        losses = m0_task_losses(out["cell_logits"], out["tissue_logits"],
                                *targets, **(loss_kwargs or {}))
        n = images[0].shape[0]
        totals["cell"] += losses["cell"].item() * n
        totals["tissue"] += losses["tissue"].item() * n
        totals["total"] += (losses["cell"] + tissue_weight
                            * losses["tissue"]).item() * n
        count += n

        tissue_iou.update(out["tissue_logits"], targets[1])
        detections = detect_cells(out["cell_logits"], **postprocess)
        for i in range(n):
            pair_id = batch["pair_id"][i]
            preds = to_full_size(detections[i], size)
            gts = truth_points(batch["cell_points"][i])
            cell_f1.update(preds, gts)

            single_f1 = CellF1()
            single_f1.update(preds, gts)
            single_iou = TissueIoU()
            single_iou.update(out["tissue_logits"][i:i + 1],
                              targets[1][i:i + 1])
            records.append({
                "pair_id": pair_id,
                "organ": organs.get(pair_id) if organs else None,
                "num_gt": len(gts),
                "num_pred": len(preds),
                **{k: round(v, 4) for k, v in single_f1.compute().items()},
                **{k: round(v, 4) for k, v in single_iou.compute().items()},
                "predictions": [[x, y, c, round(s, 4)]
                                for x, y, c, s in preds],
                "truths": gts
            })
        if keep_cell_logits:
            cached.extend(
                (batch["pair_id"][i], out["cell_logits"][i].half().cpu(),
                 truth_points(batch["cell_points"][i]))
                for i in range(n)
            )

    metrics = {f"loss_{k}": v / max(count, 1) for k, v in totals.items()}
    metrics.update(cell_f1.compute())
    metrics.update(tissue_iou.compute())
    return metrics, records, cached
