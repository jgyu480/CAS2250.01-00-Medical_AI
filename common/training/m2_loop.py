"""M2 학습·평가 스크립트가 함께 쓰는 도구 (데이터 로더, 손실, 검증 loop).

M0 공통 loop(common/training/m0_loop.py)를 고치지 않고, 다른 부분만 둔다.
- 정답: Dataset이 증강까지 적용한 세포 점(cell_points)으로 BC·TC 중심 지도를
  만든다. 뒤집기·회전 뒤의 점으로 만들므로 사진과 지도가 항상 같이 움직인다.
- 손실: 세포 = 가중 제곱오차(sigmoid), 조직 = M0와 같은 CE + Dice(UNK 제외).
- 후처리: common/metrics/heatmap_eval.py, 공식 F1은 m0_eval.CellF1 그대로.
"""

from functools import partial

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from common.data.center_heatmaps import points_to_heatmaps, check_parameters
from common.data.tensors import TorchOcelotDataset, collate_pairs
from common.losses.heatmap_losses import weighted_mse
from common.losses.m0_losses import cross_entropy_dice
from common.metrics.heatmap_eval import detect_cells_heatmap
from common.metrics.m0_eval import CellF1, TissueIoU
from common.training.m0_loop import (
    FULL_SIZE, _init_worker, autocast, truth_points, to_full_size
)


class HeatmapDataset(Dataset):
    """TorchOcelotDataset에 cell_heatmap (2, 1024, 1024) float32를 더한다."""

    def __init__(self, target, **kwargs):
        check_parameters(target["sigma"], target["lambda"], target["mode"])
        if target.get("scope", "classwise") != "classwise":
            raise ValueError("보고서 8.4: BC·TC 지도는 같은 클래스 중심으로 만든다.")
        self.target = target
        self.inner = TorchOcelotDataset(**kwargs)
        self.source = self.inner.source

    def __len__(self):
        return len(self.inner)

    def set_epoch(self, epoch):
        self.inner.set_epoch(epoch)

    def __getitem__(self, index):
        sample = self.inner[index]
        heat = points_to_heatmaps(
            sample["cell_points"].numpy(), self.target["sigma"],
            self.target["lambda"], self.target["mode"]
        )
        sample["cell_heatmap"] = torch.from_numpy(heat)
        return sample


def collate_heatmaps(samples):
    batch = collate_pairs(samples)
    batch["cell_heatmap"] = torch.stack(batch["cell_heatmap"])
    return batch


def make_loader(split, seed, batch_size, num_workers, target,
                label_source="official", augment=False,
                augmentation="default", limit=None, shuffle=False):
    dataset = HeatmapDataset(
        target, split=split, label_source=label_source, augment=augment,
        seed=seed, augmentation=augmentation
    )
    data = Subset(dataset, range(min(limit, len(dataset)))) if limit else dataset
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_heatmaps,
        generator=generator, drop_last=False,
        # M0와 같은 이유로 끈다: epoch마다 바뀐 증강 값이 worker에 전달돼야 한다.
        persistent_workers=False,
        worker_init_fn=partial(_init_worker, seed=seed)
    )
    return loader, dataset


def prepare(batch, device, size=FULL_SIZE):
    """batch를 장치로 옮긴다. size<1024이면 점검용으로 줄인다."""
    images = [batch[k].to(device, non_blocking=True)
              for k in ("cell_image", "tissue_image")]
    heat = batch["cell_heatmap"].to(device, non_blocking=True)
    tissue = batch["tissue_target"].to(device, non_blocking=True)
    if size != FULL_SIZE:
        images = [F.interpolate(x, size=(size, size), mode="bilinear",
                                align_corners=False) for x in images]
        heat = F.interpolate(heat, size=(size, size), mode="area")
        tissue = F.interpolate(tissue[:, None].float(), size=(size, size),
                               mode="nearest")[:, 0].long()
    return images, (heat, tissue), batch["cell_box"].to(device)


def task_losses(cell_logits, tissue_logits, cell_heatmap, tissue_target,
                **cell_kwargs):
    """두 손실을 따로 반환한다. 전체 손실은 cell + λ_tissue·tissue (M0와 같음)."""
    return {
        "cell": weighted_mse(cell_logits, cell_heatmap, **cell_kwargs),
        "tissue": cross_entropy_dice(tissue_logits, tissue_target)
    }


def cell_loss_kwargs(settings):
    loss = settings["cell_loss"]
    return {"foreground_weight": loss["foreground_weight"],
            "threshold": loss["foreground_threshold"],
            "dilation": loss["dilation"],
            "min_foreground_fraction": loss["min_foreground_fraction"]}


@torch.no_grad()
def evaluate(model, loader, device, postprocess, size=FULL_SIZE, amp=True,
             tissue_weight=1.0, loss_kwargs=None, keep_cell_logits=False,
             organs=None):
    """검증·평가 loop. m0_loop.evaluate와 같은 형식의 결과를 반환한다."""
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
        losses = task_losses(out["cell_logits"], out["tissue_logits"],
                             *targets, **(loss_kwargs or {}))
        n = images[0].shape[0]
        totals["cell"] += losses["cell"].item() * n
        totals["tissue"] += losses["tissue"].item() * n
        totals["total"] += (losses["cell"] + tissue_weight
                            * losses["tissue"]).item() * n
        count += n

        tissue_iou.update(out["tissue_logits"], targets[1])
        detections = detect_cells_heatmap(out["cell_logits"], **postprocess)
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
