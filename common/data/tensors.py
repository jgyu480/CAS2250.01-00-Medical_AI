import json
import numpy as np
import torch
from torch.utils.data import Dataset
from common.paths import PROJECT_ROOT
from .dataset import OcelotDataset


def image_tensor(image, settings):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("입력은 RGB uint8 배열이어야 합니다.")
    tensor = torch.from_numpy(
        np.ascontiguousarray(image)
    ).permute(2, 0, 1).float() / 255
    mean = torch.tensor(settings["mean"]).view(3, 1, 1)
    std = torch.tensor(settings["std"]).view(3, 1, 1)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("정규화 설정이 유한한 값이 아닙니다.")
    if (std <= 0).any():
        raise ValueError("표준편차는 양수여야 합니다.")
    return (tensor - mean) / std


class TorchOcelotDataset(Dataset):
    def __init__(self, **kwargs):
        self.source = OcelotDataset(**kwargs)
        settings = json.loads(
            (PROJECT_ROOT / "configs/components.json").read_text(
                encoding="utf-8"
            )
        )
        self.settings = settings["normalization"]

    def __len__(self):
        return len(self.source)

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __getitem__(self, index):
        sample = self.source[index]
        for key in ("cell_image", "tissue_image"):
            sample[key] = image_tensor(sample[key], self.settings)
        for key in ("cell_target", "tissue_target"):
            sample[key] = torch.from_numpy(sample[key]).long()
        for key in ("cell_box", "cell_points"):
            sample[key] = torch.from_numpy(sample[key]).float()
        return sample


def collate_pairs(samples):
    if not samples:
        raise ValueError("비어 있는 batch입니다.")
    stacked = {
        "cell_image", "tissue_image",
        "cell_target", "tissue_target", "cell_box"
    }
    return {
        key: torch.stack([sample[key] for sample in samples])
        if key in stacked else [sample[key] for sample in samples]
        for key in samples[0]
    }
