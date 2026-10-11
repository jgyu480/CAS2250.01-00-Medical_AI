"""세포·조직 두 갈래 사이에서 같은 위치의 특징을 주고받는 부품.

cell_box는 조직 사진 기준 [왼쪽, 위, 오른쪽, 아래] 0~1 경계 좌표다.
(common/data/geometry.py의 cell_box, 증강 후 값도 Dataset이 함께 바꿔 준다.)

- 조직→세포: 조직 특징에서 cell_box 영역을 잘라 세포 특징 크기로 확대한다.
- 세포→조직: 세포 특징을 cell_box 크기로 축소해 조직 특징의 해당 위치에
  배치하고, 나머지는 0으로 채운다. 배치된 영역은 coverage mask로 표시한다.

OCELOT 논문(arXiv 2303.13110) Fig. 5의 방식을 따른다.
보내는 쪽 특징은 3×3 conv를 거친 뒤 받는 쪽 특징과 채널 방향으로 붙는다.
붙인 결과는 1×1 conv로 원래 채널 수로 되돌려 기존 특징에 더한다.
더하는 방식(residual)은 SSL 백본 특징을 보존하기 위한 우리 구현 선택이다.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .segmentation import norm


def check_boxes(boxes, batch):
    if boxes.ndim != 2 or boxes.shape != (batch, 4):
        raise ValueError(f"cell_box는 ({batch}, 4) 크기여야 합니다.")
    if not torch.isfinite(boxes).all():
        raise ValueError("cell_box에 NaN/Inf가 있습니다.")
    if ((boxes[:, 2:] - boxes[:, :2]) <= 0).any():
        raise ValueError("cell_box의 너비·높이가 0 이하입니다.")
    if (boxes < -1e-4).any() or (boxes > 1 + 1e-4).any():
        raise ValueError("cell_box는 0~1 범위여야 합니다.")


def crop_theta(boxes):
    """세포 사진 정규화 좌표(-1~1) → 조직 사진 정규화 좌표.

    x_tissue = left + u·(right-left), u∈[0,1] 을 -1~1 좌표로 바꾼 식.
    """
    left, top, right, bottom = boxes.unbind(1)
    zero = torch.zeros_like(left)
    return torch.stack([
        torch.stack([right - left, zero, left + right - 1], dim=1),
        torch.stack([zero, bottom - top, top + bottom - 1], dim=1)
    ], dim=1)


def place_theta(boxes):
    """조직 사진 정규화 좌표 → 세포 사진 정규화 좌표 (crop_theta의 역변환)."""
    left, top, right, bottom = boxes.unbind(1)
    width, height = right - left, bottom - top
    zero = torch.zeros_like(left)
    return torch.stack([
        torch.stack([1 / width, zero, -(left + right - 1) / width], dim=1),
        torch.stack([zero, 1 / height, -(top + bottom - 1) / height], dim=1)
    ], dim=1)


def crop_to_cell(tissue_feature, boxes, size):
    """조직 특징에서 세포 사진 영역을 잘라 size(H, W)로 확대한다."""
    batch, channels = tissue_feature.shape[:2]
    check_boxes(boxes, batch)
    theta = crop_theta(boxes.to(tissue_feature.dtype))
    grid = F.affine_grid(
        theta, (batch, channels, *size), align_corners=False
    )
    # 상자는 항상 조직 사진 안에 있으므로 border는 가장자리 반 픽셀만 보정한다.
    return F.grid_sample(
        tissue_feature, grid, mode="bilinear",
        padding_mode="border", align_corners=False
    )


def place_in_tissue(cell_feature, boxes, size):
    """세포 특징을 상자 크기로 축소해 size(H, W) 조직 격자에 배치한다.

    반환: (배치된 특징, coverage mask). 상자 밖은 특징·mask 모두 0이다.
    """
    batch, channels = cell_feature.shape[:2]
    check_boxes(boxes, batch)
    height, width = size

    # 확대·축소 비율이 4배라 bilinear만 쓰면 일부 픽셀만 읽힌다.
    # 먼저 평균 pooling으로 상자 크기 근처까지 줄인 뒤 위치를 맞춘다.
    extent = (boxes[:, 2:] - boxes[:, :2]).amax(dim=0)
    pooled = F.adaptive_avg_pool2d(cell_feature, (
        max(1, round(float(extent[1]) * height)),
        max(1, round(float(extent[0]) * width))
    ))

    theta = place_theta(boxes.to(cell_feature.dtype))
    grid = F.affine_grid(
        theta, (batch, channels, height, width), align_corners=False
    )
    inside = (grid.abs() <= 1).all(dim=-1).unsqueeze(1).to(cell_feature.dtype)
    placed = F.grid_sample(
        pooled, grid, mode="bilinear",
        padding_mode="border", align_corners=False
    ) * inside
    return placed, inside


def share_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        norm(out_channels),
        nn.ReLU(inplace=True)
    )


def fuse_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        norm(out_channels)
    )


class TissueToCell(nn.Module):
    """조직 특징 → 세포 특징. 좌표에 맞춰 잘라 확대한 뒤 결합한다."""

    def __init__(self, cell_channels, tissue_channels, shared=256):
        super().__init__()
        self.share = share_conv(tissue_channels, shared)
        self.fuse = fuse_conv(cell_channels + shared, cell_channels)

    def forward(self, cell, tissue, boxes):
        cropped = crop_to_cell(self.share(tissue), boxes, cell.shape[-2:])
        return cell + self.fuse(torch.cat([cell, cropped], dim=1))


class CellToTissue(nn.Module):
    """세포 특징 → 조직 특징. 축소해 해당 위치에 배치하고 coverage를 붙인다."""

    def __init__(self, tissue_channels, cell_channels, shared=256):
        super().__init__()
        self.share = share_conv(cell_channels, shared)
        self.fuse = fuse_conv(tissue_channels + shared + 1, tissue_channels)

    def forward(self, tissue, cell, boxes):
        placed, coverage = place_in_tissue(
            self.share(cell), boxes, tissue.shape[-2:]
        )
        return tissue + self.fuse(
            torch.cat([tissue, placed, coverage], dim=1)
        )
