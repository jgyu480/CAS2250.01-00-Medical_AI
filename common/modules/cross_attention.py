"""M1: tissue-global cross-attention aligned to the tissue coordinate frame.

Inputs:
  cell:   [B, 256, Hc, Wc]  (ASPP features)
  tissue: [B, 256, Ht, Wt]  (ASPP features)
  boxes:  [B, 4]           (left, top, right, bottom), normalized tissue coords

Cell positions form Query tokens; a 16x16 grid of pooled tissue features
forms Key/Value tokens. 2-D Fourier coordinates are shared in the *tissue*
coordinate frame. No reverse cell-to-tissue information path exists here.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .exchange import check_boxes, fuse_conv


def _grid_xy(height, width, *, device):
    """Return [H*W, 2] patch-center x/y coordinates in [0, 1]."""
    ys = (torch.arange(height, dtype=torch.float32, device=device) + 0.5) / height
    xs = (torch.arange(width, dtype=torch.float32, device=device) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def tissue_token_coordinates(height, width, *, device):
    """Tissue patch centers, [1, H*W, 2] in tissue coordinates."""
    return _grid_xy(height, width, device=device).unsqueeze(0)


def cell_token_coordinates(height, width, boxes):
    """Cell query centers expressed in the corresponding tissue coordinates."""
    check_boxes(boxes, boxes.shape[0])
    uv = _grid_xy(height, width, device=boxes.device).unsqueeze(0)
    bounds = boxes.to(dtype=torch.float32)
    left_top = bounds[:, None, :2]
    extent = (bounds[:, 2:] - bounds[:, :2])[:, None, :]
    return left_top + uv * extent


def _fourier_xy(coords, channels):
    """Fixed sin/cos features, shared for tissue and cell positions.

    Coordinates must be in the same reference frame (tissue image [0,1]^2).
    Frequencies are 1..16 cycles across the entire tissue image.
    """
    if channels % 4:
        raise ValueError("Position embedding channels must be divisible by 4")
    n_freq = channels // 4
    frequencies = torch.exp(torch.linspace(
        0.0, math.log(16.0), n_freq, device=coords.device, dtype=torch.float32
    ))
    angles = (2 * math.pi) * coords.float().unsqueeze(-1) * frequencies
    x, y = angles[..., 0, :], angles[..., 1, :]
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)


class TissueGuidedCrossAttention(nn.Module):
    """ASPP fusion: cell Queries attend to pooled global tissue Keys/Values."""

    def __init__(self, channels=256, heads=4, token_grid=(16, 16),
                 use_coordinates=True):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        if use_coordinates and channels % 4:
            raise ValueError("coordinate encoding requires channels % 4 == 0")
        if len(token_grid) != 2 or any(int(x) < 1 for x in token_grid):
            raise ValueError("token_grid must contain positive (height, width)")
        self.channels = channels
        self.token_grid = tuple(int(x) for x in token_grid)
        self.use_coordinates = bool(use_coordinates)
        self.cell_norm = nn.LayerNorm(channels)
        self.tissue_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels, num_heads=heads, dropout=0.0,
            batch_first=True
        )
        # Same residual fusion pattern as the current M0 TissueToCell module.
        self.fuse = fuse_conv(channels * 2, channels)

    def forward(self, cell, tissue, boxes, *, return_attention=False):
        if cell.ndim != 4 or tissue.ndim != 4:
            raise ValueError("cell and tissue must be [B,C,H,W]")
        batch, channels, hc, wc = cell.shape
        if channels != self.channels or tissue.shape[:2] != (batch, channels):
            raise ValueError("cell/tissue batch and channels must match")
        check_boxes(boxes, batch)

        pooled = F.adaptive_avg_pool2d(tissue, self.token_grid)
        ht, wt = pooled.shape[-2:]
        q = self.cell_norm(cell.flatten(2).transpose(1, 2))
        kv = self.tissue_norm(pooled.flatten(2).transpose(1, 2))
        if self.use_coordinates:
            # Compute positions in fp32, then cast once for AMP safety.
            query_xy = cell_token_coordinates(hc, wc, boxes)
            tissue_xy = tissue_token_coordinates(ht, wt, device=cell.device)
            q = q + _fourier_xy(query_xy, channels).to(q.dtype)
            k = kv + _fourier_xy(tissue_xy, channels).to(kv.dtype)
        else:
            k = kv

        attended, weights = self.attention(
            q, k, kv, need_weights=return_attention,
            average_attn_weights=False if return_attention else True
        )
        attended = attended.transpose(1, 2).reshape(batch, channels, hc, wc)
        fused = cell + self.fuse(torch.cat((cell, attended), dim=1))
        if return_attention:
            # [B, heads, Hc*Wc, token_grid[0]*token_grid[1]]
            return fused, weights
        return fused
