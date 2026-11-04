"""Position-wise probability mixture of cell-only and tissue-aware predictions.

The gate is computed at decoder resolution, while probability mixing takes place
at the original logits resolution to preserve the M0/M1 detector interface.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SpatialProbabilityGate(nn.Module):
    def __init__(self, channels=256, projection_channels=16, hidden_channels=32):
        super().__init__()
        self.local_projection = nn.Conv2d(channels, projection_channels, 1)
        self.fused_projection = nn.Conv2d(channels, projection_channels, 1)
        inputs = projection_channels * 3 + 6  # local, fused, |difference|, 2 x 3 classes
        self.predictor = nn.Sequential(
            nn.Conv2d(inputs, hidden_channels, 3, padding=1),
            nn.GroupNorm(8 if hidden_channels % 8 == 0 else 1, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        # sigmoid(0) = 0.5 at initialization, independently of input features.
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

    def forward(self, local_feature, fused_feature, local_logits, fused_logits):
        if local_feature.shape != fused_feature.shape:
            raise ValueError("Local/fused decoder features must have identical shapes")
        low_size = local_feature.shape[-2:]
        local_small = F.interpolate(local_logits.float(), size=low_size,
                                    mode="bilinear", align_corners=False).softmax(dim=1)
        fused_small = F.interpolate(fused_logits.float(), size=low_size,
                                    mode="bilinear", align_corners=False).softmax(dim=1)
        # FP32 gate computation is more stable under CUDA fp16 autocast.
        a = self.local_projection(local_feature.float())
        b = self.fused_projection(fused_feature.float())
        x = torch.cat((a, b, (a - b).abs(), local_small, fused_small), dim=1)
        return torch.sigmoid(self.predictor(x))


def mixture_log_probs(local_logits, fused_logits, gate, fixed_weight=None):
    """Return log(P) where P=(1-g)*P_local+g*P_fused.

    The result works as `cell_logits`: downstream softmax/log_softmax reconstructs P.
    This is probability-space, NOT logit-space, blending.
    """
    if local_logits.shape != fused_logits.shape:
        raise ValueError("Local and fused logits have different shapes")
    if fixed_weight is not None:
        weight = float(fixed_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("fixed_weight must be between 0 and 1")
        if weight == 0.0:
            return local_logits.float().log_softmax(dim=1)
        if weight == 1.0:
            return fused_logits.float().log_softmax(dim=1)
        gate = torch.tensor(weight, device=local_logits.device, dtype=torch.float32)
    else:
        gate = F.interpolate(gate.float(), size=local_logits.shape[-2:],
                             mode="bilinear", align_corners=False)
    local_prob = local_logits.float().softmax(dim=1)
    fused_prob = fused_logits.float().softmax(dim=1)
    prob = torch.lerp(local_prob, fused_prob, gate).clamp_min(1e-12)
    return prob.log()
