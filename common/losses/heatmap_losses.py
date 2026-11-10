"""M2 세포 손실: 독립 sigmoid 출력과 중심 정답 지도의 가중 제곱오차.

보고서 8.4: "세포 head는 BC·TC 점수 지도를 각각 예측하도록 구성하고,
독립 sigmoid 출력에 정답 지도와의 제곱오차를 적용한다. 중심 주변과 배경의
불균형은 손실 가중치로 조정한다."

세포 검출에서 같은 방식의 선례
- Xie et al., "Beyond Classification: Structured Regression for Robust Cell
  Detection Using CNN" (MICCAI 2015), https://pmc.ncbi.nlm.nih.gov/articles/PMC5226438
  세포 중심 근접 지도를 sigmoid 출력으로 회귀하고 가중 제곱오차를 쓰며,
  지도의 지역 최댓값(threshold 이상)을 세포 중심으로 뽑는다.
  그들의 가중치 (y + λ)는 정답 값이 낮은 곳의 가중치가 낮아, H_AO가 일부러 낮춘
  두 세포 사이 골짜기에 오히려 작은 가중치가 걸린다. 그래서 아래 방식을 쓴다.

가중치
- 보고서는 가중치 형태를 정하지 않았다. 열지도 회귀에서 전경·배경 불균형을
  다루는 공개 방법인 Adaptive Wing Loss의 Weighted Loss Map을 사용한다.
  Wang et al., "Adaptive Wing Loss for Robust Face Alignment via Heatmap
  Regression" (ICCV 2019), https://arxiv.org/abs/1904.07399 (Sec. 4.3)
      손실 ⊗ (W · M + 1),  M = (3×3 grey dilation(H) ≥ 0.2)
  논문 설정 W = 10, 임계값 0.2, 3×3 팽창을 기본값으로 쓴다.
  손실 함수 자체는 AWing이 아니라 보고서대로 제곱오차다.
- 팽창 때문에 가까운 두 세포 사이의 낮은 골짜기(H_AO가 낮춘 곳)도 M에 포함되어,
  모델이 골짜기를 맞추는 데 큰 가중치가 걸린다.

정규화
- M0 Focal Loss와 같은 규칙을 쓴다 (common/losses/m0_losses.py).
  합을 전경 픽셀 수(M의 픽셀 수)로 나누고, 세포가 거의 없는 batch에서 손실이
  튀지 않도록 나누는 수의 최솟값을 전체 픽셀의 min_foreground_fraction로 둔다.
"""

import torch
from torch.nn import functional as F


def foreground_mask(target, threshold=0.2, dilation=3):
    """(B, C, H, W) 정답 지도의 전경 M (AWing: 3×3 팽창 후 임계값)."""
    if dilation > 1:
        pad = dilation // 2
        target = F.max_pool2d(target, dilation, stride=1, padding=pad)
    return target >= threshold


def weighted_mse(logits, target, foreground_weight=10.0, threshold=0.2, dilation=3,
                 min_foreground_fraction=0.005):
    """logits, target: (B, 2, H, W). 반환: 스칼라 손실."""
    if logits.ndim != 4 or logits.shape != target.shape:
        raise ValueError("logits와 정답 지도의 크기가 다릅니다.")
    if not torch.is_floating_point(target):
        raise ValueError("정답 지도는 실수형이어야 합니다.")
    if dilation < 1 or dilation % 2 == 0:
        raise ValueError("팽창 크기는 1 이상의 홀수입니다.")

    target = target.float()
    if (target < 0).any() or (target > 1).any():
        raise ValueError("정답 지도 값은 0~1이어야 합니다.")

    prob = torch.sigmoid(logits.float())
    mask = foreground_mask(target, threshold, dilation)
    weight = 1.0 + foreground_weight * mask.float()
    loss = (weight * (prob - target) ** 2).sum()

    floor = max(1.0, min_foreground_fraction * target.numel())
    count = mask.sum().float().clamp_min(floor)
    return loss / count
