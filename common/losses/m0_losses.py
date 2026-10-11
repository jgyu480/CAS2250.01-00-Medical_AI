"""M0 실험계획서의 손실: 세포 Focal Loss, 조직 Cross-Entropy + Dice.

기존 common/losses/segmentation.py(Dice만 계산)는 그대로 두고,
그 안의 dice_loss를 가져와 조직 손실에 사용한다.

근거
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection" (2017),
  https://arxiv.org/abs/1708.02002
- 세포 정답: 점을 반지름 7px 원으로 바꾼 지도 (OCELOT 부록 B).
  원 밖=0, BC=1, TC=2. 원 밖(배경) 픽셀이 대부분이므로 Focal Loss로
  쉬운 배경 픽셀의 비중을 줄인다.
"""

import torch
from torch.nn import functional as F

from .segmentation import dice_loss


def focal_loss(logits, target, gamma=2.0, normalize="foreground",
               min_foreground_fraction=0.005):
    """다중 클래스 Focal Loss. FL = -(1 - p_t)^γ · log(p_t).

    normalize
    - "foreground": 세포 원 안의 픽셀 수로 나눈다 (RetinaNet의 양성 수 정규화).
      단, 세포가 거의 없는 batch에서는 나누는 수가 너무 작아 손실이 수백~수천 배로
      커진다 (train에는 세포 0개 사진 5장, 10개 미만 25장). 1차 학습 17 epoch에서
      손실이 튀어 모델이 망가진 원인이다. 그래서 나누는 수의 최솟값을
      전체 픽셀의 min_foreground_fraction(기본 0.5%)으로 둔다.
      (평균적인 사진은 세포 원이 약 2%를 차지한다.)
    - "mean": 전체 픽셀 수로 나눈다.
    """
    if logits.ndim != 4 or target.shape != (
        logits.shape[0], *logits.shape[2:]
    ):
        raise ValueError("logits와 target의 크기가 다릅니다.")
    if target.dtype != torch.long:
        raise ValueError("정답은 torch.long이어야 합니다.")
    if ((target < 0) | (target >= logits.shape[1])).any():
        raise ValueError("정답 번호가 출력 채널 범위를 벗어납니다.")

    log_prob = F.log_softmax(logits.float(), dim=1)
    log_pt = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
    loss = -((1 - log_pt.exp()) ** gamma) * log_pt

    if normalize == "foreground":
        floor = max(1.0, min_foreground_fraction * target.numel())
        count = (target > 0).sum().float().clamp_min(floor)
    elif normalize == "mean":
        count = target.numel()
    else:
        raise ValueError("normalize는 foreground/mean 중 하나입니다.")
    return loss.sum() / count


def cross_entropy_dice(logits, target, classes=(0, 1), ignore_index=2):
    """UNK(ignore_index) 픽셀을 제외한 Cross-Entropy + Dice."""
    if not (target != ignore_index).any():
        # 전부 UNK이면 학습할 정답이 없다. 0이지만 그래프는 유지한다.
        return logits.sum() * 0
    ce = F.cross_entropy(
        logits.float(), target, ignore_index=ignore_index
    )
    return ce + dice_loss(logits, target, classes, ignore_index=ignore_index)


def m0_task_losses(
    cell_logits, tissue_logits, cell_target, tissue_target,
    gamma=2.0, normalize="foreground", min_foreground_fraction=0.005
):
    """두 손실을 따로 반환한다. 전체 손실은 cell + λ·tissue (λ=1.0)."""
    return {
        "cell": focal_loss(cell_logits, cell_target, gamma, normalize,
                           min_foreground_fraction),
        "tissue": cross_entropy_dice(tissue_logits, tissue_target)
    }
