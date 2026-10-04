import torch
from torch.nn import functional as F


def dice_loss(logits, target, classes, ignore_index=None, eps=1e-6):
    if logits.ndim != 4 or target.shape != (
        logits.shape[0], *logits.shape[2:]
    ):
        raise ValueError("logits와 target의 크기가 다릅니다.")
    if target.dtype != torch.long:
        raise ValueError("정답은 torch.long이어야 합니다.")

    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid = target != ignore_index
    safe = target.masked_fill(~valid, 0)

    if ((safe < 0) | (safe >= logits.shape[1])).any():
        raise ValueError("정답 번호가 출력 채널 범위를 벗어납니다.")

    probabilities = logits.float().softmax(dim=1)
    truth = F.one_hot(
        safe, num_classes=logits.shape[1]
    ).permute(0, 3, 1, 2)

    valid = valid.unsqueeze(1)
    probabilities = probabilities * valid
    truth = truth.to(probabilities.dtype) * valid

    intersection = (probabilities * truth).sum(dim=(2, 3))
    denominator = (probabilities + truth).sum(dim=(2, 3))
    scores = (2 * intersection + eps) / (denominator + eps)
    supervised = valid.flatten(1).any(dim=1)

    if not supervised.any():
        return logits.sum() * 0
    return 1 - scores[supervised][:, list(classes)].mean()


def task_losses(cell_logits, tissue_logits, cell_target, tissue_target):
    return {
        "cell": dice_loss(cell_logits, cell_target, (1, 2)),
        "tissue": dice_loss(
            tissue_logits, tissue_target, (0, 1), ignore_index=2
        )
    }
