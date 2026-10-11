"""M0 후처리와 평가 지표.

세포
- 후처리: 세포 확률 지도 → 점 (x, y, 클래스, 점수).
  OCELOT 챌린지 공식 예제(ocelot23algo, user/unet_example/unet.py)의 순서를 따른다.
  1) 세포일 확률 = 1 - P(배경)을 Gaussian(σ=3)으로 부드럽게 한다.
  2) 지역 최댓값(min_distance 반경)을 찾는다.
  3) 그 위치에서 배경 확률이 BC·TC 중 큰 확률보다 낮을 때만 남긴다.
  4) 클래스 = BC·TC 중 큰 쪽, 점수 = 그 확률.
  공식 예제는 skimage.feature.peak_local_max를 쓰지만, 여기서는 같은 동작을
  PyTorch max pooling으로 구현해 GPU에서 바로 계산한다. (추가 패키지 불필요)
- 지표: 공식 평가(ocelot23algo, evaluation/eval.py)와 같은 규칙.
  클래스별로 점수가 높은 예측부터, 15px(약 3µm) 안의 가장 가까운 정답과 1:1 매칭.
  전체 이미지의 TP/FP/정답 수를 합쳐 Precision·Recall·F1 → BC·TC 평균 = mF1.

조직
- UNK 정답 픽셀을 제외하고 BG·CA별 IoU를 데이터 전체 픽셀 합으로 계산한다.
  mIoU = (IoU_BG + IoU_CA) / 2.
  (이미지별 평균이 아닌 전체 합산 방식은 우리 선택이다.)
"""

import math

import numpy as np
import torch
from torch.nn import functional as F

CELL_CLASSES = {1: "BC", 2: "TC"}
DISTANCE_CUTOFF = 15  # 공식 평가 고정값 (0.2 MPP 기준 약 3µm)


def gaussian_blur(image, sigma):
    """image: (B, H, W). 반경 4σ의 분리형 Gaussian (경계는 reflect)."""
    if sigma <= 0:
        return image
    radius = max(1, int(math.ceil(4 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=image.dtype,
                     device=image.device)
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    out = F.pad(image[:, None], (radius, radius, 0, 0), mode="reflect")
    out = F.conv2d(out, kernel.view(1, 1, 1, -1))
    out = F.pad(out, (0, 0, radius, radius), mode="reflect")
    out = F.conv2d(out, kernel.view(1, 1, -1, 1))
    return out[:, 0]


@torch.no_grad()
def detect_cells(cell_logits, sigma=3.0, min_distance=3, threshold=0.0):
    """cell_logits: (B, 3, H, W). 반환: 이미지별 [(x, y, cls, score), ...]."""
    prob = cell_logits.float().softmax(dim=1)
    background, cells = prob[:, 0], prob[:, 1:]
    smooth = gaussian_blur(1 - background, sigma)

    size = 2 * min_distance + 1
    local_max = F.max_pool2d(smooth[:, None], size, stride=1,
                             padding=min_distance)[:, 0]
    best, best_class = cells.max(dim=1)
    keep = (smooth == local_max) & (smooth > threshold) & (background < best)

    results = []
    for index in range(prob.shape[0]):
        y, x = keep[index].nonzero(as_tuple=True)
        results.append(list(zip(
            x.tolist(), y.tolist(),
            (best_class[index, y, x] + 1).tolist(),
            best[index, y, x].tolist()
        )))
    return results


def match_counts(predictions, truths, cls, cutoff=DISTANCE_CUTOFF):
    """한 이미지, 한 클래스의 (TP, FP, 정답 수). 공식 매칭 규칙."""
    # 공식 코드와 같은 float32로 계산해 동점·경계 처리까지 맞춘다.
    pred = np.array([p for p in predictions if p[2] == cls],
                    dtype=np.float32).reshape(-1, 4)
    gt = np.array([g[:2] for g in truths if g[2] == cls],
                  dtype=np.float32).reshape(-1, 2)
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0

    distance = np.linalg.norm(pred[:, None, :2] - gt[None], axis=2)
    available = distance <= cutoff
    tp = fp = 0
    # 동점은 원래 순서를 유지한다 (공식 코드의 np.argsort(-score)와 같은 순서).
    for index in np.argsort(-pred[:, 3], kind="quicksort"):
        neighbors = available[index].nonzero()[0]
        if len(neighbors) == 0:
            fp += 1
            continue
        nearest = neighbors[np.argmin(distance[index, neighbors])]
        tp += 1
        available[:, nearest] = False
    return tp, fp, len(gt)


class CellF1:
    """여러 이미지의 결과를 모아 클래스별 P/R/F1과 mF1을 계산한다."""

    def __init__(self, cutoff=DISTANCE_CUTOFF):
        self.cutoff = cutoff
        self.counts = {cls: [0, 0, 0] for cls in CELL_CLASSES}

    def update(self, predictions, truths):
        for cls in CELL_CLASSES:
            for i, value in enumerate(
                match_counts(predictions, truths, cls, self.cutoff)
            ):
                self.counts[cls][i] += value

    def compute(self):
        scores = {}
        for cls, name in CELL_CLASSES.items():
            tp, fp, num_gt = self.counts[cls]
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (num_gt + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            scores[f"precision_{name}"] = precision
            scores[f"recall_{name}"] = recall
            scores[f"f1_{name}"] = f1
        scores["mF1"] = sum(
            scores[f"f1_{name}"] for name in CELL_CLASSES.values()
        ) / len(CELL_CLASSES)
        return scores


class TissueIoU:
    """UNK(ignore_index)를 제외한 BG(0)·CA(1) IoU와 mIoU."""

    def __init__(self, classes=(0, 1), ignore_index=2):
        self.classes, self.ignore_index = classes, ignore_index
        self.intersection = {c: 0 for c in classes}
        self.union = {c: 0 for c in classes}

    @torch.no_grad()
    def update(self, tissue_logits, tissue_target):
        # UNK 채널은 정답으로 쓰지 않으므로 BG·CA 중에서만 예측을 고른다.
        pred = tissue_logits[:, list(self.classes)].argmax(dim=1)
        pred = torch.as_tensor(self.classes, device=pred.device)[pred]
        valid = tissue_target != self.ignore_index
        for c in self.classes:
            p, t = (pred == c) & valid, (tissue_target == c) & valid
            self.intersection[c] += (p & t).sum().item()
            self.union[c] += (p | t).sum().item()

    def compute(self):
        names = {0: "BG", 1: "CA"}
        scores = {
            f"iou_{names[c]}": self.intersection[c] / max(self.union[c], 1)
            for c in self.classes
        }
        scores["mIoU"] = sum(scores.values()) / len(self.classes)
        return scores
