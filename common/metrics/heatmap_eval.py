"""M2 후처리: BC·TC sigmoid 점수 지도 → 세포 점 (x, y, 클래스, 점수).

M0 후처리(common/metrics/m0_eval.py의 detect_cells)와 같은 순서를 쓰되,
배경 채널이 없으므로 "세포일 확률" 자리에 두 점수 지도의 픽셀별 최댓값을 쓴다.

    1) score = max(P_BC, P_TC)를 Gaussian(σ)으로 부드럽게 한다 (σ=0이면 생략).
    2) 지역 최댓값(min_distance 반경)을 찾는다.
    3) 그 값이 threshold보다 클 때만 남긴다.
       (M0의 "배경 확률 < 세포 확률" 조건 대신. 배경 점수가 0 근처라 평평한
        영역도 지역 최댓값이 되므로 threshold는 0보다 커야 한다.)
    4) 클래스 = 그 위치에서 BC·TC 중 큰 쪽, 점수 = 그 확률.

클래스 구분 없이 봉우리를 찾는 이유: M0와 비교 가능하게 하고, 같은 세포에서
BC·TC 점이 하나씩 나오는 중복을 막기 위해서다. 공식 F1 계산은 m0_eval.CellF1을
그대로 쓴다. 근거: CenterNet(https://arxiv.org/abs/1904.07850)의 지역 최댓값 추출.
"""

import torch
from torch.nn import functional as F

from .m0_eval import gaussian_blur


@torch.no_grad()
def heatmap_scores(cell_logits):
    """(B, 2, H, W) logits → 확률, 픽셀별 최댓값, 그 클래스(0=BC, 1=TC)."""
    if cell_logits.ndim != 4 or cell_logits.shape[1] != 2:
        raise ValueError("M2 세포 출력은 (B, 2, H, W)여야 합니다.")
    prob = torch.sigmoid(cell_logits.float())
    best, best_class = prob.max(dim=1)
    return prob, best, best_class


@torch.no_grad()
def detect_cells_heatmap(cell_logits, sigma=1.0, min_distance=3, threshold=0.3):
    """반환: 이미지별 [(x, y, cls, score), ...]. cls는 1=BC, 2=TC."""
    if not threshold > 0:
        raise ValueError("M2 후처리 threshold는 0보다 커야 합니다.")
    if min_distance < 1:
        raise ValueError("min_distance는 1 이상입니다.")
    _, best, best_class = heatmap_scores(cell_logits)
    smooth = gaussian_blur(best, sigma)

    size = 2 * min_distance + 1
    local_max = F.max_pool2d(smooth[:, None], size, stride=1,
                             padding=min_distance)[:, 0]
    keep = (smooth == local_max) & (smooth > threshold)

    results = []
    for index in range(best.shape[0]):
        y, x = keep[index].nonzero(as_tuple=True)
        results.append(list(zip(
            x.tolist(), y.tolist(),
            (best_class[index, y, x] + 1).tolist(),
            best[index, y, x].tolist()
        )))
    return results
