"""M2 세포 중심 정답 지도: Gaussian G, 분리 지표 A, 겹침 계수 O.

보고서 8.4의 정의를 그대로 따른다. BC와 TC는 각각 같은 클래스의 중심만으로
지도를 만든다 (classwise). 거리는 원본 1024×1024 세포 사진 픽셀 단위다.

    G(p)    = max_i exp(-||p - c_i||^2 / (2σ^2)) = exp(-d1^2 / (2σ^2))
    A(p)    = (2·d1 / (d1 + d2))^2               (중심 0, 두 중심의 이등분선 1)
    O(p)    = exp(-d2^2 / (2σ^2))
    H_A(p)  = G · (1 - λ·A)
    H_AO(p) = G · (1 - λ·O·A)

- d1, d2: 같은 클래스에서 첫째·둘째로 가까운 정답 중심까지의 거리.
- 같은 클래스의 중심이 1개뿐이면 분리 계수를 1로 둔다 (A = 0으로 처리).
- 중심이 없으면 그 클래스 지도는 전부 0이다.
- 여러 Gaussian은 더하지 않고 최댓값으로 결합하며, 지도 합을 1로 정규화하지 않는다.
- 정답 중심에서는 d1 = 0이므로 세 지도 모두 값이 1이다.

근거
- Gaussian 중심 지도와 최댓값 결합: Zhou et al., "Objects as Points" (CenterNet),
  https://arxiv.org/abs/1904.07850
- A·O 분리 계수와 λ는 보고서 8.4에서 정한 프로젝트 설계이며 출판된 방법이 아니다.
- 같은 식을 scripts/select_gaussian_params.py(σ·λ 선정 스크립트)도 사용한다.

계산
- 픽셀마다 같은 클래스의 가장 가까운 두 중심을 KD-tree로 찾는다.
- G < exp(-cutoff^2/2) (cutoff=6σ이면 약 1.5e-8)인 픽셀은 0으로 둔다.
  이 픽셀에서는 G·S ≤ G이므로 생략에 따른 오차는 그 값 이하이다.
  나머지 픽셀은 d2 거리 제한 없이 정확히 계산한다 (H_A는 d2가 멀어도 A가 크다).
"""

import numpy as np
from scipy.spatial import cKDTree

CLASSES = (1, 2)            # 출력 채널 0 = BC(1), 1 = TC(2)
MODES = ("gaussian", "ha", "hao")
CUTOFF_SIGMAS = 6.0


def check_parameters(sigma, lam, mode):
    if mode not in MODES:
        raise ValueError(f"mode는 {MODES} 중 하나입니다: {mode}")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("σ는 양수여야 합니다.")
    # 보고서 8.4: λ는 0 이상 1 미만. λ=1이면 두 중심의 이등분선에서 점수가 0이 된다.
    if not np.isfinite(lam) or not 0 <= lam < 1:
        raise ValueError("λ는 0 이상 1 미만이어야 합니다 (보고서 8.4).")


def separation_maps(d1, d2, sigma, lam):
    """거리 배열에서 (G, H_A, H_AO)를 계산한다. d2 = inf는 중심 1개."""
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)
    g = np.exp(-d1 ** 2 / (2 * sigma ** 2))
    a = np.zeros_like(d1)
    valid = np.isfinite(d2) & ((d1 + d2) > 0)
    a[valid] = (2 * d1[valid] / (d1[valid] + d2[valid])) ** 2
    o = np.where(np.isfinite(d2), np.exp(-np.square(d2) / (2 * sigma ** 2)), 0.0)
    return g, g * (1 - lam * a), g * (1 - lam * a * o)


def class_heatmap(centers, sigma, lam, mode="hao", size=1024):
    """한 클래스의 (size, size) float32 지도. centers: (N, 2) [x, y]."""
    check_parameters(sigma, lam, mode)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    heat = np.zeros((size, size), dtype=np.float32)
    if len(centers) == 0:
        return heat

    # 중심에서 cutoff 안의 픽셀만 계산한다 (밖은 G ≈ 0).
    cutoff = CUTOFF_SIGMAS * sigma
    near = np.zeros((size, size), dtype=bool)
    r = int(np.ceil(cutoff))
    for x, y in centers:
        x0, x1 = max(0, int(np.floor(x)) - r), min(size, int(np.ceil(x)) + r + 1)
        y0, y1 = max(0, int(np.floor(y)) - r), min(size, int(np.ceil(y)) + r + 1)
        near[y0:y1, x0:x1] = True
    yy, xx = np.nonzero(near)
    pixels = np.column_stack([xx, yy]).astype(np.float64)

    tree = cKDTree(centers)
    k = 2 if len(centers) >= 2 else 1
    dist, _ = tree.query(pixels, k=k)
    dist = dist.reshape(len(pixels), k)
    d1 = dist[:, 0]
    d2 = dist[:, 1] if k == 2 else np.full(len(pixels), np.inf)

    keep = d1 <= cutoff
    g, ha, hao = separation_maps(d1[keep], d2[keep], sigma, lam)
    values = {"gaussian": g, "ha": ha, "hao": hao}[mode]
    heat[yy[keep], xx[keep]] = values.astype(np.float32)
    return heat


def points_to_heatmaps(points, sigma, lam, mode="hao", size=1024):
    """(N, 3) [x, y, 클래스] → (2, size, size) float32. 채널 0 = BC, 1 = TC."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return np.stack([
        class_heatmap(points[points[:, 2] == cls, :2], sigma, lam, mode, size)
        for cls in CLASSES
    ])
