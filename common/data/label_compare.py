"""두 라벨(작성자 A/B, 공식 정답, 모델 예측)을 같은 방식으로 비교한다.

세포: 공식 평가와 같은 15px(=3µm, 0.2 MPP) 거리 기준의 1:1 매칭.
조직: UNK를 제외한 BG/CA IoU, UNK 지정 차이, 경계 근처/내부 불일치.

두 작성자 비교는 대칭이다(F1은 A/B를 바꿔도 같다).
모델 평가에 쓸 때는 A=정답, B=예측으로 넣으면 precision/recall 의미가 맞다.
"""

import numpy as np

DISTANCE_PX = 15          # OCELOT 공식 평가: 3µm / 0.2 MPP
CLASSES = {1: 'BC', 2: 'TC'}
TISSUE_RAW = {'BG': 1, 'CA': 2, 'UNK': 255}


def greedy_match(a, b, radius=DISTANCE_PX):
    """가까운 쌍부터 1:1로 묶는다. 반환: [(i, j, 거리)]."""
    a = np.asarray(a, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1, 2)
    if len(a) == 0 or len(b) == 0:
        return []

    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    ii, jj = np.nonzero(d <= radius)
    if len(ii) == 0:
        return []
    # 거리 → 인덱스 순으로 정렬해 결과가 항상 같게 한다.
    order = np.lexsort((jj, ii, d[ii, jj]))
    used_a, used_b, pairs = set(), set(), []
    for k in order:
        i, j = int(ii[k]), int(jj[k])
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, float(d[i, j])))
    return pairs


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0)


def compare_cells(points_a, points_b, radius=DISTANCE_PX):
    """points: N×3 (x, y, label). A를 기준(정답)으로 본다."""
    a = np.asarray(points_a, dtype=float).reshape(-1, 3)
    b = np.asarray(points_b, dtype=float).reshape(-1, 3)

    # 1) 클래스 무시 위치 매칭: 누락/추가된 세포와 클래스 판단 차이를 분리한다.
    pairs = greedy_match(a[:, :2], b[:, :2], radius)
    matched_a = {i for i, _, _ in pairs}
    matched_b = {j for _, j, _ in pairs}
    confusion = np.zeros((2, 2), dtype=int)  # 행=A(BC,TC), 열=B(BC,TC)
    for i, j, _ in pairs:
        confusion[int(a[i, 2]) - 1, int(b[j, 2]) - 1] += 1

    n = confusion.sum()
    agree = np.trace(confusion)
    expected = (confusion.sum(1) @ confusion.sum(0)) / n if n else 0.0
    kappa = (agree - expected) / (n - expected) if n and n != expected else 0.0

    # 2) 공식 지표와 같은 클래스별 매칭 → mF1.
    per_class = {}
    for label, name in CLASSES.items():
        sa, sb = a[a[:, 2] == label], b[b[:, 2] == label]
        tp = len(greedy_match(sa[:, :2], sb[:, :2], radius))
        per_class[name] = dict(
            tp=tp, fp=len(sb) - tp, fn=len(sa) - tp,
            **f1(tp, len(sb) - tp, len(sa) - tp)
        )

    return dict(
        count_a=len(a), count_b=len(b),
        count_a_by_class={n_: int((a[:, 2] == k).sum()) for k, n_ in CLASSES.items()},
        count_b_by_class={n_: int((b[:, 2] == k).sum()) for k, n_ in CLASSES.items()},
        location_matched=len(pairs),
        only_a=len(a) - len(pairs),     # A만 찍음 = B 기준 누락
        only_b=len(b) - len(pairs),     # B만 찍음
        mean_offset_px=float(np.mean([d for *_, d in pairs])) if pairs else None,
        class_confusion=confusion.tolist(),
        class_agreement=float(agree / n) if n else None,
        class_kappa=float(kappa) if n else None,
        per_class=per_class,
        mf1=float(np.mean([v['f1'] for v in per_class.values()])),
        unmatched_a_index=sorted(set(range(len(a))) - matched_a),
        unmatched_b_index=sorted(set(range(len(b))) - matched_b),
        class_mismatch_pairs=[(i, j) for i, j, _ in pairs if a[i, 2] != b[j, 2]],
    )


def _shift_any(mask, radius):
    """(2r+1)×(2r+1) 정사각 이웃 팽창. 가로→세로로 나눠 scipy 없이 계산한다."""
    out = mask.copy()
    for axis in (0, 1):
        src, out = out, out.copy()
        n = src.shape[axis]
        for shift in range(1, radius + 1):
            lo = [slice(None)] * 2
            hi = [slice(None)] * 2
            lo[axis], hi[axis] = slice(0, n - shift), slice(shift, n)
            out[tuple(hi)] |= src[tuple(lo)]
            out[tuple(lo)] |= src[tuple(hi)]
    return out


def boundary_band(ca, radius):
    """CA 경계에서 radius 픽셀 안쪽/바깥쪽 띠."""
    ca = ca.astype(bool)
    return _shift_any(ca, radius) & _shift_any(~ca, radius)


def compare_tissue(mask_a, mask_b, band_px=8, region=None):
    """mask: 원본 번호(BG=1, CA=2, UNK=255). region: 비교를 제한할 bool 배열."""
    a, b = np.asarray(mask_a), np.asarray(mask_b)
    if a.shape != b.shape:
        raise ValueError('두 조직 마스크의 크기가 다릅니다.')
    region = np.ones(a.shape, bool) if region is None else region.astype(bool)

    unk_a, unk_b = (a == 255) & region, (b == 255) & region
    valid = region & ~unk_a & ~unk_b      # 공식 지표처럼 UNK는 제외
    iou = {}
    for name, value in (('BG', 1), ('CA', 2)):
        pa, pb = (a == value) & valid, (b == value) & valid
        union = (pa | pb).sum()
        iou[name] = float((pa & pb).sum() / union) if union else None

    present = [v for v in iou.values() if v is not None]
    disagree = valid & (a != b)
    band = boundary_band(a == 2, band_px) | boundary_band(b == 2, band_px)

    return dict(
        iou=iou,
        miou=float(np.mean(present)) if present else None,
        pixel_agreement=float(1 - disagree.sum() / valid.sum()) if valid.sum() else None,
        ca_share_a=float(((a == 2) & region).sum() / region.sum()),
        ca_share_b=float(((b == 2) & region).sum() / region.sum()),
        unk_share_a=float(unk_a.sum() / region.sum()),
        unk_share_b=float(unk_b.sum() / region.sum()),
        unk_only_a_px=int((unk_a & ~unk_b).sum()),
        unk_only_b_px=int((unk_b & ~unk_a).sum()),
        unk_both_px=int((unk_a & unk_b).sum()),
        disagree_px=int(disagree.sum()),
        # 경계 띠 안의 불일치 = 경계선 위치 차이, 밖 = 영역 자체 판단 차이
        disagree_near_boundary_px=int((disagree & band).sum()),
        disagree_interior_px=int((disagree & ~band).sum()),
        a_ca_b_bg_px=int((valid & (a == 2) & (b == 1)).sum()),
        a_bg_b_ca_px=int((valid & (a == 1) & (b == 2)).sum()),
        disagree_map=disagree,
    )


def box_region(box, size=1024):
    """세포 사진이 차지하는 조직 사진 영역(0~1 경계 좌표) → bool 배열."""
    x0, y0, x1, y1 = np.asarray(box) * size
    yy, xx = np.mgrid[0:size, 0:size] + 0.5
    return (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
