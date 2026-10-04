"""CSV의 세포 점을 학습용 원형 클래스 지도에 옮긴다."""

import csv
import numpy as np


def read_points(path, size=1024):
    rows = []

    with path.open(encoding='utf-8-sig', newline='') as handle:
        for line, row in enumerate(csv.reader(handle), 1):
            if not row or all(not value.strip() for value in row):
                continue

            if len(row) != 3:
                raise ValueError(
                    f'{path.name}:{line}: x,y,label 세 칸이어야 합니다.'
                )

            rows.append([float(value) for value in row])

    # 빈 CSV도 N=0인 점 목록으로 읽을 수 있다.
    points = np.asarray(rows, dtype=np.float64).reshape(-1, 3)

    if (
        not np.isfinite(points).all()
        or np.any(points[:, :2] < 0)
        or np.any(points[:, :2] > size - 1)
        or not np.isin(points[:, 2], [1, 2]).all()
    ):
        raise ValueError(
            f'{path.name}: 좌표 범위 또는 BC=1/TC=2를 확인하세요.'
        )

    if len(np.unique(points[:, :2], axis=0)) != len(points):
        raise ValueError(f'{path.name}: 같은 위치에 중복된 점이 있습니다.')

    return points


def points_to_target(points, radius=7, size=1024):
    """0=세포 원 밖, 1=BC, 2=TC. 원 밖의 0은 조직 BG와 별개다."""
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('원 반지름은 양수여야 합니다.')

    target = np.zeros((size, size), dtype=np.int64)
    distance = np.full((size, size), np.inf)

    # 가까운 중심을 사용한다. 동률은 (x,y,label) 정렬순으로 결정한다.
    for x, y, label in sorted(points.tolist()):
        left = max(0, int(np.floor(x - radius)))
        top = max(0, int(np.floor(y - radius)))
        right = min(size, int(np.ceil(x + radius)) + 1)
        bottom = min(size, int(np.ceil(y + radius)) + 1)

        yy, xx = np.mgrid[top:bottom, left:right]
        d2 = (xx - x)**2 + (yy - y)**2
        nearest = distance[top:bottom, left:right]

        keep = (d2 <= radius**2) & (d2 < nearest)
        target[top:bottom, left:right][keep] = int(label)
        nearest[keep] = d2[keep]

    return target
