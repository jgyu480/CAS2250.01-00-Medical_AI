"""두 사진과 라벨에 같은 뒤집기·90도 회전을 적용한다."""

import numpy as np


def transform_pair(cell, tissue, mask, points, box, flip=False, turns=0):
    cell, tissue, mask = cell.copy(), tissue.copy(), mask.copy()
    points = points.copy()
    box = np.asarray(box, dtype=float).copy()
    size = cell.shape[0]

    if (
        cell.shape[:2] != (size, size)
        or tissue.shape[:2] != (size, size)
    ):
        raise ValueError('두 사진은 같은 크기의 정사각형이어야 합니다.')

    if flip:
        cell = cell[:, ::-1]
        tissue = tissue[:, ::-1]
        mask = mask[:, ::-1]
        points[:, 0] = size - 1 - points[:, 0]

        x0, y0, x1, y1 = box
        box = np.array([1 - x1, y0, 1 - x0, y1])

    for _ in range(int(turns) % 4):
        cell = np.rot90(cell)
        tissue = np.rot90(tissue)
        mask = np.rot90(mask)

        x = points[:, 0].copy()
        y = points[:, 1].copy()
        points[:, 0], points[:, 1] = y, size - 1 - x

        x0, y0, x1, y1 = box
        box = np.array([y0, 1 - x1, y1, 1 - x0])

    # 뒤집기 후 음수 stride를 없애 이후 PyTorch 변환도 가능하게 한다.
    return (
        np.ascontiguousarray(cell),
        np.ascontiguousarray(tissue),
        np.ascontiguousarray(mask),
        points,
        box
    )
