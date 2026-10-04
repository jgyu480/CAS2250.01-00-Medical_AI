"""WSI 좌표에서 조직 사진 안의 세포 사진 위치를 계산한다."""

import numpy as np
from PIL import Image


def cell_box(record):
    """반환: [왼쪽, 위, 오른쪽, 아래], 조직 사진 기준 0~1 경계 좌표."""
    bounds = []

    for name in ('cell', 'tissue'):
        item = record[name]
        box = np.array(
            [item[k] for k in ('x_start', 'y_start', 'x_end', 'y_end')],
            dtype=float
        )

        if not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
            raise ValueError(f'{name}: WSI 좌표 범위가 잘못됐습니다.')

        # WSI에서 차지한 길이 × 원본 MPP ÷ 저장된 이미지 크기.
        for axis, span in zip(('x', 'y'), box[2:] - box[:2]):
            expected = span * float(record[f'mpp_{axis}']) / 1024
            actual = float(item[f'resized_mpp_{axis}'])

            if expected <= 0 or not np.isclose(
                actual, expected, rtol=0.01, atol=1e-6
            ):
                raise ValueError(
                    f'{name}: {axis}축 MPP와 WSI 범위가 불일치합니다.'
                )

        bounds.append(box)

    cell, tissue = bounds
    span = tissue[2:] - tissue[:2]

    box = np.r_[
        (cell[:2] - tissue[:2]) / span,
        (cell[2:] - tissue[:2]) / span
    ]

    if np.any(box < -1e-6) or np.any(box > 1 + 1e-6):
        raise ValueError('세포 사진이 조직 사진의 범위를 벗어납니다.')

    box = np.clip(box, 0, 1)
    center = (box[:2] + box[2:]) / 2
    offsets = [record['patch_x_offset'], record['patch_y_offset']]

    if not np.allclose(center, offsets, atol=1e-3, rtol=0):
        raise ValueError('WSI 좌표와 patch offset의 중심 위치가 다릅니다.')

    return box


def points_in_tissue(points, box, size=1024):
    """점은 픽셀 중심, box는 픽셀 경계이므로 0.5를 보정한다."""
    box = np.asarray(box)
    return (
        (np.asarray(points)[:, :2] + 0.5) / size
        * (box[2:] - box[:2]) + box[:2]
    ) * size - 0.5


def crop_tissue(image, box, size=1024):
    """미리보기용: 소수 좌표를 반올림하지 않고 해당 영역을 확대한다."""
    source = Image.fromarray(image)
    width, height = source.size
    extent = tuple(np.asarray(box) * [width, height, width, height])

    return np.asarray(
        source.transform(
            (size, size),
            Image.Transform.EXTENT,
            extent,
            resample=Image.Resampling.BILINEAR
        )
    ).copy()
