"""공식 데이터 파일을 읽어 형식과 값의 범위를 검사한다."""

import csv
import math
from collections import Counter

import numpy as np
from PIL import Image


def files_by_id(folder, suffix):
    # 파일 이름의 ID를 기준으로 JPG·CSV·PNG를 연결한다.
    if not folder.is_dir():
        raise ValueError(f'Missing directory: {folder}')

    return {
        p.stem: p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() == suffix
    }


def inspect_rgb(path):
    with Image.open(path) as image:
        # 헤더뿐 아니라 픽셀도 읽어서 손상된 파일을 확인한다.
        image.load()

        if image.size != (1024, 1024) or image.mode != 'RGB':
            raise ValueError(
                f'{path.name}: expected 1024x1024 RGB, '
                f'got {image.size}/{image.mode}'
            )


def inspect_points(path):
    counts = Counter()
    coordinates = set()
    duplicates = 0

    with path.open(encoding='utf-8-sig', newline='') as handle:
        for line, row in enumerate(csv.reader(handle), 1):
            if not row or all(not item.strip() for item in row):
                continue

            if len(row) != 3:
                raise ValueError(f'{path.name}:{line}: expected x,y,label')

            values = [float(value) for value in row]
            x, y, label = values

            if not all(math.isfinite(value) for value in values):
                raise ValueError(f'{path.name}:{line}: NaN/Inf')

            if not (0 <= x <= 1023 and 0 <= y <= 1023):
                raise ValueError(
                    f'{path.name}:{line}: coordinates outside 0..1023'
                )

            if label not in (1, 2):
                raise ValueError(
                    f'{path.name}:{line}: expected BC=1 or TC=2'
                )

            # 같은 좌표가 반복되면 기록한다. 정답을 자동 삭제하지 않는다.
            duplicates += int((x, y) in coordinates)
            coordinates.add((x, y))
            counts[int(label)] += 1

    return counts, duplicates


def inspect_mask(path):
    with Image.open(path) as image:
        image.load()
        mask = np.asarray(image)

        if image.size != (1024, 1024) or mask.ndim != 2:
            raise ValueError(
                f'{path.name}: expected a 1024x1024 single-channel mask'
            )

        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f'{path.name}: mask IDs must be integers')

        # RGB로 변환하지 않고 정수 클래스 값을 그대로 읽는다.
        values, counts = np.unique(mask, return_counts=True)
        if set(map(int, values)) - {1, 2, 255}:
            raise ValueError(f"Unexpected tissue IDs: {values}")

        return Counter({
            int(value): int(count)
            for value, count in zip(values, counts)
        })
