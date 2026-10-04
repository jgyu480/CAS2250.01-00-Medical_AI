"""원본 조직 번호와 코드에서 사용할 조직 번호를 구분한다."""

import numpy as np

TISSUE_RAW = {'BG': 1, 'CA': 2, 'UNK': 255}
TISSUE_TARGET = {'BG': 0, 'CA': 1, 'UNK': 2}
TISSUE_IGNORE_INDEX = TISSUE_TARGET['UNK']
MAPPING_SOURCE = 'https://ocelot2023.grand-challenge.org/datasets/'


def tissue_to_target(mask):
    mask = np.asarray(mask)

    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError('조직 라벨은 2차원 정수 배열이어야 합니다.')

    unknown = set(map(int, np.unique(mask))) - set(TISSUE_RAW.values())
    if unknown:
        raise ValueError(f'알 수 없는 원본 조직 번호: {unknown}')

    # 원본 PNG는 변경하지 않고, 읽은 배열에서만 번호를 바꾼다.
    target = np.full(mask.shape, TISSUE_IGNORE_INDEX, dtype=np.int64)
    target[mask == TISSUE_RAW['BG']] = TISSUE_TARGET['BG']
    target[mask == TISSUE_RAW['CA']] = TISSUE_TARGET['CA']
    return target
