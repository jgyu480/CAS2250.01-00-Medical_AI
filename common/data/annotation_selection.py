"""어노테이션 대상은 train에서만 고르고 결정된 목록은 고정한다."""

import random
import re

ORGANS = (
    'bladder', 'endometrium', 'head-and-neck',
    'kidney', 'prostate', 'stomach'
)

SELECTION_FIELDS = [
    'pair_id', 'organ', 'slide_name', 'cell_image', 'tissue_image'
]

TRACKING_FIELDS = [
    'pair_id', 'organ', 'draft_status', 'review_status',
    'annotator', 'reviewer', 'started_at', 'finished_at', 'notes'
]


def select_pairs(manifest, per_organ=4, seed=42):
    if per_organ < 1:
        raise ValueError('per_organ must be positive')

    train = [row for row in manifest if row['split'] == 'train']
    rng = random.Random(seed)
    used_cases = set()
    selected = []

    for organ in ORGANS:
        # 원래 파일 목록 순서가 달라도 결과가 같도록 먼저 정렬한다.
        candidates = sorted(
            (row for row in train if row['organ'] == organ),
            key=lambda row: row['pair_id']
        )
        rng.shuffle(candidates)
        chosen = []

        for row in candidates:
            match = re.match(
                r'^(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})(?:-|$)',
                row['slide_name']
            )
            case = match.group(1) if match else row['slide_name']

            if case in used_cases:
                continue

            used_cases.add(case)
            chosen.append({
                key: row[key]
                for key in SELECTION_FIELDS
            })

            if len(chosen) == per_organ:
                break

        if len(chosen) != per_organ:
            raise ValueError(f'{organ}: not enough distinct cases')

        selected.extend(
            sorted(chosen, key=lambda row: row['pair_id'])
        )

    return selected
