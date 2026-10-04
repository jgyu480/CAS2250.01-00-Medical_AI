"""metadata의 샘플 연결과 split 정보를 확인한다."""

import json
import re
from collections import Counter


def load_metadata(path):
    data = json.loads(path.read_text(encoding='utf-8'))

    if not isinstance(data, dict) or not data:
        raise ValueError('metadata.json must be a nonempty object')

    # sample_pairs 아래에 기록된 형식과 직접 ID로 연결된 형식을 지원한다.
    pairs = data.get('sample_pairs', data)

    if not isinstance(pairs, dict) or not pairs:
        raise ValueError(
            'Unsupported metadata schema; send the top-level keys'
        )

    result = {}

    for key, value in pairs.items():
        if not isinstance(value, dict) or not str(key).isdigit():
            raise ValueError(f'Unsupported metadata record: {key}')

        # 숫자 ID를 공식 파일 이름처럼 001 형태로 맞춘다.
        pair_id = str(key).zfill(3)

        if pair_id in result:
            raise ValueError(f'Duplicate metadata ID: {pair_id}')

        result[pair_id] = value

    return result, list(data)


def metadata_summary(pairs, split, ids, report):
    slides = set()
    patients = set()
    organs = Counter()

    for pair_id in sorted(ids):
        if pair_id not in pairs:
            report['errors'].append(
                f'{split}/{pair_id}: missing metadata'
            )
            continue

        record = pairs[pair_id]
        missing = [
            key for key in ('slide_name', 'organ', 'subset')
            if not record.get(key)
        ]

        if missing:
            report['errors'].append(
                f'{split}/{pair_id}: metadata missing {missing}'
            )
            continue

        if not isinstance(record['slide_name'], str):
            report['errors'].append(
                f'{split}/{pair_id}: slide_name must be a string'
            )
            continue

        if record['subset'] != split:
            report['errors'].append(
                f'{split}/{pair_id}: metadata subset={record["subset"]}'
            )

        slide = record['slide_name']
        slides.add(slide)
        organs[str(record['organ'])] += 1

        # TCGA 슬라이드 이름의 앞부분에서 case ID를 얻는다.
        match = re.match(
            r'^(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})(?:-|$)',
            slide
        )
        if match:
            patients.add(match.group(1))

    return slides, patients, dict(sorted(organs.items()))
