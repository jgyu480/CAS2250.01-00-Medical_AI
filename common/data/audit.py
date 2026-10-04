"""실제 파일 검사 결과를 수집한다. 데이터와 라벨은 수정하지 않는다."""

import itertools

from .audit_metadata import load_metadata
from .audit_split import inspect_split


def audit_dataset(root):
    report = {
        'splits': {},
        'errors': [],
        'warnings': [],
        'geometry_validated': False,
        'tissue_mapping_validated': True
    }
    manifest = []

    try:
        metadata, keys = load_metadata(root / 'metadata.json')
    except (OSError, ValueError) as exc:
        report['errors'].append(str(exc))
        report['ok'] = False
        return report, manifest

    # 실제 metadata 구조를 다음 단계 설계에 사용할 수 있게 기록한다.
    report['metadata_top_level_keys'] = keys
    first_id = sorted(metadata)[0]
    report['metadata_example'] = {
        'pair_id': first_id,
        'record': metadata[first_id]
    }

    id_sets, slide_sets, patient_sets = {}, {}, {}

    for split in ('train', 'val', 'test'):
        try:
            info, ids, slides, patients, rows = inspect_split(
                root, split, metadata, report
            )
            report['splits'][split] = info
            id_sets[split] = ids
            slide_sets[split] = slides
            patient_sets[split] = patients
            manifest.extend(rows)
        except (OSError, ValueError) as exc:
            report['errors'].append(f'{split}: {exc}')

    for left, right in itertools.combinations(id_sets, 2):
        for name, groups in (
            ('sample', id_sets),
            ('slide', slide_sets)
        ):
            overlap = sorted(groups[left] & groups[right])
            if overlap:
                report['errors'].append(
                    f'{left}/{right}: overlapping {name} IDs {overlap}'
                )

        overlap = sorted(patient_sets[left] & patient_sets[right])
        if overlap:
            report['warnings'].append(
                f'{left}/{right}: overlapping TCGA case IDs {overlap}'
            )

    excluded = {'586', '589', '609', '615'}
    present = sorted(id_sets.get('test', set()) & excluded)
    report['v101_excluded_test_ids_present'] = present

    if present:
        report['errors'].append(
            f'Test includes IDs excluded from v1.0.1: {present}'
        )

    all_ids = set().union(*id_sets.values()) if id_sets else set()
    report['unused_metadata_ids'] = sorted(set(metadata) - all_ids)

    # 픽셀 숫자만 보고 BG/CA/UNK 의미를 추측해서 지정하지 않는다.
    report['tissue_class_mapping'] = {'BG': 1, 'CA': 2, 'UNK': 255}

    report['ok'] = not report['errors']
    return report, manifest
