"""한 split의 네 파일이 짝을 이루는지 검사한다."""

from collections import Counter

from .audit_io import files_by_id, inspect_rgb, inspect_points, inspect_mask
from .audit_metadata import metadata_summary


def inspect_split(root, split, metadata, report):
    groups = {
        'cell_image': files_by_id(
            root / 'images' / split / 'cell', '.jpg'
        ),
        'tissue_image': files_by_id(
            root / 'images' / split / 'tissue', '.jpg'
        ),
        'cell_points': files_by_id(
            root / 'annotations' / split / 'cell', '.csv'
        ),
        'tissue_mask': files_by_id(
            root / 'annotations' / split / 'tissue', '.png'
        )
    }

    ids = set().union(*(set(group) for group in groups.values()))
    complete = set.intersection(*(set(group) for group in groups.values()))
    cell_counts, pixels, rows = Counter(), Counter(), []
    valid = 0

    if not ids:
        report['errors'].append(f'{split}: no samples found')

    for position, pair_id in enumerate(sorted(ids), 1):
        missing = [
            key for key, group in groups.items()
            if pair_id not in group
        ]
        if missing:
            report['errors'].append(
                f'{split}/{pair_id}: missing {missing}'
            )
            continue

        try:
            inspect_rgb(groups['cell_image'][pair_id])
            inspect_rgb(groups['tissue_image'][pair_id])
            counts, duplicates = inspect_points(groups['cell_points'][pair_id])
            mask_counts = inspect_mask(groups['tissue_mask'][pair_id])

            cell_counts.update(counts)
            pixels.update(mask_counts)
            valid += 1

            if duplicates:
                report['warnings'].append(
                    f'{split}/{pair_id}: {duplicates} duplicate cell coordinates'
                )

            # Dataset에서 사용할 수 있도록 네 파일 경로를 한 행에 모은다.
            record = metadata.get(pair_id, {})
            row = {'pair_id': pair_id, 'split': split}
            row.update({
                key: str(group[pair_id].relative_to(root))
                for key, group in groups.items()
            })
            row.update({
                'slide_name': record.get('slide_name', ''),
                'organ': record.get('organ', '')
            })
            rows.append(row)

        except (OSError, ValueError) as exc:
            report['errors'].append(f'{split}/{pair_id}: {exc}')

        if position % 100 == 0:
            print(f'[SCAN] {split}: {position}/{len(ids)}')

    slides, patients, organs = metadata_summary(
        metadata, split, ids, report
    )
    info = {
        'file_counts': {key: len(group) for key, group in groups.items()},
        'complete_pairs': len(complete),
        'valid_file_pairs': valid,
        'cells': {'BC': cell_counts[1], 'TC': cell_counts[2]},
        'mask_pixel_counts': dict(sorted(pixels.items())),
        'organs': organs,
        'slide_count': len(slides),
        'tcga_case_count': len(patients)
    }

    expected = {'train': 400, 'val': 137, 'test': 126}[split]
    if len(complete) != expected:
        report['warnings'].append(
            f'{split}: found {len(complete)} pairs; '
            f'v1.0.1 reference count={expected}'
        )

    print(f'[SCAN] {split}: {valid}/{len(ids)} file pairs validated')
    return info, ids, slides, patients, rows
