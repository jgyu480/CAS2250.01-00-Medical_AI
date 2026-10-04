"""대상 목록·작업 기록·입력 사진을 준비한다. 정답을 생성하지 않는다."""

import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import resolve_paths
from common.data.annotation_selection import (
    select_pairs, SELECTION_FIELDS, TRACKING_FIELDS
)
from common.data.audit_metadata import load_metadata


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    paths = resolve_paths()
    config = json.loads(
        (ROOT / 'configs/annotation.json').read_text(encoding='utf-8')
    )
    audit = paths['outputs'] / 'audit'
    report = json.loads(
        (audit / 'report.json').read_text(encoding='utf-8')
    )

    if not report.get('ok') or json.loads((ROOT / 'configs/common.json').read_text(encoding='utf-8')).get('tasks', {}).get('tissue_classes') != {'BG': 1, 'CA': 2, 'UNK': 255}:
        raise SystemExit('먼저 데이터 검사와 공식 조직 클래스 설정을 완료하세요.')

    selected = select_pairs(
        read_csv(audit / 'manifest.csv'),
        config['per_organ'],
        config['seed']
    )
    selection_path = ROOT / config['selection_csv']
    tracking_path = ROOT / config['tracking_csv']

    # 기존 작업 목록이나 기록을 조용히 덮어쓰지 않는다.
    if selection_path.exists() and read_csv(selection_path) != selected:
        raise SystemExit('Existing selection differs. It was not overwritten.')

    if tracking_path.exists():
        tracking = read_csv(tracking_path)
        if (
            len(tracking) != len(selected)
            or {row['pair_id'] for row in tracking}
            != {row['pair_id'] for row in selected}
        ):
            raise SystemExit('Existing tracking IDs differ. It was not overwritten.')

    if not selection_path.exists():
        write_csv(selection_path, SELECTION_FIELDS, selected)

    if not tracking_path.exists():
        tracking = [
            {key: '' for key in TRACKING_FIELDS}
            for _ in selected
        ]
        for state, pair in zip(tracking, selected):
            state.update(
                pair_id=pair['pair_id'],
                organ=pair['organ'],
                draft_status='not_started',
                review_status='not_started'
            )
        write_csv(tracking_path, TRACKING_FIELDS, tracking)

    metadata, _ = load_metadata(paths['data_root'] / 'metadata.json')

    for pair in selected:
        packet = paths['outputs'] / 'annotation_packets' / pair['pair_id']
        packet.mkdir(parents=True, exist_ok=True)

        for key, filename in (
            ('cell_image', 'cell.jpg'),
            ('tissue_image', 'tissue.jpg')
        ):
            if not (packet / filename).exists():
                shutil.copy2(
                    paths['data_root'] / pair[key],
                    packet / filename
                )

        (packet / 'metadata.json').write_text(
            json.dumps(metadata[pair['pair_id']], ensure_ascii=False, indent=2)
            + '\n',
            encoding='utf-8'
        )

    for version in ('draft', 'reviewed'):
        for task in ('cell', 'tissue'):
            (ROOT / 'annotations/manual' / version / task).mkdir(
                parents=True,
                exist_ok=True
            )

    print(
        '[OK] Selected pairs:',
        len(selected),
        dict(Counter(row['organ'] for row in selected))
    )
    print('[OK] Existing selection and work records are preserved on rerun.')
    print('[OK] Input photos prepared. No ground-truth files were created.')
    print('Selection:', selection_path)
    print('Packets:', paths['outputs'] / 'annotation_packets')


if __name__ == '__main__':
    main()
