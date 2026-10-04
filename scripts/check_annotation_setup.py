"""준비 완료와 실제 라벨링 완료를 구분해 검사한다."""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import resolve_paths
from common.data.annotation_selection import select_pairs
from common.data.audit_io import inspect_points, inspect_mask


def read_csv(path):
    if not path.is_file():
        raise SystemExit(f'[오류] 파일 없음: {path}\n먼저 python scripts/prepare_annotation.py를 실행하세요.')
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise SystemExit('[ERROR] ' + message)


def main():
    paths = resolve_paths()
    config = json.loads(
        (ROOT / 'configs/annotation.json').read_text(encoding='utf-8')
    )
    selection = read_csv(ROOT / config['selection_csv'])
    manifest = read_csv(paths['outputs'] / 'audit/manifest.csv')

    require(
        selection == select_pairs(
            manifest, config['per_organ'], config['seed']
        ),
        'Selection changed'
    )

    states = read_csv(ROOT / config['tracking_csv'])
    require(len(states) == len(selection), 'Tracking row count differs')
    require(
        {row['pair_id'] for row in states}
        == {row['pair_id'] for row in selection},
        'Tracking IDs differ'
    )

    allowed = {'not_started', 'in_progress', 'completed'}
    draft_done = reviewed_done = 0

    for row in states:
        pair_id = row['pair_id']

        for key in ('draft_status', 'review_status'):
            require(row[key] in allowed, f'{pair_id}: invalid {key}')

        packet = paths['outputs'] / 'annotation_packets' / pair_id
        require(
            all(
                (packet / name).is_file()
                for name in ('cell.jpg', 'tissue.jpg', 'metadata.json')
            ),
            f'{pair_id}: missing packet'
        )

        for version, key in (
            ('draft', 'draft_status'),
            ('reviewed', 'review_status')
        ):
            cell = (
                ROOT / 'annotations/manual' / version
                / 'cell' / f'{pair_id}.csv'
            )
            tissue = (
                ROOT / 'annotations/manual' / version
                / 'tissue' / f'{pair_id}.png'
            )

            if cell.exists():
                _, duplicates = inspect_points(cell)
                require(
                    not duplicates,
                    f'{pair_id}/{version}: duplicate points'
                )
            if tissue.exists():
                inspect_mask(tissue)

            if row[key] == 'completed':
                require(
                    cell.is_file() and tissue.is_file(),
                    f'{pair_id}/{version}: completion requires both label files'
                )

        if row['draft_status'] == 'completed':
            require(
                bool(row['annotator'].strip()),
                f'{pair_id}: annotator missing'
            )
            draft_done += 1

        if row['review_status'] == 'completed':
            require(
                row['draft_status'] == 'completed',
                f'{pair_id}: draft incomplete'
            )
            require(
                bool(row['reviewer'].strip())
                and row['reviewer'].strip() != row['annotator'].strip(),
                f'{pair_id}: independent reviewer required'
            )
            reviewed_done += 1

    print(
        '[OK] Annotation setup:',
        len(selection),
        dict(Counter(row['organ'] for row in selection))
    )
    print(
        f'[INFO] Actual annotation: draft={draft_done}/{len(selection)}, '
        f'reviewed={reviewed_done}/{len(selection)}'
    )

    # 어노테이션 검사 시 기존 단계 진행률을 보존한다.
    progress_path = ROOT / 'docs/progress.json'
    progress = (
        json.loads(progress_path.read_text(encoding='utf-8'))
        if progress_path.exists() else {}
    )
    completed = set(progress.get('completed_steps', []))
    later_stage = (
        any(step > 3 for step in completed)
        or progress.get('current_step', 1) > 4
    )
    progress.setdefault('total_steps', 10)
    progress['completed_steps'] = sorted(completed | {1, 2, 3})
    progress['current_step'] = max(progress.get('current_step', 1), 4)
    if not later_stage:
        progress['status'] = 'annotation_preparation_complete'
    progress.setdefault('training_started', False)
    progress.setdefault('annotation', {}).update(
        selected=len(selection),
        draft_complete=draft_done,
        reviewed_complete=reviewed_done
    )
    previous_path = ROOT / 'docs/progress.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else {}
    progress['completed_steps'] = sorted(set(previous.get('completed_steps', [])) | {1, 2, 3})
    progress['current_step'] = max(previous.get('current_step', 4), 4)
    (ROOT / 'docs/progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8'
    )

    print(
        '[PASS] Step 3 preparation passed. '
        'Annotation completion is recorded separately.'
    )


if __name__ == '__main__':
    main()
