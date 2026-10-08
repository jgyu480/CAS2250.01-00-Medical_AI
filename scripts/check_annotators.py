"""작성자별 라벨 파일과 작업 기록이 일치하는지 검사하고 진행 현황을 보여준다.

python scripts/check_annotators.py

자동 검사: 형식·좌표·클래스·중복 점·기록 일관성.
세포 누락이나 분류의 옳고 그름은 사람이 확인한다.
"""

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data.annotators import (  # noqa: E402
    annotator_ids, label_paths, load_annotation_config, read_csv_rows,
    read_notes, read_tracking
)
from common.data.targets import read_points  # noqa: E402
from common.data.audit_io import inspect_mask  # noqa: E402


def main():
    config = load_annotation_config()
    selection = read_csv_rows(ROOT / config['selection_csv'])
    annotators = annotator_ids(config)
    try:
        states = read_tracking(config)
    except ValueError as error:
        raise SystemExit(f'[오류] {error}\n먼저 python scripts/prepare_annotators.py')

    errors, done = [], Counter()
    expected = {(p['pair_id'], a) for p in selection for a in annotators}
    if set(states) != expected:
        errors.append(
            f'기록 불일치: 빠짐 {sorted(expected - set(states))[:5]}, '
            f'남음 {sorted(set(states) - expected)[:5]}'
        )

    for pair_id, annotator in sorted(expected & set(states)):
        state = states[(pair_id, annotator)]
        paths = label_paths(annotator, pair_id, config)
        try:
            if paths['cell'].exists():
                read_points(paths['cell'])
            if paths['tissue'].exists():
                inspect_mask(paths['tissue'])
            read_notes(paths['notes'])
        except ValueError as error:
            errors.append(f'{pair_id}/{annotator}: {error}')
            continue

        if state['status'] == 'completed':
            if not (paths['cell'].is_file() and paths['tissue'].is_file()):
                errors.append(f'{pair_id}/{annotator}: completed인데 파일이 없습니다.')
            else:
                done[annotator] += 1

    total = len(selection)
    for annotator in annotators:
        name = config['annotators'][annotator]
        print(f'[진행] {name}({annotator}): {done[annotator]}/{total} 완료')

    both = sum(
        all(states.get((p['pair_id'], a), {}).get('status') == 'completed'
            for a in annotators)
        for p in selection
    )
    print(f'[진행] 두 사람 모두 완료되어 비교 가능한 쌍: {both}/{total}')

    if errors:
        for message in errors[:20]:
            print('[오류]', message)
        raise SystemExit('[FAIL] 작성자별 라벨 점검 실패')
    print('[PASS] 작성자별 라벨 형식과 기록이 일치합니다.')


if __name__ == '__main__':
    main()
