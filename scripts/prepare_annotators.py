"""작성자별 라벨 폴더와 작업 기록을 준비한다. 정답을 만들지 않는다.

python scripts/prepare_annotators.py

- annotations/manual/annotators/<작성자>/{cell,tissue,notes}/ 생성
- annotations/annotator_tracking.csv 생성 (24쌍 × 작성자 수)
기존 기록은 덮어쓰지 않고, 빠진 작성자/샘플 줄만 추가한다.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data.annotators import (  # noqa: E402
    ANNOTATOR_TRACKING_FIELDS, annotator_dir, annotator_ids,
    load_annotation_config, read_csv_rows, tracking_path, write_csv_rows
)


def main():
    config = load_annotation_config()
    selection = read_csv_rows(ROOT / config['selection_csv'])
    annotators = annotator_ids(config)
    if len(annotators) < 2:
        raise SystemExit('[오류] configs/annotation.json에 작성자를 2명 이상 적으세요.')

    path = tracking_path(config)
    rows = read_csv_rows(path) if path.exists() else []
    if rows and list(rows[0]) != ANNOTATOR_TRACKING_FIELDS:
        raise SystemExit(f'[오류] {path.name}의 열 구성이 다릅니다. 수정하지 않았습니다.')

    existing = {(row['pair_id'], row['annotator']) for row in rows}
    selected_ids = {pair['pair_id'] for pair in selection}
    stray = {pid for pid, _ in existing} - selected_ids
    if stray:
        raise SystemExit(f'[오류] selection에 없는 샘플 기록: {sorted(stray)}')

    added = 0
    for pair in selection:
        for annotator in annotators:
            if (pair['pair_id'], annotator) in existing:
                continue
            rows.append(dict(
                pair_id=pair['pair_id'], organ=pair['organ'],
                annotator=annotator, status='not_started',
                started_at='', finished_at='', notes=''
            ))
            added += 1

    order = {a: i for i, a in enumerate(annotators)}
    rows.sort(key=lambda r: (r['pair_id'], order.get(r['annotator'], 99)))
    write_csv_rows(path, ANNOTATOR_TRACKING_FIELDS, rows)

    for annotator in annotators:
        for sub in ('cell', 'tissue', 'notes'):
            folder = annotator_dir(annotator, config) / sub
            folder.mkdir(parents=True, exist_ok=True)
            # 빈 폴더도 Git에 남겨 다른 팀원이 구조를 바로 볼 수 있게 한다.
            keep = folder / '.gitkeep'
            if not any(folder.iterdir()):
                keep.touch()

    print(f'[OK] 작성자 {annotators}, 대상 {len(selection)}쌍')
    print(f'[OK] 작업 기록 {path.relative_to(ROOT)}: {len(rows)}줄 (새로 추가 {added})')
    print('[안내] 정답 파일은 만들지 않았습니다. 라벨링 도구로 작성한 뒤')
    print('       python scripts/import_annotation.py 로 가져오세요.')


if __name__ == '__main__':
    main()
