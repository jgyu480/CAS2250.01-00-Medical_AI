"""작성자별로 독립 저장한 직접 라벨의 경로·기록을 다룬다.

두 작성자가 같은 24쌍을 서로의 결과를 보지 않고 따로 라벨링한다.
저장 위치: annotations/manual/annotators/<작성자ID>/{cell,tissue,notes}/<샘플ID>.*
작업 기록: annotations/annotator_tracking.csv (샘플 × 작성자 한 줄)

기존 draft/reviewed 구조는 그대로 두고, 작성자별 구조를 추가한다.
"""

import csv
import json

from common.paths import PROJECT_ROOT

ANNOTATOR_TRACKING_FIELDS = [
    'pair_id', 'organ', 'annotator', 'status',
    'started_at', 'finished_at', 'notes'
]
STATUSES = ('not_started', 'in_progress', 'completed')
NOTE_FIELDS = ['x', 'y', 'task', 'comment']
LABEL_PREFIX = 'annotator:'


def load_annotation_config():
    return json.loads(
        (PROJECT_ROOT / 'configs/annotation.json').read_text(encoding='utf-8')
    )


def annotator_ids(config=None):
    config = config or load_annotation_config()
    return list(config.get('annotators', {}))


def parse_label_source(label_source, config=None):
    """'annotator:sangwoo' → 'sangwoo'. 작성자 모드가 아니면 None."""
    if not isinstance(label_source, str) or not label_source.startswith(
        LABEL_PREFIX
    ):
        return None

    annotator = label_source[len(LABEL_PREFIX):]
    known = annotator_ids(config)
    if annotator not in known:
        raise ValueError(
            f'알 수 없는 작성자 {annotator!r}. 가능한 값: {known}'
        )
    return annotator


def annotator_dir(annotator, config=None):
    config = config or load_annotation_config()
    return PROJECT_ROOT / config['annotator_root'] / annotator


def label_paths(annotator, pair_id, config=None):
    root = annotator_dir(annotator, config)
    return dict(
        cell=root / 'cell' / f'{pair_id}.csv',
        tissue=root / 'tissue' / f'{pair_id}.png',
        notes=root / 'notes' / f'{pair_id}.csv'
    )


def read_csv_rows(path):
    if not path.is_file():
        raise ValueError(f'필요한 CSV가 없습니다: {path}')
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tracking_path(config=None):
    config = config or load_annotation_config()
    return PROJECT_ROOT / config['annotator_tracking_csv']


def read_tracking(config=None):
    """{(pair_id, annotator): row}. 기록 형식이 틀리면 중단한다."""
    config = config or load_annotation_config()
    rows = read_csv_rows(tracking_path(config))
    known = set(annotator_ids(config))
    result = {}

    for row in rows:
        key = (row['pair_id'], row['annotator'])
        if row['annotator'] not in known:
            raise ValueError(f'{key}: 설정에 없는 작성자입니다.')
        if row['status'] not in STATUSES:
            raise ValueError(f'{key}: status는 {STATUSES} 중 하나입니다.')
        if key in result:
            raise ValueError(f'{key}: 작업 기록이 중복됐습니다.')
        result[key] = row

    return result


def read_notes(path):
    """판단이 어려웠던 위치 메모. 파일이 없으면 빈 목록."""
    if not path.is_file():
        return []
    rows = read_csv_rows(path)
    for row in rows:
        if set(NOTE_FIELDS) - set(row):
            raise ValueError(f'{path.name}: 메모 열은 {NOTE_FIELDS}입니다.')
    return rows


def require_complete(annotator, pair_ids, config=None):
    """학습에 쓰기 전, 대상 전체가 완료됐고 두 파일이 있는지 확인한다."""
    config = config or load_annotation_config()
    states = read_tracking(config)

    for pair_id in sorted(pair_ids):
        state = states.get((pair_id, annotator))
        if state is None:
            raise ValueError(f'{pair_id}/{annotator}: 작업 기록이 없습니다.')
        if state['status'] != 'completed':
            raise ValueError(
                f'{pair_id}/{annotator}: 라벨이 미완료입니다 '
                f'({state["status"]}).'
            )
        for task in ('cell', 'tissue'):
            path = label_paths(annotator, pair_id, config)[task]
            if not path.is_file():
                raise ValueError(
                    f'{pair_id}/{annotator}: {task} 라벨 파일이 없습니다.'
                )
