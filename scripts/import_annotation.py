"""라벨링 도구에서 저장한 파일을 검사한 뒤 작성자 폴더로 옮기고 기록을 갱신한다.

예시
  # 다운로드 폴더에 있는 224.csv, 224.png, 224_notes.csv를 한 번에 가져오기
  python scripts/import_annotation.py --annotator sangwoo --from-dir ~/Downloads --status completed

  # 한 샘플만 지정해서 가져오기
  python scripts/import_annotation.py --annotator sangwoo --pair 224 \
      --cell ~/Downloads/224.csv --tissue ~/Downloads/224.png --status in_progress

이미 있는 라벨은 --overwrite 없이는 덮어쓰지 않는다.
형식 검사: 좌표 0~1023, BC=1/TC=2, 중복 점 없음, 1024×1024 단일 채널 PNG(1/2/255).
"""

import argparse
import filecmp
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data.annotators import (  # noqa: E402
    ANNOTATOR_TRACKING_FIELDS, annotator_ids, label_paths,
    load_annotation_config, read_csv_rows, read_notes, read_tracking,
    tracking_path, write_csv_rows
)
from common.data.targets import read_points  # noqa: E402


def check_mask(path):
    with Image.open(path) as image:
        image.load()
        if image.mode not in ('L', 'P', 'I', 'I;16'):
            raise ValueError(
                f'{path.name}: 단일 채널 PNG가 아닙니다 (mode={image.mode}). '
                '색칠된 스크린샷이 아닌 도구의 저장 파일을 쓰세요.'
            )
        mask = np.asarray(image)
    if mask.shape != (1024, 1024):
        raise ValueError(f'{path.name}: 크기가 1024×1024가 아닙니다.')
    values = set(map(int, np.unique(mask)))
    if values - {1, 2, 255}:
        raise ValueError(f'{path.name}: BG=1, CA=2, UNK=255 외 값 {sorted(values)}')
    share = {v: float((mask == v).mean()) for v in (1, 2, 255)}
    return share


def import_one(args, pair_id, cell, tissue, notes, config):
    dest = label_paths(args.annotator, pair_id, config)
    points = read_points(cell)
    share = check_mask(tissue)
    if notes is not None:
        read_notes(notes)

    for key, src in (('cell', cell), ('tissue', tissue), ('notes', notes)):
        if src is None:
            continue
        target = dest[key]
        if target.is_file() and filecmp.cmp(src, target, shallow=False):
            continue  # 같은 내용이면 기록만 갱신한다.
        if target.exists() and target.resolve() != src.resolve():
            if not args.overwrite:
                raise ValueError(
                    f'{pair_id}: {target.relative_to(ROOT)} 이미 있음 '
                    '(덮어쓰려면 --overwrite)'
                )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != src.resolve():
            shutil.copy2(src, target)

    counts = {1: int((points[:, 2] == 1).sum()), 2: int((points[:, 2] == 2).sum())}
    print(
        f'[OK] {pair_id}/{args.annotator}: 세포 BC={counts[1]} TC={counts[2]}, '
        f'조직 BG={share[1]:.1%} CA={share[2]:.1%} UNK={share[255]:.1%}'
    )
    if share[255] > 0.3:
        print('     [주의] UNK가 30%를 넘습니다. 작업하지 않은 영역을 UNK로 둔 것은 아닌지 확인하세요.')
    if len(points) == 0:
        print('     [주의] 세포 점이 0개입니다.')


def update_tracking(config, annotator, pair_ids, status, note):
    path = tracking_path(config)
    read_tracking(config)  # 형식 검사
    rows = read_csv_rows(path)
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    for row in rows:
        if row['annotator'] != annotator or row['pair_id'] not in pair_ids:
            continue
        row['status'] = status
        row['started_at'] = row['started_at'] or now
        if status == 'completed':
            row['finished_at'] = now
        if note:
            row['notes'] = (row['notes'] + ' | ' if row['notes'] else '') + note
    write_csv_rows(path, ANNOTATOR_TRACKING_FIELDS, rows)


def main():
    config = load_annotation_config()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--annotator', required=True, choices=annotator_ids(config))
    parser.add_argument('--from-dir', type=Path)
    parser.add_argument('--pair')
    parser.add_argument('--cell', type=Path)
    parser.add_argument('--tissue', type=Path)
    parser.add_argument('--notes', type=Path)
    parser.add_argument('--status', choices=['in_progress', 'completed'], default='in_progress')
    parser.add_argument('--note', default='', help='작업 기록 notes 열에 덧붙일 짧은 메모')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    selected = {row['pair_id'] for row in read_csv_rows(ROOT / config['selection_csv'])}
    jobs = []

    if args.from_dir:
        folder = args.from_dir.expanduser()
        for pair_id in sorted(selected):
            cell, tissue = folder / f'{pair_id}.csv', folder / f'{pair_id}.png'
            if not (cell.is_file() or tissue.is_file()):
                continue
            if not (cell.is_file() and tissue.is_file()):
                raise SystemExit(f'[오류] {pair_id}: csv와 png가 둘 다 있어야 합니다.')
            notes = folder / f'{pair_id}_notes.csv'
            jobs.append((pair_id, cell, tissue, notes if notes.is_file() else None))
        if not jobs:
            raise SystemExit(f'[오류] {folder}에서 대상 샘플 파일을 찾지 못했습니다.')
    else:
        if not (args.pair and args.cell and args.tissue):
            raise SystemExit('[오류] --from-dir 또는 --pair/--cell/--tissue를 지정하세요.')
        jobs.append((args.pair.zfill(3), args.cell.expanduser(),
                     args.tissue.expanduser(),
                     args.notes.expanduser() if args.notes else None))

    for pair_id, *_ in jobs:
        if pair_id not in selected:
            raise SystemExit(f'[오류] {pair_id}는 selection.csv의 대상이 아닙니다.')

    done = []
    for job in jobs:
        try:
            import_one(args, *job, config)
            done.append(job[0])
        except ValueError as error:
            print('[오류]', error)

    if done:
        update_tracking(config, args.annotator, set(done), args.status, args.note)
        print(f'[기록] {len(done)}쌍 → status={args.status}')
    if len(done) != len(jobs):
        raise SystemExit('[FAIL] 일부 파일을 가져오지 못했습니다. 위 오류를 확인하세요.')


if __name__ == '__main__':
    main()
