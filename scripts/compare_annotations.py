"""두 라벨 출처를 샘플별로 비교한다. 결과는 outputs/annotation_compare/에 저장.

라벨 출처(--a, --b)
  official             공식 정답 (data/raw)
  annotator:<ID>       작성자 폴더 (예: annotator:jinho)
  dir:<경로>           <경로>/cell/<ID>.csv, <경로>/tissue/<ID>.png 형식의 폴더
                       (모델 예측 저장 폴더. CSV 4번째 열 점수는 무시)

예시
  # 작성자 간 차이 (어노테이션 작업)
  python scripts/compare_annotations.py --a annotator:jinho --b annotator:sangwoo
  # 각 작성자와 공식 정답의 차이
  python scripts/compare_annotations.py --a official --b annotator:sangwoo
  # (M0 실험자) 같은 모델 예측을 두 작성자 정답과 각각 평가
  python scripts/compare_annotations.py --a annotator:jinho   --b dir:outputs/M0/pred_train
  python scripts/compare_annotations.py --a annotator:sangwoo --b dir:outputs/M0/pred_train

A를 기준(정답)으로 본다. 작성자끼리 비교할 때 F1과 IoU는 A/B를 바꿔도 같다.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import resolve_paths  # noqa: E402
from common.data.annotators import (  # noqa: E402
    label_paths, load_annotation_config, parse_label_source, read_csv_rows,
    read_notes, read_tracking
)
from common.data.label_compare import (  # noqa: E402
    DISTANCE_PX, box_region, compare_cells, compare_tissue
)


def r4(value):
    return '' if value is None else round(value, 4)


def read_points_loose(path):
    rows = []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.reader(handle):
            if row and any(v.strip() for v in row):
                rows.append([float(v) for v in row[:3]])
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def read_mask(path):
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.ndim != 2:
        raise ValueError(f'{path}: 단일 채널 마스크가 아닙니다.')
    return mask


class Source:
    def __init__(self, spec, config, manifest, require_completed):
        self.spec, self.config = spec, config
        self.annotator = parse_label_source(spec, config)
        self.folder = None
        self.manifest = manifest
        self.states = None
        if spec.startswith('dir:'):
            self.folder = Path(spec[4:]).expanduser()
            if not self.folder.is_absolute():
                self.folder = ROOT / self.folder
        elif spec != 'official' and self.annotator is None:
            raise SystemExit(f'[오류] 알 수 없는 라벨 출처: {spec}')
        if self.annotator and require_completed:
            self.states = read_tracking(config)

    @property
    def name(self):
        return self.spec.replace(':', '-').replace('/', '_')

    def paths(self, pair_id):
        if self.annotator:
            if self.states is not None and self.states.get(
                (pair_id, self.annotator), {}
            ).get('status') != 'completed':
                return None
            p = label_paths(self.annotator, pair_id, self.config)
            return p['cell'], p['tissue']
        if self.folder:
            return (self.folder / 'cell' / f'{pair_id}.csv',
                    self.folder / 'tissue' / f'{pair_id}.png')
        row = self.manifest.get(pair_id)
        if row is None:
            return None
        root = resolve_paths()['data_root']
        return root / row['cell_points'], root / row['tissue_mask']

    def load(self, pair_id):
        paths = self.paths(pair_id)
        if paths is None or not all(p.is_file() for p in paths):
            return None
        return read_points_loose(paths[0]), read_mask(paths[1])

    def notes(self, pair_id):
        if not self.annotator:
            return []
        return read_notes(label_paths(self.annotator, pair_id, self.config)['notes'])


def load_images(manifest, pair_id):
    row = manifest.get(pair_id)
    if row is None:
        return None, None
    root = resolve_paths()['data_root']
    try:
        return (np.asarray(Image.open(root / row['cell_image']).convert('RGB')),
                np.asarray(Image.open(root / row['tissue_image']).convert('RGB')))
    except OSError:
        return None, None


def draw_cells(image, a, b, result, path):
    """회색=위치·클래스 일치, 빨강=A만, 하늘=B만, 주황=위치 일치·클래스 다름."""
    canvas = Image.fromarray(image if image is not None else
                             np.full((1024, 1024, 3), 255, np.uint8))
    draw = ImageDraw.Draw(canvas)
    mismatch_a = {i for i, _ in result['class_mismatch_pairs']}
    for i, (x, y, _) in enumerate(a):
        if i in result['unmatched_a_index']:
            color = (230, 30, 30)
        elif i in mismatch_a:
            color = (255, 140, 0)
        else:
            color = (120, 120, 120)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=2)
    for j in result['unmatched_b_index']:
        x, y, _ = b[j]
        draw.line((x - 6, y, x + 6, y), fill=(0, 190, 255), width=2)
        draw.line((x, y - 6, x, y + 6), fill=(0, 190, 255), width=2)
    canvas.save(path)


def draw_tissue(image, a, b, disagree, box, path):
    """초록=둘 다 CA, 빨강=A만 CA, 파랑=B만 CA, 자홍=한쪽 이상 UNK."""
    base = (image if image is not None else
            np.full((1024, 1024, 3), 255, np.uint8)).astype(float)
    overlay = base.copy()
    for sel, color in (
        ((a == 2) & (b == 2), (0, 200, 80)),
        ((a == 2) & (b == 1), (230, 30, 30)),
        ((a == 1) & (b == 2), (30, 80, 230)),
        ((a == 255) | (b == 255), (230, 50, 230)),
    ):
        overlay[sel] = base[sel] * 0.45 + np.array(color) * 0.55
    canvas = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    if box is not None:
        ImageDraw.Draw(canvas).rectangle(tuple(np.asarray(box) * 1024),
                                         outline=(255, 220, 0), width=4)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--ids', nargs='*', help='비교할 샘플 ID (기본: selection.csv 24쌍)')
    parser.add_argument('--radius', type=float, default=DISTANCE_PX)
    parser.add_argument('--band', type=int, default=8, help='조직 경계 띠 반지름(px)')
    parser.add_argument('--include-unfinished', action='store_true',
                        help='completed가 아닌 작성자 라벨도 비교 (중간 점검용)')
    parser.add_argument('--no-images', action='store_true')
    args = parser.parse_args()

    config = load_annotation_config()
    outputs = resolve_paths()['outputs']
    manifest_path = outputs / 'audit/manifest.csv'
    manifest = ({r['pair_id']: r for r in read_csv_rows(manifest_path)}
                if manifest_path.is_file() else {})
    metadata = {}
    try:
        from common.data.audit_metadata import load_metadata
        from common.data.geometry import cell_box
        metadata, _ = load_metadata(resolve_paths()['data_root'] / 'metadata.json')
    except (OSError, ValueError):
        cell_box = None

    src_a = Source(args.a, config, manifest, not args.include_unfinished)
    src_b = Source(args.b, config, manifest, not args.include_unfinished)
    ids = [i.zfill(3) for i in args.ids] if args.ids else [
        r['pair_id'] for r in read_csv_rows(ROOT / config['selection_csv'])]

    folder = outputs / 'annotation_compare' / f'{src_a.name}__vs__{src_b.name}'
    (folder / 'overlays').mkdir(parents=True, exist_ok=True)
    rows, skipped, notes_md = [], [], []
    totals = dict(tp={'BC': 0, 'TC': 0}, fp={'BC': 0, 'TC': 0}, fn={'BC': 0, 'TC': 0})
    confusion = np.zeros((2, 2), int)
    inter = {'BG': 0, 'CA': 0}
    union = {'BG': 0, 'CA': 0}

    for pair_id in ids:
        la, lb = src_a.load(pair_id), src_b.load(pair_id)
        if la is None or lb is None:
            skipped.append(pair_id)
            continue
        (pa, ma), (pb, mb) = la, lb
        cell = compare_cells(pa, pb, args.radius)
        tissue = compare_tissue(ma, mb, args.band)

        box = None
        if pair_id in metadata and cell_box is not None:
            box = cell_box(metadata[pair_id])
            inbox = compare_tissue(ma, mb, args.band, box_region(box))
        else:
            inbox = None

        for name in ('BC', 'TC'):
            for k in ('tp', 'fp', 'fn'):
                totals[k][name] += cell['per_class'][name][k]
        confusion += np.array(cell['class_confusion'])
        valid = (ma != 255) & (mb != 255)
        for name, v in (('BG', 1), ('CA', 2)):
            inter[name] += int(((ma == v) & (mb == v) & valid).sum())
            union[name] += int((((ma == v) | (mb == v)) & valid).sum())

        rows.append(dict(
            pair_id=pair_id,
            organ=manifest.get(pair_id, {}).get('organ', ''),
            cells_a=cell['count_a'], cells_b=cell['count_b'],
            tc_a=cell['count_a_by_class']['TC'], tc_b=cell['count_b_by_class']['TC'],
            location_matched=cell['location_matched'],
            only_a=cell['only_a'], only_b=cell['only_b'],
            mean_offset_px=round(cell['mean_offset_px'], 2) if cell['mean_offset_px'] is not None else '',
            class_agreement=round(cell['class_agreement'], 4) if cell['class_agreement'] is not None else '',
            class_kappa=round(cell['class_kappa'], 4) if cell['class_kappa'] is not None else '',
            f1_bc=round(cell['per_class']['BC']['f1'], 4),
            f1_tc=round(cell['per_class']['TC']['f1'], 4),
            cell_mf1=round(cell['mf1'], 4),
            tissue_iou_bg=r4(tissue['iou']['BG']), tissue_iou_ca=r4(tissue['iou']['CA']),
            tissue_miou=r4(tissue['miou']),
            ca_share_a=round(tissue['ca_share_a'], 4), ca_share_b=round(tissue['ca_share_b'], 4),
            unk_share_a=round(tissue['unk_share_a'], 4), unk_share_b=round(tissue['unk_share_b'], 4),
            unk_only_a_px=tissue['unk_only_a_px'], unk_only_b_px=tissue['unk_only_b_px'],
            disagree_near_boundary_px=tissue['disagree_near_boundary_px'],
            disagree_interior_px=tissue['disagree_interior_px'],
            a_ca_b_bg_px=tissue['a_ca_b_bg_px'], a_bg_b_ca_px=tissue['a_bg_b_ca_px'],
            cellbox_tissue_miou=r4(inbox['miou']) if inbox else '',
            notes_a=len(src_a.notes(pair_id)), notes_b=len(src_b.notes(pair_id)),
        ))

        if not args.no_images:
            cimg, timg = load_images(manifest, pair_id)
            draw_cells(cimg, pa, pb, cell, folder / 'overlays' / f'{pair_id}_cell.png')
            draw_tissue(timg, ma, mb, tissue['disagree_map'], box,
                        folder / 'overlays' / f'{pair_id}_tissue.png')

        for src in (src_a, src_b):
            for note in src.notes(pair_id):
                notes_md.append(
                    f"| {pair_id} | {src.spec} | {note['task']} | "
                    f"({note['x']}, {note['y']}) | {note['comment']} |")

    if not rows:
        raise SystemExit(f'[오류] 비교 가능한 쌍이 없습니다. 건너뜀: {skipped}')

    with (folder / 'per_pair.csv').open('w', encoding='utf-8-sig', newline='') as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pooled = {}
    for name in ('BC', 'TC'):
        tp, fp, fn = (totals[k][name] for k in ('tp', 'fp', 'fn'))
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        pooled[name] = dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r,
                            f1=2 * p * r / (p + r) if p + r else 0)
    n = confusion.sum()
    summary = dict(
        a=args.a, b=args.b, pairs=len(rows), skipped=skipped,
        distance_px=args.radius, boundary_band_px=args.band,
        cell_pooled=pooled,
        cell_pooled_mf1=float(np.mean([v['f1'] for v in pooled.values()])),
        cell_mean_pair_mf1=float(np.mean([r['cell_mf1'] for r in rows])),
        cell_class_confusion_rows_a_cols_b=dict(
            labels=['BC', 'TC'], matrix=confusion.tolist()),
        cell_class_agreement=float(np.trace(confusion) / n) if n else None,
        tissue_pooled_iou={k: inter[k] / union[k] if union[k] else None for k in inter},
        tissue_pooled_miou=float(np.mean([inter[k] / union[k] for k in inter if union[k]])),
        totals=dict(
            only_a=sum(r['only_a'] for r in rows), only_b=sum(r['only_b'] for r in rows),
            disagree_near_boundary_px=sum(r['disagree_near_boundary_px'] for r in rows),
            disagree_interior_px=sum(r['disagree_interior_px'] for r in rows),
            unk_only_a_px=sum(r['unk_only_a_px'] for r in rows),
            unk_only_b_px=sum(r['unk_only_b_px'] for r in rows)),
    )
    (folder / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    if notes_md:
        (folder / 'difficult_cases.md').write_text(
            '# 판단이 어려웠던 위치 (작성자 메모)\n\n'
            '| 샘플 | 작성자 | 작업 | 위치(px) | 이유 |\n|---|---|---|---|---|\n'
            + '\n'.join(notes_md) + '\n', encoding='utf-8')

    print(f'[비교] {args.a}  vs  {args.b}: {len(rows)}쌍 (건너뜀 {len(skipped)})')
    print(f"  세포 mF1(합산) {summary['cell_pooled_mf1']:.3f} | "
          f"BC {pooled['BC']['f1']:.3f} TC {pooled['TC']['f1']:.3f} | "
          f"위치 일치 세포의 클래스 일치율 {summary['cell_class_agreement'] or 0:.3f}")
    print(f"  A만 표시 {summary['totals']['only_a']}개, B만 표시 {summary['totals']['only_b']}개")
    print(f"  조직 mIoU(합산, UNK 제외) {summary['tissue_pooled_miou']:.3f} | "
          f"경계 근처 불일치 {summary['totals']['disagree_near_boundary_px']}px, "
          f"내부 불일치 {summary['totals']['disagree_interior_px']}px")
    print('[저장]', folder)


if __name__ == '__main__':
    main()
