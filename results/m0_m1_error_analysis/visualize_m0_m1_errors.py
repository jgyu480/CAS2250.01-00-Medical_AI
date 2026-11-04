#!/usr/bin/env python3
"""Visualize paired M0->M1 cell-detection error transitions on OCELOT val.

No CUDA or inference required. Uses per_gt_comparison.csv from
analyze_m0_m1_errors.py, the val per-image evaluation JSON files, and audit
manifest.csv for original cell/tissue images. Output: case PNGs and index.md.

Visual conventions:
  white crosshair: focal ground-truth cell center
  turquoise dot: predicted BC
  orange dot: predicted TC
  yellow ring + line: detection diagnostically matched to focal GT
  small white circles: other GT cells in the shown cell crop
The diagnostic statuses are separate from official class-wise F1.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont, ImageOps

TILE = 248
HEADER = 49
PAD = 12
FONT_FILE_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
]
COLORS = {'BC': '#14D4D2', 'TC': '#FFB35C'}
YELLOW = '#FFEB61'
WHITE = '#FFFFFF'

TRANSITIONS = {
    'BC_in_CA': [
        ('wrong_class', 'correct'),
        ('correct', 'wrong_class'),
        ('correct', 'missed'),
        ('missed', 'correct'),
    ],
    'TC_in_BG': [
        ('wrong_class', 'correct'),
        ('wrong_class', 'missed'),
        ('correct', 'missed'),
        ('missed', 'correct'),
    ],
}


def font(size):
    for f in FONT_FILE_CANDIDATES:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def rows_by_id(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return {str(r['pair_id']).zfill(3): r for r in csv.DictReader(f)}


def eval_by_id(path):
    with Path(path).open(encoding='utf-8') as f:
        values = json.load(f)
    if not isinstance(values, list):
        raise ValueError(f'{path} must contain a list')
    return {str(r['pair_id']).zfill(3): r for r in values}


def load_gt_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def crop_with_transform(img, x, y, radius):
    # PIL crop outside bounds is padded with black; target remains centered.
    left, top = float(x) - radius, float(y) - radius
    cell = img.convert('RGB').crop((round(left), round(top), round(left + 2 * radius), round(top + 2 * radius)))
    cell = cell.resize((TILE, TILE), Image.Resampling.LANCZOS)
    def xy(px, py):
        return (round((float(px) - round(left)) * TILE / (2 * radius)),
                round((float(py) - round(top)) * TILE / (2 * radius)))
    return cell, xy


def draw_cross(d, x, y, color=WHITE, radius=11, width=3):
    d.ellipse((x-radius, y-radius, x+radius, y+radius), outline='#0B1020', width=width + 3)
    d.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=width)
    d.line((x-5, y, x+5, y), fill=color, width=2)
    d.line((x, y-5, x, y+5), fill=color, width=2)


def draw_pred(d, x, y, cls, matched=False, gt=None):
    color = COLORS.get(cls, '#AFAFAF')
    r = 4
    d.ellipse((x-r, y-r, x+r, y+r), fill=color, outline='#111827', width=2)
    if matched:
        rr = 10
        d.ellipse((x-rr, y-rr, x+rr, y+rr), outline=YELLOW, width=3)
        if gt is not None:
            d.line((*gt, x, y), fill=YELLOW, width=2)


def text(d, position, value, size=14, fill=WHITE):
    d.text(position, str(value), font=font(size), fill=fill)


def render_cell(img, truth_rows, predictions, focal, model_name, radius):
    x, y = int(focal['gt_x']), int(focal['gt_y'])
    base, tr = crop_with_transform(img, x, y, radius)
    d = ImageDraw.Draw(base)
    # Other GTs: small white circles; target: prominent crosshair.
    for px, py, _cls in truth_rows:
        qx, qy = tr(px, py)
        if 0 <= qx < TILE and 0 <= qy < TILE and math.hypot(px-x, py-y) > 1.0:
            d.ellipse((qx-3, qy-3, qx+3, qy+3), outline='#FFFFFF', width=1)
    gx, gy = tr(x, y)
    draw_cross(d, gx, gy)
    matched_index = focal.get(f'{model_name}_pred_index', '')
    if matched_index not in ('', None, 'None'):
        matched_index = int(matched_index)
    else:
        matched_index = None
    matched_point = None
    for i, p in enumerate(predictions):
        px, py, klass = p[:3]
        qx, qy = tr(px, py)
        if not (0 <= qx < TILE and 0 <= qy < TILE):
            continue
        label = 'BC' if int(klass) == 1 else 'TC'
        draw_pred(d, qx, qy, label, matched=False)
        if matched_index == i:
            matched_point = (qx, qy, label)
    if matched_point is not None:
        draw_pred(d, matched_point[0], matched_point[1], matched_point[2], matched=True, gt=(gx, gy))
    return base


def render_tissue(img, focal, radius):
    tx, ty = int(focal['tissue_x']), int(focal['tissue_y'])
    crop, _ = crop_with_transform(img, tx, ty, radius)
    d = ImageDraw.Draw(crop)
    draw_cross(d, TILE // 2, TILE // 2, color='#FFFFFF')
    return crop


def make_card(focal, cell_img, tissue_img, gt_truths, pred0, pred1, radius_cell, radius_tissue):
    w = TILE * 3 + PAD * 4
    h = TILE + HEADER + 70
    out = Image.new('RGB', (w, h), '#121B2B')
    d = ImageDraw.Draw(out)
    lefts = [PAD + (TILE + PAD) * i for i in range(3)]
    first = render_tissue(tissue_img, focal, radius_tissue)
    second = render_cell(cell_img, gt_truths, pred0, focal, 'm0', radius_cell)
    third = render_cell(cell_img, gt_truths, pred1, focal, 'm1', radius_cell)
    for i, im in enumerate((first, second, third)):
        out.paste(im, (lefts[i], HEADER))
    desc = f"GT={focal['gt_class']} in {focal['context']} | Pair {focal['pair_id']} | GT#{focal['gt_index']}"
    text(d, (12, 7), desc, 16)
    a, b = focal['m0_status'], focal['m1_status']
    text(d, (lefts[0]+6, 30), 'Tissue context', 12, '#D1D5DB')
    text(d, (lefts[1]+6, 30), f'M0: {a}', 12, '#D1D5DB')
    text(d, (lefts[2]+6, 30), f'M1: {b}', 12, '#D1D5DB')
    y = HEADER + TILE + 10
    text(d, (PAD, y), 'Legend: white = focal GT / other GT outlines', 13)
    text(d, (PAD, y+20), 'Turquoise = predicted BC; orange = predicted TC; yellow = diagnostic match', 12, '#D1D5DB')
    return out


def rank_row(r):
    # Prioritize confident relevant detections and those near the target GT.
    total = 0.0
    for model in ('m0', 'm1'):
        value = r.get(model + '_score', '')
        if value not in ('', None, 'None'):
            try:
                total += float(value)
            except (ValueError, TypeError):
                pass
    return total


def select_examples(items, count):
    values = sorted(items, key=rank_row, reverse=True)
    distinct = []
    used_pairs = set()
    for row in values:
        if row['pair_id'] in used_pairs:
            continue
        distinct.append(row)
        used_pairs.add(row['pair_id'])
        if len(distinct) == count:
            break
    if len(distinct) < count:
        used = {(r['pair_id'], r['gt_index']) for r in distinct}
        distinct += [r for r in values if (r['pair_id'], r['gt_index']) not in used][:count-len(distinct)]
    return distinct


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--m0', type=Path, required=True)
    p.add_argument('--m1', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-transition', type=int, default=3)
    p.add_argument('--cell-radius', type=int, default=110)
    p.add_argument('--tissue-radius', type=int, default=110)
    args = p.parse_args()
    if args.per_transition < 1 or args.cell_radius < 8 or args.tissue_radius < 8:
        p.error('Invalid counts or radii')
    manifest = rows_by_id(args.manifest)
    m0, m1 = eval_by_id(args.m0), eval_by_id(args.m1)
    raw_rows = load_gt_rows(args.csv)
    selected_groups = defaultdict(list)
    for row in raw_rows:
        if str(row.get('stable_context', '')).lower() not in ('true', '1'):
            continue
        case = f"{row['gt_class']}_in_{row['context']}"
        if case not in TRANSITIONS:
            continue
        transition = (row['m0_status'], row['m1_status'])
        if transition in TRANSITIONS[case]:
            selected_groups[(case, transition)].append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    lines = ['# M0 vs M1 qualitative error examples', '',
             'Validation stable-context GT cells only; selected by diagnostic transitions.',
             'Yellow ring denotes secondary diagnostic opposite-class matching when applicable,',
             'not the official F1 pairing. Cases are examples, not representative frequency estimates.', '',
             'Legend: white crosshair = focal GT, small white circles = other GT, turquoise = predicted BC,',
             'orange = predicted TC, yellow ring/line = diagnostic match to focal GT.', '']
    created = 0
    for case, transition_list in TRANSITIONS.items():
        lines += [f'## {case}', '']
        for before, after in transition_list:
            key = (case, (before, after))
            group = selected_groups.get(key, [])
            lines.append(f'### {before} -> {after} (available: {len(group)})')
            lines.append('')
            for j, row in enumerate(select_examples(group, args.per_transition), 1):
                pid = str(row['pair_id']).zfill(3)
                if pid not in manifest or pid not in m0 or pid not in m1:
                    raise ValueError(f'Missing image/evaluation record for pair {pid}')
                rec = manifest[pid]
                cell_path = args.data_root / rec['cell_image']
                tissue_path = args.data_root / rec['tissue_image']
                if not cell_path.is_file() or not tissue_path.is_file():
                    raise FileNotFoundError(f'{pid}: {cell_path} / {tissue_path}')
                if m0[pid]['truths'] != m1[pid]['truths']:
                    raise ValueError(f'{pid}: M0 and M1 ground-truth differs')
                with Image.open(cell_path) as im:
                    cell = im.convert('RGB').copy()
                with Image.open(tissue_path) as im:
                    tissue = im.convert('RGB').copy()
                card = make_card(row, cell, tissue, m0[pid]['truths'],
                                 m0[pid]['predictions'], m1[pid]['predictions'],
                                 args.cell_radius, args.tissue_radius)
                filename = f'{case}_{before}-to-{after}_{j:02d}_pair-{pid}_gt-{row["gt_index"]}.png'
                card.save(args.output / filename, optimize=True)
                lines.append(f'- Pair {pid}, GT#{row["gt_index"]}: [{filename}]({filename})')
                created += 1
            lines.append('')
    (args.output / 'index.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'[완료] {created} image cards -> {args.output}')
    print(f'[목록] {args.output / "index.md"}')
    if not created:
        raise SystemExit('No matched records; inspect CSV column names and stable_context.')

if __name__ == '__main__':
    main()
