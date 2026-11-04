#!/usr/bin/env python3
"""OCELOT M0 vs M1 validation error analysis (no GPU/inference needed).

Inputs are the existing eval_val_per_image.json files, official OCELOT
metadata.json, and the validation tissue-label PNGs listed in the audit manifest.

Official F1 is reproduced by class-aware, confidence-ordered matching at 15 px.
Diagnostic *wrong class* matches are performed only among still-unmatched
GTs/predictions, so diagnostic categories are disjoint. They are not official
metrics. Tissue context comes from the *ground-truth* mask at each cell's
mapped center; it is never inferred from the model's tissue prediction.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / 'common').exists():
    sys.path.insert(0, str(ROOT))

CUTOFF = 15.0
CLS = {1: 'BC', 2: 'TC'}
RAW_TISSUE = {1: 'BG', 2: 'CA', 255: 'UNK'}
STATUS = ('correct', 'wrong_class', 'missed')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_records(path):
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f'{path}: expected a list of per-image records')
    output = {}
    for record in records:
        pair_id = str(record['pair_id']).zfill(3)
        if pair_id in output:
            raise ValueError(f'{path}: duplicate pair_id={pair_id}')
        output[pair_id] = record
    return output


def match_diagnostics(record, cutoff=CUTOFF):
    """Official class-wise greedy matches, then diagnostic cross-class matches.

    Returns one mutually exclusive status for every GT; each prediction is
    either used in a correct match, used in a cross-class match, or extra.
    """
    gt = np.asarray(record['truths'], dtype=np.float32).reshape(-1, 3)
    pred = np.asarray(record['predictions'], dtype=np.float32).reshape(-1, 4)
    if len(gt) and not np.isin(gt[:, 2], [1, 2]).all():
        raise ValueError('GT classes must be 1=BC or 2=TC')
    if len(pred) and not np.isin(pred[:, 2], [1, 2]).all():
        raise ValueError('Prediction classes must be 1=BC or 2=TC')

    matches = {}  # gt index -> (prediction index, status, dist)
    used_pred = set()
    tp_counts = Counter()

    # Identical class-aware matching policy to common.metrics.m0_eval.match_counts.
    for cls in (1, 2):
        gi = np.flatnonzero(gt[:, 2] == cls)
        pi = np.flatnonzero(pred[:, 2] == cls)
        if not len(gi) or not len(pi):
            continue
        distances = np.linalg.norm(pred[pi, None, :2] - gt[None, gi, :2], axis=2)
        available = distances <= cutoff
        for pos in np.argsort(-pred[pi, 3], kind='quicksort'):
            neighbors = np.flatnonzero(available[pos])
            if not len(neighbors):
                continue
            nearest = neighbors[np.argmin(distances[pos, neighbors])]
            g, p = int(gi[nearest]), int(pi[pos])
            matches[g] = (p, 'correct', float(distances[pos, nearest]))
            used_pred.add(p)
            tp_counts[cls] += 1
            available[:, nearest] = False

    # Secondary diagnostic matching: WRONG CLASS only and unused on both sides.
    # The original official F1 counts are never altered by this operation.
    gi = [i for i in range(len(gt)) if i not in matches]
    pi = [i for i in range(len(pred)) if i not in used_pred]
    candidates = []
    if gi and pi:
        d = np.linalg.norm(pred[np.asarray(pi), None, :2] - gt[None, np.asarray(gi), :2], axis=2)
        for ppos, p in enumerate(pi):
            for gpos, g in enumerate(gi):
                if pred[p, 2] != gt[g, 2] and d[ppos, gpos] <= cutoff:
                    candidates.append((float(d[ppos, gpos]), -float(pred[p, 3]), p, g))
    for distance, _neg_score, p, g in sorted(candidates):
        if g not in matches and p not in used_pred:
            matches[g] = (p, 'wrong_class', distance)
            used_pred.add(p)

    by_gt = []
    for g in range(len(gt)):
        if g in matches:
            p, status, distance = matches[g]
            by_gt.append({
                'status': status, 'pred_index': p,
                'pred_class': CLS[int(pred[p, 2])],
                'pred_x': int(pred[p, 0]), 'pred_y': int(pred[p, 1]),
                'score': float(pred[p, 3]), 'distance_px': distance,
            })
        else:
            by_gt.append({
                'status': 'missed', 'pred_index': None, 'pred_class': '',
                'pred_x': None, 'pred_y': None,
                'score': None, 'distance_px': None,
            })

    extras = []
    for p in range(len(pred)):
        if p in used_pred:
            continue
        if len(gt):
            distances = np.linalg.norm(gt[:, :2] - pred[p, :2], axis=1)
            nearest = float(np.min(distances))
        else:
            nearest = None
        extras.append({
            'pred_index': p, 'pred_x': int(pred[p, 0]),
            'pred_y': int(pred[p, 1]), 'pred_class': CLS[int(pred[p, 2])],
            'score': float(pred[p, 3]), 'nearest_gt_px': nearest,
            'type': ('near_gt_extra' if nearest is not None and nearest <= cutoff
                     else 'isolated_false_positive'),
        })

    # Recompute official F1 counts *before* secondary wrong-class matching.
    official = {}
    for cls, name in CLS.items():
        n_gt = int(np.sum(gt[:, 2] == cls))
        n_pred = int(np.sum(pred[:, 2] == cls))
        tp = int(tp_counts[cls])
        official[name] = {'tp': tp, 'fp': n_pred - tp, 'gt': n_gt}
    return gt, pred, by_gt, extras, official


def f1_from_counts(counts):
    result = {}
    for c in ('BC', 'TC'):
        tp, fp, gt = (counts[c][k] for k in ('tp', 'fp', 'gt'))
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (gt + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        result[c] = {'precision': precision, 'recall': recall, 'f1': f1,
                     'tp': tp, 'fp': fp, 'gt': gt}
    result['mF1'] = (result['BC']['f1'] + result['TC']['f1']) / 2
    return result


def cell_box_from_metadata(record):
    """Replicate common.data.geometry.cell_box for WSI registration."""
    bounds = []
    for name in ('cell', 'tissue'):
        item = record[name]
        bounds.append(np.array([item[k] for k in ('x_start', 'y_start', 'x_end', 'y_end')],
                               dtype=np.float64))
    cell, tissue = bounds
    span = tissue[2:] - tissue[:2]
    if (span <= 0).any():
        raise ValueError('Invalid tissue WSI coordinate extent')
    box = np.r_[(cell[:2] - tissue[:2]) / span,
                (cell[2:] - tissue[:2]) / span]
    if (box < -1e-6).any() or (box > 1 + 1e-6).any():
        raise ValueError(f'Cell outside tissue: {box}')
    return np.clip(box, 0, 1)


def sample_context(mask, box, x, y, purity_radius=2, purity_threshold=0.8):
    """Map cell pixel-center to tissue pixel-center and sample GT tissue label.

    Exact convention: common.data.geometry.points_in_tissue.
    """
    h, w = mask.shape
    if (h, w) != (1024, 1024):
        raise ValueError(f'Unexpected tissue mask size: {(h, w)}')
    x_t = ((x + .5) / 1024 * (box[2] - box[0]) + box[0]) * 1024 - .5
    y_t = ((y + .5) / 1024 * (box[3] - box[1]) + box[1]) * 1024 - .5
    xi = int(np.clip(np.rint(x_t), 0, w - 1))
    yi = int(np.clip(np.rint(y_t), 0, h - 1))
    raw = int(mask[yi, xi])
    if raw not in RAW_TISSUE:
        raise ValueError(f'Unexpected tissue label value {raw}')
    r = purity_radius
    patch = mask[max(0, yi-r):min(h, yi+r+1), max(0, xi-r):min(w, xi+r+1)]
    purity = float(np.mean(patch == raw))
    label = RAW_TISSUE[raw]
    return {'context': label, 'context_purity': purity,
            'stable_context': label in ('BG', 'CA') and purity >= purity_threshold,
            'tissue_x': xi, 'tissue_y': yi}


def status_rate(rows, model):
    counts = Counter(r[f'{model}_status'] for r in rows)
    n = len(rows)
    return {'n': n,
            'correct': counts['correct'],
            'wrong_class': counts['wrong_class'],
            'missed': counts['missed'],
            'correct_rate': round(counts['correct'] / n, 6) if n else None,
            'wrong_class_rate': round(counts['wrong_class'] / n, 6) if n else None,
            'miss_rate': round(counts['missed'] / n, 6) if n else None,
            'error_rate': round((counts['wrong_class'] + counts['missed']) / n, 6)
            if n else None}


def group_compare(rows, selector):
    groups = defaultdict(list)
    for row in rows:
        groups[selector(row)].append(row)
    answer = {}
    for name, items in sorted(groups.items()):
        a, b = status_rate(items, 'm0'), status_rate(items, 'm1')
        answer[str(name)] = {
            'm0': a, 'm1': b,
            'delta_wrong_class_rate': round(b['wrong_class_rate'] - a['wrong_class_rate'], 6),
            'delta_miss_rate': round(b['miss_rate'] - a['miss_rate'], 6),
            'delta_error_rate': round(b['error_rate'] - a['error_rate'], 6),
        }
    return answer


def write_csv(path, records, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def fmt_pct(value):
    return 'N/A' if value is None else f'{value * 100:.2f}%'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m0', type=Path, default=Path('results/m0_dicefix_seed42/eval_val_per_image.json'))
    parser.add_argument('--m1', type=Path, default=Path('results/m1_dicefix_seed42/eval_val_per_image.json'))
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Path containing OCELOT metadata.json and annotation files')
    parser.add_argument('--manifest', type=Path, required=True,
                        help='Previously audited manifest.csv')
    parser.add_argument('--output', type=Path, default=Path('results/m0_m1_error_analysis'))
    parser.add_argument('--purity-radius', type=int, default=2,
                        help='Tissue mask neighborhood radius (2 => 5x5)')
    parser.add_argument('--purity-threshold', type=float, default=0.8)
    args = parser.parse_args()
    if args.purity_radius < 0 or not (0 <= args.purity_threshold <= 1):
        parser.error('Invalid purity settings')

    m0, m1 = read_records(args.m0), read_records(args.m1)
    if set(m0) != set(m1):
        raise ValueError(f'Nonidentical pair IDs: only M0={set(m0)-set(m1)}, only M1={set(m1)-set(m0)}')
    if not m0:
        raise ValueError('No image records found')
    metadata_raw = load_json(args.data_root / 'metadata.json')
    metadata = metadata_raw.get('sample_pairs', metadata_raw)
    metadata = {str(k).zfill(3): v for k, v in metadata.items()}
    with args.manifest.open(newline='', encoding='utf-8-sig') as f:
        manifests = {str(row['pair_id']).zfill(3): row for row in csv.DictReader(f)
                     if row['split'] == 'val'}
    if set(m0) - set(manifests):
        raise ValueError(f'Missing validation manifest records: {set(m0)-set(manifests)}')

    args.output.mkdir(parents=True, exist_ok=True)
    per_gt, per_pred_extra, per_image = [], [], []
    totals = {'m0': {'BC': Counter(), 'TC': Counter()},
              'm1': {'BC': Counter(), 'TC': Counter()}}

    for pair_id in sorted(m0):
        record0, record1 = m0[pair_id], m1[pair_id]
        if record0['truths'] != record1['truths']:
            raise ValueError(f'{pair_id}: M0/M1 have different GT points or their order')
        if record0.get('organ') != record1.get('organ'):
            raise ValueError(f'{pair_id}: organ metadata differs')
        if pair_id not in metadata:
            raise ValueError(f'{pair_id}: missing metadata')
        box = cell_box_from_metadata(metadata[pair_id])
        mask_path = args.data_root / manifests[pair_id]['tissue_mask']
        with Image.open(mask_path) as image:
            mask = np.asarray(image).copy()
        if mask.ndim != 2:
            raise ValueError(f'{pair_id}: tissue mask must be grayscale')
        cache = {}
        for model, record in (('m0', record0), ('m1', record1)):
            gt, pred, matched, extras, official = match_diagnostics(record)
            cache[model] = (gt, matched)
            for cell_class in ('BC', 'TC'):
                totals[model][cell_class].update(official[cell_class])
            for e in extras:
                ctx = sample_context(mask, box, e['pred_x'], e['pred_y'],
                                     args.purity_radius, args.purity_threshold)
                per_pred_extra.append({'pair_id': pair_id, 'organ': record0.get('organ'),
                                       'model': model, **e, **ctx})
        gt = cache['m0'][0]
        for idx, (x, y, c) in enumerate(gt):
            ctx = sample_context(mask, box, x, y,
                                 args.purity_radius, args.purity_threshold)
            a, b = cache['m0'][1][idx], cache['m1'][1][idx]
            per_gt.append({
                'pair_id': pair_id, 'organ': record0.get('organ'), 'gt_index': idx,
                'gt_x': int(x), 'gt_y': int(y), 'gt_class': CLS[int(c)], **ctx,
                **{f'm0_{k}': v for k, v in a.items()},
                **{f'm1_{k}': v for k, v in b.items()},
                'transition': a['status'] + ' -> ' + b['status'],
            })
        per_image.append({
            'pair_id': pair_id, 'organ': record0.get('organ'),
            'num_gt': len(record0['truths']),
            'm0_mF1': record0.get('mF1'), 'm1_mF1': record1.get('mF1'),
            'delta_mF1': round(float(record1['mF1']) - float(record0['mF1']), 6),
            'm0_BC_F1': record0.get('f1_BC'), 'm1_BC_F1': record1.get('f1_BC'),
            'm0_TC_F1': record0.get('f1_TC'), 'm1_TC_F1': record1.get('f1_TC'),
        })

    official_scores = {k: f1_from_counts(v) for k, v in totals.items()}
    for model, path in (('m0', args.m0), ('m1', args.m1)):
        summary_path = path.with_name('eval_val.json')
        if summary_path.exists():
            reported = load_json(summary_path)['overall']['mF1']
            recomputed = official_scores[model]['mF1']
            if abs(recomputed - float(reported)) > 0.0005:
                raise ValueError(f'{model}: official metric mismatch. '
                                 f'recomputed={recomputed:.6f}, eval_val.json={reported}. '
                                 'Check prediction score rounding/tie behavior.')
            print(f'[검증] {model.upper()} mF1 {recomputed:.4f} ≈ eval_val.json {reported:.4f}')

    selected = [('BC', 'CA'), ('TC', 'BG')]
    targeted = {}
    for cls, context in selected:
        subset = [r for r in per_gt if r['gt_class'] == cls and r['context'] == context]
        stable = [r for r in subset if r['stable_context']]
        targeted[f'{cls}_in_{context}'] = {
            'all': {'m0': status_rate(subset, 'm0'), 'm1': status_rate(subset, 'm1')},
            'stable': {'m0': status_rate(stable, 'm0'), 'm1': status_rate(stable, 'm1')},
        }
    summary = {
        'design': {
            'split': 'val', 'pair_count': len(m0), 'gt_count': len(per_gt),
            'official_cutoff_px': CUTOFF,
            'note': 'Official class-aware F1; diagnostics use secondary wrong-class matching only after correct-class matching.',
            'tissue_context': 'Ground-truth tissue mask at GT cell center mapped via WSI cell_box; UNK retained, never guessed.',
            'stable_context_definition': f'center label is BG/CA and its {(2*args.purity_radius+1)}x{(2*args.purity_radius+1)} neighborhood purity >= {args.purity_threshold}',
            'eval_data': {'m0': str(args.m0), 'm1': str(args.m1)},
        },
        'official': official_scores,
        'overall_gt_status': {'m0': status_rate(per_gt, 'm0'),
                              'm1': status_rate(per_gt, 'm1')},
        'by_cell_class_and_tissue': group_compare(per_gt, lambda r: r['gt_class'] + '_in_' + r['context']),
        'by_cell_class_and_stable_tissue': group_compare(
            [r for r in per_gt if r['stable_context']], lambda r: r['gt_class'] + '_in_' + r['context']),
        'targeted': targeted,
        'by_organ': group_compare(per_gt, lambda r: r['organ']),
        'gt_status_transitions': dict(sorted(Counter(r['transition'] for r in per_gt).items())),
        'extra_predictions': {model: dict(Counter(
            r['type'] for r in per_pred_extra if r['model'] == model))
            for model in ('m0', 'm1')},
    }
    (args.output / 'analysis_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    gt_fields = ['pair_id', 'organ', 'gt_index', 'gt_x', 'gt_y', 'gt_class',
                 'context', 'context_purity', 'stable_context', 'tissue_x', 'tissue_y']
    for model in ('m0', 'm1'):
        gt_fields += [f'{model}_{key}' for key in (
            'status', 'pred_index', 'pred_class', 'pred_x', 'pred_y', 'score', 'distance_px')]
    gt_fields += ['transition']
    write_csv(args.output / 'per_gt_comparison.csv', per_gt, gt_fields)
    write_csv(args.output / 'extra_predictions.csv', per_pred_extra,
              ['pair_id', 'organ', 'model', 'pred_index', 'pred_x', 'pred_y',
               'pred_class', 'score', 'nearest_gt_px', 'type',
               'context', 'context_purity', 'stable_context', 'tissue_x', 'tissue_y'])
    write_csv(args.output / 'per_image_comparison.csv', per_image,
              ['pair_id', 'organ', 'num_gt', 'm0_mF1', 'm1_mF1', 'delta_mF1',
               'm0_BC_F1', 'm1_BC_F1', 'm0_TC_F1', 'm1_TC_F1'])

    lines = [
        '# M0 vs M1 — OCELOT validation error analysis', '',
        '## Evaluation design',
        f'- {len(m0)} validation pairs, {len(per_gt)} GT cells; no test set used.',
        '- Official mF1: class-wise score-ordered 1:1 matching within 15px.',
        '- Diagnostic categories: correct / wrong_class / missed (mutually exclusive).',
        '- Tissue category is sampled from **GT tissue annotations**, not model predictions.',
        '- Stable-context subset requires a non-UNK label and neighborhood purity threshold.',
        '- These diagnostic confusion/miss rates are **not the official F1**.', '',
        '## Official evaluation (reproduced)', '',
        '| Metric | M0 | M1 | Difference |', '|---|---:|---:|---:|',
    ]
    for key in ('mF1', 'BC', 'TC'):
        a = official_scores['m0'][key] if key == 'mF1' else official_scores['m0'][key]['f1']
        b = official_scores['m1'][key] if key == 'mF1' else official_scores['m1'][key]['f1']
        lines.append(f'| {key} | {a:.4f} | {b:.4f} | {b-a:+.4f} |')
    lines += ['', '## Key context-sensitive cases (ground-truth cell denominator)', '',
              '| Case | subset | GT count | M0 wrong class | M1 wrong class | M0 miss | M1 miss |',
              '|---|---|---:|---:|---:|---:|---:|']
    for case, item in targeted.items():
        for subset_name in ('all', 'stable'):
            a, b = item[subset_name]['m0'], item[subset_name]['m1']
            lines.append(f"| {case} | {subset_name} | {a['n']} | "
                         f"{fmt_pct(a['wrong_class_rate'])} | {fmt_pct(b['wrong_class_rate'])} | "
                         f"{fmt_pct(a['miss_rate'])} | {fmt_pct(b['miss_rate'])} |")
    lines += ['', '## Interpretation and caveats', '',
              '- Wrong-class detection is a *secondary*, one-to-one, opposite-label matching after official correct-label matching; it cannot be directly equated with the official false-positive count.',
              '- Missed means no available opposite-label detection within 15 px after correct matches; ambiguity and dense-cell effects may remain.',
              '- A local tissue center label is a region-level annotation, not proof of histological cell identity; inspect near-boundary cases and UNK.',
              '- The two models have independently validation-tuned postprocessing. To isolate architectural effects, consider an additional common-postprocessing comparison.',
              '- M1 also removes the M0 cell-to-tissue pathway; observed differences cannot be attributed exclusively to cross-attention.',
              '- Validation is also used to choose the checkpoint and postprocessing, so these results are exploratory, not a final test-set estimate.', '',
              '## Other outputs', '',
              '- `analysis_summary.json`: machine-readable aggregate results',
              '- `per_gt_comparison.csv`: paired GT-level errors, tissue context and M0→M1 transitions',
              '- `extra_predictions.csv`: unmatched predictions (near-GT extra vs isolated FP)',
              '- `per_image_comparison.csv`: per-image mF1/F1 changes', '']
    (args.output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')

    print(f'[완료] {len(m0)} pairs / {len(per_gt)} GT cells | output={args.output}')
    print(f"[공식 mF1] M0 {official_scores['m0']['mF1']:.4f} / M1 {official_scores['m1']['mF1']:.4f}")
    for case in targeted:
        print(f'[{case}]')
        for subset_name in ('all', 'stable'):
            a = targeted[case][subset_name]['m0']
            b = targeted[case][subset_name]['m1']
            print(f"  {subset_name}: n={a['n']} | wrong_class M0={fmt_pct(a['wrong_class_rate'])} M1={fmt_pct(b['wrong_class_rate'])} | missed M0={fmt_pct(a['miss_rate'])} M1={fmt_pct(b['miss_rate'])}")
    print('[출력] report.md, analysis_summary.json, per_gt_comparison.csv, extra_predictions.csv, per_image_comparison.csv')


if __name__ == '__main__':
    main()
