"""Validation-only follow-up analysis. Put beside scripts/diagnose_m0.py.

One forward pass per validation image; checkpoint and official results unchanged.
Postprocessing variants are diagnostic comparisons, not a new selected result.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import diagnose_m0 as d


def read_csv(path):
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def complete_cohort(run):
    paths = sorted((run / "analysis").glob("diagnostics_cohorts_*"), reverse=True)
    for p in paths:
        if all((p / n).is_file() for n in ("cells.csv", "images.csv")):
            return p
    raise ValueError("먼저 diagnose_m0.py cohorts를 실행하세요.")


def group_cells(cells, keys):
    groups = defaultdict(list)
    for c in cells:
        groups[tuple(c[k] for k in keys)].append(c)
    result = []
    for key, rows in sorted(groups.items()):
        missed = sum(c["error"] == "missed" for c in rows)
        wrong = sum(c["error"] == "misclassified" for c in rows)
        localized = len(rows) - missed
        result.append({**dict(zip(keys, key)), "cells": len(rows),
                       "images": len({c["pair_id"] for c in rows}),
                       "clusters": len({c["cluster"] for c in rows}),
                       "miss_rate": missed / len(rows),
                       "wrong_all_gt_rate": wrong / len(rows),
                       "wrong_localized_rate": wrong / localized if localized else ""})
    return result


def contrast(cells, key_a, key_b, field, event, repeats=1000):
    """Paired patient bootstrap, four totals per patient, pooled cell rates.

    A bootstrap draw missing either group is undefined and omitted, not set to 0.
    These are observational contrasts, not adjusted causal effects.
    """
    blocks = defaultdict(lambda: np.zeros(4, dtype=float))
    image_sets = [set(), set()]
    for c in cells:
        if c[field] not in (key_a, key_b):
            continue
        side = 0 if c[field] == key_a else 1
        blocks[c["cluster"]][2 * side] += c["error"] == event
        blocks[c["cluster"]][2 * side + 1] += 1
        image_sets[side].add(c["pair_id"])
    if not blocks:
        return None
    a = np.stack(list(blocks.values()))
    total = a.sum(0)
    if not total[1] or not total[3]:
        return None
    rng = np.random.default_rng(42)
    draws = a[rng.integers(len(a), size=(repeats, len(a)))].sum(1)
    draws = draws[(draws[:, 1] > 0) & (draws[:, 3] > 0)]
    delta = draws[:, 0] / draws[:, 1] - draws[:, 2] / draws[:, 3]
    lo, hi = np.quantile(delta, [.025, .975]) if len(delta) else (None, None)
    return {"A": key_a, "B": key_b, "event": event,
            "cells_A": int(total[1]), "cells_B": int(total[3]),
            "images_A": len(image_sets[0]), "images_B": len(image_sets[1]),
            "clusters": len(a), "rate_A": total[0] / total[1],
            "rate_B": total[2] / total[3],
            "delta_A_minus_B": total[0] / total[1] - total[2] / total[3],
            "CI95_low": lo, "CI95_high": hi, "valid_bootstrap_draws": len(delta)}


def cached_analysis(args):
    source = Path(args.cohorts).resolve() if args.cohorts else complete_cohort(args.run)
    images, cells = read_csv(source / "images.csv"), read_csv(source / "cells.csv")
    image_map = {r["pair_id"]: r for r in images}
    for c in cells:
        c["cluster"] = image_map[c["pair_id"]]["cluster"]
        c["gt_index"] = int(c["gt_index"])
    if len(images) != 137 or len(image_map) != 137:
        raise ValueError("공식 validation 137개 이미지가 필요합니다.")
    d.write_csv(args.out / "joint_cells.csv", group_cells(
        cells, ["organ", "class", "region", "neighbor_bin"]))
    contrasts = []
    for organ in ["ALL", *sorted({c["organ"] for c in cells})]:
        for cls in ("BC", "TC"):
            rows = [c for c in cells if c["class"] == cls
                    and (organ == "ALL" or c["organ"] == organ)]
            for field, a, b, event in (
                ("neighbor_bin", "14-30", "30+", "missed"),
                ("region", "CA" if cls == "BC" else "BG",
                 "BG" if cls == "BC" else "CA", "misclassified")):
                r = contrast(rows, a, b, field, event)
                if r:
                    contrasts.append({"organ": organ, "class": cls, "comparison": field, **r})
    d.write_csv(args.out / "organ_contrasts.csv", contrasts)
    for r in contrasts:
        if r['organ'] == 'ALL':
            print(f"  {r['class']} {r['event']} ({r['A']} - {r['B']}): "
                  f"{r['delta_A_minus_B']:+.1%}, "
                  f"95% CI [{r['CI95_low']:+.1%}, {r['CI95_high']:+.1%}]", flush=True)
    chosen, used_images = [], set()
    for reason, cls, region, event in (
        ("close_missed_BC", "BC", None, "missed"),
        ("close_missed_TC", "TC", None, "missed"),
        ("BC_in_CA_wrong", "BC", "CA", "misclassified"),
        ("TC_in_BG_wrong", "TC", "BG", "misclassified")):
        rows = [c for c in cells if c["class"] == cls and c["error"] == event
                and (c["region"] == region if region else c["neighbor_bin"] == "14-30")]
        # Separate images give more useful examples than many cells from one patch.
        for c in rows:
            if c["pair_id"] in used_images:
                continue
            chosen.append({**c, "reason": reason})
            used_images.add(c["pair_id"])
            if sum(v["reason"] == reason for v in chosen) == 2:
                break
    d.write_csv(args.out / "selected_cells.csv", chosen)
    d.save_json(args.out / "notes.json", {
        "cohort_source": str(source), "regions": "ground-truth tissue regions",
        "rates": "All-GT denominators; location-first diagnostic matching, not official F1.",
        "CI": "Patient/slide cluster bootstrap, fixed model; exploratory, no multiple-test correction.",
        "confounding": "Within-organ comparison still does not control morphology, density and tissue simultaneously.",
        "sparse": "Check cell/image/cluster support, especially absent classes and tiny groups.",
        "probability": "7px maximum may come from a neighboring cell. Inspect crops; not proof of postprocessing failure.",
        "tissue": "Imagewise macro IoU assigns 0 to zero-union classes. Present-class IoU is a separate diagnostic, not a replacement official metric."})
    print(f"[완료] 장기별 비교 {len(contrasts)}개 / 시각화 대상 {len(chosen)}개", flush=True)
    return cells, image_map, chosen


def variants(pp):
    candidates = [("baseline", dict(pp)),
                  ("sigma_less", {**pp, "sigma": max(.5, pp["sigma"] / 2)}),
                  ("sigma_more", {**pp, "sigma": pp["sigma"] + 1}),
                  ("distance_less", {**pp, "min_distance": max(1, pp["min_distance"] - 2)}),
                  ("distance_more", {**pp, "min_distance": pp["min_distance"] + 2}),
                  ("threshold_more", {**pp, "threshold": min(.95, pp["threshold"] + .15)})]
    unique, seen = [], set()
    for name, settings in candidates:
        key = tuple(settings[k] for k in ("sigma", "min_distance", "threshold"))
        if key not in seen:
            seen.add(key)
            unique.append((name, settings))
    return unique


def save_crop(args, raw, gt, predictions, alt, prob, c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y, cls = gt[c["gt_index"]]
    h, w = prob.shape[1:]
    x0, x1, y0, y1 = max(0, x - 96), min(w, x + 97), max(0, y - 96), min(h, y + 97)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(raw["cell_image"][y0:y1, x0:x1])
    for px, py, pc in gt:
        if x0 <= px < x1 and y0 <= py < y1:
            ax[0].scatter(px-x0, py-y0, facecolors="none", edgecolors="yellow" if pc == 1 else "cyan", s=55)
    for points, color, marker in ((predictions, "red", "+"), (alt, "lime", "x")):
        for px, py, *_ in points:
            if x0 <= px < x1 and y0 <= py < y1:
                ax[0].scatter(px-x0, py-y0, c=color, marker=marker, s=22)
    ax[0].set_title("GT circles; baseline +; less sigma x")
    for a, data, title in ((ax[1], 1-prob[0], "P(any cell)"),
                            (ax[2], prob[cls], "P(true class: " + c["class"] + ")")):
        im = a.imshow(data[y0:y1, x0:x1], vmin=0, vmax=1, cmap="magma")
        a.scatter(x-x0, y-y0, facecolors="none", edgecolors="lime", s=100)
        a.set_title(title)
        fig.colorbar(im, ax=a, fraction=.04)
    for a in ax:
        a.set_axis_off()
    fig.suptitle(f"{c['reason']} | image {c['pair_id']} | GT index {c['gt_index']} | {c['organ']}")
    fig.tight_layout()
    folder = args.out / "crops"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / f"{c['reason']}_{c['pair_id']}_{c['gt_index']}.png", dpi=150)
    plt.close(fig)


def paired_delta(counts_a, counts_b, cluster_ids, repeats=1000):
    blocks_a, blocks_b = {}, {}
    for a, b, cluster in zip(counts_a, counts_b, cluster_ids):
        blocks_a.setdefault(cluster, np.zeros((2, 3), dtype=int))
        blocks_b.setdefault(cluster, np.zeros((2, 3), dtype=int))
        blocks_a[cluster] += a
        blocks_b[cluster] += b
    a, b = np.stack(list(blocks_a.values())), np.stack(list(blocks_b.values()))
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(repeats):
        ids = rng.integers(len(a), size=len(a))
        differences.append(d.score_counts(a[ids].sum(0))["mF1"] - d.score_counts(b[ids].sum(0))["mF1"])
    return np.quantile(differences, [.025, .975]).tolist()


def infer(args, cells, image_map, chosen):
    import torch
    from common.metrics.m0_eval import detect_cells, TissueIoU
    from common.training.m0_loop import make_loader, prepare, truth_points
    from common.data.geometry import points_in_tissue
    model, state, device = d.load_model(args)
    if state["settings"]["size"] != 1024:
        raise ValueError("1024px 모델이 필요합니다.")
    post = d.read_json(args.run / "postprocess.json")
    if post.get("selected_on") != "val":
        raise ValueError("validation에서 선택한 postprocess.json이 필요합니다.")
    pp = {k: post["chosen"][k] for k in ("sigma", "min_distance", "threshold")}
    configs = variants(pp)
    loader, ds = make_loader("val", state["settings"]["seed"], 1, 0, augment=False)
    if set(image_map) != {r["pair_id"] for r in ds.source.rows}:
        raise ValueError("cohorts와 현재 validation ID가 다릅니다.")
    by_cell = {(c["pair_id"], c["gt_index"]): c for c in cells}
    raw_index = {r["pair_id"]: i for i, r in enumerate(ds.source.rows)}
    all_counts = {name: [] for name, _ in configs}
    all_rates, probes, tissue_rows, baseline_records, cluster_ids = [], [], [], [], []
    total_tissue = TissueIoU()
    with torch.no_grad():
        for n, batch in enumerate(loader, 1):
            imgs, targets, box = prepare(batch, device, 1024)
            out = model(*imgs, box)
            pid = batch["pair_id"][0]
            gt = truth_points(batch["cell_points"][0])
            context = [by_cell[(pid, j)] for j in range(len(gt))]
            if any((int(float(c['x'])), int(float(c['y'])), c['class']) != (x, y, 'BC' if k == 1 else 'TC')
                   for c, (x, y, k) in zip(context, gt)):
                raise ValueError(f"{pid}: cohorts와 정답이 다릅니다.")
            cluster_ids.append(image_map[pid]["cluster"])
            predictions, labels = {}, {}
            for name, settings in configs:
                preds = detect_cells(out["cell_logits"], **settings)[0]
                lab, extras = d.localization_labels(preds, gt)
                counts = d.official_counts(preds, gt)
                all_counts[name].append(counts)
                predictions[name], labels[name] = preds, lab
                close = np.asarray([c["neighbor_bin"] == "14-30" for c in context], dtype=bool)
                atypical = np.asarray([(c["class"], c["region"]) in (("BC", "CA"), ("TC", "BG")) for c in context], dtype=bool)
                all_rates.append({"variant": name, "pair_id": pid, "organ": image_map[pid]["organ"],
                                  "num_gt": len(gt), "num_pred": len(preds),
                                  "missed": int(sum(lab == "missed")), "wrong": int(sum(lab == "misclassified")),
                                  **extras, "close_gt": int(close.sum()), "close_missed": int(sum((lab == "missed") & close)),
                                  "atypical_gt": int(atypical.sum()), "atypical_wrong": int(sum((lab == "misclassified") & atypical)),
                                  **d.score_counts(counts)})
            prob = out["cell_logits"][0].float().softmax(0).cpu().numpy()
            tp = out["tissue_logits"][0, :2].float().softmax(0).cpu().numpy()
            tm = targets[1][0].cpu().numpy()
            pred_mask = tp.argmax(0)
            valid = tm != 2
            intersections, unions, support = [], [], []
            for cls in (0, 1):
                p, t = (pred_mask == cls) & valid, (tm == cls) & valid
                intersections.append(int((p & t).sum()))
                unions.append(int((p | t).sum()))
                support.append(int(t.sum()))
            ious = np.asarray(intersections) / np.maximum(unions, 1)
            present = np.asarray(support) > 0
            image_iou = TissueIoU()
            image_iou.update(out["tissue_logits"], targets[1])
            total_tissue.update(out["tissue_logits"], targets[1])
            tissue_rows.append({"pair_id": pid, "organ": image_map[pid]["organ"],
                                "recorded_mIoU_quartile": image_map[pid].get('tissue_mIoU_recorded_quartile', ''),
                                "gt_BG_pixels": support[0], "gt_CA_pixels": support[1],
                                "both_classes_present": int(present.all()), "zero_union_classes": int(sum(np.asarray(unions) == 0)),
                                "macro_mIoU": float(ious.mean()),
                                "present_class_IoU": float(ious[present].mean()) if present.any() else "",
                                "cell_mF1": d.score_counts(all_counts['baseline'][-1])["mF1"]})
            baseline_records.append({"pair_id": pid, "organ": image_map[pid]["organ"],
                                     "num_gt": len(gt), "num_pred": len(predictions['baseline']),
                                     **d.score_counts(all_counts['baseline'][-1]), **image_iou.compute(),
                                     "predictions": predictions['baseline'], "truths": gt})
            coords = points_in_tissue(np.asarray(gt, dtype=float).reshape(-1, 3), box[0].cpu().numpy())
            for j, (x, y, cls) in enumerate(gt):
                x0, x1, y0, y1 = max(0, x-7), min(1024, x+8), max(0, y-7), min(1024, y+8)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                disk = (xx-x)**2 + (yy-y)**2 <= 49
                tx, ty = np.clip(np.rint(coords[j]).astype(int), 0, 1023)
                c = context[j]
                probes.append({**c, "error_cached": c["error"], "error_baseline": str(labels['baseline'][j]),
                               **{f"error_{name}": str(labels[name][j]) for name, _ in configs if name != 'baseline'},
                               "p_empty_at_gt": float(prob[0, y, x]), "p_BC_at_gt": float(prob[1, y, x]),
                               "p_TC_at_gt": float(prob[2, y, x]),
                               "max_foreground_7px": float((1-prob[0, y0:y1, x0:x1])[disk].max()),
                               "max_true_class_7px": float(prob[cls, y0:y1, x0:x1][disk].max()),
                               "tissue_pred_region": "BG" if pred_mask[ty, tx] == 0 else "CA",
                               "tissue_p_CA": float(tp[1, ty, tx]),
                               "tissue_region_correct": int((pred_mask[ty, tx] == (0 if c['region']=='BG' else 1))) if c['region'] != 'UNK' else ""})
            selected = [c for c in chosen if c["pair_id"] == pid]
            if selected:
                raw = ds.source[raw_index[pid]]
                for c in selected:
                    save_crop(args, raw, gt, predictions['baseline'], predictions.get('sigma_less', predictions['baseline']), prob, c)
            if n % 10 == 0 or n == len(loader):
                print(f"[추론] val {n}/{len(loader)} | {device}", flush=True)
            # Persist small tables periodically so an interrupted run can be inspected.
            if n % 20 == 0:
                d.write_csv(args.out / "cell_probes.csv", probes)
                d.write_csv(args.out / "postprocess_per_image.csv", all_rates)
            del out, prob, tp, imgs, targets
    baseline = d.score_counts(np.sum(all_counts['baseline'], axis=0))
    summaries = []
    for name, settings in configs:
        rows = [r for r in all_rates if r['variant'] == name]
        score = d.score_counts(np.sum(all_counts[name], axis=0))
        lo, hi = paired_delta(all_counts[name], all_counts['baseline'], cluster_ids)
        summaries.append({"variant": name, **settings, **score, "delta_mF1": score['mF1']-baseline['mF1'],
                          "delta_CI95_low": lo, "delta_CI95_high": hi,
                          **{k: sum(r[k] for r in rows) for k in ('num_pred', 'missed', 'wrong', 'duplicate', 'false_positive')},
                          "close_miss_rate": sum(r['close_missed'] for r in rows)/max(1, sum(r['close_gt'] for r in rows)),
                          "atypical_wrong_rate": sum(r['atypical_wrong'] for r in rows)/max(1, sum(r['atypical_gt'] for r in rows))})
    d.write_csv(args.out / "postprocess_summary.csv", summaries)
    d.write_csv(args.out / "postprocess_per_image.csv", all_rates)
    d.write_csv(args.out / "cell_probes.csv", probes)
    d.write_csv(args.out / "tissue_support.csv", tissue_rows)
    probe_groups = defaultdict(list)
    for p in probes:
        for keys in (("class", "neighbor_bin", "error_baseline"),
                     ("class", "region", "tissue_region_correct", "error_baseline")):
            probe_groups[(keys, tuple(p[k] for k in keys))].append(p)
    probe_summary = []
    for (keys, values), rows in probe_groups.items():
        probe_summary.append({"grouping": '+'.join(keys), **dict(zip(keys, values)),
                              "cells": len(rows), "images": len({p['pair_id'] for p in rows}),
                              "median_max_foreground_7px": float(np.median([p['max_foreground_7px'] for p in rows])),
                              "median_max_true_class_7px": float(np.median([p['max_true_class_7px'] for p in rows])),
                              "fraction_foreground_above_half": sum(p['max_foreground_7px'] > .5 for p in rows)/len(rows)})
    d.write_csv(args.out / "probe_summary.csv", probe_summary)
    tissue_summary = []
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        for scope in ('all', 'both_GT_classes_present'):
            rows = [r for r in tissue_rows if r['recorded_mIoU_quartile'] == q
                    and (scope == 'all' or r['both_classes_present'])]
            if not rows:
                continue
            counts = sum((all_counts['baseline'][i] for i, r in enumerate(tissue_rows) if r in rows),
                         np.zeros((2, 3), dtype=int))
            tissue_summary.append({'recorded_mIoU_quartile': q, 'scope': scope,
                                   'images': len(rows), 'both_GT_classes_present': sum(r['both_classes_present'] for r in rows),
                                   'images_with_zero_union_class': sum(r['zero_union_classes'] > 0 for r in rows),
                                   **d.score_counts(counts)})
    d.write_csv(args.out / "tissue_support_summary.csv", tissue_summary)
    print(f"[조직 지표] 정답 조직이 한 종류뿐인 이미지: {sum(not r['both_classes_present'] for r in tissue_rows)}개; "
          f"union=0인 클래스가 있는 이미지: {sum(r['zero_union_classes'] > 0 for r in tissue_rows)}개", flush=True)
    d.save_json(args.out / "baseline_per_image.json", baseline_records)
    old = d.read_json(args.run / "eval_val.json")
    old = old.get("overall", old.get("metrics", old))
    current = {**baseline, **total_tissue.compute()}
    d.save_json(args.out / "replay_comparison.json", {"images": len(baseline_records), "device": str(device),
                "batch": 1, "precision": "fp32", "baseline_postprocess": pp, "current": current,
                "difference_from_received": {k: v-old[k] for k, v in current.items() if isinstance(old.get(k), (int, float))}})
    print("[후처리 비교]", flush=True)
    for r in summaries:
        print(f"  {r['variant']}: mF1={r['mF1']:.4f}, 차이={r['delta_mF1']:+.4f}, 가까운 세포 누락={r['close_miss_rate']:.1%}, 오검출={r['false_positive']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/m0/M0_official_default_seed42")
    parser.add_argument("--project", default=".")
    parser.add_argument("--cohorts", help="특정 diagnostics_cohorts 폴더; 기본값은 최신 완성 폴더")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cached-only", action="store_true", help="추론 없이 장기별 비교만")
    args = parser.parse_args()
    args.root = Path(args.project).expanduser().resolve()
    sys.path.insert(0, str(args.root))
    args.run = Path(args.run).expanduser()
    if not args.run.is_absolute():
        args.run = args.root / args.run
    if not (args.root / "models/M0/model.py").is_file():
        parser.error("프로젝트 루트에서 실행하세요.")
    args.out = args.run / "analysis" / ("next_analysis_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    args.out.mkdir(parents=True)
    d.save_json(args.out / "status.json", {"status": "running", "cached_only": args.cached_only})
    print(f"[1/2] 장기·세포 종류·조직 영역·이웃 거리 비교 → {args.out}", flush=True)
    cells, images, selected = cached_analysis(args)
    if not args.cached_only:
        print("[2/2] validation 재추론·후처리 비교·확률 지도·조직 지표 확인", flush=True)
        infer(args, cells, images, selected)
    d.save_json(args.out / "status.json", {"status": "complete", "cached_only": args.cached_only})
    print(f"[결과] {args.out}", flush=True)


if __name__ == "__main__":
    main()
