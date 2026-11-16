"""저장된 per-image 결과를 두 개씩 짝지어 비교한다 (모델을 다시 돌리지 않는다).

seed만, σ만, 정답 종류(G/GA)만 바꾼 쌍을 같은 방법으로 비교해 "그 차이가 학습을 다시 했을 때의
흔들림보다 큰가"와 "두 모델이 같은 세포를 놓치는가"를 본다.

    python scripts/compare_m2_pairs.py \
        --metadata data/raw/ocelot2023_v1.0.1/metadata.json \
        --file G6s42=results/m2_sigma_search/gaussian/candidates/search_gaussian_s6_l0.9/eval_holdout_per_image.json \
        --file G6s43=results/m2_sigma_search/confirm/candidates/search_gaussian_s6_l0.9_seed43/eval_holdout_per_image.json \
        --pair G6s42:G6s43 --out results/m2_sigma_search/pairs_holdout.json

쌍마다 적는 값 (B − A)
- mF1: 공식 F1(15px, 클래스별 매칭)을 per-image 파일로 다시 계산한 값과 차이.
  95% 구간은 환자(TCGA)/slide 단위로 두 모델에 같은 표본을 뽑은 paired bootstrap이다.
- missed: 위치 우선 15px 매칭(보고서 6.2 진단, 공식 F1 아님)으로 놓친 정답 세포.
  both / only_A / only_B 비율과 overlap = 둘 다 놓침 ÷ 둘 중 하나라도 놓침.
  overlap이 1에 가까우면 두 모델이 같은 세포를 놓친다 (함께 써도 새로 찾는 세포가 적다).
  이웃 거리 구간(14-30px, 30px+)과 클래스별로도 적는다.

주의: 각 모델이 자기 후처리 값으로 낸 예측을 비교한다. bootstrap 구간은 평가 사진을 다시 뽑았을 때의
흔들림이고 학습 seed의 흔들림은 들어 있지 않다 (그건 seed만 다른 쌍의 차이로 본다).
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_m0 import (localization_labels, nearest_distances, official_counts,
                                 patient_id, read_json, save_json, score_counts)

GROUPS = [(cls, name, lo, hi, label) for cls, name in ((1, "BC"), (2, "TC"))
          for lo, hi, label in ((14, 30, "14-30"), (30, np.inf, "30+"))]


def load(path):
    """pair_id → (공식 매칭 개수 [BC, TC] × [tp, fp, gt], 놓침 여부, 클래스, 이웃 거리)"""
    out = {}
    for row in read_json(path):
        truths = np.asarray(row["truths"], dtype=float).reshape(-1, 3)
        predictions = [tuple(p) for p in row["predictions"]]
        missed = (localization_labels(predictions, row["truths"])[0] == "missed"
                  if len(truths) else np.zeros(0, bool))
        out[row["pair_id"]] = (official_counts(predictions, row["truths"]), missed,
                               truths[:, 2].astype(int), nearest_distances(truths[:, :2]))
    return out


def overlap(a, b):
    n, both, only_a, only_b = len(a), int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum())
    union = both + only_a + only_b
    return {"cells": n, "missed_A": round(float(a.mean()), 4), "missed_B": round(float(b.mean()), 4),
            "both": round(both / n, 4), "only_A": round(only_a / n, 4), "only_B": round(only_b / n, 4),
            "overlap": round(both / union, 4) if union else None}


def compare(a, b, cluster_of, repeats, seed):
    ids = sorted(a)
    if ids != sorted(b):
        raise SystemExit("[중단] 두 파일의 사진 목록이 다릅니다.")
    groups = defaultdict(list)
    for pid in ids:
        groups[cluster_of[pid]].append(pid)
    keys = sorted(groups)
    ca = np.array([sum(a[p][0] for p in groups[k]) for k in keys])
    cb = np.array([sum(b[p][0] for p in groups[k]) for k in keys])
    fa, fb = score_counts(ca.sum(0))["mF1"], score_counts(cb.sum(0))["mF1"]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(repeats):
        pick = rng.integers(0, len(keys), len(keys))
        diffs.append(score_counts(cb[pick].sum(0))["mF1"] - score_counts(ca[pick].sum(0))["mF1"])
    low, high = np.percentile(diffs, [2.5, 97.5])
    ma, mb = (np.concatenate([r[p][1] for p in ids]) for r in (a, b))
    cls = np.concatenate([a[p][2] for p in ids])
    near = np.concatenate([a[p][3] for p in ids])
    missed = {"all": overlap(ma, mb)}
    for c, name, lo, hi, label in GROUPS:
        keep = (cls == c) & (near >= lo) & (near < hi)
        missed[f"{name}_{label}"] = overlap(ma[keep], mb[keep])
    return {"images": len(ids), "clusters": len(keys),
            "mF1_A": round(fa, 4), "mF1_B": round(fb, 4), "diff": round(fb - fa, 4),
            "ci95": [round(float(low), 4), round(float(high), 4)], "missed": missed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--pair", action="append", required=True, metavar="A:B")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    files = dict(item.split("=", 1) for item in args.file)
    runs = {name: load(path) for name, path in files.items()}
    slides = read_json(args.metadata)["sample_pairs"]
    result = {"files": files, "bootstrap": {"unit": "TCGA patient when available, otherwise slide",
                                            "repeats": args.repeats, "seed": args.seed}, "pairs": {}}
    for item in args.pair:
        name_a, name_b = item.split(":")
        a, b = runs[name_a], runs[name_b]
        cluster_of = {p: patient_id(slides[p]["slide_name"]) for p in a}
        r = compare(a, b, cluster_of, args.repeats, args.seed)
        result["pairs"][item] = r
        m = r["missed"]["all"]
        print(f"{name_a} {r['mF1_A']:.4f} → {name_b} {r['mF1_B']:.4f}: {r['diff']:+.4f} "
              f"({r['ci95'][0]:+.4f} ~ {r['ci95'][1]:+.4f}) | 놓침 {m['missed_A']:.1%} / {m['missed_B']:.1%}, "
              f"둘 다 {m['both']:.1%}, {name_a}만 {m['only_A']:.1%}, {name_b}만 {m['only_B']:.1%}, "
              f"겹침 {m['overlap']:.2f}")
    save_json(Path(args.out), result)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
