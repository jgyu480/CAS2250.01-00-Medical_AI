"""공식 train을 환자 단위로 나눠 분할 파일을 만든다 (공식 val·test는 건드리지 않는다).

공식 val은 최종 비교, 공식 test는 마지막 1회 평가에 남겨 두고,
계수 선택·checkpoint 선택·앙상블 학습에 쓸 데이터를 공식 train 안에서만 나눈다.

    python scripts/make_train_split.py \
        --parts fit=233 select=40 holdout=127 --seed 42 \
        --out configs/train_split_seed42.json

- 같은 환자(TCGA 환자, 없으면 slide)의 사진은 한 부분에만 들어간다.
  train 400장은 환자 132명이고 한 환자에서 최대 18장이 나와, 사진 단위로 나누면
  같은 환자가 학습과 평가에 함께 들어간다.
- 장기 비율은 부분마다 공식 train과 비슷하게 맞춘다 (환자를 통째로 옮기므로 몇 장 차이가 난다).
- seed가 같으면 같은 파일이 나온다. 만든 파일은 저장소에 두고 모든 모델이 같은 id를 쓴다.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import resolve_paths
from scripts.diagnose_m0 import patient_id


def parse_parts(items):
    parts = {}
    for item in items:
        name, _, count = item.partition("=")
        if not name or not count.isdigit() or name in parts:
            raise SystemExit(f"[중단] --parts는 이름=장수 형식이고 이름이 겹치면 안 됩니다: {item}")
        parts[name] = int(count)
    return parts


def apportion(rows, cols):
    """장기별 장수(rows)와 부분별 장수(cols)를 둘 다 맞추는 정수 표 (나머지가 큰 칸부터 1씩)."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    raw = np.outer(rows, cols) / cols.sum()
    table = np.floor(raw).astype(int)
    left = raw - table
    while True:
        need_r, need_c = rows - table.sum(1), cols - table.sum(0)
        if not need_r.any():
            return table
        open_cells = np.outer(need_r > 0, need_c > 0)
        r, c = np.unravel_index(np.argmax(np.where(open_cells, left, -1)), left.shape)
        table[r, c] += 1
        left[r, c] = -1


def split_pairs(pairs, parts, seed):
    """pairs: {pair_id: (patient, organ)}. 부분별 pair_id 목록을 돌려준다."""
    names = list(parts)
    sizes = np.array([parts[n] for n in names])
    organs = sorted({organ for _, organ in pairs.values()})
    organ_count = Counter(organ for _, organ in pairs.values())

    # 장기별·부분별 목표 장수: 장기 합과 부분 크기를 모두 맞추게 배분한다.
    table = apportion([organ_count[o] for o in organs], sizes)
    target = {o: table[i] for i, o in enumerate(organs)}

    groups = defaultdict(list)
    for pair_id, (patient, organ) in pairs.items():
        groups[(organ, patient)].append(pair_id)

    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    # 큰 환자부터 넣어야 목표에 가깝게 맞출 수 있다 (같은 크기는 섞인 순서대로).
    keys.sort(key=lambda k: -len(groups[k]))

    assigned = {o: np.zeros(len(names), dtype=int) for o in organs}
    total = np.zeros(len(names), dtype=int)
    result = {n: [] for n in names}
    for organ, patient in keys:
        size = len(groups[(organ, patient)])
        need = target[organ] - assigned[organ]
        overall = sizes - total
        # 이 장기에서 가장 많이 모자란 부분, 같으면 전체에서 더 모자란 부분.
        choice = max(range(len(names)), key=lambda i: (need[i], overall[i], -i))
        assigned[organ][choice] += size
        total[choice] += size
        result[names[choice]].extend(groups[(organ, patient)])

    # 다듬기: 환자 하나 옮기기 또는 두 환자 맞바꾸기로 목표와의 차이가 줄면 반영한다.
    owner = {k: next(i for i, n in enumerate(names) if groups[k][0] in result[n]) for k in keys}

    def cost(own):
        part = np.zeros(len(names))
        cell = {o: np.zeros(len(names)) for o in organs}
        for (organ, _), i in own.items():
            part[i] += len(groups[(organ, _)])
            cell[organ][i] += len(groups[(organ, _)])
        return (np.abs(part - sizes).sum()
                + 0.5 * sum(np.abs(cell[o] - target[o]).sum() for o in organs))

    best = cost(owner)
    improved = True
    while improved:
        improved = False
        for k in keys:
            for i in range(len(names)):
                if i == owner[k]:
                    continue
                trial = {**owner, k: i}
                c = cost(trial)
                if c < best - 1e-9:
                    owner, best, improved = trial, c, True
        for a_i, a in enumerate(keys):
            for b in keys[a_i + 1:]:
                if owner[a] == owner[b] or a[0] != b[0]:
                    continue
                trial = {**owner, a: owner[b], b: owner[a]}
                c = cost(trial)
                if c < best - 1e-9:
                    owner, best, improved = trial, c, True
    result = {n: [] for n in names}
    for k, i in owner.items():
        result[names[i]].extend(groups[k])
    return {n: sorted(v) for n, v in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", nargs="+", required=True, metavar="NAME=COUNT")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata", help="OCELOT metadata.json (기본: 데이터 폴더)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    parts = parse_parts(args.parts)
    path = Path(args.metadata) if args.metadata else resolve_paths()["data_root"] / "metadata.json"
    meta = json.loads(path.read_text(encoding="utf-8"))["sample_pairs"]
    pairs = {k: (patient_id(v["slide_name"]), v["organ"])
             for k, v in meta.items() if v["subset"] == "train"}
    if sum(parts.values()) != len(pairs):
        raise SystemExit(f"[중단] 부분 장수 합 {sum(parts.values())} ≠ 공식 train {len(pairs)}장")

    result = split_pairs(pairs, parts, args.seed)

    owners = defaultdict(set)
    for name, ids in result.items():
        for pair_id in ids:
            owners[pairs[pair_id][0]].add(name)
    if any(len(v) > 1 for v in owners.values()) or sum(map(len, result.values())) != len(pairs):
        raise SystemExit("[중단] 환자가 두 부분에 나뉘었거나 빠진 사진이 있습니다.")

    organs = sorted({o for _, o in pairs.values()})
    summary = {
        name: {"pairs": len(ids), "target": parts[name],
               "patients": len({pairs[i][0] for i in ids}),
               "organs": {o: sum(pairs[i][1] == o for i in ids) for o in organs}}
        for name, ids in result.items()
    }
    payload = {
        "description": "공식 train만 환자 단위로 나눈 분할. 공식 val·test는 포함하지 않는다.",
        "source_split": "train",
        "group_by": "TCGA patient (없으면 slide)",
        "stratify_by": "organ",
        "seed": args.seed,
        "command": "python scripts/make_train_split.py --parts "
                   + " ".join(f"{n}={c}" for n, c in parts.items())
                   + f" --seed {args.seed} --out {args.out}",
        "metadata_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "summary": summary,
        "parts": result,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"[분할] 공식 train {len(pairs)}장, 환자 {len(owners)}명 → {out}")
    for name, v in summary.items():
        print(f"  {name:8s} {v['pairs']:4d}장 (목표 {v['target']}), 환자 {v['patients']}명 | "
              + ", ".join(f"{o} {c}" for o, c in v["organs"].items()))


if __name__ == "__main__":
    main()
