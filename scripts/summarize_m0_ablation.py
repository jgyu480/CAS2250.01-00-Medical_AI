"""M0 꺼보기(ablation) 결과를 한 표로 모은다. 학습·추론 없이 history.csv만 읽는다.

python scripts/summarize_m0_ablation.py outputs/m0/ablation_all_on outputs/m0/ablation_no_aspp_t2c ...

각 run에서
- epoch 0: 학습 전(연결을 끄자마자) validation mF1  (--eval-init로 기록된 경우)
- 마지막 3 epoch 평균 mF1 / mIoU: 한 epoch의 운을 줄인 비교 기준
- best mF1과 그 epoch
첫 번째 run을 기준으로 차이(Δ)를 계산한다. 보통 "다 켬"을 첫 번째로 넣는다.
결과: outputs/m0/ablation_summary.csv
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import resolve_paths


def summarize(run):
    with (Path(run) / "history.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    trained = [r for r in rows if r["epoch"] != "0"]
    if not trained:
        raise SystemExit(f"[중단] {run}: 학습된 epoch가 없습니다.")
    init = next((r for r in rows if r["epoch"] == "0"), None)
    last = trained[-3:]
    best = max(trained, key=lambda r: float(r["val_mF1"]))
    mean = lambda key: sum(float(r[key]) for r in last) / len(last)
    return {
        "run": Path(run).name,
        "epochs": len(trained),
        "mF1_epoch0": float(init["val_mF1"]) if init else None,
        "mF1_last3": mean("val_mF1"),
        "BC_last3": mean("val_f1_BC"),
        "TC_last3": mean("val_f1_TC"),
        "mIoU_last3": mean("val_mIoU"),
        "best_mF1": float(best["val_mF1"]),
        "best_epoch": int(best["epoch"]),
    }


def main():
    runs = sys.argv[1:]
    if not runs:
        raise SystemExit(__doc__)
    table = [summarize(r) for r in runs]
    base = table[0]
    for row in table:
        row["delta_mF1_last3"] = row["mF1_last3"] - base["mF1_last3"]
        row["delta_mIoU_last3"] = row["mIoU_last3"] - base["mIoU_last3"]

    out = resolve_paths()["outputs"] / "m0" / "ablation_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, list(table[0]))
        writer.writeheader()
        writer.writerows([{k: round(v, 4) if isinstance(v, float) else v
                           for k, v in r.items()} for r in table])

    print(f"[기준] {base['run']}  (Δ는 마지막 3 epoch 평균 기준)")
    print(f"{'run':32s} {'ep0':>7s} {'mF1':>7s} {'ΔmF1':>8s} {'mIoU':>7s} {'ΔmIoU':>8s} {'best':>12s}")
    for r in table:
        ep0 = f"{r['mF1_epoch0']:.4f}" if r["mF1_epoch0"] is not None else "-"
        print(f"{r['run']:32s} {ep0:>7s} {r['mF1_last3']:7.4f} "
              f"{r['delta_mF1_last3']:+8.4f} {r['mIoU_last3']:7.4f} "
              f"{r['delta_mIoU_last3']:+8.4f} {r['best_mF1']:.4f}@{r['best_epoch']:>2d}")
    print(f"[저장] {out}")
    print("[안내] seed 1회 비교입니다. 차이가 0.01 안팎이면 '차이 없음'으로 해석하세요.")


if __name__ == "__main__":
    main()
