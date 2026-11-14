"""M2 계수 탐색 (G 또는 GA 한 모델): 간격 있는 σ로 처음부터 학습 → 가장 좋은 σ 주변을
파인튜닝으로 탐색 → 고른 방법으로 공식 train 재학습 → 공식 val 계산.

    python scripts/run_m2_search.py --mode gaussian --out /root/m2_search_g
    python scripts/run_m2_search.py --mode ha --ft-lambdas 0.5 --out /root/m2_search_ga

데이터 (scripts/make_train_split.py, 공식 train만 환자 단위로 나눔)
- fit: 후보 학습 / select: checkpoint 선택·후처리 선택 / holdout: 계수 비교
- 공식 val은 마지막 확인에서만, 공식 test는 쓰지 않는다.

단계
1) coarse: σ ∈ --sigmas, λ ∈ --lambdas. fit으로 처음부터 학습 → select에서 후처리 → holdout 평가.
2) 파인튜닝: coarse에서 holdout mF1이 가장 높은 후보의 best.pt에서 시작한다.
   --ft-epochs epoch, 학습률 ×--ft-lr-scale, 백본 고정 없음.
   - 대조군: 같은 σ·λ로 같은 만큼 더 학습 (추가 학습 효과를 따로 보기 위해)
   - σ* ± --ft-step, GA는 σ*에서 --ft-lambdas
3) 선택: 모든 후보 중 holdout mF1이 가장 높은 것.
4) 공식 확인 (--no-final로 끔): 고른 방법 그대로 공식 train으로 학습 → 공식 val에서 후처리 고르기와 평가.
   파인튜닝 후보를 골랐으면 σ*로 50 epoch 학습한 뒤 같은 파인튜닝을 반복한다.

시간 제한: 다음 후보가 끝나는 예상 시각이 --budget-hours를 넘으면 (공식 확인 시간을 남기고) 건너뛴다.
결과: --out/summary.json, summary.csv, candidates/<후보>/ (설정·기록·평가),
      고른 후보의 best.pt (앙상블용, fit으로 학습), final/<run>/ (공식 확인, best.pt 포함).
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import resolve_paths

KEEP = ("config.json", "history.csv", "log.txt", "postprocess.json")
M0_VAL = "results/m0_dicefix_seed42/eval_val_per_image.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["gaussian", "ha", "hao"], required=True)
    p.add_argument("--sigmas", type=float, nargs="+", default=[4.0, 6.0, 8.0])
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.9],
                   help="coarse λ (G는 지도에 영향 없음, 이름에만 쓰인다)")
    p.add_argument("--ft-step", type=float, default=1.0)
    p.add_argument("--ft-lambdas", type=float, nargs="*", default=[],
                   help="σ*에서 파인튜닝으로 볼 다른 λ (GA)")
    p.add_argument("--ft-epochs", type=int, default=10)
    p.add_argument("--ft-lr-scale", type=float, default=0.1)
    p.add_argument("--config", default="configs/m2_search_train.json")
    p.add_argument("--split-file", default="configs/train_split_seed42.json")
    p.add_argument("--train-part", default="fit")
    p.add_argument("--val-part", default="select")
    p.add_argument("--test-part", default="holdout")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budget-hours", type=float, default=22.0)
    p.add_argument("--no-final", action="store_true", help="공식 train 재학습·공식 val 확인을 하지 않는다")
    p.add_argument("--out", required=True)
    p.add_argument("--train-extra", default="", help="train_m2.py에 더할 인자 (점검: --train-extra=--smoke)")
    p.add_argument("--eval-extra", default="", help="eval_m2.py에 더할 인자 (점검: '--eval-extra=--limit 2')")
    return p.parse_args()


class Search:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.runs = resolve_paths()["outputs"] / "m2"
        self.start = time.time()
        self.rows = []
        self.durations = {"coarse": [], "ft": [], "final": []}

    # ---------- 공통 ----------
    def hours(self):
        return (time.time() - self.start) / 3600

    def call(self, cmd):
        print(f"\n[실행 {self.hours():.2f}h] {' '.join(map(str, cmd))}", flush=True)
        return subprocess.run([str(c) for c in cmd], cwd=ROOT).returncode == 0

    def estimate(self, stage):
        a = self.args
        coarse = (sum(self.durations["coarse"]) / len(self.durations["coarse"])
                  if self.durations["coarse"] else 1.6)
        if stage == "coarse":
            return coarse
        if stage == "ft":
            if self.durations["ft"]:
                return sum(self.durations["ft"]) / len(self.durations["ft"])
            return 0.25 + coarse * a.ft_epochs / 50
        # 공식 확인: train 400장(fit의 약 1.7배) + val 137장 후처리 탐색
        return 2.0 * coarse + (self.estimate("ft") * 1.7 if self.best_is_ft() else 0)

    def best_is_ft(self):
        best = self.best()
        return bool(best and best["init_from"])

    def fits(self, stage):
        reserve = 0 if self.args.no_final else self.estimate("final")
        need = self.hours() + self.estimate(stage) + (reserve if stage != "final" else 0)
        if need > self.args.budget_hours:
            print(f"[시간] {stage} 건너뜀: 예상 {need:.1f}h > 제한 {self.args.budget_hours}h", flush=True)
            return False
        return True

    def name(self, sigma, lam, suffix=""):
        return f"search_{self.args.mode}_s{sigma:g}_l{lam:g}{suffix}"

    def train_cmd(self, run_name, sigma, lam, init_from=None, split=True):
        a = self.args
        cmd = [sys.executable, "scripts/train_m2.py", "--config", a.config, "--seed", a.seed,
               "--target-mode", a.mode, "--sigma", sigma, "--lambda", lam, "--run-name", run_name]
        if split:
            cmd += ["--split-file", a.split_file, "--train-part", a.train_part,
                    "--val-part", a.val_part]
        if init_from:
            cmd += ["--init-from", init_from, "--epochs", a.ft_epochs,
                    "--freeze-epochs", 0, "--lr-scale", a.ft_lr_scale]
        return cmd + a.train_extra.split()

    def eval_cmd(self, run_name, *where):
        return ([sys.executable, "scripts/eval_m2.py", "--checkpoint",
                 self.runs / run_name / "best.pt", *where] + self.args.eval_extra.split())

    @staticmethod
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None

    def keep(self, run_name, dest, labels, checkpoint=False):
        src, dest = self.runs / run_name, self.out / dest
        dest.mkdir(parents=True, exist_ok=True)
        names = list(KEEP) + [f"{kind}_{label}{tail}" for label in labels
                              for kind, tail in (("eval", ".json"), ("eval", "_per_image.json"),
                                                 ("close_cells", ".json"))]
        if checkpoint:
            names.append("best.pt")
        for n in names:
            if (src / n).exists():
                shutil.copy2(src / n, dest / n)

    def write_summary(self, chosen=None, final=None):
        a = self.args
        payload = {"mode": a.mode, "split_file": a.split_file,
                   "parts": {"train": a.train_part, "select": a.val_part, "compare": a.test_part},
                   "rule": f"{a.test_part} 세포 mF1이 가장 높은 후보 (공식 val·test 미사용)",
                   "hours": round(self.hours(), 2), "candidates": self.rows,
                   "chosen": chosen, "final": final}
        (self.out / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        fields = ["name", "stage", "sigma", "lambda", "init_from_name", "status",
                  "select_mF1", "mF1", "f1_BC", "f1_TC", "BC_miss_14_30", "TC_miss_14_30",
                  "BC_miss_30", "TC_miss_30", "false_positive", "postprocess", "hours"]
        with (self.out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fields, extrasaction="ignore")
            w.writeheader()
            for r in self.rows:
                w.writerow({**r, "postprocess": json.dumps(r.get("postprocess"))})

    # ---------- 후보 하나 ----------
    def candidate(self, stage, sigma, lam, init_from=None, from_name=None, suffix=""):
        a = self.args
        run_name = self.name(sigma, lam, suffix)
        row = {"name": run_name, "stage": stage, "sigma": sigma, "lambda": lam,
               "init_from": str(init_from) if init_from else None,
               "init_from_name": from_name, "status": "skipped"}
        kind = "coarse" if stage == "coarse" else "ft"
        if not self.fits(kind):
            self.rows.append(row)
            self.write_summary()
            return row
        began = time.time()
        ok = (self.call(self.train_cmd(run_name, sigma, lam, init_from))
              and self.call(self.eval_cmd(run_name, "--part", a.val_part, "--tune"))
              and self.call(self.eval_cmd(run_name, "--part", a.test_part)))
        hours = (time.time() - began) / 3600
        row["hours"] = round(hours, 2)
        if ok:
            self.durations[kind].append(hours)
            folder = self.runs / run_name
            sel = self.read(folder / f"eval_{a.val_part}.json")
            test = self.read(folder / f"eval_{a.test_part}.json")
            cells = self.read(folder / f"close_cells_{a.test_part}.json")["M2"]
            rate = {(t["class"], t["neighbor_bin"]): t["miss_rate"]
                    for t in cells["by_class_and_neighbor"]}
            row.update(status="done", select_mF1=sel["overall"]["mF1"],
                       mF1=test["overall"]["mF1"], f1_BC=test["overall"]["f1_BC"],
                       f1_TC=test["overall"]["f1_TC"], epoch=test["epoch"],
                       BC_miss_14_30=rate[("BC", "14-30")], TC_miss_14_30=rate[("TC", "14-30")],
                       BC_miss_30=rate[("BC", "30+")], TC_miss_30=rate[("TC", "30+")],
                       false_positive=cells["totals"]["false_positive"],
                       postprocess=test["postprocess"])
            print(f"[후보] {run_name}: {a.test_part} mF1 {row['mF1']:.4f} "
                  f"(select {row['select_mF1']:.4f}) | {hours:.2f}h", flush=True)
        else:
            row["status"] = "failed"
            print(f"[실패] {run_name}", flush=True)
        self.keep(run_name, f"candidates/{run_name}", [a.val_part, a.test_part])
        self.rows.append(row)
        self.write_summary()
        return row

    def best(self, stage=None):
        done = [r for r in self.rows if r["status"] == "done"
                and (stage is None or r["stage"] == stage)]
        return max(done, key=lambda r: (r["mF1"], r["select_mF1"])) if done else None

    # ---------- 전체 ----------
    def run(self):
        a = self.args
        print(f"[탐색] {a.mode} | coarse σ {a.sigmas} λ {a.lambdas} | 파인튜닝 ±{a.ft_step:g}"
              f"{' λ ' + str(a.ft_lambdas) if a.ft_lambdas else ''} | 분할 {a.split_file} "
              f"({a.train_part}/{a.val_part}/{a.test_part}) | 제한 {a.budget_hours}h", flush=True)
        for lam in a.lambdas:
            for sigma in a.sigmas:
                self.candidate("coarse", sigma, lam)
        base = self.best("coarse")
        if base is None:
            raise SystemExit("[중단] coarse 후보가 하나도 끝나지 않았습니다.")
        print(f"[coarse 선택] {base['name']} ({a.test_part} mF1 {base['mF1']:.4f})", flush=True)

        start = self.runs / base["name"] / "best.pt"
        s0, l0 = base["sigma"], base["lambda"]
        plan = [("ft-control", s0, l0, "_ft-control")]
        plan += [("ft", s, l0, f"_ft-from-s{s0:g}") for s in (s0 - a.ft_step, s0 + a.ft_step) if s > 0]
        plan += [("ft", s0, lam, f"_ft-from-l{l0:g}") for lam in a.ft_lambdas if lam != l0]
        for stage, sigma, lam, suffix in plan:
            self.candidate(stage, sigma, lam, start, base["name"], suffix)

        chosen = self.best()
        self.keep(chosen["name"], f"candidates/{chosen['name']}", [a.val_part, a.test_part],
                  checkpoint=True)
        control = next((r for r in self.rows if r["stage"] == "ft-control"), None)
        print(f"[최종 선택] {chosen['name']} ({a.test_part} mF1 {chosen['mF1']:.4f})"
              + (f" | 파인튜닝 대조군 {control['mF1']:.4f}" if control and control.get("mF1") else ""),
              flush=True)
        self.write_summary(chosen=chosen)
        if a.no_final:
            return

        final = self.final(chosen, base)
        self.write_summary(chosen=chosen, final=final)
        print(f"[완료] {self.hours():.2f}h. 결과: {self.out}", flush=True)

    def final(self, chosen, base):
        """고른 방법 그대로 공식 train으로 학습 → 공식 val (M0와 비교)."""
        a = self.args
        began = time.time()
        steps = []
        if chosen["init_from"]:
            base_name = f"final_{a.mode}_s{base['sigma']:g}_l{base['lambda']:g}_base"
            steps.append((base_name, base["sigma"], base["lambda"], None))
            name = (f"final_{a.mode}_s{chosen['sigma']:g}_l{chosen['lambda']:g}"
                    f"_ft-from-s{base['sigma']:g}-l{base['lambda']:g}")
            steps.append((name, chosen["sigma"], chosen["lambda"], self.runs / base_name / "best.pt"))
        else:
            name = f"final_{a.mode}_s{chosen['sigma']:g}_l{chosen['lambda']:g}"
            steps.append((name, chosen["sigma"], chosen["lambda"], None))
        result = {"name": name, "recipe": [s[0] for s in steps], "status": "failed"}
        if not self.fits("final"):
            # Run 시간 제한(24h)에 걸려 강제 종료되면 결과 내보내기도 못 할 수 있다.
            result["status"] = "skipped"
            return result
        for run_name, sigma, lam, init in steps:
            if not self.call(self.train_cmd(run_name, sigma, lam, init, split=False)):
                return result
        compare = ["--compare", M0_VAL] if (ROOT / M0_VAL).exists() else []
        if not self.call(self.eval_cmd(name, "--split", "val", "--tune", *compare)):
            return result
        for run_name, *_ in steps:
            self.keep(run_name, f"final/{run_name}", ["val"], checkpoint=True)
        ev = self.read(self.runs / name / "eval_val.json")
        result.update(status="done", hours=round((time.time() - began) / 3600, 2),
                      val=ev["overall"], postprocess=ev["postprocess"], epoch=ev["epoch"])
        print(f"[공식 val] {name}: mF1 {ev['overall']['mF1']:.4f}", flush=True)
        return result


if __name__ == "__main__":
    Search(parse_args()).run()
