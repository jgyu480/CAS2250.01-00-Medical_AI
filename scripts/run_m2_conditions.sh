#!/bin/bash
# M2 세 조건(G, H_A, H_AO)을 1회씩 학습하고 validation 평가까지 한다. σ=6, λ=0.9, seed 42.
# 데이터 다운로드·검사 4개 → 조건별 학습 → eval_m2.py --tune --compare(M0) → 결과 모으기.
# 결과(학습 기록·평가·진단·best.pt)는 OUT(기본 /root/m2_results)에 조건별 폴더로 모인다.
#
# VESSL Run (vessl/m2_run.yaml):  bash scripts/run_m2_conditions.sh
# 다른 환경:                      OUT=outputs/m2_results bash scripts/run_m2_conditions.sh
# 점검용 (데이터 준비 생략, 짧은 학습): SKIP_PREP=1 TRAIN_EXTRA=--smoke bash scripts/run_m2_conditions.sh
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
OUT=${OUT:-/root/m2_results}
TRAIN_EXTRA=${TRAIN_EXTRA:-}      # 점검용 (예: --smoke)
SKIP_PREP=${SKIP_PREP:-0}         # 점검용: 데이터 준비·검사 건너뛰기
ZIP_URL="https://zenodo.org/records/8417503/files/ocelot2023_v1.0.1.zip?download=1"

mkdir -p "$OUT"
exec > >(tee -a "$OUT/run_all.log") 2>&1
cd "$REPO" || exit 1
echo "[시작] $(date '+%F %T')"
nvidia-smi || true
python -c "import sys, torch; print(sys.version.split()[0], torch.__version__, 'cuda', torch.cuda.is_available())"
git log --oneline -1 2>/dev/null || true

python -c "import scipy" 2>/dev/null || python -m pip install -q "scipy<1.16" || exit 1

if [ "$SKIP_PREP" != "1" ]; then
  mkdir -p data/downloads data/raw
  echo "[데이터] OCELOT 다운로드"
  python -c "import urllib.request as u; u.urlretrieve('$ZIP_URL', 'data/downloads/ocelot2023_v1.0.1.zip')" || exit 1
  python -c "import zipfile; zipfile.ZipFile('data/downloads/ocelot2023_v1.0.1.zip').extractall('data/raw')" || exit 1
  python scripts/check_data.py --zip data/downloads/ocelot2023_v1.0.1.zip || { echo "[중단] check_data 실패"; exit 1; }
  python scripts/check_dataset.py || { echo "[중단] check_dataset 실패"; exit 1; }
  python scripts/check_components.py || { echo "[중단] check_components 실패"; exit 1; }
  python scripts/check_m2.py --data || { echo "[중단] check_m2 실패"; exit 1; }
fi

# 학습 결과가 저장되는 폴더 (OUTPUT_ROOT 환경변수 또는 기본 outputs/)
OUTPUT_DIR=$(python -c "from common.paths import resolve_paths; print(resolve_paths()['outputs'])") || exit 1
echo "[경로] 학습 결과 $OUTPUT_DIR/m2"

for MODE in gaussian ha hao; do
  echo "================ [$MODE] 학습 시작 $(date '+%F %T') ================"
  if ! python scripts/train_m2.py --seed 42 --target-mode "$MODE" $TRAIN_EXTRA; then
    echo "[실패] $MODE 학습. 다음 조건으로 넘어갑니다."
    continue
  fi
  RUN_DIR=$(ls -td "$OUTPUT_DIR"/m2/M2_${MODE}_s6_l0.9_* 2>/dev/null | head -1)
  if [ -z "$RUN_DIR" ] || [ ! -f "$RUN_DIR/best.pt" ]; then
    echo "[실패] $MODE 결과 폴더를 찾지 못했습니다."
    continue
  fi
  echo "[$MODE] 평가 $(date '+%F %T')"
  python scripts/eval_m2.py --checkpoint "$RUN_DIR/best.pt" --split val --tune \
    --compare results/m0_dicefix_seed42/eval_val_per_image.json || echo "[실패] $MODE 평가"
  DEST="$OUT/$(basename "$RUN_DIR")"
  mkdir -p "$DEST"
  for f in config.json history.csv log.txt postprocess.json eval_val.json \
           eval_val_per_image.json close_cells_val.json best.pt; do
    [ -f "$RUN_DIR/$f" ] && cp "$RUN_DIR/$f" "$DEST/"
  done
  echo "[$MODE] 결과 저장: $DEST"
done
echo "[완료] $(date '+%F %T')"
ls -la "$OUT"/*
