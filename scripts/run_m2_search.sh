#!/bin/bash
# M2 계수 탐색 한 모델 (VESSL Run 하나). G와 GA를 각각 다른 Run으로 동시에 돌린다.
# 데이터 다운로드·검사 → scripts/run_m2_search.py (coarse σ → 파인튜닝 → 공식 train 재학습 → 공식 val)
# 결과는 OUT(기본 /root/m2_search_<MODE>)에 모인다 (VESSL artifact로 내보냄).
#
# VESSL Run:  MODE=gaussian bash scripts/run_m2_search.sh      (vessl/m2_search_g.yaml)
#             MODE=ha bash scripts/run_m2_search.sh            (vessl/m2_search_ga.yaml)
# 점검용:     SKIP_PREP=1 SEARCH_EXTRA="--train-extra=--smoke" MODE=gaussian bash scripts/run_m2_search.sh
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
MODE=${MODE:?MODE=gaussian 또는 MODE=ha}
OUT=${OUT:-/root/m2_search_${MODE}}
SPLIT=${SPLIT:-configs/train_split_seed42.json}
SEARCH_EXTRA=${SEARCH_EXTRA:-}
SKIP_PREP=${SKIP_PREP:-0}
ZIP_URL="https://zenodo.org/records/8417503/files/ocelot2023_v1.0.1.zip?download=1"

mkdir -p "$OUT"
exec > >(tee -a "$OUT/run_all.log") 2>&1
cd "$REPO" || exit 1
echo "[시작] $(date '+%F %T') MODE=$MODE"
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

# 분할 파일이 이 데이터의 metadata.json으로 만든 것인지 확인한다.
python - "$SPLIT" <<'PY' || { echo "[중단] 분할 파일 확인 실패"; exit 1; }
import hashlib, json, sys
from common.paths import resolve_paths
split = json.load(open(sys.argv[1], encoding="utf-8"))
meta = resolve_paths()["data_root"] / "metadata.json"
if hashlib.sha256(meta.read_bytes()).hexdigest() != split["metadata_sha256"]:
    raise SystemExit("metadata.json이 분할 파일을 만들 때와 다릅니다.")
print("[분할]", sys.argv[1], {k: v["pairs"] for k, v in split["summary"].items()})
PY

EXTRA=""
[ "$MODE" = "ha" ] && EXTRA="--ft-lambdas 0.5"
python scripts/run_m2_search.py --mode "$MODE" --split-file "$SPLIT" --out "$OUT" $EXTRA $SEARCH_EXTRA
echo "[종료] $(date '+%F %T')"
ls -la "$OUT" "$OUT"/candidates/* "$OUT"/final/* 2>/dev/null
