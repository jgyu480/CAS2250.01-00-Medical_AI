#!/bin/bash
# M2 계수 탐색 한 모델 (VESSL Run 하나). G와 GA를 각각 다른 Run으로 돌린다 (한 번에 하나).
# 데이터 다운로드·검사 → scripts/run_m2_search.py (coarse σ → 파인튜닝 → 공식 train 재학습 → 공식 val)
# 결과는 OUT(기본 /root/m2_search_<MODE>)에 모인다 (VESSL artifact로 내보냄).
#
# VESSL Run:  MODE=gaussian bash scripts/run_m2_search.sh      (vessl/m2_search_g.yaml)
#             MODE=ha bash scripts/run_m2_search.sh            (vessl/m2_search_ga.yaml)
# 확인 단계:  RUNS="gaussian:6:0.9:43 gaussian:8:0.9:43" bash scripts/run_m2_search.sh   (vessl/m2_confirm.yaml)
#             적어 준 후보(모델:σ:λ:seed)만 학습·평가한다. 후처리 격자는 GRID만큼 넓혀서도 본다.
# 점검용:     SKIP_PREP=1 SEARCH_EXTRA="--train-extra=--smoke" MODE=gaussian bash scripts/run_m2_search.sh
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
RUNS=${RUNS:-}
if [ -n "$RUNS" ]; then
  MODE=confirm
  OUT=${OUT:-/root/m2_confirm}
  GRID=${GRID-sigma=1 min_distance=11 threshold=0.6,0.7}
else
  MODE=${MODE:?MODE=gaussian 또는 MODE=ha (확인 단계는 RUNS=...)}
  OUT=${OUT:-/root/m2_search_${MODE}}
fi
SPLIT=${SPLIT:-configs/train_split_seed42.json}
SEARCH_EXTRA=${SEARCH_EXTRA:-}
SKIP_PREP=${SKIP_PREP:-0}
ZIP_URL="https://zenodo.org/records/8417503/files/ocelot2023_v1.0.1.zip?download=1"

# 파일 하나를 받는다. 연결이 끊기거나 2분 넘게 멈추면 받은 곳부터 이어 받는다 (최대 30회).
# 받은 ZIP이 공식 파일과 같은지는 뒤의 check_data.py가 MD5로 확인한다.
fetch() {
  python - "$1" "$2" <<'PY'
import os, sys, time, urllib.error, urllib.request
url, dest = sys.argv[1], sys.argv[2]
for attempt in range(1, 31):
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            resumed = bool(have) and response.status == 206   # 200이면 서버가 이어받기를 무시한 것 → 처음부터
            length = response.headers.get("Content-Length")
            expected = (have if resumed else 0) + int(length) if length else None
            with open(dest, "ab" if resumed else "wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
        size = os.path.getsize(dest)
        if expected is None or size >= expected:
            print(f"[다운로드] 완료 {size:,} bytes (시도 {attempt}회)", flush=True)
            sys.exit(0)
        print(f"[다운로드] 시도 {attempt}: {size:,} / {expected:,} bytes에서 끊김 → 이어 받기", flush=True)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:   # 요청한 위치가 파일 끝 = 이미 다 받음
            print(f"[다운로드] 이미 받은 파일 {have:,} bytes", flush=True)
            sys.exit(0)
        print(f"[다운로드] 시도 {attempt}: HTTP {exc.code} → 다시 시도", flush=True)
    except Exception as exc:
        print(f"[다운로드] 시도 {attempt}: {type(exc).__name__}: {exc} → 다시 시도", flush=True)
    time.sleep(min(60, 5 * attempt))
sys.exit("[중단] 다운로드 실패 (30회 시도)")
PY
}

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
  fetch "$ZIP_URL" data/downloads/ocelot2023_v1.0.1.zip || exit 1
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

if [ -n "$RUNS" ]; then
  python scripts/run_m2_search.py --confirm $RUNS ${GRID:+--extend-grid $GRID} \
    --split-file "$SPLIT" --out "$OUT" $SEARCH_EXTRA
  STATUS=$?
else
  EXTRA=""
  [ "$MODE" = "ha" ] && EXTRA="--ft-lambdas 0.5"
  python scripts/run_m2_search.py --mode "$MODE" --split-file "$SPLIT" --out "$OUT" $EXTRA $SEARCH_EXTRA
  STATUS=$?
fi
echo "[종료] $(date '+%F %T') (탐색 종료 코드 $STATUS)"
# 내려받기 쉽게 작은 결과 파일(가중치 제외)을 하나로 묶는다. 묶는 동안 폴더가 바뀌지 않게 밖에서 만든 뒤 옮긴다.
sleep 1
tar -czf "$OUT.small.tar.gz" -C "$OUT" --exclude='*.pt' --exclude='*.tar.gz' . 2>/dev/null
[ -s "$OUT.small.tar.gz" ] && mv "$OUT.small.tar.gz" "$OUT/results_small.tar.gz" && echo "[묶음] $OUT/results_small.tar.gz"
ls -la "$OUT" "$OUT"/candidates/* "$OUT"/final/* 2>/dev/null
exit $STATUS
