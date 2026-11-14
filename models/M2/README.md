# M2: 세포 중심 반응의 분리를 통한 밀집 세포 검출

담당: 신상우 · 계획: 보고서 8.4 · 작업 순서: 민규 요청(σ·λ 선정 → 1회 학습 → 점검 → 방향 결정)

## 왜 하는가

M0는 가까운 세포를 놓친다. validation에서 최근접 정답 세포까지 14–30px인 세포의 누락률이
30px 이상인 세포보다 BC 11.3%p, TC 17.2%p 높았다 (보고서 6.2).
M0 정답은 중심마다 반지름 7px 원을 그린 지도라서 원 안이 모두 같은 값이고,
두 원이 붙으면 한 덩어리가 된다. M2는 **정답 지도만** 바꿔서 두 중심 사이 점수를 낮춘다.

## 정답 지도 (`common/data/center_heatmaps.py`)

BC와 TC 각각, **같은 클래스의 중심만**으로 만든다. d1, d2 = 같은 클래스에서 첫째·둘째로 가까운 중심까지 거리.

| 이름 | 식 | 뜻 |
|---|---|---|
| G | `exp(-d1² / 2σ²)` (= Gaussian들의 픽셀별 최댓값) | 중심 1, 멀수록 작다. 더하지 않고 최댓값, 합 정규화 없음 |
| A | `(2·d1 / (d1 + d2))²` | 중심 0, 두 중심의 이등분선 1 |
| O | `exp(-d2² / 2σ²)` | 두 번째 중심이 멀면 작다 |
| H_A | `G · (1 − λ·A)` | A만 적용 |
| **H_AO** | `G · (1 − λ·O·A)` | **이번 1회 실험의 정답** |

- 중심 값은 세 지도 모두 1이다. λ는 0 이상 1 미만이다.
- 같은 클래스 중심이 1개면 분리 계수는 1(억제 없음)이고, 0개면 그 클래스 지도 전체가 0이다.
- 증강(뒤집기·90° 회전) 뒤의 점으로 지도를 만들기 때문에 사진과 지도가 항상 같이 움직인다.
- 식은 `scripts/select_gaussian_params.py`(σ·λ 선정 스크립트)의 식과 같다. `check_m2.py`가 차이를 검사한다(최대 3e-8).

## 모델·손실·후처리

| 항목 | M0 | M2 |
|---|---|---|
| encoder, ASPP, decoder, 특징 전달 ①②③, 조직 head | — | **M0와 같음** (`models/M0/model.py`의 M0 클래스를 그대로 사용) |
| 세포 head | 1×1 conv 3채널 (배경/BC/TC), softmax | 1×1 conv **2채널 (BC/TC), 독립 sigmoid** |
| 세포 정답 | 반지름 7px 원형 클래스 지도 | BC·TC 중심 지도 H_AO (2채널, 0~1) |
| 세포 손실 | Focal Loss | **가중 제곱오차** `Σ (W·M+1)(σ(z) − H)² / 전경 수` |
| 조직 손실, 가중치 | CE + Dice, 1:1 | 같음 |
| 학습 설정 | 50 epoch, 고정 3, AdamW, lr 1e-5/1e-4, cosine, batch 2, clip 10, 기본 증강 | 같음 |
| 후처리 | `1 − P(배경)` smoothing → 지역 최댓값 → 배경 < 세포 | `max(P_BC, P_TC)` smoothing → 지역 최댓값 → **threshold 초과** |
| 평가 | 공식 15px 매칭 F1 (`m0_eval.CellF1`) | 같음 |

- 손실 가중치: 보고서는 "손실 가중치로 조정"이라고만 적었다. 그래서 Adaptive Wing Loss의 Weighted Loss Map을 따른다.
  - `M = (3×3 팽창한 H ≥ 0.2)`, `W = 10`
  - 팽창 덕분에 두 세포 사이 골짜기에도 큰 가중치가 걸린다.
  - 나누는 수는 M0 Focal과 같은 규칙이다: 전경 픽셀 수, 최소 전체의 0.5%.
- 후처리 threshold: 배경 채널이 없어 0보다 커야 한다.
  - 학습 중에는 0.3으로 고정한다. σ·λ 선정 스크립트의 골짜기 기준(중간점 점수 0.30 이하)과 맞춘 값이다.
  - 최종 후처리 값은 validation에서 격자 탐색으로 고른다. 격자는 σ 0/1/2/3, 간격 3/5/7, threshold 0.1~0.5이다.

## σ·λ 선정

```bash
python scripts/select_gaussian_params.py --repo . --scope classwise
```

보고서 정답 지도는 classwise이므로 `--scope classwise` 결과를 쓴다.
비교용 pooled(기본값) 결과도 함께 기록한다.
고른 값은 `configs/m2_train.json`의 `cell_target.sigma`, `cell_target.lambda`에 적는다.

실행 결과 (train 400쌍, 11/6):

| scope | 기본 기준 (골짜기 0.30) | 골짜기 0.20 | σ=6 Gaussian만으로 기준 통과 비율 |
|---|---|---|---|
| classwise | σ=6, **λ=0** | σ=6, λ=0.9 | 94.6% (BC-BC 94.2%, TC-TC 94.8%) |
| pooled (민규 명령 그대로) | σ=6, **λ=0** | σ=6, λ=0.9 | 94.7% |

- 규칙은 "M0 원 크기에 맞춘 σ, 그다음 가장 작은 λ"다.
- σ=6이면 14–30px 쌍의 94.6%가 Gaussian만으로 기준(중간점 0.30 이하)을 넘는다. train 14–30px 쌍의 95%가 18px 이상 떨어져 있기 때문이다.
- 그래서 λ=0이 뽑힌다. λ=0이면 H_AO = G가 되어 A·O가 적용되지 않는다.
- **1회 실험은 σ=6, λ=0.9로 정했다 (11/7).** 골짜기 0.2 기준의 결과이며 보고서 그림 10과 같은 λ다. λ=0(Gaussian)은 이후 세 조건 비교에서 따로 돌린다.
- 비교 그림: `outputs/gaussian_parameter_audit_classwise/lambda0_vs_0.9.png` (Git 제외, 신상우 맥북)

## 필요한 설정

- **경로:** M0와 같은 데이터·SSL 가중치·결과 경로를 쓴다 (`OCELOT_DATA_ROOT`, `SSL_WEIGHTS_PATH`, `OUTPUT_ROOT`).
- **사전 검사:** `check_data.py`가 만든 `outputs/audit/manifest.csv`, `report.json`이 있어야 한다.
- **패키지:** M0 환경에 더해 `scipy`가 필요하다 (정답 지도의 KD-tree). 민규 안내 명령 `python -m pip install numpy scipy matplotlib Pillow`에 들어 있다.

## 실행

```bash
python scripts/check_m2.py                     # 데이터 없이 점검 (CPU 약 20초)
python scripts/check_m2.py --data              # 실제 데이터로 정답 지도 점검 (config의 σ·λ)
python scripts/train_m2.py --smoke             # 짧은 동작 확인 (성능 의미 없음)
python scripts/train_m2.py --seed 42           # 본 학습 (VESSL GPU)
python scripts/train_m2.py --resume outputs/m2/<run>/last.pt
bash scripts/run_m2_conditions.sh             # 세 조건 학습 + 평가 한 번에 (VESSL Run용)
python scripts/eval_m2.py --checkpoint outputs/m2/<run>/best.pt --split val --tune \
  --compare results/m0_dicefix_seed42/eval_val_per_image.json
python scripts/eval_m2.py --checkpoint outputs/m2/<run>/best.pt --split test   # 설정 고정 후 1회
```

- 결과 폴더는 `OUTPUT_ROOT/m2/<run>`(기본 `outputs/m2/<run>`)이다.
- run 이름 기본값은 `M2_hao_s<σ>_l<λ>_official_seed42`이다.
- `--target-mode gaussian|ha|hao`로 세 조건을 바꿀 수 있다. 비교할 때는 σ·λ·나머지 설정을 같게 둔다.

## VESSL에서 세 조건 학습 (G, H_A, H_AO · σ=6, λ=0.9, seed 42)

`scripts/run_m2_conditions.sh`가 데이터 다운로드·검사 4개 → 조건별 학습 → `eval_m2.py --tune --compare`(M0 비교)를
순서대로 실행하고, 결과를 `/root/m2_results/<run 이름>/`에 모은다.

1. VESSL(app.vessl.ai, 조직 AI-LEC-03) → **Run → 새 Run 만들기**
2. 오른쪽 **YAML 에디터**에 `vessl/m2_run.yaml` 내용을 붙여 넣는다.
   - 클러스터 `yonsei-ai-gpu`, 리소스 `gpu-1`(RTX 3090), 이미지 `torch:2.3.1-cuda12.1-r5`
   - import: 이 저장소 `feature/m2-center-heatmap` → `/root/repo`
   - export: `/root/m2_results` → VESSL artifact
   - 명령: `bash scripts/run_m2_conditions.sh`
3. **만들기** → Run의 로그 탭에서 진행을 확인한다. 세 조건이면 약 7~8시간이다.

- Workspace(JupyterLab) 대신 Run을 쓰는 이유: Run은 서버에 직접 접속하지 않아 교외 네트워크에서도 실행된다. Workspace는 교외에서 JupyterLab 포트 접속이 막혔다(11/9 확인).
- YAML 명령에 여러 줄 스크립트를 직접 넣으면 VESSL 실행 파일 안에서 문법 오류가 난다. 그래서 명령은 한 줄로 두고 스크립트는 저장소 파일로 둔다.
- 조건별 결과 파일
  - `config.json`, `history.csv`, `log.txt`
  - `postprocess.json`, `eval_val.json`, `eval_val_per_image.json`, `close_cells_val.json`
  - `best.pt`
- test 평가(`--split test`)는 점검 후 설정을 고정하고 한 번만 한다.

직접 터미널에서 돌릴 때 (교내망 Workspace 등):

```bash
git clone -b feature/m2-center-heatmap https://github.com/jgyu480/CAS2250.01-00-Medical_AI.git
cd CAS2250.01-00-Medical_AI
OUT=outputs/m2_results nohup bash scripts/run_m2_conditions.sh > outputs/run_m2.log 2>&1 &
```

## 2차: G·GA 계수 탐색 (11/14~, 담당 신상우)

민규 지시(11/14): G와 GA를 앙상블에 쓰므로 둘 다 성능을 올린다. 계수는 **각 모델 단독 성능이 가장 높은 값**으로 고르고,
고를 때 **test 지표를 쓰지 않는다**. σ는 간격을 두고 처음부터 학습한 뒤, 가장 좋은 값 주변은 파인튜닝으로 본다.
Run 하나는 최대 24시간이고 동시에 여러 개 돌릴 수 있다.

### 데이터 분할 (`configs/train_split_seed42.json`)

공식 val은 최종 비교(앙상블 평가 포함), 공식 test는 마지막 1회에 남겨 두고, **공식 train 400장 안에서만** 나눈다.
처음 계획(train+val 537장을 다시 나눔)은 val 정답이 계수 선택에 쓰여, val로 하는 M2 결과와 앙상블 평가가 함께 낙관적으로 나올 수 있어 바꿨다.

| 부분 | 장수 | 환자 | 용도 |
|---|---|---|---|
| fit | 233 | 63 | 후보 학습 |
| select | 40 | 22 | checkpoint·후처리 선택 |
| holdout | 127 | 47 | 계수 비교 (가장 높은 holdout mF1을 고른다) |

- 같은 환자의 사진은 한 부분에만 넣었다 (train은 환자 132명, 한 환자에서 최대 18장).
- 장기 비율은 공식 train과 같게 맞췄다 (`scripts/make_train_split.py`, seed 42, 다시 실행하면 같은 파일).
- 앙상블을 273/127로 나눈다면 fit+select = 273, holdout = 127과 같은 구성이다 (팀 확인 필요).

### 탐색 단계 (`scripts/run_m2_search.py`)

1. **처음부터 학습:** σ = 4, 6, 8 (GA는 λ = 0.9). fit 학습 → select에서 checkpoint·후처리 → holdout 평가.
2. **파인튜닝:** holdout mF1이 가장 높은 σ*의 best.pt에서 10 epoch (학습률 ×0.1, 백본 고정 없음).
   - 대조군: σ* 그대로 10 epoch 더. overnight 실험(11/10)에서 σ를 바꾸지 않고 더 학습만 해도 mF1이 떨어졌기 때문에, 파인튜닝 후보는 대조군과 비교한다.
   - σ*−1, σ*+1, GA는 σ*에서 λ = 0.5도 본다.
3. **선택:** 모든 후보 중 holdout mF1이 가장 높은 것. 이 checkpoint(fit으로 학습)를 앙상블용으로 내보낸다.
4. **공식 확인:** 고른 방법 그대로 공식 train 400장으로 학습 → 공식 val에서 후처리 고르기·평가 (M0와 비교).
   파인튜닝 후보를 골랐으면 σ*로 50 epoch 학습한 뒤 같은 파인튜닝을 반복한다.

- 다음 후보가 끝나는 예상 시각이 22시간(공식 확인 시간을 남기고)을 넘으면 건너뛴다. Run이 24시간에 강제 종료되면 결과를 내보내지 못할 수 있어서다.
- 설정 `configs/m2_search_train.json`: 학습 설정은 `m2_train.json`과 같고, 학습 중 후처리(3/7/0.4)와 후처리 격자(σ 2–5, d 3–9, thr 0.2–0.5, 64개)만 1회 실험에서 고른 값이 격자 끝이던 것에 맞춰 옮겼다.

### 후보를 고른 이유

- **σ=6:** `select_gaussian_params.py`에서 반높이 반지름 약 7px 기준으로 나온 값이고 1회 실험 값이다.
- **σ=4:** 더 날카롭다. 14px 떨어진 같은 클래스 두 세포 사이 점수가 0.22(σ=6은 0.51)로 낮아 가까운 세포 분리에 유리하다.
- **σ=8:** 더 넓다 (같은 위치 0.68). 중심이 조금 어긋나도 너그럽고, 반응이 약한 세포를 살릴 수 있다.
- **λ:** 0.9(1회 실험, 보고서 그림 10)와 0.5(약한 억제). λ=0이면 GA는 G와 같다.

### 실행

VESSL에서 Run 두 개를 동시에 만든다 (YAML 에디터에 붙여 넣기).

- `vessl/m2_search_g.yaml` → `MODE=gaussian bash scripts/run_m2_search.sh` (약 10시간)
- `vessl/m2_search_ga.yaml` → `MODE=ha bash scripts/run_m2_search.sh` (약 10–11시간)
- 결과: `/root/m2_search_<mode>/summary.json`, `summary.csv`, `candidates/<후보>/`, `final/<run>/`
  - `best.pt`는 고른 후보(앙상블용)와 공식 확인 모델만 내보낸다.

직접 돌릴 때와 점검:

```bash
python scripts/make_train_split.py --parts fit=233 select=40 holdout=127 --seed 42 --out configs/train_split_seed42.json
python scripts/train_m2.py --config configs/m2_search_train.json --split-file configs/train_split_seed42.json \
  --train-part fit --val-part select --target-mode gaussian --sigma 4 --lambda 0.9
python scripts/eval_m2.py --checkpoint outputs/m2/<run>/best.pt --part select --tune
python scripts/eval_m2.py --checkpoint outputs/m2/<run>/best.pt --part holdout
python scripts/train_m2.py ... --init-from outputs/m2/<run>/best.pt --epochs 10 --freeze-epochs 0 --lr-scale 0.1
SKIP_PREP=1 SEARCH_EXTRA="--train-extra=--smoke" MODE=gaussian bash scripts/run_m2_search.sh   # 짧은 점검
```

- 분할로 학습한 모델은 공식 val에서 후처리를 고르지 않는다. `eval_m2.py`가 막는다 (`--split val`은 select에서 고른 값으로 평가만).
- 파인튜닝 옵션(`--init-from`, `--lr-scale`, `--freeze-epochs`)은 `train_m0.py`와 같은 이름이다. overnight 실험 코드(`results/m2-overnight-20261110-121336`)의 파인튜닝 방식(10 epoch, 학습률 ×0.1, 고정 없음)을 따랐다.

## 평가 항목 (보고서 8.4)

`eval_m2.py`는 M0와 같은 `eval_<split>.json`을 만들고, `close_cells_<split>.json`에 다음을 기록한다.

- **최근접 거리 구간별 누락률·오분류율**: 0–14 / 14–30 / 30+ px. 보고서 6.2와 같은 정의이고 `scripts/diagnose_m0.py`의 함수를 그대로 쓴다.
- **중복 검출·오검출 수**
- **놓친 세포의 원인 구분**
  - 피크 합침: 정답 위치 점수가 threshold보다 큰데 봉우리가 없음
  - 약한 반응: 정답 위치 점수가 threshold 이하
- `--compare`를 주면 M0 저장 결과도 같은 코드로 다시 계산해 나란히 기록한다.
- 위치 우선 매칭과 원인 구분은 진단용이며 공식 F1이 아니다.

## 확인한 결과

### 세 조건 1회 학습 (VESSL, σ=6, λ=0.9, seed 42, val 137쌍, 11/9~10)

- 결과 파일: `results/m2_s6_l0.9_seed42/{gaussian,ha,hao}/` (eval·진단·후처리·history), 실행 기록 `run_all.log` (코드 `9a2abf9`)
- 비교 요약: `results/m2_s6_l0.9_seed42/comparison.json` (아래 명령으로 다시 만든다)
- `best.pt`는 저장소에 올리지 않는다 (VESSL Run artifact에 있다).

```bash
python scripts/compare_m2.py --m0 results/m0_dicefix_seed42 \
  --run G=results/m2_s6_l0.9_seed42/gaussian --run H_A=results/m2_s6_l0.9_seed42/ha \
  --run H_AO=results/m2_s6_l0.9_seed42/hao \
  --metadata data/raw/ocelot2023_v1.0.1/metadata.json \
  --out results/m2_s6_l0.9_seed42/comparison.json
```

| | M0 | G | H_A | H_AO |
|---|---|---|---|---|
| best epoch | 36 | 17 | 17 | 16 |
| 후처리 σ / d / thr | 2 / 7 / 0 | 3 / 7 / 0.4 | 3 / 7 / 0.3 | 3 / 7 / 0.4 |
| mF1 | 0.7033 | 0.7330 | 0.7342 | 0.7324 |
| BC F1 (P / R) | 0.6272 (0.666 / 0.593) | 0.6662 (0.629 / 0.708) | 0.6672 (0.599 / 0.754) | 0.6701 (0.658 / 0.683) |
| TC F1 (P / R) | 0.7795 (0.845 / 0.724) | 0.7997 (0.824 / 0.777) | 0.8012 (0.780 / 0.824) | 0.7947 (0.776 / 0.815) |
| mIoU | 0.8214 | 0.8056 | 0.8157 | 0.8041 |
| 누락률 BC 14–30 / 30+ px | 36.7% / 26.3% | 26.2% / 14.1% | 17.4% / 6.9% | 25.2% / 12.3% |
| 누락률 TC 14–30 / 30+ px | 32.2% / 14.8% | 25.1% / 9.0% | 20.3% / 6.7% | 22.6% / 7.7% |
| 가까움−멂 누락률 차이 BC / TC | +10.3 / +17.4%p | +12.1 / +16.1%p | +10.5 / +13.6%p | +12.9 / +14.9%p |
| 오검출 / 중복 / 전체 예측 | 2480 / 87 / 21464 | 3776 / 153 / 24832 | 5576 / 183 / 27811 | 4362 / 165 / 25847 |

95% 구간은 환자/slide 46개 단위로 모든 run에 같은 표본을 뽑은 bootstrap 1000회다 (`comparison.json`의 `paired_bootstrap`).

- **정답을 중심 지도로 바꾼 효과(G)가 가장 크다.** G − M0: mF1 +0.0296 (+0.0115 ~ +0.0454), 14–30px 누락률 BC −10.4%p, TC −7.0%p.
- **A·O 억제(H_AO)는 같은 후처리에서 G와 mF1 차이가 없다.** H_AO − G: mF1 −0.0006 (−0.0127 ~ +0.0123).
  - TC는 14–30px 누락률 −2.5%p (−3.9 ~ −1.2), 가까움−멂 차이 −1.3%p (−2.3 ~ −0.4)로 조금 줄었다. 재현율은 오르고 정밀도는 내려가 TC F1은 그대로다.
  - BC는 14–30px 누락률 −1.1%p (−5.1 ~ +2.3)로 차이가 없다.
- **H_A의 낮은 누락률은 threshold 0.3 영향이 크다.** 격자 결과에서 같은 3/7/0.4로 보면 H_A는 mF1 0.7312, BC 재현율 0.671 (G 0.708, H_AO 0.683)이다. 3/7/0.3에서는 G의 BC 재현율이 0.775로 H_A(0.754)보다 높다. 거리 구간 진단은 각 run이 고른 후처리 기준이라 H_A만 thr이 다르다.
- **가까운 세포 누락은 줄었지만, 먼 세포와의 차이는 남아 있다.** 먼 세포 누락도 함께 줄어 가까움−멂 차이가 M0와 비슷하다 (TC 17.4 → G 16.1 / H_AO 14.9%p, BC 10.3 → 12.1 / 12.9%p).
- **남은 누락은 대부분 약한 반응이다.** 14–30px에서 놓친 세포 중 피크 합침은 BC 2.0% (G) / 6.9% (H_AO), TC 3.4% / 11.5%다.
  - 놓친 가까운 TC의 이웃은 G 80%, H_AO 87%가 검출되었다 (검출된 TC의 이웃은 71%, 72%).
  - 즉 두 봉우리가 합쳐지기보다 이웃만 반응하고 자기 위치 반응이 threshold보다 약한 형태가 많다.
- **오검출이 늘었다** (M0 2480 → G 3776, H_AO 4362). 약 2/3는 가장 가까운 정답이 30px보다 먼 예측이다.
- **학습 곡선**: val 세포 loss는 epoch 9–11에서 가장 낮고 이후 오른다 (train loss는 계속 감소). 학습 중 val mF1은 epoch 16–17에 가장 높고, 이후 0.62–0.71에서 오르내리다 마지막 10 epoch는 0.69–0.70이다 (M0도 마지막 10 epoch 0.69–0.70).
- **mIoU 차이는 정답 변경 효과로 보기 어렵다.** 조직 branch는 같고, 세포 mF1로 고른 epoch의 값이다. 학습 중 최고 mIoU는 0.832–0.834로 M0(0.834)와 비슷하다.
- 세 조건 모두 후처리 σ=3, d=7이 격자 끝 값이다. 격자를 넓히면 더 나을 수 있다.
- 주의: checkpoint와 후처리를 모두 val로 골랐고 seed가 1개다. bootstrap 구간에는 seed 편차가 들어 있지 않다.

### 이전 점검 (11/6)

- 실제 데이터 (신상우 맥북, 11/6):
  - `check_data.py`(zip MD5 일치)와 `check_dataset.py` 모두 PASS
  - `check_m2.py --data --sigma 6 --lambda 0.9` PASS: 증강 뒤 점 위치에서 해당 클래스 지도 값이 1이고, M0 원형 정답과 위치가 일치한다.
  - 사진 한 장 읽는 시간: M0 Dataset 0.55초, M2 0.68초. 정답 지도 계산이 약 0.1초 더 걸린다.
- M0 기준선과 같은 환경(Python 3.10, torch 2.3.1)에서 점검·학습·이어서 학습·후처리 선택·test 평가가 동작한다.
- `check_m2.py` 결과:
  - 정답 지도가 선정 스크립트의 식과 일치하고, 보고서 그림 10 숫자(0.084→0.203, 0.050→0.275)를 재현한다.
  - 뒤집기·회전 뒤 지도가 일치한다.
  - 손실 가중치가 골짜기를 포함한다.
  - 정답 지도로 만든 출력에서 12~14px 쌍까지 모든 중심을 찾는다.
  - 가중치 이름이 M0와 같고, 다른 것은 세포 head뿐이다.
- 가짜 데이터로 확인한 것:
  - 학습 → 이어서 학습 → val 후처리 선택 → test 보호장치 → 진단까지 끝까지 동작한다.
  - 사진 2장 과적합 시 train mF1 0.95에 도달한다.
- M0 저장 결과(`results/m0_dicefix_seed42`, val)를 같은 코드로 다시 계산한 누락률:
  - BC: 14–30px 36.7%, 30+ 26.3%
  - TC: 14–30px 32.2%, 30+ 14.8%
  - 보고서 6.2 수치와 0.1~1.2%p 차이가 난다. 보고서 계산에 쓴 후처리 값 차이로 보인다. M2와는 같은 코드로 비교한다.

## 남은 작업

- [x] `select_gaussian_params.py` 실행 (classwise·pooled 모두 σ=6, λ=0)
- [x] 1회 실험 σ=6, λ=0.9 확정 → `configs/m2_train.json` 기록
- [x] VESSL에서 seed 42로 G / H_A / H_AO 각 1회 학습 → M0와 비교 (위 결과, `scripts/compare_m2.py`)
- [ ] 민규 점검 후 방향 결정 (계속 / 수정)
- [ ] 후처리 격자 넓히기 (σ 4·5, d 9·11). 저장된 `best.pt`로 평가만 하면 된다.
- [ ] H_A를 G·H_AO와 같은 후처리(3/7/0.4)로 거리 구간 재진단 (`eval_m2.py`에 후처리 값 지정 옵션 필요)
- [ ] M0 확률지도에서 피크 합침과 약한 반응 구분 (보고서 8.4 첫 단계, M0 `best.pt` 필요)
- [ ] 방향을 정한 조건만 seed 43·44
- [ ] 설정 고정 후 test 1회

## 참고 자료

- Zhou et al., Objects as Points (CenterNet), https://arxiv.org/abs/1904.07850 — Gaussian 중심 지도, 최댓값 결합, 지역 최댓값 추출 (보고서 [6])
- Xie et al., Beyond Classification: Structured Regression for Robust Cell Detection (MICCAI 2015), https://pmc.ncbi.nlm.nih.gov/articles/PMC5226438 — 세포 중심 지도를 sigmoid 회귀 + 가중 제곱오차로 학습, threshold 이상 지역 최댓값
- Wang et al., Adaptive Wing Loss (ICCV 2019), https://arxiv.org/abs/1904.07399 — Weighted Loss Map (W=10, 3×3 팽창, 0.2)
- A·O 분리 계수와 λ는 보고서 8.4의 프로젝트 설계이며 출판된 방법이 아니다.
