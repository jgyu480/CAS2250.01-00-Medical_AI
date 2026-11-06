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

## 실행

```bash
python scripts/check_m2.py                     # 데이터 없이 점검 (CPU 약 15초)
python scripts/check_m2.py --data              # 실제 데이터로 정답 지도 점검
python scripts/train_m2.py --smoke             # 짧은 동작 확인 (성능 의미 없음)
python scripts/train_m2.py --seed 42           # 본 학습 (VESSL GPU)
python scripts/train_m2.py --resume outputs/m2/<run>/last.pt
python scripts/eval_m2.py --checkpoint outputs/m2/<run>/best.pt --split val --tune \
  --compare results/m0_dicefix_seed42/eval_val_per_image.json
python scripts/eval_m2.py --checkpoint outputs/m2/<run>/best.pt --split test   # 설정 고정 후 1회
```

- 결과 폴더는 `OUTPUT_ROOT/m2/<run>`(기본 `outputs/m2/<run>`)이다.
- run 이름 기본값은 `M2_hao_s<σ>_l<λ>_official_seed42`이다.
- `--target-mode gaussian|ha|hao`로 세 조건을 바꿀 수 있다. 비교할 때는 σ·λ·나머지 설정을 같게 둔다.

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

- [ ] `select_gaussian_params.py --scope classwise` 실행 → σ·λ 확정, `configs/m2_train.json` 기록
- [ ] VESSL에서 seed 42 1회 학습 → `eval_m2.py --tune --compare`로 M0와 비교
- [ ] 민규 점검 후 방향 결정 (계속 / 수정)
- [ ] M0 확률지도에서 피크 합침과 약한 반응 구분 (보고서 8.4 첫 단계, M0 `best.pt` 필요)
- [ ] 이후: Gaussian / H_A / H_AO 세 조건 비교, seed 43·44

## 참고 자료

- Zhou et al., Objects as Points (CenterNet), https://arxiv.org/abs/1904.07850 — Gaussian 중심 지도, 최댓값 결합, 지역 최댓값 추출 (보고서 [6])
- Xie et al., Beyond Classification: Structured Regression for Robust Cell Detection (MICCAI 2015), https://pmc.ncbi.nlm.nih.gov/articles/PMC5226438 — 세포 중심 지도를 sigmoid 회귀 + 가중 제곱오차로 학습, threshold 이상 지역 최댓값
- Wang et al., Adaptive Wing Loss (ICCV 2019), https://arxiv.org/abs/1904.07399 — Weighted Loss Map (W=10, 3×3 팽창, 0.2)
- A·O 분리 계수와 λ는 보고서 8.4의 프로젝트 설계이며 출판된 방법이 아니다.
