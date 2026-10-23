# M0 구현·실험 기록

M0는 OCELOT 논문(arXiv 2303.13110)의 Cell-Tissue Feature Sharing 구조를
Lunit SSL ResNet-50으로 구현한 기준 모델(Base Model)이다.
M1~M3는 이 결과를 기준으로 비교한다.

## 1. 추가·수정한 파일

공용 파일은 수정하지 않고 새 파일만 추가하였다.
예외: `models/M0/model.py`는 M0 구현 자리로 비워 둔 파일이라 내용을 채웠다.

| 파일 | 상태 | 역할 |
|---|---|---|
| models/M0/model.py | 내용 채움 | M0 조립, build_model |
| common/modules/exchange.py | 새 파일 | 좌표 맞춤 특징 전달 (crop / 배치 + coverage) |
| common/losses/m0_losses.py | 새 파일 | 세포 Focal, 조직 CE+Dice |
| common/metrics/m0_eval.py | 새 파일 | 세포 점 찾기, 세포 F1, 조직 mIoU |
| common/training/m0_loop.py | 새 파일 | 데이터 로더, 검증 loop |
| configs/m0_train.json | 새 파일 | 학습 설정 |
| scripts/check_m0.py | 새 파일 | 정렬·출력·손실·gradient·짧은 학습 점검 |
| scripts/train_m0.py | 새 파일 | 학습 (꺼보기용 옵션: --model-config, --init-from, --lr-scale, --freeze-epochs, --eval-init) |
| scripts/eval_m0.py | 새 파일 | validation 후처리 선택, test 평가 |
| scripts/analyze_m0.py | 새 파일 | 학습 곡선, 오류 유형, 예측 사례, 라벨 비교 |
| docs/m0_annotation_check.md | 새 파일 | 직접 라벨링 24쌍의 공식 라벨 일치도 |
| models/M0/ablation/*.json | 새 파일 | 전달 경로를 끈 M0 설정 4개 (9장) |
| scripts/summarize_m0_ablation.py | 새 파일 | 꺼보기 결과 요약표 |

## 2. 구조

| 위치 | 방향      | 방법                                                                            |
|---|---------|-------------------------------------------------------------------------------|
| ① encoder 뒤 (2048ch) | 세포 → 조직 | 세포 특징을 상자 크기로 평균 pooling → 조직 격자의 cell_box 위치에 배치, 밖은 0. coverage mask 1채널 추가 |
| ② ASPP 뒤 (256ch) | 조직 → 세포 | 조직 특징에서 cell_box 영역을 bilinear로 잘라 세포 특징 크기로 확대                                |
| ③ decoder 뒤 (256ch) | 조직 → 세포 | 조직 특징에서 cell_box 영역을 bilinear로 잘라 세포 특징 크기로 확대(②와 동일)                         |

위치·방향은 models/M0/config.json(팀 설정)을 그대로 읽는다. "none"으로 끌 수 있다.
세 곳을 모두 끄면 두 백본이 각자 학습하는 독립 모델이 된다.

전달 블록: 보내는 특징 → 3×3 conv(256ch, GroupNorm, ReLU) → 위치 맞춤
→ 받는 특징과 채널 결합 → 1×1 conv(GroupNorm)로 원래 채널 → 받는 특징에 더함.

- 3×3 conv 후 채널 결합: OCELOT 논문 Fig. 5.
- 축소+zero padding(세포→조직), 잘라서 확대(조직→세포): 논문 Fig. 5.
- 결과를 더하는 방식(residual)과 coverage mask: 우리 구현 선택.
  SSL 백본 특징을 유지하고, 조직 갈래가 세포 정보가 있는 영역을 알 수 있게 한다.
- 좌표 계산: affine_grid + grid_sample, align_corners=False.
  cell_box가 픽셀 경계 기준 0~1 좌표이므로 이 설정이 맞다.
- 논문 Fig. 6의 화살표 방향과 완전히 같은지는 그림으로 별도 확인이 필요하다.
  README에 적힌 대로 원 논문을 그대로 재현하는 실험은 아니다.

## 3. 손실

| 작업 | 손실 | 설정 |
|---|---|---|
| 세포 | Focal Loss | γ=2, 세포 원 안 픽셀 수로 정규화 (RetinaNet 방식), 분모 하한 0.5% |
| 조직 | CE + Dice | UNK(2) 제외, Dice는 BG·CA 평균 |
| 전체 | cell + 1.0·tissue | 실험계획서 λ=1.0 |

Focal Loss: https://arxiv.org/abs/1708.02002

### 손실 안정화 (1차 학습에서 발견한 문제)

- 증상: 1차 학습 17 epoch에서 세포 손실이 0.93 → 2.60으로 튀고 val mF1이 0.05로 떨어졌다.
- 원인: 세포 손실을 "세포 원 안 픽셀 수"로 나누는데, train에는 세포 0개 사진이 5장,
  10개 미만 사진이 25장 있다. 이런 사진끼리 한 batch가 되면 분모가 거의 0이 된다.
  같은 출력에서 계산한 세포 손실: 세포 많은 batch 2.9 / 세포 1개 184 / 세포 0개 1,632.
- 수정
  - 분모 하한: 전체 픽셀의 0.5% (평균적인 사진은 세포 원이 약 2%라 평소에는 영향 없음).
    수정 후 세포 1개·0개 batch의 손실은 0.16.
  - gradient norm clipping 10 (안전장치).
- 1차 결과는 지우지 않고 outputs/m0/M0_official_default_seed42_v1_loss_spike에 보관했다.

## 4. 평가 (common/metrics/m0_eval.py)

| 항목 | 방법 | 근거 |
|---|---|---|
| 세포 점 찾기 | 1-P(배경)을 Gaussian(σ) → 반경 min_distance 지역 최댓값 → 배경 확률이 BC·TC 최대 확률보다 낮은 곳만 → 클래스 = argmax(BC, TC), 점수 = 그 확률 | 공식 예제 ocelot23algo/user/unet_example (peak_local_max를 max pooling으로 구현) |
| 세포 F1 | 클래스별 점수 높은 순으로 15px(약 3µm) 안 가장 가까운 정답과 1:1 매칭, 전체 이미지 TP/FP 합산, BC·TC F1 평균 | 공식 ocelot23algo/evaluation/eval.py. 무작위 300회 대조에서 결과 동일 확인 |
| 조직 mIoU | UNK 제외, BG·CA IoU를 데이터 전체 픽셀 합으로 계산 후 평균 | 우리 선택 (공식 챌린지는 조직 평가 없음) |

공식 저장소에 라이선스 파일이 없어 코드를 복사하지 않고 같은 규칙으로 다시 구현했다.
정답 지도로 만든 완벽한 예측을 넣으면 mF1 = 1.0이 나오는 것도 확인했다 (val 10쌍).

## 5. 학습 설정 (scripts/train_m0.py, configs/m0_train.json)

| 항목 | 값 | 근거 |
|---|---|---|
| 백본 | Lunit BT ResNet-50 (bt_rn50_ep200.torch), 세포·조직 각각 | 팀 설정, SHA256 확인 |
| 데이터 | train 400 / val 137 / test 126 (OCELOT v1.0.1) | 공식 split |
| 학습 라벨 | 공식 라벨 400쌍 | 라벨 품질 비교의 기준선 |
| 입력 | 1024×1024, Lunit 공식 mean/std 정규화 | 팀 설정 |
| 증강 | 뒤집기·90도 회전 + RandStainNA + Color Jitter (`default`), 매 epoch 실시간 적용 | 실험계획서, 팀원 구현 |
| 옵티마이저 | AdamW, encoder 1e-5 / 새 부품 1e-4, weight decay 0.01 | 팀 설정 (weight decay는 PyTorch 기본값) |
| 스케줄 | Cosine, warmup 없음 | 실험계획서 |
| 학습 길이 | 50 epoch, 처음 3 epoch 백본 고정 (BN 통계는 항상 고정) | 팀 설정 |
| batch | 2 (accumulation 없음) | 미지정 항목. 맥 메모리 기준 |
| seed | 42 | 팀 설정 seeds 42/43/44 중 첫 번째 |
| checkpoint 선택 | validation 세포 mF1 최고 epoch | 팀 설정 |
| 기기 | MacBook (Apple MPS), 혼합 정밀도 미사용 | |
| 시간 | 백본 고정 epoch 약 4.4분, 이후 약 7분, 전체 약 5.5시간 | |

증강은 Dataset.set_epoch로 epoch마다 바뀐다. DataLoader worker는 epoch마다 새로 만든다
(persistent_workers=False). 
*이진호님이 저장하신 증강 사진 96쌍은 확인용이며 학습에 사용하지 않음.

## 6. 평가 절차 (scripts/eval_m0.py)

1. `--split val --tune`: best.pt로 validation을 한 번 추론하고 후처리 격자
   (σ 2/3/4 × d 3/5/7 × thr 0~0.5, 54개) 중 val mF1 최고값을 postprocess.json에 저장
2. `--split test`: postprocess.json이 있어야 실행된다. test로 설정을 고르지 않는다.
3. 결과: eval_<split>.json (전체 + 장기별), eval_<split>_per_image.json (이미지별)

## 7. 결과 (공식 라벨, seed 42)

### 학습 경과

- best epoch 36, val mF1 0.7017 (학습 중 고정 후처리 σ=3, d=3)
- 26 epoch 이후 val mF1 0.66~0.70에서 안정, 마지막 10 epoch 0.684~0.690
- train 손실 2.61 → 0.80으로 꾸준히 감소. 수정 후에는 손실 폭발 없음
- val 손실은 9 epoch(1.50)부터 50 epoch(2.09)까지 상승, mF1은 유지
  → 틀린 예측에 대한 확신이 커지는 형태의 과적합. checkpoint는 mF1로 골라 영향을 줄였다.

### 후처리 선택 (validation)

- 선택: σ=2, min_distance=5, threshold=0 → val mF1 0.7026
- 54개 조합의 차이는 대부분 0.001 이내 (σ=3: 0.7016~0.7018, σ=4: 0.6992~0.6993)
- threshold 0~0.4는 결과가 같았다. 검출 조건(배경 확률 < 세포 클래스 확률) 때문에
  검출 위치의 세포 확률이 이미 약 0.5 이상이기 때문이다. threshold 0.5에서는 하락.

### 최종 점수

| | validation (137쌍) | test (126쌍) |
|---|---|---|
| 세포 mF1 | 0.7026 | **0.6893** |
| BC F1 (정밀도 / 재현율) | 0.6280 (0.667 / 0.593) | 0.6340 (0.737 / 0.557) |
| TC F1 (정밀도 / 재현율) | 0.7772 (0.837 / 0.726) | 0.7446 (0.774 / 0.717) |
| 조직 mIoU | 0.8271 | **0.8672** |
| BG / CA IoU | 0.856 / 0.798 | 0.876 / 0.859 |

### 장기별 세포 mF1

| 장기 | val (쌍) | test (쌍) |
|---|---|---|
| bladder | 0.6230 (29) | 0.6427 (24) |
| endometrium | 0.7263 (29) | 0.6863 (23) |
| head-and-neck | 0.6202 (9) | 0.7027 (10) |
| kidney | 0.7189 (41) | 0.7280 (41) |
| prostate | 0.6392 (17) | 0.6303 (16) |
| stomach | 0.7326 (12) | 0.7188 (12) |

head-and-neck은 쌍 수가 적어(9~10) val과 test 차이가 크다.

### 논문과 참고 비교 (OCELOT test)

| 모델 | 세포 mF1 | 조직 mIoU |
|---|---|---|
| 논문 Cell-only | 64.44 | – |
| 논문 Feature-sharing | 68.91 | 81.97 |
| 논문 Pred-to-inter-2 (논문 최고) | 71.23 | – |
| M0 (seed 42) | 68.93 | 86.72 |

직접 비교의 한계: 논문은 test 130쌍(v1.0.0)·5회 평균, ResNet-34·Dice 손실.
M0는 126쌍(v1.0.1)·1회, SSL ResNet-50·Focal/CE+Dice.

## 8. 오류 분석 (test, scripts/analyze_m0.py errors)

클래스를 무시하고 위치로 먼저 매칭해 오류 유형을 나눴다 (공식 F1과 별도의 보조 분석).

| 맞춤 | 누락 | 오분류 | 오검출 (근처 정답 없음) | 중복 |
|---|---|---|---|---|
| 64.8% | 25.2% | 10.0% | 2,286개 | 75개 |

(비율은 정답 세포 22,477개 기준)

- 가장 큰 오류는 누락이다. 두 클래스 모두 재현율이 정밀도보다 낮다.
- 세포가 빽빽할수록 많이 놓친다. 예측 수 ÷ 정답 수: 50~150개 0.93,
  150~300개 0.87, 300개 이상 0.74.
- 오분류는 BC→TC(1,487)가 TC→BC(756)의 2배다. endometrium은 407 대 38.
  암 영역 안의 세포를 모두 종양으로 부르는 경향일 수 있다.
- prostate 634·635·636번은 정답 19~94개 중 0~2개만 찾았다. 사진 전체를 놓친 사례.
- 장기별: endometrium·prostate는 누락(31~33%), bladder는 오분류(13.9%)가 가장 많다.
- 정답이 0~2개인 사진은 이미지 단위 F1이 0으로 계산되지만 실제 오류는 거의 없다.

## 9. 추가 실험: 전달 경로 끄기 (validation)

### 한눈에 보기

M0는 세포 사진과 함께 넓은 조직 사진(어디가 암 영역인지 보여주는 "지도")을 같이 본다.
세포와 조직 사이의 연결을 하나씩, 또는 두 개를 함께 끊어 보고 성능이 어떻게 변하는지 확인했다.

- **세포→조직 연결(①)은 없어도 된다.** 끊어도 성능이 떨어지지 않는다.
- **지도(조직 정보)는 꼭 필요하다.** 조직→세포 연결(②③)을 모두 끊으면 세포를 훨씬 많이 놓친다.
- **암 영역에서는 지도에 너무 기댄다.** "암 영역이면 종양세포"라고 판단해서
  암 영역의 정상세포(BC)를 종양세포(TC)로 틀린다. → M1-1 Gate의 근거
- **비암 영역의 종양세포(TC) 오분류는 지도 탓이 아니다.** 지도를 끊어도 그대로다.
  → Gate로는 고치기 어렵다

### 9.1 왜 했나

M0의 세포 branch와 조직 branch는 세 곳에서 정보를 주고받는다(2장 표).

| 번호 | 위치 | 방향 | 쉽게 말하면 |
|---|---|---|---|
| ① | Encoder 뒤 | 세포 → 조직 | 조직을 판단할 때 확대된 세포 사진의 정보를 참고 |
| ② | ASPP 뒤 | 조직 → 세포 | 세포를 판단할 때 주변 조직(지도) 정보를 참고 |
| ③ | Decoder 뒤 | 조직 → 세포 | 최종 판단 직전에 지도를 한 번 더 참고 |

궁금했던 것은 두 가지다.

1. **세 연결이 정말 다 필요한가?** 팀 설계는 셋 다 켜는 것이지만, 실제로 도움이 되는지는 확인하지 않았다.
2. **세포 오분류가 지도 때문인가?** 보고서 7.3절에서 BC는 암 영역(CA)에서, TC는 비암 영역(BG)에서
   반대 클래스로 자주 틀렸다. 모델이 세포 자체보다 "주변이 암 영역이면 종양세포"처럼
   지도만 보고 판단했기 때문일 수 있다. 일반 영상에서도 모델이 주변 맥락에 기대다가
   맥락이 평소와 다를 때 틀리는 현상(contextual bias)이 보고되어 있다
   (Singh et al., CVPR 2020).

### 9.2 어떻게 했나

연결을 끈 설정 파일(models/M0/ablation/)을 만들고, 이미 학습된 M0(best.pt, epoch 36)에서 시작해
두 번 평가했다.

1. **끄자마자 평가:** 다시 학습하지 않고 연결만 끊는다. 지금 모델이 그 연결에 얼마나 기대고 있는지 보여준다.
2. **10 epoch 더 학습한 뒤 평가 (fine-tuning):** 끊은 상태로 더 학습시킨다. 연결 없이도 적응해서
   회복할 수 있는지 보여준다. 학습률은 원래의 1/10이고 나머지 설정(데이터, 증강, 손실, seed 42)은 M0와 같다.

공정한 비교를 위해 두 가지를 맞췄다.

- **기준도 똑같이 10 epoch 더 학습했다.** 더 학습하는 것만으로도 점수가 변하기 때문이다
  (실제로 0.7017 → 0.6847). 그래서 비교 대상은 원래 best.pt가 아니라 "모두 켜고 똑같이 더 학습한 모델"이다.
- **마지막 3 epoch 점수의 평균을 썼다.** 한 epoch 점수는 들쭉날쭉해서다.

조건은 5개다: 기준(모두 켬), ①만 끔, ②만 끔, ③만 끔, ②+③ 모두 끔.
②+③을 함께 끈 이유: ②만 꺼도 ③으로 조직 정보가 계속 들어간다.
세포 쪽에 조직 정보가 전혀 안 들어가게 하려면 둘 다 꺼야 한다.

### 9.3 결과 1: 점수

| 설정 파일 (models/M0/ablation/) | 끈 연결 | 끄자마자 mF1 | 10 epoch 후 mF1 | Δ mF1 | 조직 mIoU |
|---|---|---|---|---|---|
| (models/M0/config.json) | 없음 (기준) | 0.7017 | 0.6847 | – | 0.8280 |
| no_encoder_c2t.json | ① 세포→조직 | 0.6913 | 0.6975 | +0.013 | 0.8300 |
| no_aspp_t2c.json | ② 조직→세포 (ASPP 뒤) | 0.5742 | 0.6721 | −0.013 | 0.8263 |
| no_decoder_t2c.json | ③ 조직→세포 (Decoder 뒤) | 0.6666 | 0.6716 | −0.013 | 0.8283 |
| no_t2c_both.json | ②+③ 조직→세포 모두 | 0.5150 | 0.6433 | **−0.041** | 0.8267 |

**읽는 법:** Δ mF1은 기준과의 차이다. 음수면 껐더니 나빠졌다는 뜻이고, 그 연결이 도움이 됐다는 뜻이다.
seed를 바꾸면 점수가 0.01 정도 흔들리므로, ±0.01 안팎의 차이는 "거의 차이 없음"으로 본다.

**쉽게 말하면**
- **①:** 꺼도 나빠지지 않았다(+0.013). 끄자마자 조직 점수가 0.827 → 0.785로 떨어졌지만
  1 epoch 만에 원래대로 돌아왔다. → 필요 없는 연결이다.
- **②, ③:** 하나씩 끄면 각각 −0.013으로 조금 나빠진다. 이것만으로는 확실하다고 말하기 어렵다.
- **②+③:** 함께 끄면 −0.041로 확실히 나빠진다. 하나씩 끌 때 조금만 떨어진 건,
  남은 하나가 조직 정보를 대신 전달해 줬기 때문이다.
- ②③을 꺼도 조직 점수는 그대로다. 이 연결은 세포 쪽으로만 정보를 보내므로 당연하다.

### 9.4 결과 2: 어떤 실수가 늘고 줄었나 (②+③ 모두 끔)

점수만 봐서는 왜 떨어졌는지 알 수 없다. 그래서 실수를 두 종류로 나눴다.

- **누락률:** 정답 세포를 아예 못 찾은 비율
- **오분류율:** 찾은 세포 중 종류(BC/TC)를 틀린 비율.
  찾은 세포 수로 나눴다. 정답 전체로 나누면 누락이 늘어날 때 오분류가 저절로 줄어 보여서다.

그리고 세포를 "종류 × 그 세포가 놓인 조직 영역"으로 나눴다.
BC가 BG에, TC가 CA에 있는 경우가 흔한 경우(지도와 맞음)이고,
BC가 CA에, TC가 BG에 있는 경우가 드문 경우(지도와 어긋남)다. 7.3절에서 많이 틀린 곳이 어긋난 경우다.

| 정답 세포 / 놓인 영역 | 지도와 | 누락률 (켬 → 끔) | 오분류율 (켬 → 끔) |
|---|---|---|---|
| BC / BG | 맞음 | 29.9% → 36.8% | 13.8% → 12.1% |
| TC / CA | 맞음 | 23.2% → 30.5% | 7.7% → **9.8%** (늘어남) |
| BC / CA | 어긋남 | 29.1% → 38.3% | 41.1% → **32.6%** (줄어듦) |
| TC / BG | 어긋남 | 20.7% → 30.2% | 48.4% → 49.2% (그대로) |

(validation 137쌍, 마지막 epoch 모델, 후처리는 M0와 같은 σ=2, d=5. scripts/diagnose_m0.py cohorts로 계산)

**쉽게 말하면**
1. **누락은 모든 줄에서 7~10%p 늘었다.** 조직 정보의 가장 큰 역할은 세포를 "찾는" 것이다.
   9.3에서 점수가 떨어진 주된 이유가 이것이다.
2. **암 영역(CA)에서는 지도가 판단을 한쪽으로 밀고 있었다.** 지도를 끊으니 CA 안의 BC는 덜 틀리고
   (41.1% → 32.6%), CA 안의 진짜 TC는 조금 더 틀렸다(7.7% → 9.8%).
   즉 지도가 "암 영역이면 TC"라고 밀어서, 진짜 TC에는 도움이 되지만 같은 영역의 BC에는 오답을 만들었다.
   9.1의 질문 2에 대한 답은 **암 영역에서는 "그렇다"**이다.
3. **비암 영역(BG)의 TC는 지도를 끊어도 똑같이 틀린다**(48.4% → 49.2%).
   이 실수는 지도 탓이 아니다. 세포 모양만으로 구별하기 어렵거나, 정답 라벨 자체가 애매한 경우로 보인다.
   질문 2에 대한 답은 **비암 영역에서는 "아니다"**이다.

### 9.5 그래서 다음 모델에서는

- ① 세포→조직 연결은 빼거나 단순화해도 된다.
- 조직 정보는 빼면 안 된다. 대신 위치마다 **조직 정보를 얼마나 믿을지 조절**해야 한다. → M1-1 Gate
- 단, Gate로는 비암 영역의 TC 오분류를 고치기 어렵다. 세포 자체의 특징을 개선하거나 라벨을 재검토하는
  다른 방법이 필요하다.

### 9.6 한계

- seed 1회 결과다. 하나씩 끈 차이(±0.013)는 seed 변동(약 0.01)과 비슷한 크기라 경향으로만 해석한다.
  ②+③의 −0.041은 변동보다 충분히 크다.
- 처음부터 학습한 모델이 아니라 학습된 M0를 10 epoch 더 학습한 결과다.
- 세포 사진에도 세포 주변 조직이 조금 보인다. 그래서 ②③을 모두 꺼도 주변 정보가 완전히 사라지지는 않는다.
- 조직 Dice 문제(11장)를 고치기 전 M0 기준이다. 모든 조건이 같은 손실을 써서 조건끼리의 비교는 유효하지만,
  조직 학습과 직접 관련된 ①의 결론은 Dice 수정 후 다시 확인한다.

## 10. 동작 확인

- check_m0.py: 좌표 정렬(crop 오차 < 1e-6, 배치·coverage, 뒤집기 일치), 출력 크기,
  gradient 경로(조직 손실 → 세포 백본, 세포 손실 → 조직 백본), BN 통계 고정,
  짧은 학습 후 손실 감소. 맥·클라우드 모두 PASS
- train_m0.py --smoke, 이어서 학습(--resume), DataLoader worker 확인
- 같은 2쌍 60스텝 반복 학습: TC F1 0 → 0.661, 조직 mIoU 0.51 → 0.88
- 본 학습 중 맥이 잠들었다 깨어난 epoch(44)도 정상적으로 이어졌다

## 11. 남은 것

- 조직 Dice 수정: `common/losses/segmentation.py`의 dice_loss는 정답에 없는 클래스(CA가 없는
  사진, train 84/400장)도 평균에 넣어 손실이 0.5 아래로 내려가지 않는다. 정답에 있는 클래스만
  평균하도록 새 파일로 고친 뒤 M0를 다시 학습하고 9장 실험을 반복한다. 공용 파일은 그대로 둔다.
- seed 43, 44 반복 → 평균 ± 편차
- 라벨 품질 비교: 신상우 24쌍 + 공식, 이진호 24쌍 + 공식으로 학습해 공식 라벨 결과와 비교
  (라벨 일치도는 docs/m0_annotation_check.md)
- M1~M3와 함께 쓸 메인 학습 라벨 결정 (팀)
- 결과 파일 공유 위치 결정 (outputs/는 Git 제외)

## 실행 명령

```
python scripts/check_m0.py
caffeinate -i python scripts/train_m0.py --seed 42 --num-workers 4
python scripts/eval_m0.py --checkpoint outputs/m0/M0_official_default_seed42/best.pt --split val --tune
python scripts/eval_m0.py --checkpoint outputs/m0/M0_official_default_seed42/best.pt --split test
python scripts/analyze_m0.py errors --run outputs/m0/M0_official_default_seed42 --split test
```

9장 꺼보기 (경로 하나당 맥 MPS 약 70분)

```
COMMON="--init-from outputs/m0/M0_official_default_seed42/best.pt --lr-scale 0.1 --freeze-epochs 0 --eval-init --epochs 10 --seed 42 --num-workers 4"
python scripts/train_m0.py $COMMON --run-name ablation_all_on
for name in no_encoder_c2t no_aspp_t2c no_decoder_t2c no_t2c_both; do
  python scripts/train_m0.py $COMMON --model-config models/M0/ablation/$name.json --run-name ablation_$name
done
python scripts/summarize_m0_ablation.py outputs/m0/ablation_all_on outputs/m0/ablation_no_encoder_c2t outputs/m0/ablation_no_aspp_t2c outputs/m0/ablation_no_decoder_t2c outputs/m0/ablation_no_t2c_both

# 오류 분석 (각 run마다)
cp outputs/m0/M0_official_default_seed42/postprocess.json outputs/m0/ablation_<run>/
python scripts/eval_m0.py --checkpoint outputs/m0/ablation_<run>/last.pt --split val
python scripts/diagnose_m0.py cohorts --run outputs/m0/ablation_<run> --records outputs/m0/ablation_<run>/eval_val_per_image.json
```

---

M0 실험 담당: 임수빈
