# M0 구현·실험 기록

M0는 OCELOT 논문(arXiv 2303.13110)의 Cell-Tissue Feature Sharing 구조를
Lunit SSL ResNet-50으로 구현한 기준 모델(Base Model)이다.
M1~M3는 이 결과를 기준으로 비교한다.

## 1. 추가·수정한 파일

공용 파일은 수정하지 않고 새 파일만 추가했다.
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
| scripts/train_m0.py | 새 파일 | 학습 |
| scripts/eval_m0.py | 새 파일 | validation 후처리 선택, test 평가 |
| scripts/analyze_m0.py | 새 파일 | 학습 곡선, 오류 유형, 예측 사례, 라벨 비교 |
| docs/m0_annotation_check.md | 새 파일 | 직접 라벨링 24쌍의 공식 라벨 일치도 |

## 2. 구조

| 위치 | 방향 | 방법 |
|---|---|---|
| ① encoder 뒤 (2048ch) | 세포→조직 | 세포 특징을 상자 크기로 평균 pooling → 조직 격자의 cell_box 위치에 배치, 밖은 0. coverage mask 1채널 추가 |
| ② ASPP 뒤 (256ch) | 조직→세포 | 조직 특징에서 cell_box 영역을 bilinear로 잘라 세포 특징 크기로 확대 |
| ③ decoder 뒤 (256ch) | 조직→세포 | ②와 같음 |

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
(persistent_workers=False). 이진호님이 저장한 증강 사진 96쌍은 확인용이며 학습에 쓰지 않았다.

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

## 9. 동작 확인

- check_m0.py: 좌표 정렬(crop 오차 < 1e-6, 배치·coverage, 뒤집기 일치), 출력 크기,
  gradient 경로(조직 손실 → 세포 백본, 세포 손실 → 조직 백본), BN 통계 고정,
  짧은 학습 후 손실 감소. 맥·클라우드 모두 PASS
- train_m0.py --smoke, 이어서 학습(--resume), DataLoader worker 확인
- 같은 2쌍 60스텝 반복 학습: TC F1 0 → 0.661, 조직 mIoU 0.51 → 0.88
- 본 학습 중 맥이 잠들었다 깨어난 epoch(44)도 정상적으로 이어졌다

## 10. 남은 것

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

---

M0 실험 담당: 임수빈
