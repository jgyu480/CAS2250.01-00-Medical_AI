# M0 분기별 기울기 게이트 실험

## 방법

모델 구조와 추론 과정은 유지하고 학습 기울기의 결합 비율을 변경한다.
세포 손실은 Focal, 조직 손실은 CE+Dice이며 조직 UNK는 제외한다.

두 손실이 모두 영향을 주는 파라미터에 다음 기울기를 사용한다.

- 세포 분기: `2*(1-a_C)*g_C + 2*a_C*g_T`
- 조직 분기: `2*(1-a_T)*g_T + 2*a_T*g_C`

`g_T`에는 기존 조직 손실 가중치가 포함된다.
독립적인 sigmoid 게이트 두 개는 0.5에서 시작하므로 초기에는 기존 합과 같다.
한 손실만 영향을 주는 파라미터는 기존 기울기를 유지한다.
전달 모듈은 수신 분기에 소속시키며 고정된 파라미터는 제외한다.
출력하는 total loss는 기존 합이며 실제 업데이트에 분기별 가중치를 적용한다.

## 게이트 학습

현재 학습률과 gradient clipping을 사용하는 한 단계 SGD 임시 업데이트를
미분한다. 세포 게이트는 게이트용 세포 손실, 조직 게이트는 조직 손실을
각각 줄이는 방향으로 학습한다. 실제 모델 옵티마이저는 AdamW이다.
임시 업데이트는 Adam 모멘텀과 weight decay를 생략한 근사이다.

게이트는 Adam(lr=0.01), 기울기 norm 상한 1을 사용해 5배치마다 갱신한다.
게이트용 forward에서는 dropout을 끄고 복제한 버퍼를 사용한다.
전체 학습 이력을 미분하지 않으며 충돌 제거를 보장하는 방식은 아니다.

## 데이터와 대조군

공식 train에서 TCGA 환자 단위로 약 40쌍을 게이트용으로 분리한다.
TCGA 환자를 식별할 수 없으면 slide name 단위로 묶는다.
실제 분리 수와 ID는 gate_split.json에 저장한다.
게이트용 이미지는 증강하지 않는다. 공식 val/test는 게이트 학습에 쓰지 않는다.

학습 게이트와 0.5 고정 게이트를 같은 분할·SSL 초기값·seed에서 비교한다.
각 50에포크, 처음 3에포크 백본 고정, 배치 2, 기존 학습률 정책을 사용한다.
기존 train 400쌍 전체 M0 결과와의 차이는 게이트만의 효과로 해석하지 않는다.
단일 seed 결과는 탐색적 비교이다.

## 실행과 기록

- check_dual_gate.py: 합성 데이터 CPU 검사.
- check_dual_gate_gpu.py: 실제 데이터 1024px CUDA 검사.
- train_m0_dual_gate.py: 단일 조건 학습.
- run_dual_gate_suite.py: 학습 게이트 이후 고정 게이트를 순차 실행.
  기본은 dry-run이며 --execute로 실행하고 --resume으로 재개한다.

매 배치의 게이트·적용 가중치·게이트 기울기·메타 손실·데이터 ID는
dual_gate/batches.csv, 에포크 집계는 dual_gate/epochs.csv에 저장한다.
gradient_conflict에는 게이트 적용 전 두 손실의 기울기 통계를 저장한다.
원시 기울기 벡터와 게이트 적용 후 충돌 통계는 저장하지 않는다.

last.pt에는 모델·두 옵티마이저·게이트·scaler·난수 상태·설정을 저장한다.
best.pt는 validation Cell mF1으로 선택한다. 재개에는 last.pt를 사용한다.
두 실험이 끝나면 dual_gate_comparison_seed42.csv를 생성한다.
sum_epoch_seconds는 체크포인트 쓰기와 초기 준비 시간을 제외한다.

## 근거

- Auto-Lambda, TMLR 2022: https://shikun.io/projects/auto-lambda
- Learning to Reweight Examples, ICML 2018:
  https://proceedings.mlr.press/v80/ren18a.html

독립적인 분기별 게이트와 각자의 목표 손실은 위 원리의 실험적 확장이다.
논문의 완전한 재현이나 PCGrad보다 우수하다고 검증된 방법은 아니다.
