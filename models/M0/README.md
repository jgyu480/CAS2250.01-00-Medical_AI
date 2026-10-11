# M0

좌표를 맞춘 양방향 특징 전달을 사용하는 기준 모델이다.
Lunit SSL ResNet-50 백본으로 세포 검출·분류와 조직 분할을 함께 학습한다.

## 구조

- Encoder 뒤: 세포 특징을 조직 좌표에 맞춰 축소·배치한다.
- ASPP 뒤: 조직 특징에서 세포 FoV를 잘라 세포 갈래에 전달한다.
- Decoder 뒤: 조직 특징을 동일한 좌표로 다시 세포 갈래에 전달한다.
- 세포 손실은 Focal Loss, 조직 손실은 UNK를 제외한 Cross-Entropy + Dice를 사용한다.
- 전체 손실은 `L_cell + 1.0 * L_tissue`이다.

## 관련 파일

- `config.json`: M0 특징 전달 위치와 방식.
- `model.py`: M0 모델 조립.
- `../../common/modules/exchange.py`: 좌표 기반 특징 전달.
- `../../common/losses/m0_losses.py`: 세포·조직 손실.
- `../../common/metrics/m0_eval.py`: 세포 F1·조직 mIoU.
- `../../common/training/m0_loop.py`: 데이터 로더와 공통 평가 loop.
- `../../configs/m0_train.json`: 학습 설정.
- `../../scripts/check_m0.py`: 정렬·출력·gradient·짧은 학습 점검.
- `../../scripts/train_m0.py`: 학습·checkpoint 저장.
- `../../scripts/eval_m0.py`: validation 후처리 선택과 test 평가.
- `../../scripts/analyze_m0.py`: 학습 곡선·오류·예측 사례·작성자별 라벨 비교.

## 실행

```bash
python scripts/check_m0.py
caffeinate -i python scripts/train_m0.py --seed 42 --num-workers 4
python scripts/eval_m0.py \
  --checkpoint outputs/m0/M0_official_default_seed42/best.pt \
  --split val --tune
python scripts/eval_m0.py \
  --checkpoint outputs/m0/M0_official_default_seed42/best.pt \
  --split test
python scripts/analyze_m0.py errors \
  --run outputs/m0/M0_official_default_seed42 --split test
```

`data/`, `weights/`, `outputs/`는 Git에 올리지 않는다. 재현에 필요한 경로는
루트 `README.md`의 경로 설정과 `configs/m0_train.json`을 따른다.

## Seed 42 기준선 결과

- 학습: 공식 train 400쌍, 기본 증강, 1024×1024, 50 epoch.
- best checkpoint: 36 epoch의 validation 세포 mF1 기준.
- validation: 세포 mF1 0.7026, 조직 mIoU 0.8271.
- test 126쌍: 세포 mF1 0.6893, 조직 mIoU 0.8672.
- 후처리는 validation에서 선택한 후 test에 고정해 적용했다.

세부 구현 근거, 학습 경과, 장기별 점수와 오류 분석은
[`../../docs/m0_implementation.md`](../../docs/m0_implementation.md)에 기록한다.

## 남은 작업

- seed 43·44 반복 학습 및 평균±표준편차 계산.
- 두 작성자의 직접 라벨과 공식 라벨을 사용한 품질 비교.
- 증강 조건별 비교 실험.
- Git에서 제외된 checkpoint·로그·예측 결과의 팀 공유 위치 결정.
