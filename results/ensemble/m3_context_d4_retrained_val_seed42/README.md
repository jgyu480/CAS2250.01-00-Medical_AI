# OCELOT M3 공식 validation 비교

- G는 공식 train 내부 holdout으로 σ 6 → 5 파인튜닝을 선택하였다. holdout mF1 0.7247, 공식 val mF1 0.7339.
- GA는 같은 절차로 σ 6과 λ 0.9 → 0.5 파인튜닝을 선택하였다. holdout mF1 0.7241, 공식 val mF1 0.7337.
- 위 공식 val 점수는 원 탐색 기록의 단일 모델 점수이다. 새로 학습한 G·GA의 별도 val 결과는 `M2_training.csv`에 있다. 이 실행의 모든 앙상블은 새 G·GA와 기존 M0·M1·M1-1의 D4 TTA를 사용한다.
- `summary.csv`는 TTA 평균, 15채널 결합기, M3-1, M3-2의 `val_tune` 지표이다. `shared`는 TTA 평균으로 고른 동일 후처리, `per_method`는 각 방법의 validation 최적 후처리이다.
- `per_pair.csv`에는 세포 매칭 오류 및 14–30px 누락 집계가 있고, `validation_grid.csv`에는 후처리 탐색 전체 결과가 있다.
- `val_tune`은 후처리 설정 선택에도 사용하였다. 독립 test 점수는 이 실행에 포함되지 않는다.
- `run.json`에서 모델 경로와 SHA-256, validation fit/tune ID, M2 탐색 커밋을 확인할 수 있다.
