# OCELOT 앙상블 비교 자료

| 자료 | 용도 |
|---|---|
| ensemble_comparison.csv | 기존 평균·D4 TTA·픽셀 게이트, M3 및 ASPP 비교 |
| validation_selected_test.csv | 각 실험에서 validation으로 선택한 방식의 공식 test 점수 |
| M2_training.csv | 새 G·GA 단독 공식 validation 점수 |
| official_test_*/frozen_before_test.json | test 이전에 고정한 선택 기준, 후처리, 체크포인트 해시 |
| official_test_m3/tta_leave_one_out.csv | 단독 TTA와 모델 한 개씩 제외한 D4 비교 |
| official_test_m3/tta_error_overlap.csv | 각 모델 쌍의 누락 세포 교집합·합집합 |
| official_test_m3/per_pair.csv | M3 TP·FP·FN, 근접 누락 및 조직 영역별 오류 |
| official_test_legacy/legacy_val/summary.csv | 기존 단독 모델·평균·픽셀 MLP·ASPP validation 비교 |

historical은 기존 D4 실험의 구현과 원래 validation 분할을 재사용했다. 원래 픽셀 게이트에는 증강하지 않은 모델 확률맵을 사용했다. 각 실험의 test 점수 산출에는 validation에서 미리 고른 후처리만 사용했다. 보고서에서는 validation-selected 행을 대표 수치로 사용한다. Leave-one-out은 재사용한 모델별 D4 캐시(FP16)를 사용하고, 모든 행에 전체 TTA의 validation 후처리를 적용했다.
