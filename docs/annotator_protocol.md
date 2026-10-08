# 두 작성자 독립 어노테이션 (이진호·신상우, ~10/8)

## 목적

같은 24쌍을 두 사람이 서로의 결과를 보지 않고 각각 라벨링한다.
세포 위치·누락·클래스, 조직 경계·UNK 지정의 차이를 비교해
의료 라벨링의 주관성과 모호성을 확인한다(수업 학습 목표 03).

기존 draft/reviewed 구조(작성 → 다른 사람이 검토)는 그대로 두었다.
이번 작업은 "검토"가 아니라 "독립 작성 두 벌"이므로 작성자별 폴더를 추가했다.

## 저장 구조

```
annotations/
  selection.csv                 대상 24쌍 (기존, 변경 없음)
  annotator_tracking.csv        샘플 × 작성자 작업 기록 (48줄)
  manual/annotators/
    jinho/   cell/<ID>.csv  tissue/<ID>.png  notes/<ID>.csv
    sangwoo/ cell/<ID>.csv  tissue/<ID>.png  notes/<ID>.csv
```

- 세포 CSV: 헤더 없음, `x,y,label`, BC=1, TC=2, cell.jpg 픽셀 좌표(0~1023).
- 조직 PNG: 1024×1024 단일 채널, BG=1, CA=2, UNK=255. 공식 정답과 같은 형식.
- 메모 CSV: 헤더 `x,y,task,comment`. 판단이 어려웠던 위치와 이유.
- 작성자 ID는 `configs/annotation.json`의 `annotators`에 있다(jinho=이진호, sangwoo=신상우).

## 작업 순서

1. 처음 한 번: `python scripts/prepare_annotation.py` (작업 사진 준비, 기존)
   → `python scripts/prepare_annotators.py` (작성자 폴더·기록 생성)
2. `tools/annotator/index.html`을 Chrome/Edge로 연다(더블클릭).
3. 작성자 칸에 본인 ID 입력 → "작업 폴더 열기"로 `outputs/annotation_packets/<ID>` 선택.
4. 세포 탭: 모든 세포핵 중심에 점. 기본 TC, `1`=BC, `2`=TC.
   왼쪽 "조직 맥락"의 빨간 상자로 이 세포 사진이 조직 어디에 있는지 확인하며 분류한다.
5. 조직 탭: 시작은 전부 BG. 암 영역을 다각형/붓으로 CA, 판정 불가 영역만 UNK.
6. 애매한 곳은 `N`(메모)으로 위치와 이유를 남긴다. 자동으로 BC로 정하지 않는다.
7. "모두 저장" → `<ID>.csv`, `<ID>.png`, `<ID>_notes.csv`가 다운로드된다.
8. 가져오기:
   `python scripts/import_annotation.py --annotator sangwoo --from-dir ~/Downloads --status completed`
   (작업 중이면 `--status in_progress`. 이어서 할 때는 도구의 "저장했던 라벨 불러오기")
9. `python scripts/check_annotators.py`로 형식과 진행 현황 확인.

도구는 1.2초마다 브라우저에 임시 저장한다. 다시 열면 복원 여부를 묻는다.
임시 저장은 같은 브라우저에서만 유지되므로 중간중간 "모두 저장"을 한다.

## 독립성 지키기

- 두 사람 모두 끝나기 전에는 상대 폴더를 열거나 비교 스크립트를 돌리지 않는다.
- PR 없이 저장소에 직접 반영하므로, 상대가 끝나기 전에 push하면 서로 볼 수 있다.
  **두 사람 모두 completed가 된 뒤에 함께 push하거나, 각자 끝난 뒤 상대에게 알리고 push한다.**
- 공식 정답(data/raw/.../annotations)은 작업 중 열지 않는다. 도구도 공식 정답을 읽지 않는다.

## 비교 (두 사람 모두 끝난 뒤)

```
python scripts/compare_annotations.py --a annotator:jinho --b annotator:sangwoo
python scripts/compare_annotations.py --a official --b annotator:jinho
python scripts/compare_annotations.py --a official --b annotator:sangwoo
```

결과: `outputs/annotation_compare/<A>__vs__<B>/`

| 파일 | 내용 |
|---|---|
| per_pair.csv | 샘플별 세포 수, A만/B만 찍은 세포(누락), 위치 오차, 클래스 일치율·kappa, BC/TC F1, 조직 IoU, UNK 차이, 경계 근처/내부 불일치 |
| summary.json | 24쌍 합산 지표, 위치가 맞은 세포의 BC/TC 혼동표 |
| overlays/<ID>_cell.png | 회색=일치, 빨강=A만, 하늘색 십자=B만, 주황=위치 같고 클래스 다름 |
| overlays/<ID>_tissue.png | 초록=둘 다 CA, 빨강=A만 CA, 파랑=B만 CA, 자홍=UNK, 노랑 상자=세포 사진 위치 |
| difficult_cases.md | 두 사람의 메모를 한 표로 모음 |

지표 정의
- 세포 매칭: 공식 평가와 같은 15px(3µm, 0.2 MPP) 이내, 가까운 쌍부터 1:1.
- "A만"은 B가 놓친 세포, "B만"은 A가 놓친 세포로 읽는다.
- 클래스 일치율·kappa: 위치가 맞은 세포끼리 TC/BC가 같은 비율. 위치 차이와 분류 차이를 분리한다.
- 조직: UNK는 IoU에서 제외(공식 지표와 동일). UNK 지정 차이는 따로 센다.
- 경계 근처 불일치: 어느 한쪽 CA 경계에서 8px(조직 사진 기준) 안의 불일치 = 경계선 위치 차이.
  내부 불일치: 그 밖 = 영역 자체의 판단 차이.

각자 판단이 어려웠던 부분과 이유는 메모 CSV + `annotator_tracking.csv`의 notes에 남기고,
비교 결과의 큰 차이(예: per_pair.csv에서 kappa가 낮은 샘플)를 함께 보며 원인을 정리한다.

## 학습에서 작성자 라벨 선택

```python
OcelotDataset('train', label_source='annotator:jinho')    # 이진호 라벨 24쌍 + 나머지 공식
OcelotDataset('train', label_source='annotator:sangwoo')  # 신상우 라벨 24쌍 + 나머지 공식
```

- 24쌍 전체가 completed이고 두 파일이 있어야 한다. 빠지면 공식 라벨로 대체하지 않고 중단한다.
- 미리보기: `python scripts/preview_dataset.py --id 224 --labels annotator:jinho`

## M0 실험자: 같은 모델 예측을 두 사람 정답과 각각 평가

모델 예측을 `<폴더>/cell/<ID>.csv`(x,y,label[,score]), `<폴더>/tissue/<ID>.png`(1/2/255)로 저장한 뒤:

```
python scripts/compare_annotations.py --a annotator:jinho   --b dir:<예측 폴더>
python scripts/compare_annotations.py --a annotator:sangwoo --b dir:<예측 폴더>
```

A가 정답, B가 예측이므로 precision/recall 방향이 평가 의미와 맞는다.
두 결과의 차이가 "정답을 누가 만들었는가"에 따른 평가 변동이다.

주의: 이 24쌍은 train이다. 비교에 쓰는 모델이 이 24쌍의 어느 라벨로 학습했는지 함께 기록한다.
학습에 쓴 라벨과 같은 작성자의 정답으로 평가하면 그쪽 점수가 유리하게 나올 수 있다.
