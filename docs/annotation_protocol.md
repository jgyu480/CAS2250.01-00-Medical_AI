# 직접 어노테이션 작업 기준

## 1. 대상

annotations/selection.csv에 기록된 train 24쌍을 사용한다.
장기별 4쌍이며 M0~M3에서 같은 대상과 같은 라벨 버전을 사용한다.
목록이 확정된 뒤 편한 사진으로 임의 교체하지 않는다.

작업 사진은 outputs/annotation_packets/샘플ID에 있다.
cell.jpg, tissue.jpg, metadata.json이 함께 제공된다.
공식 점 정답과 조직 마스크는 작업 폴더에 복사하지 않는다.

## 2. 세포 어노테이션

cell.jpg를 보고 세포핵의 중심에 점을 표시한다.
TC는 파란색, BC는 노란색으로 표시한다.

저장 형식은 헤더 없는 CSV이며 한 줄에 x,y,label을 기록한다.
BC=1, TC=2를 사용한다.
좌표는 왼쪽 위를 원점으로 하는 cell.jpg 자체의 픽셀 좌표다.
x는 오른쪽, y는 아래쪽으로 증가하며 범위는 0~1023이다.

예시 120,300,2는 x=120, y=300의 종양세포 중심점이라는 뜻이다.
예시 값을 실제 샘플의 정답으로 저장하지 않는다.

사진 전체를 확인한다. 일부 영역만 찍고 전체 완료로 표시하지 않는다.
판단이 어려운 세포를 자동으로 BC로 정하지 않는다.
검토할 위치와 이유를 notes에 기록한 뒤 분류를 확인한다.
세포 정답에는 UNK나 empty_pixel 클래스를 추가하지 않는다.

## 3. 조직 어노테이션

tissue.jpg 전체에서 암조직·비암조직·불확실한 영역을 표시한다.
CA는 초록색으로 표시한다.

저장 파일은 1024×1024 단일 채널 정수 PNG다.
BG=1, CA=2, UNK=255를 사용한다.
화면에 표시하는 색과 PNG에 저장하는 정수 값은 구분한다.
색칠된 RGB 스크린샷을 정답 마스크로 사용하지 않는다.

UNK는 실제로 판단이 불확실한 영역에 사용한다.
작업하지 않은 부분을 모두 UNK로 채워 완료로 표시하지 않는다.

## 4. 저장 위치

초안 세포: annotations/manual/draft/cell/샘플ID.csv
초안 조직: annotations/manual/draft/tissue/샘플ID.png

검토본 세포: annotations/manual/reviewed/cell/샘플ID.csv
검토본 조직: annotations/manual/reviewed/tissue/샘플ID.png

첫 작성이 끝나면 초안을 보존한다.
검토 과정의 수정은 reviewed에 별도로 저장한다.
원본 data/raw의 공식 정답은 수정하지 않는다.

## 5. 검토와 작업 기록

tracking.csv의 draft_status와 review_status를 관리한다.

not_started: 아직 시작하지 않음.
in_progress: 작성 또는 검토 중.
completed: 해당 단계의 두 정답 파일을 모두 작성하고 확인함.

annotator와 reviewer에는 실제 작업자를 기록한다.
reviewer는 초안 작성자와 다른 사람으로 둔다.
started_at, finished_at에는 작업 시간을 기록한다.
notes에는 애매한 위치, 수정 이유, 판단이 다른 부분을 기록한다.

동료 검토본을 전문의 정답과 같은 품질로 간주하지 않는다.
초안과 검토본의 차이, 공식 train 정답과의 차이를 따로 분석한다.
공식 정답과의 비교는 독립 작성·검토 후 품질 분석에 사용한다.

## 6. 학습 라벨 사용 계획

본 실험:
선택한 24쌍은 reviewed, 나머지 376쌍은 공식 train 정답.

품질 비교 실험:
같은 24쌍에서 draft를 사용하고 나머지 조건은 동일하게 유지.

선택된 24쌍의 수동 정답이 빠지면 학습 전에 중단한다.
해당 샘플만 공식 정답으로 조용히 대체하지 않는다.
Validation과 test 정답은 공식 정답을 그대로 사용한다.

## 7. 완료 확인

python scripts/check_annotation_setup.py

자동 검사는 파일 형식·좌표 범위·클래스·기록의 일관성을 확인한다.
모든 세포가 빠짐없이 표시됐는지와 분류의 정확성은 사람이 검토한다.
