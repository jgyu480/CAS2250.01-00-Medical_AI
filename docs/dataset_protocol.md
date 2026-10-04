# 4단계: 데이터를 읽고 위치를 맞추기

## 이번 단계의 목적

M0~M3가 함께 사용할 Dataset을 만든다.
같은 입력과 라벨을 사용해야 모델 구조 차이에 따른 성능을 비교할 수 있다.

현재는 NumPy 배열을 반환한다. GPU와 PyTorch가 필요하지 않다.
이미지 정규화와 PyTorch 연결은 SSL 백본을 준비하는 5단계에서 진행한다.

## 한 샘플에 들어 있는 내용

| 이름 | 내용 |
| --- | --- |
| cell_image / tissue_image | 1024x1024x3, RGB uint8 사진 |
| cell_points | N행 3열, x,y,label 점 목록 |
| cell_target | 1024x1024, 원 밖=0 / BC=1 / TC=2 |
| tissue_target | 1024x1024, BG=0 / CA=1 / UNK=2 |
| cell_box | 조직 사진 안의 세포 사진 위치. 왼쪽·위·오른쪽·아래 |
| label_source | 실제 사용한 official/draft/reviewed |
| transform | 적용한 뒤집기와 90도 회전 횟수 |

cell_target의 0은 세포 중심 원 밖이라는 뜻이다.
비종양 세포는 BC=1이다.

조직 PNG 원본의 BG=1, CA=2, UNK=255는 읽을 때만 번호를 변환한다.
원본 파일 자체는 변경하지 않는다.
내부 UNK=2는 이후 손실과 조직 지표 계산에서 제외한다.

## 논문과 공식 구현 반영

OCELOT 논문 Appendix B의 세포 중심 원형 정답 방식을 사용한다.
기본 반지름은 저장된 cell 사진에서 7픽셀이다.
실제 MPP가 조금씩 달라 물리적 반지름이 모든 사진에서 정확히 같지는 않다.

원이 겹치면 가까운 중심의 클래스를 사용한다.
동률은 좌표 정렬순으로 결정한다.
겹침 규칙은 우리의 구현 선택이다.

논문 부록의 세포 번호 예시와 배포 데이터의 번호가 다르다.
파일 해석과 출력은 공식 배포 기준 BC=1, TC=2를 따른다.

ocelot23algo는 챌린지 입출력·추론 예제다.
논문 전체 학습 구현이라고 표현하지 않는다.

metadata의 WSI 경계로 실제 위치를 계산한다.
cell 사진이 항상 tissue 사진 가운데 있다고 가정하지 않는다.

뒤집기와 90도 회전은 두 사진·점·조직 라벨·위치 상자에 함께 적용한다.
색 변화·잡음·흐림은 학습 설정에서 추가 검토한다.
현재 구현이 논문의 모든 증강을 재현했다고 주장하지 않는다.

## 라벨 출처

official은 공식 라벨을 읽는 점검용 기본 모드다.

draft/reviewed는 선택된 train 24쌍만 직접 작성한 라벨로 교체한다.
나머지 train은 공식 라벨을 사용한다.

선택된 전체 목록의 완료 기록과 두 라벨 파일이 필요하다.
미완료 샘플에는 공식 정답을 자동 대입하지 않고 실행을 중단한다.

val/test는 공식 라벨과 원본 사진만 사용한다.

## 확인 명령

python scripts/check_dataset.py
python scripts/preview_dataset.py --id 001
python scripts/preview_dataset.py --id 001 --augment

## 그림 읽기

왼쪽 위: 세포 사진과 점.
오른쪽 위: 조직 라벨과 빨간 위치 상자.
왼쪽 아래: 빨간 상자 영역을 확대한 조직 사진.
오른쪽 아래: 점을 반지름 7픽셀 원으로 바꾼 정답.

왼쪽 위와 왼쪽 아래에서 같은 조직 구조가 보이는지 확인한다.
조직 사진은 해상도가 낮아 확대하면 더 흐릿하다.

BC는 노란색, TC는 파란색, CA는 초록색이다.
UNK의 자홍색은 미리보기에서 구분하기 위한 표시 색이다.

## 진척도

전체 검사가 PASS이면 구현 4/10단계 완료다.
이 비율은 모델 성능이나 실제 라벨링 비율을 뜻하지 않는다.
실제 라벨링 완료 수는 docs/progress.json에 따로 기록한다.

## 근거 자료

- [OCELOT 논문: Section 5.1, Appendix B](https://arxiv.org/html/2303.13110v2)
- [공식 데이터 설명: Metadata](https://lunit-io.github.io/research/ocelot_dataset/)
- [공식 클래스 정의](https://ocelot2023.grand-challenge.org/datasets/)
- [공식 추론 입출력 구현](https://github.com/lunit-io/ocelot23algo)
