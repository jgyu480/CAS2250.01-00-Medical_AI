# 2단계 — 환경과 데이터 점검

## 근거

OCELOT 논문은 같은 조직의 서로 겹치는 cell/tissue 사진을 사용한다.
두 입력과 두 정답이 같은 샘플 ID로 연결되는지 먼저 확인한다.

공식 데이터 설명의 파일 경로와 metadata 필드를 검사 기준으로 사용한다.
공식 알고리즘 인터페이스의 세포 클래스는 BC=1, TC=2다.
v1.0.1은 test 586/589/609/615를 제외했다.

## 이번 구현

- 프로젝트 전용 Python 3.11 환경.
- NumPy와 Pillow 버전을 requirements-audit.txt에 기록.
- JPG를 실제로 읽어 크기·RGB 형식 확인.
- 세포 CSV의 좌표 범위·유한값·클래스 확인.
- 조직 PNG의 단일 채널·정수 값과 픽셀 수 확인.
- 네 파일의 샘플 ID 연결과 metadata subset 확인.
- Split 사이의 샘플·슬라이드 중복 확인.
- TCGA case ID 중복은 검토할 경고로 기록.
- ZIP의 MD5를 공식 릴리스와 비교.
- 검사 보고서와 상대 경로 manifest 저장.

## 아직 확정하지 않은 항목

조직 픽셀 값의 BG/CA/UNK 의미는 릴리스 설명서와 대조한다.
숫자가 작다는 이유만으로 BG나 CA를 지정하지 않는다.
Metadata를 이용한 실제 좌표 정렬은 4단계에서 구현한다.
SSL 가중치의 모델 로딩 검증은 5단계에서 수행한다.
파일 점검 통과는 어노테이션의 의학적 정확성을 보장하지 않는다.

## 자료

- https://arxiv.org/abs/2303.13110
- https://lunit-io.github.io/research/ocelot_dataset/
- https://zenodo.org/records/8417503
- https://github.com/lunit-io/ocelot23algo
- https://arxiv.org/abs/2212.04690
