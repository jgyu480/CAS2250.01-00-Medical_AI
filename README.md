# CAS2250 Medical AI — OCELOT 세포·조직 모델

같은 조직에서 나온 두 사진을 사용한다.
세포 사진에서는 세포핵 중심을 찾고 BC/TC를 분류한다.
조직 사진에서는 픽셀마다 BG/CA/UNK를 구분한다.
M0~M3로 두 작업 사이의 정보 전달 방식과 gradient 처리 효과를 비교한다.

## 현재 진행 상태

1단계: 모델별 폴더·설정·설명·경로 처리·구조 점검을 준비했다.
실제 모델과 학습·평가 코드는 아직 구현하지 않았다.
지금 실행하는 점검 명령은 학습을 시작하지 않는다.

## 데이터 이해

| 구성 | 의미 |
|---|---|
| cell 이미지 | 좁은 영역을 자세히 본 1024×1024 사진 |
| tissue 이미지 | 넓은 4096×4096 영역을 1024×1024로 줄인 사진 |
| cell CSV | 세포 중심 x,y 좌표와 BC/TC 클래스 |
| tissue PNG | 픽셀별 조직 클래스 정답 |
| metadata.json | 두 사진의 상대 위치·슬라이드·장기 정보 |

BC는 비종양 세포, TC는 종양 세포다. 세포 없는 픽셀과 BC는 구분한다.
BG는 비암조직, CA는 암조직, UNK는 불확실한 영역이다.
UNK 정답 픽셀은 조직 손실과 평가에서 제외한다.
라벨의 표시 색상과 파일에 저장된 정수 값은 구분한다.
실제 조직 라벨 값은 2단계에서 확인하므로 현재 설정은 null이다.

두 사진은 저장 크기가 같아도 보는 영역이 다르다.
세포 사진이 조직 사진의 중앙이라고 가정하지 않고 metadata를 사용한다.
v1.0.1은 test 586/589/609/615를 제외했으므로 실제 split 수를 검사한다.

## 공통 모델 구조

각 사진을 ResNet-50 encoder → ASPP → decoder → head로 처리한다.
Encoder는 특징을 추출하고 ASPP는 여러 범위의 주변 정보를 모은다.
Decoder는 위치별 예측을 위해 특징 지도의 해상도를 복원한다.
Head는 특징을 클래스별 예측으로 바꾼다.

두 encoder는 별도 파라미터를 가지며 같은 Lunit BT SSL 가중치로 시작한다.
Decoder와 head는 직접 구현하며 SSL 사전학습을 처음부터 반복하지 않는다.
세포 점 정답은 원형 영역으로 바꿔 학습하고, 예측 지도에서 중심점을 찾는다.

## 비교 모델

| 모델 | ASPP 뒤 조직→세포 | Encoder 뒤 세포→조직 | 학습 방식 |
|---|---|---|---|
| M0 | 좌표에 맞춰 잘라 확대 | 좌표에 맞춰 축소·배치 | 두 손실 합 |
| M1 | 전체 조직 cross-attention | 없음 | 두 손실 합 |
| M2 | M1과 동일 | M0와 동일 | 두 손실 합 |
| M3 | M2와 동일 | M2와 동일 | PCGrad |

Decoder 뒤의 조직→세포 좌표 전달은 네 모델에서 동일하다.
Attention의 Query는 세포, Key/Value는 전체 조직 특징이다.
초기 토큰 격자는 16×16이며 위치 정보를 같은 조직 좌표계로 맞춘다.
토큰 격자는 첫 구현 설정이며 점검 후 검토한다.
역방향 전달은 관측된 ROI에 배치하고 coverage mask로 그 영역을 표시한다.

- M0↔M2: ASPP 뒤 attention 효과.
- M1↔M2: 세포→조직 역방향 전달 효과.
- M2↔M3: 같은 구조에서 gradient 충돌 조정 효과.

M0는 OCELOT 구조를 참고해 SSL ResNet-50으로 구현하는 기준 모델이다.
원 논문의 모든 설정과 성능을 그대로 재현하는 실험은 아니다.
PCGrad는 두 작업의 gradient가 충돌할 때 해당 성분을 조정한다.
두 손실 모두 연결된 파라미터에 적용하고 M2와 업데이트 규모를 맞춘다.

## 폴더

| 위치 | 역할 |
|---|---|
| models/M0~M3 | 모델별 구조·설정·설명 |
| common/backbones | SSL 가중치 로딩과 특징 추출 |
| common/data | Dataset·정답 생성·좌표 정렬·증강 |
| common/modules | ASPP·decoder·좌표 전달·attention |
| common/losses | 세포·조직 손실 |
| common/metrics | 공식 세포 F1·UNK 제외 조직 mIoU |
| common/training | 공통 학습 loop·checkpoint·PCGrad |
| configs | 공통 설정 |
| scripts | 점검·시각화·실행 명령 |
| docs | 단계별 논문 검토·어노테이션 기록 |
| vessl | 추후 클라우드 실행 설정 |
| data, weights, outputs | 로컬 데이터·가중치·결과, Git 제외 |

## 구현 순서

1. 디렉터리·설정·README·경로 처리.
2. 환경·데이터·metadata·라벨·split 점검.
3. 어노테이션 형식·작업 이력·라벨 선택 정책.
4. Dataset·target·좌표 정렬·증강·미리보기.
5. SSL encoder·ASPP·decoder·head·loss.
6. M0 구현 및 forward 점검.
7. M1 attention 구현 및 점검.
8. M2 양방향 전달 구현 및 점검.
9. M3 PCGrad 구현 및 gradient 점검.
10. 학습·평가 명령과 VESSL 설정 준비.

각 단계에서 논문과 공식 구현을 확인하고 docs에 기록한다.
진행률은 구현 단계의 완료 비율이다. 실제 학습은 별도 실행한다.
현재 점검 명령: python scripts/check_structure.py

## 어노테이션 계획

초기 계획은 train에서 장기별 4쌍, 총 24쌍을 직접 라벨링하는 것이다.
교수님이 수량을 별도로 지정하면 해당 지시를 따른다.
초안과 검토본을 모두 보관한다. 실제 라벨링은 사람이 수행한다.

본 실험에서는 검토본 24쌍과 나머지 공식 train 정답을 네 모델에 공통 사용한다.
M0에서 같은 이미지의 초안/검토본만 바꿔 품질 영향을 추가 확인한다.
Validation과 test의 정답은 수정하지 않는다.

## 학습·평가 계획

초기 학습 설정은 총 50 epoch이며 encoder 고정 3 epoch를 포함한다.
AdamW, encoder 학습률 1e-5, 새 모듈 학습률 1e-4를 사용한다.
Cosine scheduler와 seed 42/43/44를 계획한다.
세포 Focal Loss와 UNK를 제외한 조직 CE·Dice를 사용한다.
세포와 조직 손실 가중치는 1:1로 시작한다.
Batch size와 gradient accumulation은 실제 GPU 확인 후 확정한다.

Validation에서 checkpoint와 검출 threshold를 선택한다.
Test 전에 설정을 고정하고 공식 세포 F1과 조직 mIoU를 기록한다.
이 값들은 첫 실험의 시작 설정이며 최적 성능을 보장하지 않는다.

## 맥북과 VESSL의 경로

기본 경로는 프로젝트 기준 상대 경로다.
데이터: data/raw/ocelot2023_v1.0.1
가중치: weights/bt_rn50_ep200.torch
결과: outputs

VESSL에서는 다음 환경변수로 경로를 바꾼다.

- OCELOT_DATA_ROOT: metadata.json이 바로 있는 데이터 폴더.
- SSL_WEIGHTS_PATH: SSL 가중치 파일.
- OUTPUT_ROOT: 로그·checkpoint·예측 저장 폴더.

개인 맥북의 절대 경로를 모델 코드에 넣지 않는다.
VESSL 실행 YAML·GPU·결과 보존은 10단계에서 구성한다.

## 참고 자료

- [SSL 백본](https://github.com/lunit-io/benchmark-ssl-pathology)
- [SSL 논문](https://arxiv.org/abs/2212.04690)
- [OCELOT 논문](https://arxiv.org/abs/2303.13110)
- [OCELOT 데이터](https://zenodo.org/records/8417503)
- [공식 평가 연결](https://github.com/lunit-io/ocelot23algo)
- [PCGrad](https://arxiv.org/abs/2001.06782)
- [VESSL](https://docs.vessl.ai/reference/yaml/run-yaml)

데이터와 가중치는 원 제공자의 이용조건을 따른다.
## 2단계 실행 환경과 데이터 점검

1단계 구조 점검 완료. 현재는 2단계 데이터 검사 결과를 확인하는 중이다.

이 프로젝트의 로컬 Python 환경은 cas2250-medical-ai다.
터미널을 새로 열면 conda activate cas2250-medical-ai를 실행한다.
데이터 검사 패키지는 requirements-audit.txt에 기록한다.

검사 명령:
python scripts/check_data.py --zip data/downloads/ocelot2023_v1.0.1.zip

출력:
- outputs/audit/report.json: 개수·클래스 값·metadata·오류·환경 정보.
- outputs/audit/manifest.csv: 샘플별 네 파일의 상대 경로.

PASS는 파일 형식과 연결 검사 통과를 뜻한다.
조직 클래스 의미와 좌표 정렬은 별도로 확인한다.
검사 결과에 FAIL이 있으면 해당 manifest를 학습에 사용하지 않는다.

<!-- DATASET_STEP4_START -->
## Dataset과 좌표 정렬

M0~M3가 공통으로 사용할 데이터 로더를 준비한다.
세포 사진·조직 사진·세포 점·조직 마스크·metadata를 함께 읽는다.

- 실행 검사: `python scripts/check_dataset.py`
- 원본 미리보기: `python scripts/preview_dataset.py --id 001`
- 증강 미리보기: `python scripts/preview_dataset.py --id 001 --augment`
- 설명: [Dataset 작업 안내](docs/dataset_protocol.md)

공식 라벨 모드는 데이터 연결 점검에 사용한다.
직접 작성한 라벨 모드는 대상 목록 전체의 작성·검토 완료 기록이 필요하다.

진척도는 [progress.json](docs/progress.json)에 기록된다.
구현 완료 비율과 실제 어노테이션 완료 수를 구분한다.
현재 단계에서는 모델 학습을 실행하지 않는다.

데이터와 결과 경로는 기존 공통 경로 설정을 사용한다.
VESSL에서는 OCELOT_DATA_ROOT와 OUTPUT_ROOT로 경로를 바꿀 수 있다.
<!-- DATASET_STEP4_END -->
