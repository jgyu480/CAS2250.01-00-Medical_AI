# 학습용 데이터 증강 (~10/8)

## 요약

| 증강 | 기본 | 근거 | 라벨 처리 |
|---|---|---|---|
| Random Flip + 90° Rotate | 켜짐 | 수업 필수, 병리 영상은 방향 의미 없음 | 두 사진·점·마스크·위치 상자 함께 변환(기존 구현) |
| Random Stain Normalization (RandStainNA) | 켜짐, p=1.0 | 수업 필수, Shen et al. 2022 | 사진만 변경 |
| Color Jitter | 켜짐, p=0.8, 약하게 | 수업 필수, Tellez et al. 2019 | 사진만 변경 |
| HED Jitter | 꺼짐 | Tellez et al. 2019 | 사진만 변경 |
| Gaussian Blur | 꺼짐 | Tellez et al. 2019 (초점 차이) | 사진만 변경 |

설정 파일: `configs/augmentation.json` · 코드: `common/data/photometric.py`, `common/data/augment.py`
val/test에는 증강을 적용하지 않는다(Dataset이 막는다).

## 사용법 (M0 실험 담당자)

```python
from common.data.dataset import OcelotDataset
train = OcelotDataset('train', augment=True, seed=42, augmentation='default')
train.set_epoch(epoch)   # 매 epoch 호출 → epoch마다 다른, 재현 가능한 증강
```

처음 한 번 RandStainNA 통계를 만든다(train 400쌍만 사용, 빈 유리 제외, 몇 분). 코드가 바뀌면 다시 만든다:

```
python scripts/fit_randstainna.py
```

→ `configs/randstainna_lab_stats.json` (작은 숫자 파일, Git에 올린다)

증강은 학습 중에 즉석으로 만든다(온라인 증강). 파일로 확인하거나 넘길 때:

```
python scripts/export_augmented.py --preset default --ids 001 224 --copies 4
```

→ `outputs/augmented/default/` (cell, tissue, cell_points, tissue_mask, manifest.csv)

## 증강별 효과 비교 (ablation)

각 증강을 하나씩 뺀 preset이 준비돼 있다.

| preset | 내용 |
|---|---|
| default | 기본(Flip/Rotate + RandStainNA + Color Jitter) |
| none | 증강 없음 |
| geometric_only | Flip/Rotate만 |
| no_geometric / no_randstainna / no_color_jitter | 해당 증강만 제외 |
| plus_hed_jitter / plus_blur | 후보 증강 추가 |
| randstainna_strong | RandStainNA 분포 폭 2배(논문 기본값 1.0) |

공정한 비교를 위해 다음을 지켰다.
- 모델·데이터 분할·나머지 증강·학습 설정·seed가 같으면, 한 증강을 꺼도
  **나머지 증강의 파라미터는 샘플마다 완전히 같다**(꺼진 증강도 난수를 같은 수만큼 소모).
  `python scripts/check_augmentation.py`가 이것을 검사한다.
- 비교 지표: 검증셋 세포 mF1, 조직 mIoU. 예측 그림에서 세포 누락·오분류, 조직 경계 오류 증가 여부.
- 한 번의 차이로 판단하지 않고 seed 42/43/44 반복에서도 같은 방향인지 확인한 뒤
  강도를 낮추거나 제외한다.
- 수업에서 필수로 지정한 증강(Color Jitter, Random Stain Normalization, Rotate/Flip)을 빼야 한다면
  근거를 정리해 교수님께 먼저 확인한다.

## 눈으로 확인

```
python scripts/check_augmentation.py --id 001 --repeat 6
```

`outputs/augmentation_preview/001_single.png`: 원본과 증강 하나씩만 적용한 결과.
`outputs/augmentation_preview/001_default_repeat.png`: 기본 설정을 epoch만 바꿔 반복.
위 줄=세포 사진과 점(노랑 BC, 파랑 TC), 아래 줄=조직 사진, CA 경계(초록), 세포 위치(빨강).
확인할 것: 세포 형태·조직 구조가 과하게 변하지 않았는지, 점과 경계가 사진의 세포·조직과 맞는지,
색이 실제 H&E 범위를 벗어나지 않는지(지나친 파랑/초록 등).

## 설계 판단

**RandStainNA를 Random Stain Normalization으로 선택.** 기존 염색 정규화는 고정 템플릿 하나에 맞추고,
염색 증강은 무작위로 흔든다. RandStainNA는 학습셋의 LAB 평균·표준편차 분포에서 가상 템플릿을
매번 뽑아 Reinhard 변환으로 맞춘다. 두 접근을 합친 방법이며 논문에서 조직 분류와 핵 분할 모두에서
고정 정규화보다 좋거나 비슷했다. 논문 설정을 따라 LAB, 채널별 독립 정규분포를 쓴다.
템플릿 표준편차가 0 이하가 되지 않도록 평균의 20% 아래는 자른다(우리 구현 선택).

**RandStainNA를 조직에만, 약하게 적용(10/8 실제 사진 확인 후 수정).**
처음 구현(사진 전체 통계, 분포 폭 1.0)을 실제 OCELOT 사진에 적용해 보니
빈 유리가 많은 사진(예: 037)에서 흰 배경이 회색·청록으로 물들고,
일부 템플릿에서 사진 전체가 갈색·청록으로 어두워지는 비현실적인 색이 나왔다.
원인은 (1) 통계에 빈 유리 픽셀이 섞여 유리 비율에 따라 통계가 크게 흔들리고,
(2) 분포 폭이 넓어 극단 템플릿이 자주 뽑히기 때문이었다. 다음과 같이 고쳤다.
- 통계 계산과 Reinhard 변환을 조직 픽셀에만 적용하고, 빈 유리는 원래 색을 유지한다.
- 분포 폭(std_scale)을 0.5로 줄이고, 뽑은 값은 평균 ±2σ 안으로 자른다.
- 논문 기본 폭(1.0)은 `randstainna_strong` preset으로 남겨 비교 실험에 쓸 수 있게 했다.
수정 후 037·380·343에서 배경은 흰색으로 유지되고 H&E 범위 안에서 색만 바뀌는 것을 확인했다.

**cell/tissue 사진에 같은 색 파라미터.** 두 사진은 같은 슬라이드·같은 스캔에서 나와 염색이 같다.
같은 가상 템플릿에 각자 맞추므로 두 시야의 색 관계가 유지된다. 이것도 우리 판단이며
`shared_photometric_between_fovs: false`로 바꿔 비교할 수 있다.

**Color Jitter는 약하게.** 밝기·대비·채도 ±10%, 색조 ±0.02. RandStainNA가 염색 차이의 대부분을 다루므로
Jitter는 보조 역할이다. 색조를 크게 돌리면 H(보라)와 E(분홍)의 구분이 깨질 수 있다.

**HED Jitter·Blur는 기본 꺼짐.** HED Jitter는 RandStainNA와 역할이 겹친다. Blur는 3µm 기준 세포
중심 검출에서 작은 핵 경계를 흐릴 수 있다. 둘 다 preset으로 추가 실험만 한다.

**제외한 증강.** Elastic deformation, 임의 각도 회전, 크기 변경은 세포 형태와 MPP(크기 기준)를 바꾸고,
두 시야의 좌표 정렬과 3µm 매칭 거리를 함께 바꿔야 해서 제외했다. 90° 회전·좌우 뒤집기는 픽셀을
보간 없이 옮기므로 형태가 변하지 않는다.

## 구현 확인 결과

- 색 공간 왕복 오차 ≤1(0~255), Reinhard 자기 통계 변환 = 원본, 강도 0 Jitter = 원본.
- 같은 seed·epoch·index → 같은 결과, epoch가 바뀌면 다른 결과.
- 색 증강 적용 전후로 점·마스크·위치 상자가 완전히 같음.
- 기존 뒤집기·회전 결과는 이전 코드와 같은 난수 순서를 유지(기존 Dataset 검사 통과).
- 합성 데이터로 Dataset·내보내기·미리보기를 실행해 확인. 실제 OCELOT 데이터에서의 그림 확인과
  RandStainNA 통계 계산은 데이터가 있는 컴퓨터에서 실행해야 한다.
- 속도: 1024×1024 두 장 기준 샘플당 약 1초(CPU, numpy). DataLoader worker를 여러 개 쓰면 된다.

## 참고 자료

- Shen et al., RandStainNA: Learning Stain-Agnostic Features from Histology Slides by Bridging
  Stain Augmentation and Normalization, MICCAI 2022. https://arxiv.org/abs/2206.12694
- Tellez et al., Quantifying the effects of data augmentation and stain color normalization in
  convolutional neural networks for computational pathology, Medical Image Analysis 2019.
  https://arxiv.org/abs/1902.06543
- Reinhard et al., Color transfer between images, IEEE CG&A 2001.
- Ruifrok & Johnston, Quantification of histochemical staining by color deconvolution, 2001 (HED 분해 행렬).
