# 5단계: SSL 백본과 공통 부품

사진 → SSL ResNet-50 → ASPP → decoder → head.

세포용과 조직용 백본은 같은 BT 가중치로 시작하며 파라미터는 독립적이다.
가중치는 strict=True로 로딩한다.
마지막 stride를 dilation으로 바꿔 output stride를 16으로 유지한다.
새 부품은 GroupNorm을 사용하며 백본의 BN 통계는 고정한다.
BN 통계 고정은 백본 전체 파라미터 동결과 다르다.

세포: 원 밖=0, BC=1, TC=2.
조직: BG=0, CA=1, UNK=2.
조직 손실에서는 UNK 픽셀을 제외한다.
현재 손실은 두 과제의 Dice를 각각 반환한다.
기존 계획서의 Cross Entropy 설정과 차이가 있으므로 본 학습 전에
공통 손실 설정을 확정한다.

ImageNet 평균·표준편차는 프로젝트의 공통 초기 선택이다.
Lunit 사전학습 당시의 정확한 정규화 설정으로 확인한 값은 아니다.
체크포인트 SHA256은 공식 릴리스 파일을 내려받아 계산한 대조값이다.

128×128 축소는 연결 검사에만 사용한다.
Dataset 원본은 1024×1024로 유지한다.
backward와 optimizer는 실행하지 않는다.
M0~M3 정보 교환은 이후 단계에서 구현한다.

근거:
https://github.com/lunit-io/benchmark-ssl-pathology
https://arxiv.org/abs/2303.13110
https://arxiv.org/abs/1802.02611


## 정규화 설정 업데이트

RGB를 0~1로 바꾸고 Lunit 공식 릴리스의 평균·표준편차로 정규화한다.
RGB mean: [0.70322989, 0.53606487, 0.66096631]
RGB std: [0.21716536, 0.26081574, 0.20723464]
모든 모델에 동일하게 적용한다.
출처: https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights
