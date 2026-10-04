# 1단계 설계 근거

OCELOT §5.1~5.3의 dual branch·DeepLabV3+·feature sharing을 참고한다.
원 논문의 ResNet-34를 과제의 Lunit SSL ResNet-50으로 변경한다.
Attention은 우리 추가 비교 실험이다.
M3는 PCGrad 효과를 비교하기 위해 M2의 모델 코드를 재사용한다.
공통 데이터 처리·평가·decoder를 사용해 모델별 구현 차이를 줄인다.
VESSL을 고려해 개인 절대 경로 대신 설정과 환경변수를 사용한다.

이 단계는 구조만 점검한다.
데이터 내용과 모델은 이후 단계에서 검증한다.

자료:
- https://arxiv.org/abs/2303.13110
- https://github.com/lunit-io/benchmark-ssl-pathology
- https://arxiv.org/abs/2001.06782
- https://docs.vessl.ai/reference/yaml/run-yaml
