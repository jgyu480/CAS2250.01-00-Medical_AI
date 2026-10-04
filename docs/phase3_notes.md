# 3단계 설계 근거

OCELOT은 세포핵 중심점과 조직의 픽셀별 정답을 제공한다.
Appendix A의 annotation protocol과 consensus 절차를 참고한다.
본 실습에서는 초안 작성과 독립 동료 검토를 구분해 기록한다.

총 24쌍은 과제의 초기 작업량 선택이며 논문에서 요구한 수량은 아니다.
여섯 장기에서 각각 4쌍을 선택해 특정 장기에 치우치는 것을 줄인다.
같은 case를 반복 선택하지 않고 seed 42로 대상을 고정한다.
공식 정답 내용은 대상 선택에 사용하지 않는다.

초안과 검토본을 보존해 라벨 품질 변경 효과를 비교한다.
모델 구조 비교에서는 네 모델의 입력과 라벨을 동일하게 유지한다.
준비 완료와 실제 어노테이션 완료를 별도로 기록한다.

공식 데이터 분할은 WSI 단위다.
실제 데이터에서 동일 WSI의 split 중복은 발견되지 않았다.
TCGA case ID 중복은 존재하므로 환자 단위 완전 독립 평가로 표현하지 않는다.
과제의 공식 평가 조건을 유지하며 이 한계를 보고서에 기록한다.

참고:
- https://arxiv.org/html/2303.13110v2
- https://lunit-io.github.io/research/ocelot_dataset/
- https://ocelot2023.grand-challenge.org/datasets/
