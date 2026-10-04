# M3

M2와 같은 모델 구조에 PCGrad 학습 처리를 적용.

- config.json: 모델별 구조와 학습 방식 설정.
- model.py: 모델을 구성하는 코드. 현재는 구현 위치만 준비.
- 공통 데이터·백본·decoder·loss·평가는 common에서 사용.

M0/M1/M2/M3는 각각 6/7/8/9단계에서 구현한다.
