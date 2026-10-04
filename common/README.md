# 공통 코드

backbones는 SSL 백본, data는 데이터 처리, modules는 모델 부품을 담는다.
losses는 손실, metrics는 평가, training은 학습 처리를 담는다.
네 모델이 이 코드를 공통 사용한다.

현재 구현된 기능은 paths.py의 경로 처리다.
나머지는 이후 단계에서 작성한다.
