"""M3는 M2의 forward 구조를 그대로 사용한다."""

from models.M2.model import build_model as build_m2

def build_model(config):
    # 구조를 복사하지 않아 M2와 동일하게 유지한다.
    return build_m2(config)
