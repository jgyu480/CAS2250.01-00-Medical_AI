"""M2: M0와 같은 구조에 세포 head만 BC·TC 중심 점수 지도 2채널로 바꾼 모델.

보고서 8.4
- 세포 정답을 반지름 7px 원형 클래스 지도에서 BC·TC별 중심 지도(H_AO)로 바꾼다.
- 세포 head는 BC·TC 점수 지도를 각각 예측하고 독립 sigmoid를 적용한다.

M0와 같은 것: 두 SSL ResNet-50 encoder, ASPP, decoder, 세 위치의 좌표 기반
특징 전달(models/M0/config.json과 같은 설정), 조직 head(BG/CA/UNK 3채널).
M0와 다른 것: 세포 head의 출력 채널 3(배경/BC/TC, softmax) → 2(BC/TC, sigmoid).
sigmoid는 손실(common/losses/heatmap_losses.py)과 후처리에서 적용하므로
forward는 M0처럼 logits를 반환한다.
"""

import json
from pathlib import Path

from torch import nn

from common.paths import resolve_paths
from common.backbones.ssl_resnet import load_ssl_encoder
from models.M0.model import M0

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
CELL_OUTPUTS = ["BC", "TC"]


def read_config(config=None):
    if config is None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("model") != "M2":
        raise ValueError("M2 모델 설정이 아닙니다.")
    head = config.get("cell_head", {})
    if head.get("outputs") != CELL_OUTPUTS or head.get("activation") != "sigmoid":
        raise ValueError("M2 세포 head는 BC·TC 2채널 sigmoid입니다.")
    return config


class M2(M0):
    def __init__(self, cell_encoder, tissue_encoder, config=None):
        config = read_config(config)
        # 특징 전달 설정 검사와 조립은 M0 코드를 그대로 쓴다.
        super().__init__(cell_encoder, tissue_encoder, config)
        self.cell.head = nn.Conv2d(256, len(CELL_OUTPUTS), 1)


def build_model(config=None, weights=None):
    """두 갈래 모두 같은 Lunit BT 가중치로 시작하는 M2를 만든다."""
    weights = weights or resolve_paths()["ssl_weights"]
    return M2(load_ssl_encoder(weights), load_ssl_encoder(weights), config)
