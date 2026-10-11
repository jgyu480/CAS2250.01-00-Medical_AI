"""좌표를 맞춘 양방향 특징 전달을 사용하는 기준 모델 (M0).

OCELOT 논문의 Cell-Tissue Feature Sharing 모델을 참고했다.
https://arxiv.org/abs/2303.13110 (Section 5.3, Fig. 5·6)

    세포 사진 → encoder ─────────────→ ASPP ─(②)→ decoder ─(③)→ head → 세포 예측
                   │(①)                  ↑               ↑
    조직 사진 → encoder ─(①)→ ASPP ──────┘── decoder ───┘──→ head → 조직 예측

    ① encoder 뒤 세포→조직: 세포 특징을 축소해 조직 특징의 해당 위치에 배치
    ② ASPP 뒤 조직→세포: 조직 특징에서 세포 영역을 잘라 확대
    ③ decoder 뒤 조직→세포: ②와 같은 방식

전달 위치와 방향은 models/M0/config.json을 따른다.
"none"으로 바꾸면 해당 전달을 끈다 (비교 실험용).
"""

import json
from pathlib import Path

from torch import nn
from torch.nn import functional as F

from common.paths import resolve_paths
from common.backbones.ssl_resnet import load_ssl_encoder
from common.modules.segmentation import TaskBranch
from common.modules.exchange import TissueToCell, CellToTissue

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
HIGH, CONTEXT = 2048, 256


def read_config(config=None):
    if config is None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    t2c, c2t = config["tissue_to_cell"], config["cell_to_tissue"]
    allowed = {
        ("after_aspp", t2c.get("after_aspp", "none")): {"coordinate_crop", "none"},
        ("after_decoder", t2c.get("after_decoder", "none")): {"coordinate_crop", "none"},
        ("after_encoder", c2t.get("after_encoder", "none")): {"coordinate_place", "none"}
    }
    for (where, value), options in allowed.items():
        if value not in options:
            raise ValueError(f"M0가 지원하지 않는 전달 방식: {where}={value}")
    if config.get("attention") is not None:
        raise ValueError("M0는 attention을 사용하지 않습니다.")
    return config


class M0(nn.Module):
    def __init__(self, cell_encoder, tissue_encoder, config=None):
        super().__init__()
        config = read_config(config)
        self.config = config
        self.cell = TaskBranch(cell_encoder)
        self.tissue = TaskBranch(tissue_encoder)

        t2c, c2t = config["tissue_to_cell"], config["cell_to_tissue"]
        self.encoder_c2t = (
            CellToTissue(HIGH, HIGH)
            if c2t.get("after_encoder") == "coordinate_place" else None
        )
        self.aspp_t2c = (
            TissueToCell(CONTEXT, CONTEXT)
            if t2c.get("after_aspp") == "coordinate_crop" else None
        )
        self.decoder_t2c = (
            TissueToCell(CONTEXT, CONTEXT)
            if t2c.get("after_decoder") == "coordinate_crop" else None
        )

    def forward(self, cell_image, tissue_image, cell_box):
        cell = self.cell.encoder(cell_image)
        tissue = self.tissue.encoder(tissue_image)

        # ① encoder 뒤: 세포 → 조직
        tissue_high = tissue["high"]
        if self.encoder_c2t is not None:
            tissue_high = self.encoder_c2t(tissue_high, cell["high"], cell_box)

        cell_context = self.cell.aspp(cell["high"])
        tissue_context = self.tissue.aspp(tissue_high)

        # ② ASPP 뒤: 조직 → 세포
        if self.aspp_t2c is not None:
            cell_context = self.aspp_t2c(cell_context, tissue_context, cell_box)

        cell_decoded = self.cell.decoder(cell_context, cell["low"])
        tissue_decoded = self.tissue.decoder(tissue_context, tissue["low"])

        # ③ decoder 뒤: 조직 → 세포
        if self.decoder_t2c is not None:
            cell_decoded = self.decoder_t2c(
                cell_decoded, tissue_decoded, cell_box
            )

        def logits(branch, decoded, image):
            return F.interpolate(
                branch.head(decoded), size=image.shape[-2:],
                mode="bilinear", align_corners=False
            )

        return {
            "cell_logits": logits(self.cell, cell_decoded, cell_image),
            "tissue_logits": logits(self.tissue, tissue_decoded, tissue_image)
        }

    def encoder_parameters(self):
        return [
            *self.cell.encoder.parameters(),
            *self.tissue.encoder.parameters()
        ]

    def new_module_parameters(self):
        encoder = {id(p) for p in self.encoder_parameters()}
        return [p for p in self.parameters() if id(p) not in encoder]


def build_model(config=None, weights=None):
    """두 갈래 모두 같은 Lunit BT 가중치로 시작하는 M0를 만든다."""
    weights = weights or resolve_paths()["ssl_weights"]
    return M0(load_ssl_encoder(weights), load_ssl_encoder(weights), config)
