"""M1: unidirectional tissue-to-cell *global* cross-attention.

Compared with M0:
  - no encoder-level cell-to-tissue feature placing
  - replace ASPP-level tissue-to-cell coordinate crop with cross-attention
  - retain decoder-level tissue-to-cell coordinate crop
Both branches retain the same backbone, ASPP, decoder and prediction heads.
"""

import json
from pathlib import Path

from torch import nn
from torch.nn import functional as F

from common.paths import resolve_paths
from common.backbones.ssl_resnet import load_ssl_encoder
from common.modules.segmentation import TaskBranch
from common.modules.exchange import TissueToCell
from common.modules.cross_attention import TissueGuidedCrossAttention

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
CONTEXT = 256


def read_config(config=None):
    if config is None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("model") != "M1":
        raise ValueError("Expected M1 model config")
    t2c, c2t = config["tissue_to_cell"], config["cell_to_tissue"]
    if t2c.get("after_aspp") != "global_cross_attention":
        raise ValueError("M1 requires ASPP global_cross_attention")
    if t2c.get("after_decoder") != "coordinate_crop":
        raise ValueError("M1 retains decoder-level coordinate_crop")
    if c2t.get("after_encoder") != "none":
        raise ValueError("M1 excludes encoder-level cell_to_tissue")
    attention = config["attention"]
    if attention.get("query") != "cell" or attention.get("key_value") != "full_tissue":
        raise ValueError("M1 attention must use cell Q and full-tissue KV")
    return config


class M1(nn.Module):
    def __init__(self, cell_encoder, tissue_encoder, config=None):
        super().__init__()
        self.config = read_config(config)
        self.cell = TaskBranch(cell_encoder)
        self.tissue = TaskBranch(tissue_encoder)
        attn = self.config["attention"]
        self.aspp_cross_attention = TissueGuidedCrossAttention(
            channels=CONTEXT,
            heads=int(attn.get("num_heads", 4)),
            token_grid=attn.get("token_grid", [16, 16]),
            use_coordinates=bool(attn.get("use_coordinates", True))
        )
        self.decoder_t2c = TissueToCell(CONTEXT, CONTEXT)

    def forward(self, cell_image, tissue_image, cell_box):
        cell = self.cell.encoder(cell_image)
        tissue = self.tissue.encoder(tissue_image)
        # No cell -> tissue path (the tissue task is independent of cell input).
        cell_context = self.cell.aspp(cell["high"])
        tissue_context = self.tissue.aspp(tissue["high"])
        cell_context = self.aspp_cross_attention(
            cell_context, tissue_context, cell_box
        )
        cell_decoded = self.cell.decoder(cell_context, cell["low"])
        tissue_decoded = self.tissue.decoder(tissue_context, tissue["low"])
        # Keep the fine-scale tissue -> cell coordinate crop from M0.
        cell_decoded = self.decoder_t2c(cell_decoded, tissue_decoded, cell_box)

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
        return [*self.cell.encoder.parameters(), *self.tissue.encoder.parameters()]

    def new_module_parameters(self):
        enc = {id(p) for p in self.encoder_parameters()}
        return [p for p in self.parameters() if id(p) not in enc]


def build_model(config=None, weights=None):
    """Create both SSL ResNet-50 encoders initialized from the same Lunit BT weights."""
    weights = weights or resolve_paths()["ssl_weights"]
    return M1(load_ssl_encoder(weights), load_ssl_encoder(weights), config)
