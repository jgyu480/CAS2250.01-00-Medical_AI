"""M1-1: frozen M1 teacher, independent cell-only decoder/head, spatial gate.

Both tissue->cell paths of M1 (ASPP attention and decoder coordinate crop)
remain entirely in the tissue-aware head. Neither is used by cell-only head.
"""

import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from common.modules.gated_fusion import SpatialProbabilityGate, mixture_log_probs

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def read_config(config=None):
    if config is None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("model") != "M1_1":
        raise ValueError("Expected M1_1 model configuration")
    if config.get("freeze_m1") is not True:
        raise ValueError("M1-1 initial experiment requires frozen M1")
    return config


class M1_1(nn.Module):
    def __init__(self, m1, config=None):
        super().__init__()
        self.config = read_config(config)
        self.teacher = m1
        self.local_decoder = copy.deepcopy(m1.cell.decoder)
        self.local_head = copy.deepcopy(m1.cell.head)
        settings = self.config["gate"]
        self.gate = SpatialProbabilityGate(
            channels=int(settings.get("channels", 256)),
            projection_channels=int(settings.get("projection_channels", 16)),
            hidden_channels=int(settings.get("hidden_channels", 32)),
        )
        self.stage = "gate"
        self.freeze_teacher()

    def initialize_local_from_teacher(self):
        """Call immediately AFTER loading the trained M1 checkpoint."""
        self.local_decoder.load_state_dict(self.teacher.cell.decoder.state_dict(), strict=True)
        self.local_head.load_state_dict(self.teacher.cell.head.state_dict(), strict=True)

    def freeze_teacher(self):
        self.teacher.requires_grad_(False)
        self.teacher.eval()  # ASPP has dropout; freezing parameters alone is insufficient.

    def set_stage(self, stage):
        if stage not in ("local", "gate"):
            raise ValueError("stage must be 'local' or 'gate'")
        self.stage = stage
        self.local_decoder.requires_grad_(True)
        self.local_head.requires_grad_(True)
        self.gate.requires_grad_(stage == "gate")
        self.freeze_teacher()
        if stage == "local":
            self.gate.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        if self.stage == "local":
            self.gate.eval()
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, cell_image, tissue_image, cell_box):
        # The pretrained encoders and their ASPPs are intentionally frozen.
        # These outputs are independent of the gate/local decoder updates.
        with torch.no_grad():
            cell_feats = self.teacher.cell.encoder(cell_image)
            tissue_feats = self.teacher.tissue.encoder(tissue_image)
            cell_context = self.teacher.cell.aspp(cell_feats["high"])
            tissue_context = self.teacher.tissue.aspp(tissue_feats["high"])
            fused_context = self.teacher.aspp_cross_attention(
                cell_context, tissue_context, cell_box
            )
            tissue_decoded = self.teacher.tissue.decoder(
                tissue_context, tissue_feats["low"]
            )
            fused_decoded = self.teacher.cell.decoder(
                fused_context, cell_feats["low"]
            )
            fused_decoded = self.teacher.decoder_t2c(
                fused_decoded, tissue_decoded, cell_box
            )
            teacher_small_logits = self.teacher.cell.head(fused_decoded)
            tissue_small_logits = self.teacher.tissue.head(tissue_decoded)
            fused_logits = F.interpolate(teacher_small_logits,
                                         size=cell_image.shape[-2:],
                                         mode="bilinear", align_corners=False)
            tissue_logits = F.interpolate(tissue_small_logits,
                                          size=tissue_image.shape[-2:],
                                          mode="bilinear", align_corners=False)

        # No tissue tensor participates in this decoder or head.
        local_decoded = self.local_decoder(cell_context, cell_feats["low"])
        local_logits = F.interpolate(self.local_head(local_decoded),
                                     size=cell_image.shape[-2:],
                                     mode="bilinear", align_corners=False)
        if self.stage == "local":
            # Stage A: train only the independent local head.
            gate_map = None
            final_logits = local_logits
        else:
            gate_map = self.gate(local_decoded, fused_decoded,
                                 local_logits, fused_logits)
            final_logits = mixture_log_probs(local_logits, fused_logits, gate_map)
        return {
            "cell_logits": final_logits,
            "tissue_logits": tissue_logits,
            "local_logits": local_logits,
            "m1_logits": fused_logits,
            "gate_map": gate_map,
        }


def build_model(config=None, weights=None):
    from models.M1.model import build_model as build_m1
    config = read_config(config)
    base_config = json.loads((Path(__file__).resolve().parents[1] /
                              "M1" / "config.json").read_text(encoding="utf-8"))
    return M1_1(build_m1(base_config, weights=weights), config)
