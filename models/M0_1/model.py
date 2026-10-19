"""SegFormer-style decoder on the unchanged M0 SSL ResNet-50 architecture.

Retains M0's three exchange directions. Deep projected features replace the ASPP
exchange port. This is a ResNet hybrid, not the original MiT-based SegFormer.
M0 already uses a stage-1 skip; this variant additionally fuses stages 2 and 3.
"""
import torch
from torch import nn
from torch.nn import functional as F

from common.backbones.ssl_resnet import SSLResNet50
from common.modules.exchange import CellToTissue, TissueToCell
from common.modules.segmentation import norm


class MultiscaleSSLResNet50(SSLResNet50):
    # Identical parameters and stride/dilation to M0, including frozen BN stats.
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]


class MultiLevelDecoder(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        # 1x1 convolution is a per-pixel linear channel projection.
        self.projections = nn.ModuleList(
            nn.Conv2d(c, width, 1) for c in (256, 512, 1024, 2048))
        self.fusion = nn.Sequential(
            nn.Conv2d(4*width, width, 1, bias=False), norm(width),
            nn.ReLU(inplace=True), nn.Dropout2d(0.1))

    def project(self, features):
        return [layer(x) for layer, x in zip(self.projections, features)]

    def fuse(self, projected):
        size = projected[0].shape[-2:]
        # Deepest-to-shallowest concatenation, as in the reference head.
        features = [F.interpolate(x, size=size, mode="bilinear", align_corners=False)
                    for x in reversed(projected)]
        return self.fusion(torch.cat(features, dim=1))


class Branch(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.encoder = MultiscaleSSLResNet50()
        self.decoder = MultiLevelDecoder(width)
        self.head = nn.Conv2d(width, 3, 1)


class M0_1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config.get("decoder_width", 256)
        self.cell, self.tissue = Branch(width), Branch(width)
        self.encoder_c2t = CellToTissue(2048, 2048)
        self.context_t2c = TissueToCell(width, width)
        self.decoder_t2c = TissueToCell(width, width)

    def forward(self, cell_image, tissue_image, cell_box):
        cell = self.cell.encoder(cell_image)
        tissue = self.tissue.encoder(tissue_image)
        tissue[-1] = self.encoder_c2t(tissue[-1], cell[-1], cell_box)
        cell = self.cell.decoder.project(cell)
        tissue = self.tissue.decoder.project(tissue)
        cell[-1] = self.context_t2c(cell[-1], tissue[-1], cell_box)
        cell = self.cell.decoder.fuse(cell)
        tissue = self.tissue.decoder.fuse(tissue)
        cell = self.decoder_t2c(cell, tissue, cell_box)
        return {
            "cell_logits": F.interpolate(self.cell.head(cell), cell_image.shape[-2:],
                                          mode="bilinear", align_corners=False),
            "tissue_logits": F.interpolate(self.tissue.head(tissue), tissue_image.shape[-2:],
                                            mode="bilinear", align_corners=False)}

    def encoder_parameters(self):
        return [*self.cell.encoder.parameters(), *self.tissue.encoder.parameters()]

    def new_module_parameters(self):
        ids = {id(p) for p in self.encoder_parameters()}
        return [p for p in self.parameters() if id(p) not in ids]
