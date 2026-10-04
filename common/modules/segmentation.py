import torch
from torch import nn
from torch.nn import functional as F


def norm(channels):
    return nn.GroupNorm(32 if channels % 32 == 0 else 8, channels)


def project(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        norm(out_channels),
        nn.ReLU(inplace=True)
    )


def separable(in_channels, out_channels, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, in_channels, 3,
            padding=dilation, dilation=dilation,
            groups=in_channels, bias=False
        ),
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        norm(out_channels),
        nn.ReLU(inplace=True)
    )


class ASPP(nn.Module):
    def __init__(self, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            project(2048, 256),
            *[separable(2048, 256, rate) for rate in rates]
        ])
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), project(2048, 256)
        )
        self.merge = nn.Sequential(
            project(256 * (len(rates) + 2), 256),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        parts = [branch(x) for branch in self.branches]
        global_feature = F.interpolate(
            self.global_branch(x), size=x.shape[-2:],
            mode="bilinear", align_corners=False
        )
        return self.merge(torch.cat([*parts, global_feature], dim=1))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.low_projection = project(256, 48)
        self.refine = nn.Sequential(
            separable(256 + 48, 256),
            separable(256, 256)
        )

    def forward(self, context, low):
        context = F.interpolate(
            context, size=low.shape[-2:],
            mode="bilinear", align_corners=False
        )
        return self.refine(
            torch.cat([context, self.low_projection(low)], dim=1)
        )


class TaskBranch(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.aspp = ASPP()
        self.decoder = Decoder()
        self.head = nn.Conv2d(256, 3, 1)

    def forward(self, x):
        features = self.encoder(x)
        context = self.aspp(features["high"])
        decoded = self.decoder(context, features["low"])
        logits = F.interpolate(
            self.head(decoded), size=x.shape[-2:],
            mode="bilinear", align_corners=False
        )
        return {
            "low": features["low"],
            "high": features["high"],
            "context": context,
            "decoded": decoded,
            "logits": logits
        }
