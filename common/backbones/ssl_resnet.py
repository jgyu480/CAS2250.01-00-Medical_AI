from pathlib import Path
import torch
from torch import nn
from torchvision.models.resnet import Bottleneck, ResNet


class SSLResNet50(ResNet):
    def __init__(self, freeze_bn_stats=True):
        super().__init__(
            Bottleneck, [3, 4, 6, 3],
            replace_stride_with_dilation=[False, False, True]
        )
        del self.fc
        self.freeze_bn_stats = freeze_bn_stats
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn_stats:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        high = self.layer4(x)
        return {"low": low, "high": high}


def load_ssl_encoder(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SSL 가중치가 없습니다: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("공식 state_dict 형식이 아닙니다.")
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor)
               for k, v in state.items()):
        raise ValueError("가중치 파일 형식이 다릅니다.")
    encoder = SSLResNet50()
    encoder.load_state_dict(state, strict=True)
    if not all(torch.equal(value, state[key])
               for key, value in encoder.state_dict().items()):
        raise ValueError("파일과 로딩한 파라미터가 다릅니다.")
    return encoder
