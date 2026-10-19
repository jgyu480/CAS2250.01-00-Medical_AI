"""Model construction and explicit, checked M0-backbone transfer."""
import torch

from common.backbones.ssl_resnet import SSLResNet50, load_ssl_encoder
from common.paths import resolve_paths
from models.M0.model import M0
from models.M0_1.model import M0_1


def build_model(config, initialize_encoders=True):
    family = config.get("model", "M0")
    if family == "M0":
        model = M0(SSLResNet50(), SSLResNet50(), config)
    elif family == "M0_1":
        model = M0_1(config)
    else:
        raise ValueError(f"Unsupported experiment model: {family}")
    if initialize_encoders:
        encoder = load_ssl_encoder(resolve_paths()["ssl_weights"])
        for branch in (model.cell, model.tissue):
            branch.encoder.load_state_dict(encoder.state_dict(), strict=True)
    return model


def initialize_from(model, checkpoint, scope):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = state["model"]
    if scope == "full":
        model.load_state_dict(source, strict=True)
        return {"scope": scope, "checkpoint": str(checkpoint)}
    prefixes = ("cell.encoder.", "tissue.encoder.")
    report = {"scope": "encoders", "checkpoint": str(checkpoint), "keys": {}}
    for prefix, branch in zip(prefixes, (model.cell, model.tissue)):
        weights = {k[len(prefix):]: v for k, v in source.items() if k.startswith(prefix)}
        if not weights:
            raise RuntimeError(f"Checkpoint missing {prefix}")
        branch.encoder.load_state_dict(weights, strict=True)
        report["keys"][prefix] = len(weights)
    return report
