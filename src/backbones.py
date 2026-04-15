import inspect

import torch.nn as nn
from torchvision.models import (
    ResNet101_Weights,
    ResNet152_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet101,
    resnet152,
    resnet18,
    resnet34,
    resnet50,
)
from transformers import AutoModel


def _get_resnet(name):
    name = str(name).lower()
    if name == "resnet18":
        return resnet18(weights=ResNet18_Weights.DEFAULT)
    if name == "resnet34":
        return resnet34(weights=ResNet34_Weights.DEFAULT)
    if name == "resnet50":
        return resnet50(weights=ResNet50_Weights.DEFAULT)
    if name == "resnet101":
        return resnet101(weights=ResNet101_Weights.DEFAULT)
    if name == "resnet152":
        return resnet152(weights=ResNet152_Weights.DEFAULT)
    raise ValueError(f"Unsupported backbone: {name}")


class ResNetBackbone(nn.Module):
    def __init__(self, name="resnet18"):
        super().__init__()
        backbone = _get_resnet(name)
        self.body = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.out_channels = backbone.fc.in_features
        self.norm_layer = getattr(backbone, "_norm_layer", nn.BatchNorm2d)
        self.input_size_multiple = 32

    def forward(self, x):
        return self.body(x)


class ViTBackbone(nn.Module):
    def __init__(self, name="facebook/dinov2-with-registers-small"):
        super().__init__()
        self.body = AutoModel.from_pretrained(name, trust_remote_code=False)

        self.out_channels = self.body.config.hidden_size
        self.patch_size = self.body.config.patch_size
        self.input_size_multiple = int(self.patch_size)
        self.num_register_tokens = getattr(self.body.config, "num_register_tokens", 0)

    def forward(self, x, return_pooled_output=False):
        if x.ndim != 4:
            raise ValueError("expected x to be a 4D tensor (B,C,H,W)")

        b, _c, h, w = x.shape
        if (h % self.patch_size) != 0 or (w % self.patch_size) != 0:
            raise ValueError(
                f"input spatial size ({h},{w}) must be divisible by patch_size ({self.patch_size})"
            )

        kwargs = {}
        try:
            if (
                "interpolate_pos_encoding"
                in inspect.signature(self.body.forward).parameters
            ):
                kwargs["interpolate_pos_encoding"] = True
        except (TypeError, ValueError):
            pass

        out = self.body(pixel_values=x, **kwargs)
        tokens = out.last_hidden_state

        gh = h // self.patch_size
        gw = w // self.patch_size
        num_patches = gh * gw
        num_special_tokens = tokens.shape[1] - num_patches
        if num_special_tokens < 0:
            raise ValueError(
                f"unexpected token sequence length {tokens.shape[1]} for input ({h},{w}) "
                f"with patch_size={self.patch_size} (expected at least {num_patches})"
            )
        if num_special_tokens != 1 + self.num_register_tokens:
            raise ValueError("unexpected number of special tokens; try another model")

        if num_special_tokens > 0:
            tokens = tokens[:, num_special_tokens:]

        feature_map = (
            tokens.transpose(1, 2).contiguous().view(b, self.out_channels, gh, gw)
        )
        if return_pooled_output:
            return feature_map, getattr(out, "pooler_output", None)
        return feature_map
