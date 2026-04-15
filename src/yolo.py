import torch
import torch.nn as nn
from .backbones import ResNetBackbone


# These layers are based on DetNet: A Backbone network for Object Detection,
# https://arxiv.org/pdf/1804.06215.pdf
class DetNetBottleneck(nn.Module):
    # We keep the same grid size in the output. (SxS)
    # Layer structure is 1x1 conv, dilated 3x3 conv, 1x1 conv, with a skip connection
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type="A", norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=2,
            bias=False,
            dilation=2,
        )
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = norm_layer(self.expansion * planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes or block_type == "B":
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = self.relu(out)
        return out


class DetNet(nn.Module):
    def __init__(
        self,
        name="resnet18",
        num_classes=20,
        boxes_per_cell=2,
        head_channels=640,
    ):
        super().__init__()
        self.backbone = ResNetBackbone(name=name)
        self.head_channels = head_channels

        norm_layer = self.backbone.norm_layer
        out_channels = self.backbone.out_channels
        self.layer5 = self._make_detnet_layer(out_channels, head_channels, norm_layer)

        self.refine = nn.Sequential(
            nn.Conv2d(
                head_channels, head_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_channels, head_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(head_channels),
        )
        self.refine_relu = nn.ReLU(inplace=True)

        # Final YOLO prediction head: 2 boxes * 5 values + 20 class scores = 30 channels
        output_channels = boxes_per_cell * 5 + num_classes
        self.conv_end = nn.Conv2d(
            head_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn_end = norm_layer(output_channels)
        self._init_detector_layers()

    def _make_detnet_layer(self, in_channels, planes, norm_layer, num_blocks=3):
        layers = [
            DetNetBottleneck(
                in_planes=in_channels,
                planes=planes,
                block_type="B",
                norm_layer=norm_layer,
            )
        ]
        for _ in range(1, num_blocks):
            layers.append(
                DetNetBottleneck(
                    in_planes=planes,
                    planes=planes,
                    block_type="A",
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def _init_detector_layers(self):
        modules = [self.layer5, self.refine, self.conv_end, self.bn_end]

        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.layer5(x)
        residual = x
        x = self.refine(x)
        x = self.refine_relu(x + residual)
        x = self.conv_end(x)
        x = self.bn_end(x)
        x = torch.sigmoid(x)
        x = x.permute(0, 2, 3, 1)
        return x
