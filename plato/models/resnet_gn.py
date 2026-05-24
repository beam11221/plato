"""
The ResNet model with GroupNorm (for the CIFAR-10 dataset only).

Drop-in replacement for PLATO's resnet.py that swaps every BatchNorm2d
for GroupNorm to address client drift under non-IID partitions.

Reference:
    Hsieh et al., "The Non-IID Data Quagmire of Decentralized Machine
    Learning", ICML 2020.

The bn1/bn2/bn3 attribute names are retained for structural parity with
upstream PLATO (cut-layer split, log compatibility). The layers
themselves are GroupNorm, not BatchNorm.
"""

import collections
from typing import Any, cast

import torch.nn as nn
import torch.nn.functional as F


DEFAULT_NUM_GROUPS = 32


def _norm(channels: int, num_groups: int = DEFAULT_NUM_GROUPS) -> nn.GroupNorm:
    """GroupNorm with a safe fallback if channels < num_groups or
    are not divisible by num_groups. For ResNet-18 with channel counts
    in {64, 128, 256, 512}, this always resolves to exactly 32 groups."""
    g = min(num_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=channels)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = _norm(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = _norm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _norm(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = _norm(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = _norm(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = _norm(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _norm(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Model(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cut_layer=None):
        super().__init__()

        self.in_planes = cast(Any, 64)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = _norm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        # Preparing named layers so the model can be split across client/server
        self.layers = cast(Any, [])
        self.layerdict = cast(Any, collections.OrderedDict())
        self.layerdict["conv1"] = self.conv1
        self.layerdict["bn1"] = self.bn1
        self.layerdict["relu"] = F.relu
        self.layerdict["layer1"] = self.layer1
        self.layerdict["layer2"] = self.layer2
        self.layerdict["layer3"] = self.layer3
        self.layerdict["layer4"] = self.layer4
        self.layers.append("conv1")
        self.layers.append("bn1")
        self.layers.append("relu")
        self.layers.append("layer1")
        self.layers.append("layer2")
        self.layers.append("layer3")
        self.layers.append("layer4")
        self.cut_layer = cast(Any, cut_layer)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward_to(self, x):
        """Forward pass, but only to the layer specified by cut_layer."""
        layer_index = self.layers.index(self.cut_layer)
        for i in range(0, layer_index + 1):
            x = self.layerdict[self.layers[i]](x)
        return x

    def forward_from(self, x):
        """Forward pass, starting from the layer specified by cut_layer."""
        layer_index = self.layers.index(self.cut_layer)
        for i in range(layer_index + 1, len(self.layers)):
            x = self.layerdict[self.layers[i]](x)

        out = F.avg_pool2d(x, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    @staticmethod
    def is_valid_model_type(model_type):
        return (
            model_type.startswith("resnet_")
            and len(model_type.split("_")) == 2
            and int(model_type.split("_")[1]) in [18, 34, 50, 101, 152]
        )

    @staticmethod
    def get(model_name=None, num_classes=None, cut_layer=None, **kwargs):
        """Returns a suitable ResNet model according to its type."""
        if not isinstance(model_name, str):
            raise ValueError("A valid ResNet model name must be provided.")

        if not Model.is_valid_model_type(model_name):
            raise ValueError(f"Invalid Resnet model name: {model_name}")

        resnet_type = int(model_name.split("_")[1])

        if num_classes is None:
            num_classes = 10

        if resnet_type == 18:
            return Model(BasicBlock, [2, 2, 2, 2], num_classes, cut_layer)
        elif resnet_type == 34:
            return Model(BasicBlock, [3, 4, 6, 3], num_classes, cut_layer)
        elif resnet_type == 50:
            return Model(Bottleneck, [3, 4, 6, 3], num_classes, cut_layer)
        elif resnet_type == 101:
            return Model(Bottleneck, [3, 4, 23, 3], num_classes, cut_layer)
        elif resnet_type == 152:
            return Model(Bottleneck, [3, 8, 36, 3], num_classes, cut_layer)

        return None