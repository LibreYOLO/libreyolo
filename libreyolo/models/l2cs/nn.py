"""L2CS-Net gaze estimation network.

Ported from Ahmednull/L2CS-Net (MIT, IEEE-ICIP 2022, Abdelrahman & Hempel).
Two-stage gaze pipeline: a face crop goes through a ResNet trunk and then
two parallel classification heads — one for yaw, one for pitch — each
producing logits over `num_bins` angular bins. Continuous angles are
recovered via softmax expectation outside this module (see ``utils.py``).
"""

from __future__ import annotations

import math
from typing import List, Tuple, Type

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck


_RESNET_LAYERS: dict[str, Tuple[Type[nn.Module], List[int]]] = {
    "r18": (BasicBlock, [2, 2, 2, 2]),
    "r34": (BasicBlock, [3, 4, 6, 3]),
    "r50": (Bottleneck, [3, 4, 6, 3]),
    "r101": (Bottleneck, [3, 4, 23, 3]),
    "r152": (Bottleneck, [3, 8, 36, 3]),
}


class L2CS(nn.Module):
    """ResNet trunk with two parallel angle-bin heads (yaw, pitch).

    The ``fc_finetune`` linear layer in the upstream checkpoint is vestigial
    (defined but unused in upstream forward); we omit it here and rely on
    ``strict=False`` weight loading to silently drop those keys.
    """

    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        num_bins: int = 90,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = 512 * block.expansion
        self.fc_yaw_gaze = nn.Linear(feat_dim, num_bins)
        self.fc_pitch_gaze = nn.Linear(feat_dim, num_bins)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self,
        block: Type[nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers: List[nn.Module] = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(yaw_logits, pitch_logits)``, each shape ``(B, num_bins)``."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc_yaw_gaze(x), self.fc_pitch_gaze(x)


def build_l2cs(size: str, num_bins: int = 90) -> L2CS:
    """Instantiate an L2CS network for a given ResNet size code."""
    size = size.lower()
    if size not in _RESNET_LAYERS:
        valid = ", ".join(sorted(_RESNET_LAYERS))
        raise ValueError(f"Unknown L2CS size {size!r}. Valid sizes: {valid}.")
    block, layers = _RESNET_LAYERS[size]
    return L2CS(block, layers, num_bins=num_bins)


def detect_size_from_state_dict(state_dict: dict) -> str | None:
    """Infer the ResNet size code from a state dict's layer-depth fingerprint."""
    block_counts = []
    for stage in range(1, 5):
        i = 0
        while f"layer{stage}.{i}.conv1.weight" in state_dict:
            i += 1
        block_counts.append(i)
    is_bottleneck = "layer1.0.conv3.weight" in state_dict
    candidates = {k: v for k, v in _RESNET_LAYERS.items()
                  if (v[0] is Bottleneck) == is_bottleneck and list(v[1]) == block_counts}
    if len(candidates) == 1:
        return next(iter(candidates))
    return None
