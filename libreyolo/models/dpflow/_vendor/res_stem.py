# =============================================================================
# Copyright 2025 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch.nn as nn
from .conv import Conv2dBlock


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_layer, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv2dBlock(
            in_planes, planes, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = Conv2dBlock(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm_layer(num_channels=planes)
        self.norm2 = norm_layer(num_channels=planes)
        if not stride == 1:
            self.norm3 = norm_layer(num_channels=planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                Conv2dBlock(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class ResStem(nn.Module):
    def __init__(self, hidden_chs, norm_layer):
        super(ResStem, self).__init__()
        self.norm_fn = norm_layer

        self.norm1 = norm_layer(num_channels=hidden_chs[0])

        self.conv1 = Conv2dBlock(3, hidden_chs[0], kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = hidden_chs[0]
        self.layer1 = self._make_layer(hidden_chs[0], stride=1)
        self.layer2 = self._make_layer(hidden_chs[1], stride=2)

        self.conv2 = Conv2dBlock(hidden_chs[1], hidden_chs[2], kernel_size=1)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        #         if m.weight is not None:
        #             nn.init.constant_(m.weight, 1)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.conv2(x)

        return x
