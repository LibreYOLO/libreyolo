# Vendored from https://github.com/princeton-vl/SEA-RAFT/blob/main/core/extractor.py
# Copyright (c) 2024, Princeton Vision & Learning Lab. BSD-3-Clause (see LICENSE).
# The torchvision ImageNet initialisation (training only) is removed.
import torch.nn as nn

from .layer import BasicBlock, conv1x1


class ResNetFPN(nn.Module):
    """
    ResNet18, output resolution is 1/8.
    Each block has 2 layers.
    """
    def __init__(self, args, input_dim=3, output_dim=256, ratio=1.0, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # Config
        block = BasicBlock
        block_dims = [int(d * ratio) for d in args.block_dims]
        initial_dim = args.initial_dim
        self.input_dim = input_dim
        # Class Variable
        self.in_planes = initial_dim
        # Networks
        self.conv1 = nn.Conv2d(input_dim, initial_dim, kernel_size=7, stride=2, padding=3)
        self.bn1 = norm_layer(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        if args.pretrain == 'resnet34':
            n_block = [3, 4, 6]
        elif args.pretrain == 'resnet18':
            n_block = [2, 2, 2]
        else:
            raise NotImplementedError
        self.layer1 = self._make_layer(block, block_dims[0], stride=1, norm_layer=norm_layer, num=n_block[0])  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2, norm_layer=norm_layer, num=n_block[1])  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2, norm_layer=norm_layer, num=n_block[2])  # 1/8
        self.final_conv = conv1x1(block_dims[2], output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1, norm_layer=nn.BatchNorm2d, num=2):
        layers = []
        layers.append(block(self.in_planes, dim, stride=stride, norm_layer=norm_layer))
        for i in range(num - 1):
            layers.append(block(dim, dim, stride=1, norm_layer=norm_layer))
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x = self.relu(self.bn1(self.conv1(x)))
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
        # Output
        output = self.final_conv(x)
        return output
