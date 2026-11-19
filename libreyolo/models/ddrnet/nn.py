"""DDRNet network definition for LibreYOLO.

Ported from the MIT-licensed official implementation,
https://github.com/ydhongHIT/DDRNet at commit
``de0db317c5af4b0946231f475cf74ca4f51b8ac2``
(``segmentation/DDRNet_23_slim.py``, ``DDRNet_23.py``, ``DDRNet_39.py``).

Module names follow upstream so official checkpoints load after stripping the
``model.`` training-wrapper prefix and dropping the ``loss.`` buffers. The
residual blocks, DAPPM and segmentation head are reused from the in-tree PIDNet
port: PIDNet inherited them from DDRNet and their definitions are identical
(same layers, names, momentum and ``align_corners=False`` interpolation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..pidnet.nn import (
    ALIGN_CORNERS,
    BN_MOM,
    DAPPM,
    BasicBlock,
    BatchNorm2d,
    Bottleneck,
    SegmentHead,
)

# The graph adds the stride-8 high-resolution branch to stride-2 projections
# of itself and resizes the context branch to ``input // 8``, so both input
# sides must be multiples of 8 for the branch shapes to agree.
STRIDE = 8
IGNORE_INDEX = 255


@dataclass(frozen=True)
class DDRNetConfig:
    layers: tuple[int, int, int, int]
    planes: int
    spp_planes: int
    head_planes: int
    split_stage3: bool


# s = DDRNet-23-slim, m = DDRNet-23, l = DDRNet-39 (upstream factory arguments).
ARCH_CONFIGS = {
    "s": DDRNetConfig(layers=(2, 2, 2, 2), planes=32, spp_planes=128, head_planes=64, split_stage3=False),
    "m": DDRNetConfig(layers=(2, 2, 2, 2), planes=64, spp_planes=128, head_planes=128, split_stage3=False),
    "l": DDRNetConfig(layers=(3, 4, 6, 3), planes=64, spp_planes=128, head_planes=256, split_stage3=True),
}

# ``imgsz`` is the whole-frame Cityscapes evaluation canvas (the upstream
# test pipeline runs 1024x2048 frames). Training samples 1024x1024 crops from
# the source frame rescaled by a factor in ``rescale_range`` (paper section
# IV-B: random scaling 0.5 to 2.0, random 1024x1024 crop, horizontal flip).
SIZE_CONFIGS = {
    size: {
        "arch": arch,
        "imgsz": (1024, 2048),
        "train_crop": (1024, 1024),
        "rescale_range": (0.5, 2.0),
    }
    for size, arch in ARCH_CONFIGS.items()
}


def _make_layer(
    block: type[BasicBlock] | type[Bottleneck],
    inplanes: int,
    planes: int,
    blocks: int,
    stride: int = 1,
) -> nn.Sequential:
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOM),
        )

    layers: list[nn.Module] = [block(inplanes, planes, stride, downsample)]
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes, stride=1, no_relu=i == blocks - 1))
    return nn.Sequential(*layers)


def _compression(inplanes: int, outplanes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        BatchNorm2d(outplanes, momentum=BN_MOM),
    )


def _down(inplanes: int, outplanes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=2, padding=1, bias=False),
        BatchNorm2d(outplanes, momentum=BN_MOM),
    )


class LibreDDRNetNet(nn.Module):
    """DDRNet-23-slim / DDRNet-23 / DDRNet-39 semantic segmentation network.

    Inputs are RGB float tensors in ``[0, 1]``; the ImageNet standardization
    upstream applies in its data pipeline happens here, in non-persistent
    buffers, so state-dict keys stay upstream-compatible and exported graphs
    carry it. ``forward`` returns stride-8 logits in eval mode and the
    ``(main, auxiliary)`` stride-8 pair in train mode; the auxiliary head reads
    the high-resolution branch after stage 3, as upstream's ``augment=True``.
    """

    IGNORE_INDEX = IGNORE_INDEX

    def __init__(self, size: str = "s", num_classes: int = 19) -> None:
        super().__init__()
        if size not in ARCH_CONFIGS:
            raise ValueError(f"Unknown DDRNet size {size!r}; choose from {tuple(ARCH_CONFIGS)}.")
        cfg = ARCH_CONFIGS[size]
        self.size = size
        self.num_classes = int(num_classes)
        self.split_stage3 = cfg.split_stage3
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        layers = cfg.layers
        planes = cfg.planes
        highres_planes = planes * 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = _make_layer(BasicBlock, planes, planes, layers[0])
        self.layer2 = _make_layer(BasicBlock, planes, planes * 2, layers[1], stride=2)

        if cfg.split_stage3:
            half = layers[2] // 2
            self.layer3_1 = _make_layer(BasicBlock, planes * 2, planes * 4, half, stride=2)
            self.layer3_2 = _make_layer(BasicBlock, planes * 4, planes * 4, half)
            self.layer4 = _make_layer(BasicBlock, planes * 4, planes * 8, layers[3], stride=2)
            self.compression3_1 = _compression(planes * 4, highres_planes)
            self.compression3_2 = _compression(planes * 4, highres_planes)
            self.compression4 = _compression(planes * 8, highres_planes)
            self.down3_1 = _down(highres_planes, planes * 4)
            self.down3_2 = _down(highres_planes, planes * 4)
        else:
            self.layer3 = _make_layer(BasicBlock, planes * 2, planes * 4, layers[2], stride=2)
            self.layer4 = _make_layer(BasicBlock, planes * 4, planes * 8, layers[3], stride=2)
            self.compression3 = _compression(planes * 4, highres_planes)
            self.compression4 = _compression(planes * 8, highres_planes)
            self.down3 = _down(highres_planes, planes * 4)

        self.down4 = nn.Sequential(
            nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes * 4, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes * 8, momentum=BN_MOM),
        )

        if cfg.split_stage3:
            half = layers[2] // 2
            self.layer3_1_ = _make_layer(BasicBlock, planes * 2, highres_planes, half)
            self.layer3_2_ = _make_layer(BasicBlock, highres_planes, highres_planes, half)
            self.layer4_ = _make_layer(BasicBlock, highres_planes, highres_planes, layers[3])
        else:
            # Upstream DDRNet-23 hard-codes two blocks in each high-res stage.
            self.layer3_ = _make_layer(BasicBlock, planes * 2, highres_planes, 2)
            self.layer4_ = _make_layer(BasicBlock, highres_planes, highres_planes, 2)

        self.layer5_ = _make_layer(Bottleneck, highres_planes, highres_planes, 1)
        self.layer5 = _make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)
        self.spp = DAPPM(planes * 16, cfg.spp_planes, planes * 4)
        self.seghead_extra = SegmentHead(highres_planes, cfg.head_planes, self.num_classes)
        self.final_layer = SegmentHead(planes * 4, cfg.head_planes, self.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        # Upstream initialization: Kaiming-normal conv weights, unit BN scale.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def replace_num_classes(self, num_classes: int) -> None:
        """Re-head the main and auxiliary classifiers; keep every other tensor."""
        num_classes = int(num_classes)
        self.num_classes = num_classes
        for head in (self.final_layer, self.seghead_extra):
            in_channels = head.conv2.in_channels
            head.conv2 = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)

    def _fuse(
        self,
        x: torch.Tensor,
        x_: torch.Tensor,
        down: nn.Module,
        compression: nn.Module,
        low: torch.Tensor,
        size: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bilateral fusion between the low- and high-resolution branches."""
        x = x + down(self.relu(x_))
        x_ = x_ + F.interpolate(
            compression(self.relu(low)),
            size=size,
            mode="bilinear",
            align_corners=ALIGN_CORNERS,
        )
        return x, x_

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = (x - self.pixel_mean.to(dtype=x.dtype)) / self.pixel_std.to(dtype=x.dtype)
        size = [x.shape[-2] // 8, x.shape[-1] // 8]

        x = self.conv1(x)
        x = self.layer1(x)
        stage2 = self.layer2(self.relu(x))

        if self.split_stage3:
            x = self.layer3_1(self.relu(stage2))
            x_ = self.layer3_1_(self.relu(stage2))
            x, x_ = self._fuse(x, x_, self.down3_1, self.compression3_1, x, size)
            x = self.layer3_2(self.relu(x))
            x_ = self.layer3_2_(self.relu(x_))
            x, x_ = self._fuse(x, x_, self.down3_2, self.compression3_2, x, size)
        else:
            x = self.layer3(self.relu(stage2))
            x_ = self.layer3_(self.relu(stage2))
            x, x_ = self._fuse(x, x_, self.down3, self.compression3, x, size)
        stage3_highres = x_

        x = self.layer4(self.relu(x))
        x_ = self.layer4_(self.relu(x_))
        x, x_ = self._fuse(x, x_, self.down4, self.compression4, x, size)

        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
            self.spp(self.layer5(self.relu(x))),
            size=size,
            mode="bilinear",
            align_corners=ALIGN_CORNERS,
        )
        main = self.final_layer(x + x_)
        if self.training:
            return main, self.seghead_extra(stage3_highres)
        return main


__all__ = [
    "ARCH_CONFIGS",
    "IGNORE_INDEX",
    "SIZE_CONFIGS",
    "STRIDE",
    "DDRNetConfig",
    "LibreDDRNetNet",
]
