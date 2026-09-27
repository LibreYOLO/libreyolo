"""Native DEKR-W32-NO-DC pose graph.

Adapted from ``src/super_gradients/training/models/pose_estimation_models/dekr_hrnet.py``
in ``Deci-AI/super-gradients`` at commit
``63de22c404d5740f34f7706c302b37fce3c8fe5d`` (Apache-2.0). That file carries an
upstream Microsoft MIT header and is itself based on
``HRNet/HigherHRNet-Human-Pose-Estimation`` (MIT), modified by Zigang Geng.

Attribute names and forward arithmetic intentionally match upstream so the
released checkpoint loads strictly and inference is bit-identical.

The shipped variant replaces the paper's deformable ``ADAPTIVE`` offset block
with a standard ``BASIC`` block at dilation 5, which is what makes the graph
exportable. It is a different architecture from the original deformable
DEKR-W32 and its checkpoints are not interchangeable.

Copyright (c) Microsoft. Licensed under the MIT License.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "DEKR_W32_NO_DC_SPEC",
    "BasicBlock",
    "Bottleneck",
    "HighResolutionModule",
    "LibreDEKRModel",
]

# Mirrors recipes/arch_params/pose_dekr_w32_no_dc_arch_params.yaml at the pinned
# SuperGradients commit. Kept as a literal so the port does not depend on
# SuperGradients at runtime.
DEKR_W32_NO_DC_SPEC: dict = {
    "FINAL_CONV_KERNEL": 1,
    "STAGES": {
        "NUM_STAGES": 3,
        "NUM_MODULES": [1, 4, 3],
        "NUM_BRANCHES": [2, 3, 4],
        "BLOCK": ["BASIC", "BASIC", "BASIC"],
        "NUM_BLOCKS": [[4, 4], [4, 4, 4], [4, 4, 4, 4]],
        "NUM_CHANNELS": [[32, 64], [32, 64, 128], [32, 64, 128, 256]],
        "FUSE_METHOD": ["SUM", "SUM", "SUM"],
    },
    "HEAD_HEATMAP": {
        "BLOCK": "BASIC",
        "NUM_BLOCKS": 1,
        "NUM_CHANNELS": 32,
        "DILATION_RATE": 1,
        # False upstream: the graph emits raw logits and the decoder applies
        # the sigmoid. Nothing in this port flips it.
        "HEATMAP_APPLY_SIGMOID": False,
    },
    "HEAD_OFFSET": {
        # Upstream swapped ADAPTIVE (deformable) for BASIC at dilation 5 so the
        # graph exports to TensorRT/ONNX without custom operators.
        "BLOCK": "BASIC",
        "DILATION_RATE": 5,
        "NUM_BLOCKS": 2,
        "NUM_CHANNELS_PERKPT": 15,
    },
}


class BasicBlock(nn.Module):
    """Two-convolution residual block with a configurable dilation rate.

    ``conv2`` takes ``inplanes`` (not ``planes``) input channels upstream. Every
    call site builds the block with ``inplanes == planes``, so the two spellings
    are shape-identical; the upstream spelling is kept so a reader comparing the
    two files sees no unexplained difference.
    """

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        return self.relu(out)


class Bottleneck(nn.Module):
    """ResNet bottleneck used by the stride-four stem stack."""

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        return self.relu(out)


BLOCKS: dict[str, type[nn.Module]] = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class HighResolutionModule(nn.Module):
    """Parallel resolution branches followed by repeated summation fusion."""

    def __init__(
        self,
        num_branches: int,
        block: type[nn.Module],
        num_blocks: list[int],
        num_inchannels: list[int],
        num_channels: list[int],
        fuse_method: str,
        multi_scale_output: bool = True,
    ) -> None:
        super().__init__()
        for name, values in (
            ("NUM_BLOCKS", num_blocks),
            ("NUM_CHANNELS", num_channels),
            ("NUM_INCHANNELS", num_inchannels),
        ):
            if num_branches != len(values):
                raise ValueError(
                    f"NUM_BRANCHES({num_branches}) <> {name}({len(values)})"
                )

        self.num_inchannels = list(num_inchannels)
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_one_branch(
        self,
        branch_index: int,
        block: type[nn.Module],
        num_blocks: list[int],
        num_channels: list[int],
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        out_channels = num_channels[branch_index] * block.expansion
        if stride != 1 or self.num_inchannels[branch_index] != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers: list[nn.Module] = [
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        ]
        self.num_inchannels[branch_index] = out_channels
        for _ in range(1, num_blocks[branch_index]):
            layers.append(
                block(self.num_inchannels[branch_index], num_channels[branch_index])
            )
        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches: int,
        block: type[nn.Module],
        num_blocks: list[int],
        num_channels: list[int],
    ) -> nn.ModuleList:
        return nn.ModuleList(
            [
                self._make_one_branch(i, block, num_blocks, num_channels)
                for i in range(num_branches)
            ]
        )

    def _make_fuse_layers(self) -> nn.ModuleList | None:
        if self.num_branches == 1:
            return None

        num_inchannels = self.num_inchannels
        fuse_layers: list[nn.ModuleList] = []
        output_branches = self.num_branches if self.multi_scale_output else 1
        for i in range(output_branches):
            fuse_layer: list[nn.Module | None] = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s: list[nn.Module] = []
                    for k in range(i - j):
                        is_last = k == i - j - 1
                        out_channels = (
                            num_inchannels[i] if is_last else num_inchannels[j]
                        )
                        parts: list[nn.Module] = [
                            nn.Conv2d(
                                num_inchannels[j], out_channels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(out_channels),
                        ]
                        if not is_last:
                            parts.append(nn.ReLU(True))
                        conv3x3s.append(nn.Sequential(*parts))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self) -> list[int]:
        """Return branch widths after this module was constructed."""
        return self.num_inchannels

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        x = [self.branches[i](x[i]) for i in range(self.num_branches)]

        assert self.fuse_layers is not None
        x_fuse: list[torch.Tensor] = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                y = y + x[j] if i == j else y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class LibreDEKRModel(nn.Module):
    """DEKR-W32-NO-DC: HRNet trunk, heatmap head, per-keypoint offset heads.

    Forward returns ``(heatmap_logits, offsets)`` with shapes
    ``(B, K + 1, H / 4, W / 4)`` and ``(B, 2K, H / 4, W / 4)``. The final
    heatmap channel is the person centre. No sigmoid, peak finding, coordinate
    decode, NMS, or test-time augmentation lives in this graph.
    """

    def __init__(self, num_keypoints: int = 17, in_channels: int = 3) -> None:
        super().__init__()
        if num_keypoints < 1:
            raise ValueError(f"num_keypoints must be >= 1, got {num_keypoints}")

        spec = DEKR_W32_NO_DC_SPEC
        self.spec = spec
        self.stages_spec = spec["STAGES"]
        self.num_stages = int(self.stages_spec["NUM_STAGES"])

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        num_channels_last = [256]
        for i in range(self.num_stages):
            num_channels = list(self.stages_spec["NUM_CHANNELS"][i])
            setattr(
                self,
                f"transition{i + 1}",
                self._make_transition_layer(num_channels_last, num_channels),
            )
            stage, num_channels_last = self._make_stage(i, num_channels)
            setattr(self, f"stage{i + 2}", stage)

        self.head_inp_channels = int(sum(self.stages_spec["NUM_CHANNELS"][-1]))
        self.config_heatmap = spec["HEAD_HEATMAP"]
        self.config_offset = spec["HEAD_OFFSET"]
        self.num_joints = int(num_keypoints)
        self.num_offset = self.num_joints * 2
        self.num_joints_with_center = self.num_joints + 1
        self.offset_prekpt = int(self.config_offset["NUM_CHANNELS_PERKPT"])

        self.transition_heatmap = self._make_transition_for_head(
            self.head_inp_channels, int(self.config_heatmap["NUM_CHANNELS"])
        )
        self.transition_offset = self._make_transition_for_head(
            self.head_inp_channels, self.num_joints * self.offset_prekpt
        )
        self.head_heatmap = self._make_heatmap_head()
        (
            self.offset_feature_layers,
            self.offset_final_layer,
        ) = self._make_separate_regression_head()

    # -- construction helpers ------------------------------------------------

    @staticmethod
    def _make_transition_for_head(inplanes: int, outplanes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True),
        )

    def _final_conv_padding(self) -> int:
        return 1 if int(self.spec["FINAL_CONV_KERNEL"]) == 3 else 0

    def _make_heatmap_head(self) -> nn.ModuleList:
        config = self.config_heatmap
        channels = int(config["NUM_CHANNELS"])
        feature_conv = self._make_layer(
            BLOCKS[config["BLOCK"]],
            channels,
            channels,
            int(config["NUM_BLOCKS"]),
            dilation=int(config["DILATION_RATE"]),
        )
        heatmap_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=self.num_joints_with_center,
            kernel_size=int(self.spec["FINAL_CONV_KERNEL"]),
            stride=1,
            padding=self._final_conv_padding(),
        )
        return nn.ModuleList([feature_conv, heatmap_conv])

    def _make_separate_regression_head(self) -> tuple[nn.ModuleList, nn.ModuleList]:
        config = self.config_offset
        channels = int(config["NUM_CHANNELS_PERKPT"])
        offset_feature_layers: list[nn.Module] = []
        offset_final_layer: list[nn.Module] = []
        for _ in range(self.num_joints):
            offset_feature_layers.append(
                self._make_layer(
                    BLOCKS[config["BLOCK"]],
                    channels,
                    channels,
                    int(config["NUM_BLOCKS"]),
                    dilation=int(config["DILATION_RATE"]),
                )
            )
            offset_final_layer.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=2,
                    kernel_size=int(self.spec["FINAL_CONV_KERNEL"]),
                    stride=1,
                    padding=self._final_conv_padding(),
                )
            )
        return nn.ModuleList(offset_feature_layers), nn.ModuleList(offset_final_layer)

    @staticmethod
    def _make_layer(
        block: type[nn.Module],
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
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
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers: list[nn.Module] = [
            block(inplanes, planes, stride, downsample, dilation=dilation)
        ]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_transition_layer(
        num_channels_pre_layer: list[int],
        num_channels_cur_layer: list[int],
    ) -> nn.ModuleList:
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers: list[nn.Module | None] = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s: list[nn.Module] = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else in_channels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_stage(
        self,
        stage_index: int,
        num_inchannels: list[int],
    ) -> tuple[nn.Sequential, list[int]]:
        spec = self.stages_spec
        num_modules = int(spec["NUM_MODULES"][stage_index])
        num_branches = int(spec["NUM_BRANCHES"][stage_index])
        num_blocks = list(spec["NUM_BLOCKS"][stage_index])
        num_channels = list(spec["NUM_CHANNELS"][stage_index])
        block = BLOCKS[spec["BLOCK"][stage_index]]
        fuse_method = spec["FUSE_METHOD"][stage_index]

        modules: list[nn.Module] = []
        for _ in range(num_modules):
            module = HighResolutionModule(
                num_branches,
                block,
                num_blocks,
                num_inchannels,
                num_channels,
                fuse_method,
                True,
            )
            modules.append(module)
            num_inchannels = module.get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    # -- head rebuild --------------------------------------------------------

    def replace_head(self, num_keypoints: int) -> None:
        """Rebuild both heads for a different keypoint count, keeping the trunk."""
        if num_keypoints < 1:
            raise ValueError(f"num_keypoints must be >= 1, got {num_keypoints}")
        self.num_joints = int(num_keypoints)
        self.num_offset = self.num_joints * 2
        self.num_joints_with_center = self.num_joints + 1
        self.transition_offset = self._make_transition_for_head(
            self.head_inp_channels, self.num_joints * self.offset_prekpt
        )
        self.head_heatmap = self._make_heatmap_head()
        (
            self.offset_feature_layers,
            self.offset_final_layer,
        ) = self._make_separate_regression_head()

    # -- forward -------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        y_list = [x]
        for i in range(self.num_stages):
            transition = getattr(self, f"transition{i + 1}")
            x_list = []
            for j in range(int(self.stages_spec["NUM_BRANCHES"][i])):
                if transition[j] is not None:
                    x_list.append(transition[j](y_list[-1]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, f"stage{i + 2}")(x_list)

        x0_h, x0_w = y_list[0].shape[2], y_list[0].shape[3]
        # F.upsample(mode="bilinear") upstream, which is F.interpolate with
        # align_corners=False. Spelled explicitly here; the deprecated alias
        # emits a warning and would otherwise hide the corner convention.
        features = torch.cat(
            [y_list[0]]
            + [
                F.interpolate(
                    branch, size=(x0_h, x0_w), mode="bilinear", align_corners=False
                )
                for branch in y_list[1:]
            ],
            dim=1,
        )

        heatmap = self.head_heatmap[1](self.head_heatmap[0](self.transition_heatmap(features)))

        offset_feature = self.transition_offset(features)
        per_joint = [
            self.offset_final_layer[j](
                self.offset_feature_layers[j](
                    offset_feature[
                        :, j * self.offset_prekpt : (j + 1) * self.offset_prekpt
                    ]
                )
            )
            for j in range(self.num_joints)
        ]
        offset = torch.cat(per_joint, dim=1)
        return heatmap, offset
