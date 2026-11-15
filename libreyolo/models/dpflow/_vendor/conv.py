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

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import GroupNorm2d, BatchNorm2d, LayerNorm2d


def conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    pre_act_fn=None,
    post_act_fn=None,
    pre_norm=None,
    post_norm=None,
    is_transpose: bool = False,
) -> torch.Tensor:
    if pre_norm is not None:
        x = pre_norm(x)

    if pre_act_fn is not None:
        x = pre_act_fn(x)

    if is_transpose:
        x = F.conv_transpose2d(
            x,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    else:
        x = F.conv2d(
            x,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    if post_norm is not None:
        x = post_norm(x)

    if post_act_fn is not None:
        x = post_act_fn(x)

    return x


class ConvBase(nn.Module):
    def __init__(
        self,
        is_transpose,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        pre_norm=None,
        post_norm=None,
        group_norm_groups=8,
        pre_act_fn=None,
        post_act_fn=None,
    ) -> None:
        super().__init__()
        self.is_transpose = is_transpose
        self.in_channels = in_channels
        self.out_channels = out_channels
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype
        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.pre_act_fn = pre_act_fn
        self.post_act_fn = post_act_fn

        if self.is_transpose:
            self.register_parameter(
                "weight",
                nn.Parameter(
                    torch.zeros(
                        in_channels // groups,
                        out_channels,
                        kernel_size[0],
                        kernel_size[1],
                        device=device,
                        dtype=dtype,
                    )
                ),
            )
        else:
            self.register_parameter(
                "weight",
                nn.Parameter(
                    torch.zeros(
                        out_channels,
                        in_channels // groups,
                        kernel_size[0],
                        kernel_size[1],
                        device=device,
                        dtype=dtype,
                    )
                ),
            )

        if bias:
            self.register_parameter(
                "bias",
                nn.Parameter(torch.zeros(out_channels, device=device, dtype=dtype)),
            )
        else:
            self.bias = None

        if isinstance(pre_norm, str):
            if pre_norm == "instance":
                self.pre_norm = nn.InstanceNorm2d(1, track_running_stats=False)
            elif pre_norm == "group":
                self.pre_norm = GroupNorm2d(
                    group_norm_groups, out_channels, affine=False
                )
            elif pre_norm == "batch":
                self.pre_norm = BatchNorm2d(out_channels)
            elif pre_norm == "layer":
                self.pre_norm = LayerNorm2d(out_channels)
        else:
            self.pre_norm = pre_norm

        if isinstance(post_norm, str):
            if post_norm == "instance":
                self.post_norm = nn.InstanceNorm2d(1, track_running_stats=False)
            elif post_norm == "group":
                self.post_norm = GroupNorm2d(
                    group_norm_groups, out_channels, affine=False
                )
            elif post_norm == "batch":
                self.post_norm = BatchNorm2d(out_channels)
            elif post_norm == "layer":
                self.post_norm = LayerNorm2d(out_channels)
        else:
            self.post_norm = post_norm

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv2d(
            x=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            pre_act_fn=self.pre_act_fn,
            post_act_fn=self.post_act_fn,
            pre_norm=self.pre_norm,
            post_norm=self.post_norm,
            is_transpose=self.is_transpose,
        )


class Conv2dBlock(ConvBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        pre_norm=None,
        post_norm=None,
        group_norm_groups=8,
        pre_act_fn=None,
        post_act_fn=None,
    ) -> None:
        super().__init__(
            is_transpose=False,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            pre_norm=pre_norm,
            post_norm=post_norm,
            group_norm_groups=group_norm_groups,
            pre_act_fn=pre_act_fn,
            post_act_fn=post_act_fn,
        )


class ConvTranspose2dBlock(ConvBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        pre_norm=None,
        post_norm=None,
        group_norm_groups=8,
        pre_act_fn=None,
        post_act_fn=None,
    ) -> None:
        super().__init__(
            is_transpose=True,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            pre_norm=pre_norm,
            post_norm=post_norm,
            group_norm_groups=group_norm_groups,
            pre_act_fn=pre_act_fn,
            post_act_fn=post_act_fn,
        )
