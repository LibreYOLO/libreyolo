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

from .cgu import CGUStage
from .local_timm.weight_init import trunc_normal_
from .res_stem import ResStem
from .conv import Conv2dBlock, ConvTranspose2dBlock
from .norm import GroupNorm2d
from .update import ConvGRU


class CGUBidirDualEncoder(nn.Module):
    def __init__(
        self,
        pyramid_levels: Optional[int],
        hidden_chs: int,
        out_1x1_abs_chs: int,
        out_1x1_factor: Optional[float],
        num_out_stages: int = 0,
        gru_mode: str = "gru",
        activation_function: Optional[nn.Module] = None,
        norm_layer=GroupNorm2d,
        depth: int = 2,
        mlp_ratio: float = 4,
        mlp_use_dw_conv: bool = False,
        mlp_dw_kernel_size: int = 7,
        mlp_in_kernel_size: int = 1,
        mlp_out_kernel_size: int = 1,
        cgu_layer_scale_init_value: float = 1e-2,
    ):
        super().__init__()

        self.pyramid_levels = pyramid_levels
        self.hidden_chs = hidden_chs
        self.out_1x1_abs_chs = out_1x1_abs_chs
        self.out_1x1_factor = out_1x1_factor
        self.num_out_stages = num_out_stages
        self.gru_mode = gru_mode

        self.forward_gru = ConvGRU(hidden_chs[-1], hidden_chs[-1])
        self.down_gru = Conv2dBlock(
            hidden_chs[-1], hidden_chs[-1], 3, stride=2, padding=1, bias=True
        )

        self.backward_gru = ConvGRU(hidden_chs[-1], hidden_chs[-1])
        self.up_gru = ConvTranspose2dBlock(
            hidden_chs[-1], hidden_chs[-1], 4, stride=2, padding=1, bias=True
        )

        self.stem = self._make_stem(
            [hidden_chs[0], hidden_chs[1], 2 * hidden_chs[2]],
            norm_layer=norm_layer,
        )

        self.lowres_stem = self._make_stem(
            hidden_chs,
            norm_layer=norm_layer,
        )

        if self.out_1x1_abs_chs > 0:
            self.out_1x1 = self._make_out_1x1_layer(
                hidden_chs[-1], self.out_1x1_abs_chs
            )

        self.rec_stage = self._make_stage(
            hidden_chs[-1],
            out_chs=hidden_chs[-1],
            activation_function=activation_function,
            norm_layer=norm_layer,
            depth=depth,
            mlp_ratio=mlp_ratio,
            mlp_use_dw_conv=mlp_use_dw_conv,
            mlp_dw_kernel_size=mlp_dw_kernel_size,
            mlp_in_kernel_size=mlp_in_kernel_size,
            mlp_out_kernel_size=mlp_out_kernel_size,
            cgu_layer_scale_init_value=cgu_layer_scale_init_value,
        )

        self.back_stage = self._make_stage(
            hidden_chs[-1],
            out_chs=hidden_chs[-1],
            activation_function=activation_function,
            norm_layer=norm_layer,
            depth=depth,
            mlp_ratio=mlp_ratio,
            mlp_use_dw_conv=mlp_use_dw_conv,
            mlp_dw_kernel_size=mlp_dw_kernel_size,
            mlp_in_kernel_size=mlp_in_kernel_size,
            mlp_out_kernel_size=mlp_out_kernel_size,
            stride=1,
            cgu_layer_scale_init_value=cgu_layer_scale_init_value,
        )

        if self.num_out_stages > 0:
            self.out_merge_conv = Conv2dBlock(
                3 * hidden_chs[-1], hidden_chs[-1], kernel_size=1, pre_act_fn=nn.ReLU()
            )
            self.out_stages = self._make_out_stages(
                self.num_out_stages,
                hidden_chs[-1],
                out_chs=None,
                activation_function=activation_function,
                norm_layer=norm_layer,
                depth=depth,
                mlp_ratio=mlp_ratio,
                mlp_use_dw_conv=mlp_use_dw_conv,
                mlp_dw_kernel_size=mlp_dw_kernel_size,
                mlp_in_kernel_size=mlp_in_kernel_size,
                mlp_out_kernel_size=mlp_out_kernel_size,
                cgu_layer_scale_init_value=cgu_layer_scale_init_value,
            )

        self._init_weights()

    def _make_stem(self, hidden_chs: int, norm_layer):
        if not isinstance(hidden_chs, (list, tuple)):
            return

        return ResStem([hidden_chs[0], hidden_chs[1], hidden_chs[2]], norm_layer)

    def _make_stage(
        self,
        hidden_chs: int,
        out_chs=None,
        activation_function=None,
        norm_layer=GroupNorm2d,
        depth=2,
        mlp_ratio=4,
        mlp_use_dw_conv=True,
        mlp_dw_kernel_size=7,
        mlp_in_kernel_size=1,
        mlp_out_kernel_size=1,
        stride=2,
        cgu_layer_scale_init_value=1e-2,
    ):
        if out_chs is None:
            out_chs = hidden_chs

        return CGUStage(
            hidden_chs,
            out_chs,
            stride=stride,
            activation_function=activation_function,
            norm_layer=norm_layer,
            depth=depth,
            use_cross=True,
            mlp_ratio=mlp_ratio,
            mlp_use_dw_conv=mlp_use_dw_conv,
            mlp_dw_kernel_size=mlp_dw_kernel_size,
            mlp_in_kernel_size=mlp_in_kernel_size,
            mlp_out_kernel_size=mlp_out_kernel_size,
            layer_scale_init_value=cgu_layer_scale_init_value,
        )

    def _make_out_stages(
        self,
        num_out_stages: int,
        hidden_chs: int,
        out_chs=None,
        activation_function=None,
        norm_layer=GroupNorm2d,
        depth=2,
        mlp_dw_kernel_size=7,
        mlp_ratio=4,
        mlp_use_dw_conv=True,
        mlp_in_kernel_size=1,
        mlp_out_kernel_size=1,
        cgu_layer_scale_init_value=1e-2,
    ):
        if out_chs is None:
            out_chs = hidden_chs
        return CGUStage(
            hidden_chs,
            out_chs,
            stride=1,
            activation_function=activation_function,
            norm_layer=norm_layer,
            depth=num_out_stages * depth,
            use_cross=True,
            mlp_ratio=mlp_ratio,
            mlp_use_dw_conv=mlp_use_dw_conv,
            mlp_dw_kernel_size=mlp_dw_kernel_size,
            mlp_in_kernel_size=mlp_in_kernel_size,
            mlp_out_kernel_size=mlp_out_kernel_size,
            layer_scale_init_value=cgu_layer_scale_init_value,
        )

    def _make_out_1x1_layer(self, hidden_chs: int, out_chs: int):
        return Conv2dBlock(hidden_chs, out_chs, kernel_size=1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor, y: torch.Tensor, pyr_levels: int):
        input_x = x
        input_y = y

        x_pyramid = []
        y_pyramid = []

        pyr_iters = pyr_levels + 1
        for i in range(pyr_iters):
            if i == 0:
                x = self.stem(x)
                y = self.stem(y)

                x, hx = torch.split(x, [x.shape[1] // 2, x.shape[1] // 2], 1)
                y, hy = torch.split(y, [y.shape[1] // 2, y.shape[1] // 2], 1)
                hx = torch.tanh(hx)
                hy = torch.tanh(hy)
            else:
                hx = self.forward_gru(hx, x)
                hy = self.forward_gru(hy, y)

                x, y = self.rec_stage(hx, hy)
                x = x.contiguous()
                y = y.contiguous()
                if i < (pyr_iters - 1):
                    hx = self.down_gru(hx)
                    hx = torch.tanh(hx)
                    hy = self.down_gru(hy)
                    hy = torch.tanh(hy)

            if i >= 1:
                x_pyramid.append(x)
                y_pyramid.append(y)

        hx = torch.zeros_like(x_pyramid[-1])
        hy = torch.zeros_like(y_pyramid[-1])
        for i in range(len(x_pyramid) - 1, -1, -1):
            x = x_pyramid[i]
            y = y_pyramid[i]

            hx = self.backward_gru(hx, x)
            hy = self.backward_gru(hy, y)

            x2, y2 = self.back_stage(hx, hy)

            input_x_lowres = F.interpolate(
                input_x,
                scale_factor=(1.0 / 2.0 ** (i + 1)),
                mode="bilinear",
                align_corners=True,
            )
            x_lowres = self.lowres_stem(input_x_lowres)

            input_y_lowres = F.interpolate(
                input_y,
                scale_factor=(1.0 / 2.0 ** (i + 1)),
                mode="bilinear",
                align_corners=True,
            )
            y_lowres = self.lowres_stem(input_y_lowres)

            x_pyramid[i] = torch.cat([x, x2, x_lowres], 1)
            y_pyramid[i] = torch.cat([y, y2, y_lowres], 1)

            if i > 0:
                hx = self.up_gru(hx)
                hx = torch.tanh(hx)
                hy = self.up_gru(hy)
                hy = torch.tanh(hy)

        for i, (x, y) in enumerate(zip(x_pyramid, y_pyramid)):
            if self.num_out_stages > 0:
                x = self.out_merge_conv(x)
                y = self.out_merge_conv(y)
                x, y = self.out_stages(x, y)
            if self.out_1x1_abs_chs > 0:
                if self.out_1x1_factor is None:
                    x = self.out_1x1(x)
                    y = self.out_1x1(y)
                else:
                    x = self.out_1x1(x, int(self.out_1x1_factor * x.shape[1]))
                    y = self.out_1x1(y, int(self.out_1x1_factor * y.shape[1]))
            x_pyramid[i] = x
            y_pyramid[i] = y

        return x_pyramid[::-1], y_pyramid[::-1]
