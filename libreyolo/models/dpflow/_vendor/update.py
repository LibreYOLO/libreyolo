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

import torch
import torch.nn as nn

from typing import Callable

from .conv import Conv2dBlock
from .cgu import CGUStage
from .norm import LayerNorm2d


class FlowHead(nn.Module):
    def __init__(
        self, input_dim=128, hidden_dim=256, activation_function=None, info_pred=False
    ):
        super(FlowHead, self).__init__()
        self.conv1 = Conv2dBlock(input_dim, hidden_dim, 3, padding=1)
        out_ch = 6 if info_pred else 2
        self.conv2 = Conv2dBlock(hidden_dim, out_ch, 3, padding=1)

        act = nn.ReLU if activation_function is None else activation_function
        self.act = act(inplace=True)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = Conv2dBlock(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = Conv2dBlock(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = Conv2dBlock(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = self.convz(hx)
        z = torch.sigmoid(z)

        r = self.convr(hx)
        r = torch.sigmoid(r)

        q = self.convq(torch.cat([r * h, x], dim=1))
        q = torch.tanh(q)

        h = (1 - z) * h + z * q
        return h


class CGUGRU(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        input_dim=192 + 128,
        activation_function=None,
        norm_layer=LayerNorm2d,
        depth=2,
        mlp_ratio=4,
        mlp_use_dw_conv=True,
        mlp_dw_kernel_size=7,
        mlp_in_kernel_size=1,
        mlp_out_kernel_size=1,
        layer_scale_init_value=1e-2,
    ):
        super(CGUGRU, self).__init__()

        self.convz = CGUStage(
            hidden_dim + input_dim,
            hidden_dim,
            stride=1,
            activation_function=activation_function,
            norm_layer=norm_layer,
            depth=depth,
            use_cross=False,
            mlp_ratio=mlp_ratio,
            mlp_use_dw_conv=mlp_use_dw_conv,
            mlp_dw_kernel_size=mlp_dw_kernel_size,
            mlp_in_kernel_size=mlp_in_kernel_size,
            mlp_out_kernel_size=mlp_out_kernel_size,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.convr = CGUStage(
            hidden_dim + input_dim,
            hidden_dim,
            stride=1,
            activation_function=activation_function,
            norm_layer=norm_layer,
            depth=depth,
            use_cross=False,
            mlp_ratio=mlp_ratio,
            mlp_use_dw_conv=mlp_use_dw_conv,
            mlp_dw_kernel_size=mlp_dw_kernel_size,
            mlp_in_kernel_size=mlp_in_kernel_size,
            mlp_out_kernel_size=mlp_out_kernel_size,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.convq = CGUStage(
            hidden_dim + input_dim,
            hidden_dim,
            stride=1,
            activation_function=activation_function,
            norm_layer=norm_layer,
            depth=depth,
            use_cross=False,
            mlp_ratio=mlp_ratio,
            mlp_use_dw_conv=mlp_use_dw_conv,
            mlp_dw_kernel_size=mlp_dw_kernel_size,
            mlp_in_kernel_size=mlp_in_kernel_size,
            mlp_out_kernel_size=mlp_out_kernel_size,
            layer_scale_init_value=layer_scale_init_value,
        )

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = self.convz(hx)
        z = torch.sigmoid(z)

        r = self.convr(hx)
        r = torch.sigmoid(r)

        q = self.convq(torch.cat([r * h, x], dim=1))
        q = torch.tanh(q)

        h = (1 - z) * h + z * q
        return h


class ConvexMask(nn.Module):
    def __init__(self, net_chs, pred_stride, activation_function=None):
        super(ConvexMask, self).__init__()
        self.conv1 = Conv2dBlock(net_chs, net_chs * 2, 3, padding=1)
        self.conv2 = Conv2dBlock(net_chs * 2, pred_stride**2 * 9, 1, padding=0)

        act = nn.ReLU if activation_function is None else activation_function
        self.act = act(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class MotionEncoder(nn.Module):
    def __init__(
        self,
        dec_motenc_corr_hidden_chs: int,
        dec_motenc_corr_out_chs: int,
        dec_motenc_flow_hidden_chs: int,
        dec_motenc_flow_out_chs: int,
        corr_levels: int,
        corr_range: int,
        dec_flow_kernel_size: int,
        dec_motion_chs: int,
        activation_function: Callable[..., nn.Module],
    ):
        super(MotionEncoder, self).__init__()

        c_hidden = dec_motenc_corr_hidden_chs
        c_out = dec_motenc_corr_out_chs
        f_hidden = dec_motenc_flow_hidden_chs
        f_out = dec_motenc_flow_out_chs

        cor_planes = corr_levels * (2 * corr_range + 1) ** 2
        self.convc1 = Conv2dBlock(cor_planes, c_hidden, 1, padding=0)
        self.convc2 = Conv2dBlock(c_hidden, c_out, 3, padding=1)

        self.convf1 = Conv2dBlock(
            2,
            f_hidden,
            dec_flow_kernel_size,
            padding=dec_flow_kernel_size // 2,
        )
        self.convf2 = Conv2dBlock(f_hidden, f_out, 3, padding=1)

        in_ch = f_out + c_out
        out_ch = dec_motion_chs - 2
        self.conv = Conv2dBlock(in_ch, out_ch, 3, padding=1)

        act = nn.ReLU if activation_function is None else activation_function
        self.act = act(inplace=True)

    def forward(self, flow, corr):
        cor = self.act(self.convc1(corr))
        cor = self.act(self.convc2(cor))

        outs = [cor]

        flo = self.act(self.convf1(flow))
        flo = self.act(self.convf2(flo))
        outs.append(flo)

        outs = torch.cat(outs, dim=1)
        out = self.act(self.conv(outs))
        out_t = [out, flow]
        return torch.cat(out_t, dim=1)


class UpdateBlock(nn.Module):
    def __init__(
        self,
        dec_motenc_corr_hidden_chs: int,
        dec_motenc_corr_out_chs: int,
        dec_motenc_flow_hidden_chs: int,
        dec_motenc_flow_out_chs: int,
        corr_levels: int,
        corr_range: int,
        dec_flow_kernel_size: int,
        dec_motion_chs: int,
        activation_function: Callable[..., nn.Module],
        net_chs_fixed: int,
        inp_chs_fixed: int,
        dec_gru_norm_layer: Callable[..., nn.Module],
        dec_gru_depth: int,
        dec_gru_iters: int,
        dec_gru_mlp_ratio: float,
        cgu_mlp_use_dw_conv: bool,
        cgu_mlp_dw_kernel_size: int,
        dec_gru_mlp_in_kernel_size: int,
        dec_gru_mlp_out_kernel_size: int,
        cgu_layer_scale_init_value: float,
        dec_flow_head_chs: int,
        loss: str,
        use_upsample_mask: bool,
        upmask_gradient_scale: float,
    ):
        super(UpdateBlock, self).__init__()
        self.use_upsample_mask = use_upsample_mask
        self.upmask_gradient_scale = upmask_gradient_scale

        self.encoder = MotionEncoder(
            dec_motenc_corr_hidden_chs=dec_motenc_corr_hidden_chs,
            dec_motenc_corr_out_chs=dec_motenc_corr_out_chs,
            dec_motenc_flow_hidden_chs=dec_motenc_flow_hidden_chs,
            dec_motenc_flow_out_chs=dec_motenc_flow_out_chs,
            corr_levels=corr_levels,
            corr_range=corr_range,
            dec_flow_kernel_size=dec_flow_kernel_size,
            dec_motion_chs=dec_motion_chs,
            activation_function=activation_function,
        )

        self.gru_list = nn.ModuleList(
            [
                CGUGRU(
                    hidden_dim=net_chs_fixed,
                    input_dim=dec_motion_chs + inp_chs_fixed,
                    activation_function=activation_function,
                    norm_layer=dec_gru_norm_layer,
                    depth=dec_gru_depth,
                    mlp_ratio=dec_gru_mlp_ratio,
                    mlp_use_dw_conv=cgu_mlp_use_dw_conv,
                    mlp_dw_kernel_size=cgu_mlp_dw_kernel_size,
                    mlp_in_kernel_size=dec_gru_mlp_in_kernel_size,
                    mlp_out_kernel_size=dec_gru_mlp_out_kernel_size,
                    layer_scale_init_value=cgu_layer_scale_init_value,
                )
                for _ in range(dec_gru_iters)
            ]
        )

        self.flow_head = FlowHead(
            net_chs_fixed,
            hidden_dim=dec_flow_head_chs,
            activation_function=activation_function,
            info_pred=(loss == "laplace"),
        )

        if use_upsample_mask:
            pred_stride = 8
            self.mask = ConvexMask(
                net_chs_fixed,
                pred_stride,
                activation_function=activation_function,
            )

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)

        inp = torch.cat([inp, motion_features], dim=1)

        for gru in self.gru_list:
            net = gru(net, inp)

        delta_flow = self.flow_head(net)

        mask = None
        if self.use_upsample_mask:
            mask = self.upmask_gradient_scale * self.mask(net)

        return delta_flow, net, mask
