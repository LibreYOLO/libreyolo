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
#
# Modified by the LibreYOLO contributors (2026): inference-only port of
# ptlflow/models/dpflow/dpflow.py. The class derives from torch.nn.Module
# instead of ptlflow's Lightning BaseModel; the training loss, the model
# registry decorators, loguru logging, warm start (prev_preds) and the
# training-only outputs are removed. ptlflow's preprocess_images and
# postprocess_predictions are inlined for the exact settings DPFlow uses
# (two-sided replicate padding to the output stride, images scaled to
# [-1, 1]). The corr_mode dispatch keeps "allpairs" and the pure-PyTorch
# "torch" engine, including upstream's fallback from "allpairs" to "torch"
# on CUDA out-of-memory. The dual-pyramid network, the adaptive pyramid
# level rule and the recurrent refinement are unchanged.

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cgu_bidir_dual_encoder import CGUBidirDualEncoder
from .corr import CorrBlock
from .correlation import IterativeCorrBlock
from .padder import InputPadder
from .pwc_modules import rescale_flow, upsample2d_as
from .update import UpdateBlock
from .utils import compute_pyramid_levels, get_activation, get_norm

logger = logging.getLogger(__name__)


class DPFlow(nn.Module):
    def __init__(
        self,
        pyramid_levels: Optional[int] = None,
        iters_per_level: int = 4,
        detach_flow: bool = True,
        use_norm_affine: bool = False,
        group_norm_num_groups: int = 8,
        corr_mode: str = "allpairs",
        corr_levels: int = 1,
        corr_range: int = 4,
        activation_function: str = "orig",
        enc_norm_type: str = "group",
        enc_depth: int = 4,
        enc_mlp_ratio: float = 2.0,
        enc_mlp_in_kernel_size: int = 1,
        enc_mlp_out_kernel_size: int = 1,
        enc_hidden_chs: tuple = (64, 96, 128),
        enc_num_out_stages: int = 1,
        enc_out_1x1_chs: str = "384",
        dec_gru_norm_type: str = "layer",
        dec_gru_iters: int = 1,
        dec_gru_depth: int = 4,
        dec_gru_mlp_ratio: float = 2.0,
        dec_gru_mlp_in_kernel_size: int = 1,
        dec_gru_mlp_out_kernel_size: int = 1,
        dec_net_chs: int = 128,
        dec_inp_chs: int = 128,
        dec_motion_chs: int = 128,
        dec_flow_kernel_size: int = 7,
        dec_flow_head_chs: int = 256,
        dec_motenc_corr_hidden_chs: int = 256,
        dec_motenc_corr_out_chs: int = 192,
        dec_motenc_flow_hidden_chs: int = 128,
        dec_motenc_flow_out_chs: int = 64,
        use_upsample_mask: bool = True,
        upmask_gradient_scale: float = 1.0,
        cgu_mlp_dw_kernel_size: int = 7,
        cgu_mlp_use_dw_conv: bool = True,
        cgu_layer_scale_init_value: float = 0.01,
        loss: str = "laplace",
    ):
        super().__init__()
        if pyramid_levels is not None:
            assert pyramid_levels > 2, "Only pyramid_levels >= 3 is supported."
            self.output_stride = int(2 ** (pyramid_levels + 2))
        else:
            self.output_stride = None
        self.extra_output_stride = 0

        if corr_mode not in ("allpairs", "torch"):
            raise ValueError(
                f"corr_mode must be 'allpairs' or 'torch', got {corr_mode!r}."
            )

        self.pyramid_levels = pyramid_levels
        self.iters_per_level = iters_per_level
        self.corr_mode = corr_mode
        self.corr_range = corr_range
        self.corr_levels = corr_levels
        self.detach_flow = detach_flow
        self.loss = loss

        activation_function = get_activation(activation_function)

        enc_out_1x1_chs = (
            float(enc_out_1x1_chs)
            if (isinstance(enc_out_1x1_chs, str) and "." in enc_out_1x1_chs)
            else int(enc_out_1x1_chs)
        )

        if isinstance(enc_out_1x1_chs, float):
            out_1x1_factor = enc_out_1x1_chs
            out_1x1_abs_chs = int(enc_out_1x1_chs * enc_hidden_chs[-1])
        else:
            out_1x1_factor = None
            out_1x1_abs_chs = enc_out_1x1_chs

        self.max_feat_chs = max(enc_hidden_chs[-1], out_1x1_abs_chs)

        net_chs = dec_net_chs
        inp_chs = dec_inp_chs
        if net_chs is None or inp_chs is None:
            base_chs = out_1x1_abs_chs
            if base_chs < 1:
                base_chs = enc_hidden_chs[-1]

            base_chs = base_chs // 3 * 2

            if net_chs is None and inp_chs is None:
                net_chs = inp_chs = base_chs // 2
            elif net_chs is None and inp_chs is not None:
                net_chs = base_chs - inp_chs
            elif net_chs is not None and inp_chs is None:
                inp_chs = base_chs - net_chs
        net_chs_fixed = net_chs
        inp_chs_fixed = inp_chs

        enc_norm_layer = get_norm(
            enc_norm_type,
            affine=use_norm_affine,
            num_groups=group_norm_num_groups,
        )
        self.fnet = CGUBidirDualEncoder(
            pyramid_levels=pyramid_levels,
            hidden_chs=enc_hidden_chs,
            out_1x1_abs_chs=out_1x1_abs_chs,
            out_1x1_factor=out_1x1_factor,
            num_out_stages=enc_num_out_stages,
            activation_function=activation_function,
            norm_layer=enc_norm_layer,
            depth=enc_depth,
            mlp_ratio=enc_mlp_ratio,
            mlp_use_dw_conv=cgu_mlp_use_dw_conv,
            mlp_dw_kernel_size=cgu_mlp_dw_kernel_size,
            mlp_in_kernel_size=enc_mlp_in_kernel_size,
            mlp_out_kernel_size=enc_mlp_out_kernel_size,
            cgu_layer_scale_init_value=cgu_layer_scale_init_value,
        )

        self.dim_corr = (corr_range * 2 + 1) ** 2 * corr_levels

        dec_gru_norm_layer = get_norm(
            dec_gru_norm_type,
            affine=use_norm_affine,
            num_groups=group_norm_num_groups,
        )
        self.update_block = UpdateBlock(
            dec_motenc_corr_hidden_chs=dec_motenc_corr_hidden_chs,
            dec_motenc_corr_out_chs=dec_motenc_corr_out_chs,
            dec_motenc_flow_hidden_chs=dec_motenc_flow_hidden_chs,
            dec_motenc_flow_out_chs=dec_motenc_flow_out_chs,
            corr_levels=corr_levels,
            corr_range=corr_range,
            dec_flow_kernel_size=dec_flow_kernel_size,
            dec_motion_chs=dec_motion_chs,
            activation_function=activation_function,
            net_chs_fixed=net_chs_fixed,
            inp_chs_fixed=inp_chs_fixed,
            dec_gru_norm_layer=dec_gru_norm_layer,
            dec_gru_depth=dec_gru_depth,
            dec_gru_iters=dec_gru_iters,
            dec_gru_mlp_ratio=dec_gru_mlp_ratio,
            cgu_mlp_use_dw_conv=cgu_mlp_use_dw_conv,
            cgu_mlp_dw_kernel_size=cgu_mlp_dw_kernel_size,
            dec_gru_mlp_in_kernel_size=dec_gru_mlp_in_kernel_size,
            dec_gru_mlp_out_kernel_size=dec_gru_mlp_out_kernel_size,
            cgu_layer_scale_init_value=cgu_layer_scale_init_value,
            dec_flow_head_chs=dec_flow_head_chs,
            loss=loss,
            use_upsample_mask=use_upsample_mask,
            upmask_gradient_scale=upmask_gradient_scale,
        )

        act = nn.ReLU if activation_function is None else activation_function
        self.input_act = act(inplace=True)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def coords_grid(self, batch, ht, wd):
        coords = torch.meshgrid(
            torch.arange(ht, dtype=self.dtype, device=self.device),
            torch.arange(wd, dtype=self.dtype, device=self.device),
            indexing="ij",
        )
        coords = torch.stack(coords[::-1], dim=0).to(dtype=self.dtype)
        return coords[None].repeat(batch, 1, 1, 1)

    def upsample_flow(self, flow, mask, factor, ch=2):
        """Upsample flow field [H/f, W/f, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, ch, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, ch, factor * H, factor * W)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """Flow from ``image1`` to ``image2``.

        Inputs are RGB ``(B, 3, H, W)`` in [0, 1] at their native resolution.
        Returns the flow ``(B, 2, H, W)`` in pixels on the input canvas.
        """
        try:
            return self.forward_flow(image1, image2)
        except torch.OutOfMemoryError:
            if self.corr_mode != "allpairs":
                raise
            logger.warning(
                "DPFlow: CUDA out of memory with input size %s; switching "
                "corr_mode to 'torch' (lower memory, slower) and retrying.",
                list(image1.shape[-2:]),
            )
            self.corr_mode = "torch"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self.forward_flow(image1, image2)

    def forward_flow(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        images = torch.stack([image1, image2], dim=1)
        if self.pyramid_levels is None:
            pyr_levels = compute_pyramid_levels(images)
            output_stride = 2 ** (pyr_levels + 2 + self.extra_output_stride)
        else:
            pyr_levels = self.pyramid_levels
            output_stride = self.output_stride

        # ptlflow preprocess_images(bgr_add=-0.5, bgr_mult=2.0, bgr_to_rgb=True,
        # resize_mode="pad", pad_mode="replicate", pad_two_side=True). The
        # inputs here are already RGB, so the channel flip is not needed.
        images = (images - 0.5) * 2.0
        image_resizer = InputPadder(
            images.shape,
            stride=output_stride,
            two_side_pad=True,
            pad_mode="replicate",
        )
        images = image_resizer.fill(images).contiguous()
        x1_raw = images[:, 0]
        x2_raw = images[:, 1]
        return self.predict(x1_raw, x2_raw, pyr_levels, image_resizer)

    def predict(self, x1_raw, x2_raw, pyr_levels, image_resizer):
        b, _, height_im, width_im = x1_raw.size()

        x1_pyramid, x2_pyramid = self.fnet(x1_raw, x2_raw, pyr_levels=pyr_levels)

        b_size, _, h_x1, w_x1 = x1_pyramid[0].size()
        flow = torch.zeros(
            b_size, 2, h_x1, w_x1, dtype=self.dtype, device=x1_pyramid[0].device
        )

        out_flow = None
        net = None
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # Split feature channels into matching (x) and context (c)
            xh = x1.shape[1]
            ch = xh // 3
            x1, cn1 = torch.split(x1, [xh - ch, ch], dim=1)
            x2, cn2 = torch.split(x2, [xh - ch, ch], dim=1)
            halfch = ch // 2
            i1, n1 = torch.split(cn1, [ch - halfch, halfch], dim=1)
            i2, n2 = torch.split(cn2, [ch - halfch, halfch], dim=1)
            inp = torch.cat([i1, i2], 1)
            inp = self.input_act(inp)
            net_tmp = torch.cat([n1, n2], 1)

            coords0 = self.coords_grid(x1.shape[0], x1.shape[2], x1.shape[3])

            corr_cls = CorrBlock if self.corr_mode == "allpairs" else IterativeCorrBlock
            corr_fn = corr_cls(
                fmap1=x1,
                fmap2=x2,
                num_levels=self.corr_levels,
                radius=self.corr_range,
            )

            if l > 0:
                flow = rescale_flow(flow, x1.shape[-1], x1.shape[-2], to_local=False)
                flow = upsample2d_as(flow, x1, mode="bilinear")

            net = torch.tanh(net_tmp)

            for it in range(self.iters_per_level):
                if self.detach_flow:
                    flow = flow.detach()

                out_corr = corr_fn(coords0 + flow)

                flow_res, net, mask = self.update_block(net, inp, out_corr, flow)

                if self.loss == "laplace":
                    flow_res = flow_res[:, :2]

                flow = flow + flow_res

                if l == len(x1_pyramid) - 1 and it == self.iters_per_level - 1:
                    out_flow = rescale_flow(flow, width_im, height_im, to_local=False)
                    if mask is not None:
                        out_flow = self.upsample_flow(out_flow, mask, factor=8)
                    out_flow = upsample2d_as(out_flow, x1_raw, mode="bilinear")
                    out_flow = image_resizer.unfill(out_flow)

        return out_flow
