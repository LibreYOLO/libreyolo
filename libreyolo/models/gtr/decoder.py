"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# Adapted for LibreYOLO from Intellindust-AI-Lab/GTR
# revision 782e737efe2e6437ac537fbdcee089673d3376c1 (MIT).
# Changes: native construction, local imports and portable attention. See NOTICE.

import copy
import functools
import math
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from .denoising import get_contrastive_denoising_training_group
from .segmentation_head import SegmentationHead
from .utils import (
    bias_init_with_prob,
    deformable_attention_core_func_v2,
    distance2bbox,
    get_activation,
    inverse_sigmoid,
    weighting_function,
)

__all__ = ["GTRTransformer"]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Gate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class Integral(nn.Module):
    """
    A static layer that calculates integral results from a distribution.

    This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`,
    where Pr(n) is the softmax probability vector representing the discrete
    distribution, and W(n) is the non-uniform Weighting Function.

    Args:
        reg_max (int): Max number of the discrete bins. Default is 32.
                       It can be adjusted based on the dataset or task requirements.
    """

    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max, act="relu"):
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method="default",
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention"""
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ""
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer(
            "num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32)
        )

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self.ms_deformable_attn_core = functools.partial(
            deformable_attention_core_func_v2, method=self.method
        )

        self._reset_parameters()

        if method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def convert_to_deploy(self):
        # Merge sampling_offsets + attention_weights into one Linear so the two
        # tiny per-layer GEMMs become one (both read the same query tensor).
        if getattr(self, "_fused_offset_weight", False):
            return
        ow = nn.Linear(
            self.embed_dim,
            self.total_points * 3,
            device=self.sampling_offsets.weight.device,
            dtype=self.sampling_offsets.weight.dtype,
        )
        with torch.no_grad():
            ow.weight.copy_(
                torch.cat(
                    [self.sampling_offsets.weight, self.attention_weights.weight], dim=0
                )
            )
            ow.bias.copy_(
                torch.cat(
                    [self.sampling_offsets.bias, self.attention_weights.bias], dim=0
                )
            )
        self.offset_weight_proj = ow
        self._fused_offset_weight = True
        del self.sampling_offsets
        del self.attention_weights

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile(
            [1, sum(self.num_points_list), 1]
        )
        scaling = torch.concat(
            [torch.arange(1, n + 1) for n in self.num_points_list]
        ).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: list[int],
    ):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]

        if getattr(self, "_fused_offset_weight", False):
            ow = self.offset_weight_proj(query)
            sampling_offsets, attention_weights = ow.split(
                [self.total_points * 2, self.total_points], dim=-1
            )
        else:
            sampling_offsets = self.sampling_offsets(query)
            attention_weights = self.attention_weights(query)
        sampling_offsets = sampling_offsets.reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list), 2
        )

        attention_weights = attention_weights.reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list)
        )
        attention_weights = F.softmax(attention_weights, dim=-1)

        if (
            getattr(self, "_atlas_mode", False)
            and not self.training
            and isinstance(value, tuple)
            and len(value) == 4
            and value[1].dim() == 2
            and reference_points.shape[-1] == 4
        ):
            # Deploy fast path: all levels live in one zero-gutter atlas, so the
            # per-level grid_sample loop + cat collapses into ONE grid_sample.
            # Out-of-level samples are first clamped to a point that is still
            # fully outside the level (>= margin+1px past the border), where
            # bilinear taps read only gutter zeros — exactly what the per-level
            # padding_mode='zeros' returned.
            canvas, va_scale, va_offset, va_bound = value
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = (
                sampling_offsets
                * num_points_scale
                * reference_points[:, :, None, :, 2:]
                * self.offset_scale
            )
            sampling_locations = reference_points[:, :, None, :, :2] + offset
            g = 2.0 * sampling_locations.float() - 1.0
            g = torch.clamp(g, -va_bound, va_bound) * va_scale + va_offset
            g = g.to(canvas.dtype).permute(0, 2, 1, 3, 4).flatten(0, 1)
            sampled = F.grid_sample(
                canvas, g, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            attn = attention_weights.permute(0, 2, 1, 3).reshape(
                bs * self.num_heads, 1, Len_q, sum(self.num_points_list)
            )
            out = (
                (sampled * attn)
                .sum(-1)
                .reshape(bs, self.num_heads * self.head_dim, Len_q)
            )
            return out.permute(0, 2, 1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2
            )
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = (
                sampling_offsets
                * num_points_scale
                * reference_points[:, :, None, :, 2:]
                * self.offset_scale
            )
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        output = self.ms_deformable_attn_core(
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
            self.num_points_list,
        )

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
        cross_attn_method="default",
        layer_scale=None,
        group_detr=1,
    ):
        super().__init__()

        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        self.group_detr = group_detr

        # self attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # cross attention
        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points, method=cross_attn_method
        )
        self.dropout2 = nn.Dropout(dropout)

        self.gateway = Gate(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # self.ffn = SwiGLUFFN(d_model, dim_feedforward, d_model)

        self.dropout4 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        target,
        reference_points,
        value,
        spatial_shapes,
        attn_mask=None,
        query_pos_embed=None,
        num_dn=0,
        pos_qk=None,
    ):

        bs = target.shape[0]

        # self attention
        if pos_qk is not None and attn_mask is None and not self.training:
            # Deploy fast path: q = k = target + pos folded into the projections.
            # qkv comes from ONE packed GEMM on `target` (instead of three separate
            # projections, since q is k but k is not v); the position contribution
            # pos @ [Wq;Wk]^T arrives precomputed (query_pos_embed is constant
            # across layers at inference, so the decoder batches it once).
            mha = self.self_attn
            d = mha.embed_dim
            nh = mha.num_heads
            n = target.shape[1]
            qkv = F.linear(target, mha.in_proj_weight, mha.in_proj_bias)
            q, k, v = qkv.split(d, dim=-1)
            pos_q, pos_k = pos_qk.split(d, dim=-1)
            q = (q + pos_q).view(bs, n, nh, d // nh).transpose(1, 2)
            k = (k + pos_k).view(bs, n, nh, d // nh).transpose(1, 2)
            v = v.view(bs, n, nh, d // nh).transpose(1, 2)
            # NOTE: an explicit bmm-softmax-bmm decomposition (baddbmm form, to
            # dodge inductor's fuse_attention re-fusion) was measured SLOWER
            # than flash splitkv at n=300: 40.4us vs 36.5us per forward across
            # the 4 layers. Keep SDPA.
            target2 = F.scaled_dot_product_attention(q, k, v)
            target2 = mha.out_proj(target2.transpose(1, 2).reshape(bs, n, d))
            target = target + self.dropout1(target2)
            target = self.norm1(target)
            return self._forward_post_self_attn(
                target, reference_points, value, spatial_shapes, query_pos_embed
            )

        q = k = self.with_pos_embed(target, query_pos_embed)

        if self.training and self.group_detr > 1:
            num_det = target.shape[1] - num_dn
            group_size = num_det // self.group_detr

            if num_dn > 0:
                # Denoising queries: standard self-attention with their portion of attn_mask
                dn_q, det_q = q[:, :num_dn], q[:, num_dn:]
                dn_k, det_k = k[:, :num_dn], k[:, num_dn:]
                dn_v, det_v = target[:, :num_dn], target[:, num_dn:]

                dn_mask = attn_mask[:num_dn, :num_dn] if attn_mask is not None else None
                dn_out, _ = self.self_attn(dn_q, dn_k, dn_v, attn_mask=dn_mask)

                # Detection queries: group self-attention (each group attends only to itself)
                det_q_g = torch.cat(det_q.split(group_size, dim=1), dim=0)
                det_k_g = torch.cat(det_k.split(group_size, dim=1), dim=0)
                det_v_g = torch.cat(det_v.split(group_size, dim=1), dim=0)
                det_out, _ = self.self_attn(det_q_g, det_k_g, det_v_g)
                det_out = torch.cat(det_out.split(bs, dim=0), dim=1)

                target2 = torch.cat([dn_out, det_out], dim=1)
            else:
                # No denoising: group split all queries
                q_g = torch.cat(q.split(group_size, dim=1), dim=0)
                k_g = torch.cat(k.split(group_size, dim=1), dim=0)
                v_g = torch.cat(target.split(group_size, dim=1), dim=0)
                target2, _ = self.self_attn(q_g, k_g, v_g)
                target2 = torch.cat(target2.split(bs, dim=0), dim=1)
        else:
            target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)

        target = target + self.dropout1(target2)
        target = self.norm1(target)

        return self._forward_post_self_attn(
            target, reference_points, value, spatial_shapes, query_pos_embed
        )

    def _forward_post_self_attn(
        self, target, reference_points, value, spatial_shapes, query_pos_embed
    ):
        # cross attention
        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes,
        )

        target = self.gateway(target, self.dropout2(target2))

        # ffn
        target2 = self.linear2(self.dropout3(self.activation(self.linear1(target))))
        # target2 = self.ffn(target)

        target = target + self.dropout4(target2)
        target = self.norm2(target.clamp(min=-65504, max=65504))

        return target


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers,
    utilizing attention mechanisms, location quality estimators, and distribution refinement techniques
    to improve bounding box accuracy and robustness.
    """

    def __init__(
        self,
        hidden_dim,
        decoder_layer,
        decoder_layer_wide,
        num_layers,
        num_head,
        reg_max,
        reg_scale,
        up,
        eval_idx=-1,
        layer_scale=2,
        act="relu",
        segmentation_head=None,
    ):
        super().__init__()
        self.emit_loss_outputs = False
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [
                copy.deepcopy(decoder_layer_wide)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )
        self.segmentation_head = segmentation_head
        self.lqe_layers = nn.ModuleList(
            [copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)]
        )

    def value_op(
        self,
        memory,
        value_proj,
        value_scale,
        memory_mask,
        memory_spatial_shapes,
        value_atlas=None,
    ):
        """
        Preprocess values for MSDeformableAttention.

        Returns a tuple of per-level value tensors already reshaped to
        ``[bs*n_head, c, h, w]`` and contiguous, so the cross-attn loop can
        feed them directly into ``F.grid_sample`` without an extra reshape
        (which would copy out of non-contiguous memory once per layer × level).
        """
        if value_atlas is not None and memory_mask is None and value_scale is None:
            # Deploy fast path: all levels already live in one zero-gutter atlas
            # (built in-graph by GTRTransformer._build_value_atlas). Read-only here.
            return (value_atlas, self._va_scale, self._va_offset, self._va_bound)
        value = value_proj(memory) if value_proj is not None else memory
        value = (
            F.interpolate(memory, size=value_scale)
            if value_scale is not None
            else value
        )
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        bs = value.shape[0]
        # [bs, sumHW, num_head, head_dim] -> [bs, num_head, head_dim, sumHW]
        value = value.reshape(bs, value.shape[1], self.num_head, -1).permute(0, 2, 3, 1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        per_level = value.split(split_shape, dim=-1)
        c = value.shape[2]
        # Pre-reshape once: each [bs, nh, c, hw] view becomes a contiguous [bs*nh, c, h, w].
        return tuple(
            v.reshape(bs * self.num_head, c, h, w).contiguous()
            for v, (h, w) in zip(per_level, memory_spatial_shapes)
        )

    def build_value_atlas(
        self, spatial_shapes, num_points_list, batch_size=1, gutter=6, margin=3
    ):
        """Precompute the multi-level value atlas used by the MSDA deploy path.

        Levels are placed on the diagonal of one canvas so that level l's row
        band intersects the other levels' column bands only in zero gutter.
        Sampling grids are clamped to margin+1px outside their level box before
        the affine remap into atlas coordinates, so every bilinear tap of an
        out-of-level sample lands on gutter zeros (= per-level padding zeros).
        ``gutter`` must be >= margin + 3 for the worst-case tap to stay clear
        of the neighbouring level.
        """
        origins = []
        r = c = 0
        for h, w in spatial_shapes:
            origins.append((r, c))
            r += h + gutter
            c += w + gutter
        H_A, W_A = r, c
        scale, offset, bound = [], [], []
        for (h, w), (r0, c0), n in zip(spatial_shapes, origins, num_points_list):
            for _ in range(n):
                scale.append((w / W_A, h / H_A))
                offset.append(((2 * c0 + w) / W_A - 1, (2 * r0 + h) / H_A - 1))
                bound.append((1 + (2 * margin + 1) / w, 1 + (2 * margin + 1) / h))
        self._va_hw = (H_A, W_A)
        # Kept 2D [P, 2]: broadcasts against [..., P, 2] grids, and Module.to(
        # memory_format=channels_last) rejects non-conv-shaped 5D buffers.
        self.register_buffer("_va_scale", torch.tensor(scale), persistent=False)
        self.register_buffer("_va_offset", torch.tensor(offset), persistent=False)
        self.register_buffer("_va_bound", torch.tensor(bound), persistent=False)
        self._va_shapes = tuple(tuple(s) for s in spatial_shapes)
        self._va_origins = origins
        self._va_ready = True
        for layer in self.layers:
            layer.cross_attn._atlas_mode = True

    def convert_to_deploy(self):
        project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        # Register as buffer so model.to(dtype/device) casts/moves it; needed for fp16 inference
        # and for CUDA-graph capture (everything must live on the right device/dtype).
        self.register_buffer("project", project, persistent=False)
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList(
            [nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]]
        )
        # Stack the q/k halves of every layer's in_proj so the position contribution
        # pos @ [Wq;Wk]^T becomes ONE batched GEMM per forward (query_pos_embed is
        # constant across layers at inference); each layer then folds q=k=target+pos
        # into a single packed qkv projection of `target`.
        d = self.hidden_dim
        wqk = torch.cat(
            [l.self_attn.in_proj_weight[: 2 * d] for l in self.layers], dim=0
        )
        self.register_buffer("_sa_pos_qk_w", wqk.detach().clone(), persistent=False)

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
        attn_mask=None,
        memory_mask=None,
        dn_meta=None,
        spatial_features=None,
        value_atlas=None,
    ):
        amp_safe_mode = os.getenv("GTR_AMP_SAFE_MODE", "").lower()
        selective_amp = self.training and amp_safe_mode in {
            "selective",
            "heads_fp32",
            "matcher_fp32",
        }
        device_type = memory.device.type
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(
            memory, None, None, memory_mask, spatial_shapes, value_atlas
        )

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        dec_out_hidden = []
        if not hasattr(self, "project"):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        if selective_amp:
            with torch.autocast(device_type=device_type, enabled=False):
                ref_points_detach = F.sigmoid(ref_points_unact.float())
                query_pos_embed = query_pos_head(ref_points_detach).clamp(
                    min=-10, max=10
                )
        else:
            ref_points_detach = F.sigmoid(ref_points_unact)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

        num_dn = (
            dn_meta["dn_num_split"][0] if (dn_meta is not None and self.training) else 0
        )

        pos_qk_list = None
        if hasattr(self, "_sa_pos_qk_w") and not self.training and attn_mask is None:
            pos_qk_all = F.linear(query_pos_embed, self._sa_pos_qk_w)
            pos_qk_list = pos_qk_all.split(2 * self.hidden_dim, dim=-1)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)

            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(
                    query_pos_embed, scale_factor=self.layer_scale
                )
                value = self.value_op(
                    memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes
                )
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(
                output,
                ref_points_input,
                value,
                spatial_shapes,
                attn_mask,
                query_pos_embed,
                num_dn=num_dn,
                pos_qk=pos_qk_list[i] if pos_qk_list is not None else None,
            )

            if selective_amp:
                with torch.autocast(device_type=device_type, enabled=False):
                    output_for_head = output.float()
                    output_detach_for_head = _float_if_tensor(output_detach)
                    pred_corners_prev = _float_if_tensor(pred_corners_undetach)

                    if i == 0:
                        # Initial bounding box predictions with inverse sigmoid refinement.
                        pre_bboxes = F.sigmoid(
                            pre_bbox_head(output_for_head)
                            + inverse_sigmoid(ref_points_detach)
                        )
                        pre_scores = score_head[0](output_for_head)
                        ref_points_initial = pre_bboxes.detach()

                    pred_corners = (
                        bbox_head[i](output_for_head + output_detach_for_head)
                        + pred_corners_prev
                    )
                    inter_ref_bbox = distance2bbox(
                        ref_points_initial, integral(pred_corners, project), reg_scale
                    )
            else:
                if i == 0:
                    # Initial bounding box predictions with inverse sigmoid refinement
                    pre_bboxes = F.sigmoid(
                        pre_bbox_head(output) + inverse_sigmoid(ref_points_detach)
                    )
                    pre_scores = score_head[0](output)
                    ref_points_initial = pre_bboxes.detach()

                # Refine bounding box corners using FDR, integrating previous layer's corrections
                pred_corners = (
                    bbox_head[i](output + output_detach) + pred_corners_undetach
                )
                inter_ref_bbox = distance2bbox(
                    ref_points_initial, integral(pred_corners, project), reg_scale
                )

            if self.training or self.emit_loss_outputs or i == self.eval_idx:
                if selective_amp:
                    with torch.autocast(device_type=device_type, enabled=False):
                        scores = score_head[i](output.float())
                        # Lqe does not affect the performance here.
                        scores = self.lqe_layers[i](scores, pred_corners)
                else:
                    scores = score_head[i](output)
                    # Lqe does not affect the performance here.
                    scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                dec_out_hidden.append(output)

                if not (self.training or self.emit_loss_outputs):
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        # Segmentation: feed per-layer query features through SegmentationHead.
        if self.segmentation_head is not None and spatial_features is not None:
            mask_logits = self.segmentation_head(
                spatial_features=spatial_features,
                query_features=dec_out_hidden,
            )
            dec_out_masks = torch.stack(mask_logits)
            pre_segs = mask_logits[0]
        else:
            dec_out_masks = None
            pre_segs = None

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
            dec_out_hidden,
            dec_out_masks,
            pre_segs,
        )


def _float_if_tensor(value):
    return value.float() if isinstance(value, torch.Tensor) else value


class GTRTransformer(nn.Module):
    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=None,
        feat_strides=None,
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="silu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eval_output_layer=None,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        share_bbox_head=False,
        share_score_head=False,
        group_detr=1,
        mask_downsample_ratio=None,
    ):
        if feat_strides is None:
            feat_strides = [8, 16, 32]
        if feat_channels is None:
            feat_channels = [512, 1024, 2048]
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.emit_loss_outputs = False
        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.emit_loss_outputs = False
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        if eval_output_layer is None:
            self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
            self.eval_output_layer = self.eval_idx + 1
        else:
            if eval_output_layer < 0 or eval_output_layer > num_layers:
                raise ValueError(
                    f"eval_output_layer must be in [0, {num_layers}], got {eval_output_layer}"
                )
            self.eval_output_layer = eval_output_layer
            self.eval_idx = max(eval_output_layer - 1, 0)
        if self.eval_idx < 0 or self.eval_idx >= num_layers:
            raise ValueError(
                f"eval_idx resolves to {self.eval_idx}, expected [0, {num_layers - 1}]"
            )

        assert query_select_method in ("default", "one2many", "agnostic"), ""
        assert cross_attn_method in ("default", "discrete"), ""
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        self.group_detr = group_detr
        # Force float dtype: YAML can pass `reg_scale: 4` (int), which would create an int64
        # tensor and silently upcast every fp16 op that touches it to fp32 (e.g. distance2bbox).
        self.up = nn.Parameter(
            torch.tensor([0.5], dtype=torch.float32), requires_grad=False
        )
        self.reg_scale = nn.Parameter(
            torch.tensor([float(reg_scale)], dtype=torch.float32), requires_grad=False
        )
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            group_detr=group_detr,
        )
        decoder_layer_wide = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
            group_detr=group_detr,
        )
        # SegmentationHead (instance segmentation); one DepthwiseConvBlock per decoder layer.
        self.mask_downsample_ratio = mask_downsample_ratio
        segmentation_head = (
            SegmentationHead(
                hidden_dim,
                num_layers,
                downsample_ratio=mask_downsample_ratio,
                image_size=eval_spatial_size,
            )
            if mask_downsample_ratio
            else None
        )
        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            self.eval_idx,
            layer_scale,
            act=activation,
            segmentation_head=segmentation_head,
        )
        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        if query_select_method == "agnostic":
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=activation)

        # Per-group prediction heads for Group DETR (matches the issue author's spec:
        # "multiple pairs of classification and regression prediction heads in the first
        # stage, each pair of which provides initialization for the object queries in
        # the corresponding group"). Encoder feature space stays shared so that groups
        # remain similar to each other; only the (cls, reg) pair differentiates them.
        if group_detr > 1:
            self.enc_score_head_group = nn.ModuleList(
                [copy.deepcopy(self.enc_score_head) for _ in range(group_detr)]
            )
            self.enc_bbox_head_group = nn.ModuleList(
                [copy.deepcopy(self.enc_bbox_head) for _ in range(group_detr)]
            )

        self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3, act=activation)

        # decoder head
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=activation)
        self.integral = Integral(self.reg_max)

        dec_score_head = nn.Linear(hidden_dim, num_classes)
        self.dec_score_head = nn.ModuleList(
            [
                dec_score_head if share_score_head else copy.deepcopy(dec_score_head)
                for _ in range(self.eval_idx + 1)
            ]
            + [
                copy.deepcopy(dec_score_head)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )

        # Share the same bbox head for all layers
        dec_bbox_head = MLP(
            hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3, act=activation
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                dec_bbox_head if share_bbox_head else copy.deepcopy(dec_bbox_head)
                for _ in range(self.eval_idx + 1)
            ]
            + [
                MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3, act=activation)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)
        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList(
            [nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]]
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity()
                for i in range(len(self.dec_bbox_head))
            ]
        )
        # Static eval shape -> pack the per-level MSDA value maps into one atlas
        # (single grid_sample per decoder layer; see TransformerDecoder.build_value_atlas).
        # Identity input_proj required: _build_value_atlas copies the raw encoder
        # outputs, which must therefore equal the MSDA value tensor.
        if (
            self.eval_spatial_size is not None
            and self.cross_attn_method == "default"
            and all(isinstance(p, nn.Identity) for p in self.input_proj)
        ):
            eval_h, eval_w = self.eval_spatial_size
            spatial_shapes = [
                [int(eval_h / s), int(eval_w / s)] for s in self.feat_strides
            ]
            num_points_list = self.decoder.layers[0].cross_attn.num_points_list
            assert all(
                l.cross_attn.num_points_list == num_points_list
                for l in self.decoder.layers
            )
            self.decoder.build_value_atlas(spatial_shapes, num_points_list)

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        if self.group_detr > 1:
            # Re-initialize each per-group prediction head with a fresh random draw so
            # that groups start from DIFFERENT weights. Without this, ``copy.deepcopy``
            # + identical bias reset would leave all groups producing identical top-K
            # selections and identical gradients, collapsing Group DETR to a single group.
            for head in self.enc_score_head_group:
                init.xavier_uniform_(head.weight)
                init.constant_(head.bias, bias)
            for bbox_head in self.enc_bbox_head_group:
                for layer in bbox_head.layers[:-1]:
                    init.xavier_uniform_(layer.weight)
                    init.constant_(layer.bias, 0)
                init.constant_(bbox_head.layers[-1].weight, 0)
                init.constant_(bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, "layers"):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        init.xavier_uniform_(self.query_pos_head.layers[-1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(
                                        in_channels, self.hidden_dim, 1, bias=False
                                    ),
                                ),
                                (
                                    "norm",
                                    nn.BatchNorm2d(
                                        self.hidden_dim,
                                    ),
                                ),
                            ]
                        )
                    )
                )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(
                                        in_channels,
                                        self.hidden_dim,
                                        3,
                                        2,
                                        padding=1,
                                        bias=False,
                                    ),
                                ),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: list[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes, proj_feats

    def _generate_anchors(
        self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"
    ):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h), torch.arange(w), indexing="ij"
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(
            -1, keepdim=True
        )
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes,
        denoising_logits=None,
        denoising_bbox_unact=None,
    ):
        selective_amp = self.training and os.getenv(
            "GTR_AMP_SAFE_MODE", ""
        ).lower() in {"selective", "heads_fp32", "matcher_fp32"}

        # prepare input for decoder
        if (
            self.training
            or self.eval_spatial_size is None
            or memory.shape[1] != self.anchors.shape[1]
        ):
            anchors, valid_mask = self._generate_anchors(
                spatial_shapes, device=memory.device
            )
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        memory_masked = valid_mask.to(memory.dtype) * memory
        memory_for_select = memory_masked.float() if selective_amp else memory_masked
        anchors_for_select = anchors.float() if selective_amp else anchors

        enc_topk_bboxes_list, enc_topk_logits_list, enc_topk_memory_list = [], [], []

        if self.training and self.group_detr > 1:
            # Per-group cls/reg heads pick their own top-K from the SHARED encoder memory.
            # Group diversity comes from different per-group head weights, not from a
            # per-group feature space — consistent with Group DETR's goal of keeping
            # queries across groups similar to each other.
            content_list, bbox_unact_list = [], []
            all_enc_bboxes, all_enc_logits, all_enc_memory = [], [], []
            for g_idx in range(self.group_detr):
                if selective_amp:
                    with torch.autocast(device_type=memory.device.type, enabled=False):
                        enc_outputs_logits_g = self.enc_score_head_group[g_idx](
                            memory_for_select
                        )
                        enc_topk_memory_g, enc_topk_logits_g, enc_topk_anchors_g = (
                            self._select_topk(
                                memory_for_select,
                                enc_outputs_logits_g,
                                anchors_for_select,
                                self.num_queries,
                            )
                        )
                        enc_topk_bbox_unact_g = (
                            self.enc_bbox_head_group[g_idx](enc_topk_memory_g)
                            + enc_topk_anchors_g
                        )
                else:
                    enc_outputs_logits_g = self.enc_score_head_group[g_idx](
                        memory_masked
                    )
                    enc_topk_memory_g, enc_topk_logits_g, enc_topk_anchors_g = (
                        self._select_topk(
                            memory_masked,
                            enc_outputs_logits_g,
                            anchors,
                            self.num_queries,
                        )
                    )
                    enc_topk_bbox_unact_g = (
                        self.enc_bbox_head_group[g_idx](enc_topk_memory_g)
                        + enc_topk_anchors_g
                    )

                all_enc_bboxes.append(F.sigmoid(enc_topk_bbox_unact_g))
                all_enc_logits.append(enc_topk_logits_g)
                all_enc_memory.append(enc_topk_memory_g)

                if self.learn_query_content:
                    content_g = self.tgt_embed.weight.unsqueeze(0).tile(
                        [memory.shape[0], 1, 1]
                    )
                else:
                    content_g = enc_topk_memory_g.detach()
                content_list.append(content_g)
                bbox_unact_list.append(enc_topk_bbox_unact_g.detach())

            enc_topk_bboxes_list.append(torch.cat(all_enc_bboxes, dim=1))
            enc_topk_logits_list.append(torch.cat(all_enc_logits, dim=1))
            enc_topk_memory_list.append(torch.cat(all_enc_memory, dim=1))
            content = torch.cat(content_list, dim=1)
            enc_topk_bbox_unact = torch.cat(bbox_unact_list, dim=1)
        else:
            # Inference (or group_detr == 1 training): use a single pair of stage-1 heads.
            # When group_detr > 1, ``self.enc_score_head`` / ``self.enc_bbox_head`` only
            # served as templates for ``copy.deepcopy`` and never received any gradient,
            # so we must fall back to group 0's trained heads instead.
            if self.group_detr > 1:
                enc_score_head = self.enc_score_head_group[0]
                enc_bbox_head = self.enc_bbox_head_group[0]
            else:
                enc_score_head = self.enc_score_head
                enc_bbox_head = self.enc_bbox_head

            if selective_amp:
                with torch.autocast(device_type=memory.device.type, enabled=False):
                    enc_outputs_logits: torch.Tensor = enc_score_head(memory_for_select)
                    enc_topk_memory, enc_topk_logits, enc_topk_anchors = (
                        self._select_topk(
                            memory_for_select,
                            enc_outputs_logits,
                            anchors_for_select,
                            self.num_queries,
                        )
                    )
                    enc_topk_bbox_unact: torch.Tensor = (
                        enc_bbox_head(enc_topk_memory) + enc_topk_anchors
                    )
            else:
                enc_outputs_logits: torch.Tensor = enc_score_head(memory_masked)
                enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
                    memory_masked, enc_outputs_logits, anchors, self.num_queries
                )
                enc_topk_bbox_unact: torch.Tensor = (
                    enc_bbox_head(enc_topk_memory) + enc_topk_anchors
                )

            if self.training or self.emit_loss_outputs:
                enc_topk_bboxes_list.append(F.sigmoid(enc_topk_bbox_unact))
                enc_topk_logits_list.append(enc_topk_logits)
                enc_topk_memory_list.append(enc_topk_memory)

            if self.learn_query_content:
                content = self.tgt_embed.weight.unsqueeze(0).tile(
                    [memory.shape[0], 1, 1]
                )
            else:
                content = enc_topk_memory.detach()
            enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat(
                [denoising_bbox_unact, enc_topk_bbox_unact], dim=1
            )
            content = torch.concat([denoising_logits, content], dim=1)

        return (
            content,
            enc_topk_bbox_unact,
            enc_topk_bboxes_list,
            enc_topk_logits_list,
            enc_topk_memory_list,
        )

    def _select_topk(
        self,
        memory: torch.Tensor,
        outputs_logits: torch.Tensor,
        outputs_anchors_unact: torch.Tensor,
        topk: int,
    ):
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == "agnostic":
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_ind: torch.Tensor

        topk_anchors = outputs_anchors_unact.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]),
        )

        topk_logits = (
            outputs_logits.gather(
                dim=1,
                index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]),
            )
            if (self.training or self.emit_loss_outputs)
            else None
        )

        topk_memory = memory.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1])
        )

        return topk_memory, topk_logits, topk_anchors

    @staticmethod
    def _split(x, dim, s_idx):
        return torch.split(x, s_idx, dim=dim) if x is not None else (None, None)

    def _build_value_atlas(self, feats):
        # Functional, in-graph atlas build: three level copies into a fresh zero
        # canvas, which inductor fuses into one kernel. Nothing is mutated, so the
        # whole model traces with fullgraph=True and captures into a single CUDA
        # graph (an eager fill into a persistent buffer would split the graph, and
        # a traced fill of that buffer gets functionalized into rebuild + copy-back).
        # Valid because input_proj is Identity (checked at deploy), so the MSDA
        # value IS the encoder output, and splitting C into (num_head, head_dim)
        # matches value_op's channel split.
        dec = self.decoder
        bs = feats[0].shape[0]
        nh = self.nhead
        c = self.hidden_dim // nh
        H_A, W_A = dec._va_hw
        canvas = feats[0].new_zeros((bs * nh, c, H_A, W_A))
        for f, (h, w), (r0, c0) in zip(feats, dec._va_shapes, dec._va_origins):
            canvas[:, :, r0 : r0 + h, c0 : c0 + w] = f.view(bs, nh, c, h, w).flatten(
                0, 1
            )
        return canvas

    def forward(self, feats, targets=None, spatial_feat=None):
        value_atlas = None
        if (
            getattr(self.decoder, "_va_ready", False)
            and not self.training
            and len(feats) == len(self.input_proj)
            and tuple((f.shape[2], f.shape[3]) for f in feats)
            == self.decoder._va_shapes
        ):
            value_atlas = self._build_value_atlas(feats)
        # input projection and embedding
        memory, spatial_shapes, _proj_feats = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                    group_detr=self.group_detr if self.training else 1,
                )
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                None,
                None,
                None,
                None,
            )

        (
            init_ref_contents,
            init_ref_points_unact,
            enc_topk_bboxes_list,
            enc_topk_logits_list,
            _enc_topk_memory_list,
        ) = self._get_decoder_input(
            memory, spatial_shapes, denoising_logits, denoising_bbox_unact
        )

        # decoder
        (
            out_bboxes,
            out_logits,
            out_corners,
            out_refs,
            pre_bboxes,
            pre_logits,
            _dec_hidden,
            out_masks,
            pre_segs,
        ) = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
            spatial_features=spatial_feat,
            value_atlas=value_atlas,
        )

        s_idx = dn_meta["dn_num_split"] if dn_meta is not None else None

        if self.training and dn_meta is not None:
            # the output from the first decoder layer, only one
            dn_pre_logits, pre_logits = self._split(pre_logits, 1, s_idx)
            dn_pre_bboxes, pre_bboxes = self._split(pre_bboxes, 1, s_idx)
            dn_pre_segs, pre_segs = self._split(pre_segs, 1, s_idx)

            dn_out_logits, out_logits = self._split(out_logits, 2, s_idx)
            dn_out_bboxes, out_bboxes = self._split(out_bboxes, 2, s_idx)
            dn_out_masks, out_masks = self._split(out_masks, 2, s_idx)

            dn_out_corners, out_corners = self._split(out_corners, 2, s_idx)
            dn_out_refs, out_refs = self._split(out_refs, 2, s_idx)

        if self.training or self.emit_loss_outputs:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_corners": out_corners[-1],
                "pred_masks": out_masks[-1] if out_masks is not None else None,
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
            }
        else:
            if self.eval_output_layer == 0:
                out = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            else:
                out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
            if out_masks is not None:
                out["pred_masks"] = out_masks[-1]

        if (self.training or self.emit_loss_outputs) and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_masks[:-1] if out_masks is not None else None,
                out_corners[-1],
                out_logits[-1],
            )
            out["enc_aux_outputs"] = self._set_aux_loss(
                enc_topk_logits_list, enc_topk_bboxes_list
            )
            out["pre_outputs"] = {
                "pred_logits": pre_logits,
                "pred_boxes": pre_bboxes,
                "pred_masks": pre_segs,
            }
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}

            if dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_masks,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                )
                out["dn_pre_outputs"] = {
                    "pred_logits": dn_pre_logits,
                    "pred_boxes": dn_pre_bboxes,
                    "pred_masks": dn_pre_segs,
                }
                out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class, outputs_coord)
        ]

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class,
        outputs_coord,
        outputs_corners,
        outputs_ref,
        outputs_masks=None,
        teacher_corners=None,
        teacher_logits=None,
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_masks is None:
            res = zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)
        else:
            res = zip(
                outputs_class,
                outputs_coord,
                outputs_corners,
                outputs_ref,
                outputs_masks,
            )

        results = []
        for items in res:
            result = {
                "pred_logits": items[0],
                "pred_boxes": items[1],
                "pred_corners": items[2],
                "ref_points": items[3],
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            if outputs_masks is not None:
                result["pred_masks"] = items[4]
            results.append(result)
        return results
