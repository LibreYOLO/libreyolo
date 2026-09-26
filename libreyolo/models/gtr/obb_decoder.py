"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
OBB decoder: GTRTransformer adapted to oriented boxes following
"Real-Time Oriented Object Detection Transformer in Remote Sensing Images".

Changes vs. the HBB decoder:
* boxes are (cx, cy, w, h, a) in sigmoid domain, a = theta / (pi/2);
* Angle Distribution Refinement: FDR extended from 4 to 6 distributions
  (4 external-rectangle edges + 2 vertex offsets), decoded via distance2rbox;
* rotation-aware deformable cross-attention (sampling offsets rotated by theta);
* Oriented Contrastive Denoising (box-noise mode, angle unchanged).
"""

# Adapted for LibreYOLO from Intellindust-AI-Lab/GTR
# revision 782e737efe2e6437ac537fbdcee089673d3376c1 (MIT).
# Changes: local imports, no registry, anchors regenerated for non-default
# input sizes. See NOTICE.

import copy
import functools
import math
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .decoder import MLP, Gate
from .obb_denoising import get_obb_contrastive_denoising_training_group
from .obb_rbox import PI, distance2rbox
from .utils import (
    bias_init_with_prob,
    deformable_attention_core_func_v2,
    get_activation,
    inverse_sigmoid,
    weighting_function,
)

__all__ = ["OBBGTRTransformer"]

NUM_DIST = 6  # 4 external-rectangle edges + 2 vertex offsets


class OBBIntegral(nn.Module):
    """Integral over 6 discrete distributions: sum{Pr(n) * W(n)} per distance."""

    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, NUM_DIST)
        return x.reshape(list(shape[:-1]) + [-1])


class OBBLQE(nn.Module):
    """Location quality estimator over the 6 ADR distributions."""

    def __init__(self, k, hidden_dim, num_layers, reg_max, act="relu"):
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(NUM_DIST * (k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, NUM_DIST, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class OBBMSDeformableAttention(nn.Module):
    """Multi-scale deformable attention with rotation-aware sampling for 5-dim references."""

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method="default",
        offset_scale=0.5,
        orthogonal_attn=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale
        # Rotation-Rectified Orthogonal Attention: split heads so the second half
        # samples along the box minor axis (theta + pi/2) instead of aligning every
        # head with the major axis (paper Sec 3.2). Off -> all heads align with theta.
        assert (not orthogonal_attn) or num_heads % 2 == 0, (
            "orthogonal_attn requires an even num_heads"
        )
        self.orthogonal_attn = orthogonal_attn

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

    def _reset_parameters(self):
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

        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: List[int],
    ):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 5] sigmoid-domain rboxes
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2]
        """
        bs, Len_q = query.shape[:2]

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list), 2
        )

        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list)
        )
        attention_weights = F.softmax(attention_weights, dim=-1)

        assert reference_points.shape[-1] == 5, (
            f"OBB reference points must be 5-dim, got {reference_points.shape[-1]}"
        )
        num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
        offset = (
            sampling_offsets
            * num_points_scale
            * reference_points[:, :, None, :, 2:4]
            * self.offset_scale
        )
        # rotate offsets around the box center by the predicted angle
        theta = reference_points[:, :, None, :, 4:5] * PI  # [bs, Len_q, 1, n_ref, 1]
        if self.orthogonal_attn:
            # Second half of the heads sample orthogonally (theta + pi/2) to cover the
            # box minor axis; broadcasts theta from the head-shared dim to per-head.
            head_phase = torch.zeros(
                self.num_heads, dtype=theta.dtype, device=theta.device
            )
            head_phase[self.num_heads // 2 :] = math.pi / 2
            theta = theta + head_phase.view(
                1, 1, self.num_heads, 1, 1
            )  # [bs, Len_q, H, n_ref, 1]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        ox, oy = offset[..., 0:1], offset[..., 1:2]
        offset = torch.cat([cos_t * ox - sin_t * oy, sin_t * ox + cos_t * oy], dim=-1)
        sampling_locations = reference_points[:, :, None, :, :2] + offset

        output = self.ms_deformable_attn_core(
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
            self.num_points_list,
        )

        return output


class OBBTransformerDecoderLayer(nn.Module):
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
        orthogonal_attn=False,
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
        # cross attention (rotation-aware)
        self.cross_attn = OBBMSDeformableAttention(
            d_model,
            n_head,
            n_levels,
            n_points,
            method=cross_attn_method,
            orthogonal_attn=orthogonal_attn,
        )
        self.dropout2 = nn.Dropout(dropout)

        self.gateway = Gate(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

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
    ):

        bs = target.shape[0]

        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        if self.training and self.group_detr > 1:
            num_det = target.shape[1] - num_dn
            group_size = num_det // self.group_detr

            if num_dn > 0:
                dn_q, det_q = q[:, :num_dn], q[:, num_dn:]
                dn_k, det_k = k[:, :num_dn], k[:, num_dn:]
                dn_v, det_v = target[:, :num_dn], target[:, num_dn:]

                dn_mask = attn_mask[:num_dn, :num_dn] if attn_mask is not None else None
                dn_out, _ = self.self_attn(dn_q, dn_k, dn_v, attn_mask=dn_mask)

                det_q_g = torch.cat(det_q.split(group_size, dim=1), dim=0)
                det_k_g = torch.cat(det_k.split(group_size, dim=1), dim=0)
                det_v_g = torch.cat(det_v.split(group_size, dim=1), dim=0)
                det_out, _ = self.self_attn(det_q_g, det_k_g, det_v_g)
                det_out = torch.cat(det_out.split(bs, dim=0), dim=1)

                target2 = torch.cat([dn_out, det_out], dim=1)
            else:
                q_g = torch.cat(q.split(group_size, dim=1), dim=0)
                k_g = torch.cat(k.split(group_size, dim=1), dim=0)
                v_g = torch.cat(target.split(group_size, dim=1), dim=0)
                target2, _ = self.self_attn(q_g, k_g, v_g)
                target2 = torch.cat(target2.split(bs, dim=0), dim=1)
        else:
            target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)

        target = target + self.dropout1(target2)
        target = self.norm1(target)

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

        target = target + self.dropout4(target2)
        target = self.norm2(target.clamp(min=-65504, max=65504))

        return target


class OBBTransformerDecoder(nn.Module):
    """Decoder implementing Angle Distribution Refinement over 6 distributions."""

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
        decouple_angle_pe=False,
        decouple_angle_refine=False,
        angle_decay_alpha0=1.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        # Geometry-Decoupled Query Encoding: feed only (cx,cy,w,h) to query_pos_head so
        # the positional prior stays rotation-invariant and theta is content-driven.
        self.decouple_angle_pe = decouple_angle_pe
        # Decoupled Periodic Refinement (DPR-1): refine theta with a per-layer bounded
        # coarse-to-fine periodic update, decoupled from the ADR angle output. alpha_i =
        # angle_decay_alpha0^{-i} gives large early corrections, fine later ones.
        assert (not decouple_angle_refine) or angle_decay_alpha0 >= 1.0, (
            "angle_decay_alpha0 must be >= 1 so alpha_i = alpha0^{-i} is non-increasing (coarse-to-fine)"
        )
        self.decouple_angle_refine = decouple_angle_refine
        self.angle_decay_alpha0 = angle_decay_alpha0
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [
                copy.deepcopy(decoder_layer_wide)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )
        self.lqe_layers = nn.ModuleList(
            [
                copy.deepcopy(OBBLQE(4, 64, 2, reg_max, act=act))
                for _ in range(num_layers)
            ]
        )

    def value_op(
        self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes
    ):
        """Per-level values pre-reshaped to ``[bs*n_head, c, h, w]`` and contiguous --
        the layout deformable_attention_core_func_v2 expects on its ``value_shape='default'``
        path (same contract as the det TransformerDecoder.value_op, minus the deploy atlas)."""
        value = value_proj(memory) if value_proj is not None else memory
        value = (
            F.interpolate(memory, size=value_scale)
            if value_scale is not None
            else value
        )
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        bs = value.shape[0]
        value = value.reshape(bs, value.shape[1], self.num_head, -1).permute(0, 2, 3, 1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        c = value.shape[2]
        return tuple(
            v.reshape(bs * self.num_head, c, h, w).contiguous()
            for v, (h, w) in zip(
                value.split(split_shape, dim=-1), memory_spatial_shapes
            )
        )

    def convert_to_deploy(self):
        # Register as buffer so model.to(device/dtype) casts/moves it (fp16 inference, CUDA-graph
        # capture); a plain attribute would silently stay on the CPU in fp32.
        self.register_buffer(
            "project",
            weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True),
            persistent=False,
        )
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList(
            [nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]]
        )

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
        angle_head=None,
    ):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        if not hasattr(self, "project"):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = F.sigmoid(ref_points_unact)
        ref_for_pe = (
            ref_points_detach[..., :4] if self.decouple_angle_pe else ref_points_detach
        )
        query_pos_embed = query_pos_head(ref_for_pe).clamp(min=-10, max=10)

        num_dn = (
            dn_meta["dn_num_split"][0] if (dn_meta is not None and self.training) else 0
        )

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
            )

            if i == 0:
                # Initial oriented boxes from a traditional regression head (paper: first layer)
                pre_bboxes = F.sigmoid(
                    pre_bbox_head(output) + inverse_sigmoid(ref_points_detach)
                )
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            # Refine the 6 ADR distributions in a residual manner, then decode to an rbox
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2rbox(
                ref_points_initial, integral(pred_corners, project), reg_scale
            )

            if self.decouple_angle_refine:
                # DPR-1: drop the ADR-decoded angle; refine theta by a bounded coarse-to-fine
                # periodic update off the reference angle. theta_ref carries across layers.
                if i == 0:
                    theta_ref_rad = ref_points_initial[..., 4:5] * PI
                dtheta = torch.tanh(angle_head[i](output + output_detach)) * (
                    self.angle_decay_alpha0 ** (-i)
                )
                theta_new_rad = torch.remainder(theta_ref_rad + dtheta, PI)
                inter_ref_bbox = torch.cat(
                    [inter_ref_bbox[..., :4], theta_new_rad / PI], dim=-1
                )
                theta_ref_rad = theta_new_rad.detach()

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
        )


class OBBGTRTransformer(nn.Module):
    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        num_classes=15,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
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
        decouple_angle_pe=False,
        orthogonal_attn=False,
        decouple_angle_refine=False,
        angle_decay_alpha0=1.5,
    ):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

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
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = OBBTransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            group_detr=group_detr,
            orthogonal_attn=orthogonal_attn,
        )
        decoder_layer_wide = OBBTransformerDecoderLayer(
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
            orthogonal_attn=orthogonal_attn,
        )
        self.decoder = OBBTransformerDecoder(
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
            decouple_angle_pe=decouple_angle_pe,
            decouple_angle_refine=decouple_angle_refine,
            angle_decay_alpha0=angle_decay_alpha0,
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
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 5, 3, act=activation)

        if group_detr > 1:
            self.enc_score_head_group = nn.ModuleList(
                [copy.deepcopy(self.enc_score_head) for _ in range(group_detr)]
            )
            self.enc_bbox_head_group = nn.ModuleList(
                [copy.deepcopy(self.enc_bbox_head) for _ in range(group_detr)]
            )

        # Geometry-Decoupled Query Encoding masks theta out of the positional prior.
        self.query_pos_head = MLP(
            4 if decouple_angle_pe else 5, hidden_dim, hidden_dim, 3, act=activation
        )

        # decoder head
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 5, 3, act=activation)
        self.integral = OBBIntegral(self.reg_max)

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

        dec_bbox_head = MLP(
            hidden_dim, hidden_dim, NUM_DIST * (self.reg_max + 1), 3, act=activation
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                dec_bbox_head if share_bbox_head else copy.deepcopy(dec_bbox_head)
                for _ in range(self.eval_idx + 1)
            ]
            + [
                MLP(
                    scaled_dim,
                    scaled_dim,
                    NUM_DIST * (self.reg_max + 1),
                    3,
                    act=activation,
                )
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )

        # DPR-1 per-layer angle head (predicts the raw angle offset delta-theta). Mirrors
        # dec_bbox_head widths; zero-init the last layer so the update starts at theta_ref.
        if decouple_angle_refine:
            self.angle_head = nn.ModuleList(
                [
                    MLP(hidden_dim, hidden_dim, 1, 3, act=activation)
                    for _ in range(self.eval_idx + 1)
                ]
                + [
                    MLP(scaled_dim, scaled_dim, 1, 3, act=activation)
                    for _ in range(num_layers - self.eval_idx - 1)
                ]
            )
            for m in self.angle_head:
                init.constant_(m.layers[-1].weight, 0)
                init.constant_(m.layers[-1].bias, 0)
        else:
            self.angle_head = None

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)

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
        if self.angle_head is not None:
            self.angle_head = nn.ModuleList(
                [
                    self.angle_head[i] if i <= self.eval_idx else nn.Identity()
                    for i in range(len(self.angle_head))
                ]
            )

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        if self.group_detr > 1:
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

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])

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
        # angle channel: unact 0 -> sigmoid 0.5 -> theta = pi/4 (midpoint of [0, pi/2))
        anchors = torch.concat([anchors, torch.zeros_like(anchors[..., :1])], dim=-1)
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes,
        denoising_logits=None,
        denoising_bbox_unact=None,
    ):
        # prepare input for decoder
        # LibreYOLO: regenerate anchors when the input differs from the eval size.
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

        enc_topk_bboxes_list, enc_topk_logits_list, enc_topk_memory_list = [], [], []

        if self.training and self.group_detr > 1:
            content_list, bbox_unact_list = [], []
            all_enc_bboxes, all_enc_logits, all_enc_memory = [], [], []
            for g_idx in range(self.group_detr):
                enc_outputs_logits_g = self.enc_score_head_group[g_idx](memory_masked)
                enc_topk_memory_g, enc_topk_logits_g, enc_topk_anchors_g = (
                    self._select_topk(
                        memory_masked, enc_outputs_logits_g, anchors, self.num_queries
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
            if self.group_detr > 1:
                enc_score_head = self.enc_score_head_group[0]
                enc_bbox_head = self.enc_bbox_head_group[0]
            else:
                enc_score_head = self.enc_score_head
                enc_bbox_head = self.enc_bbox_head

            enc_outputs_logits: torch.Tensor = enc_score_head(memory_masked)
            enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
                memory_masked, enc_outputs_logits, anchors, self.num_queries
            )
            enc_topk_bbox_unact: torch.Tensor = (
                enc_bbox_head(enc_topk_memory) + enc_topk_anchors
            )

            if self.training:
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
            if self.training
            else None
        )

        topk_memory = memory.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1])
        )

        return topk_memory, topk_logits, topk_anchors

    @staticmethod
    def _split(x, dim, s_idx):
        return torch.split(x, s_idx, dim=dim) if x is not None else (None, None)

    def forward(self, feats, targets=None):
        # input projection and embedding
        memory, spatial_shapes, proj_feats = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                get_obb_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
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
            enc_topk_memory_list,
        ) = self._get_decoder_input(
            memory, spatial_shapes, denoising_logits, denoising_bbox_unact
        )

        # decoder
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = (
            self.decoder(
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
                angle_head=self.angle_head,
            )
        )

        s_idx = dn_meta["dn_num_split"] if dn_meta is not None else None

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = self._split(pre_logits, 1, s_idx)
            dn_pre_bboxes, pre_bboxes = self._split(pre_bboxes, 1, s_idx)

            dn_out_logits, out_logits = self._split(out_logits, 2, s_idx)
            dn_out_bboxes, out_bboxes = self._split(out_bboxes, 2, s_idx)

            dn_out_corners, out_corners = self._split(out_corners, 2, s_idx)
            dn_out_refs, out_refs = self._split(out_refs, 2, s_idx)

        if self.training:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
            }
        else:
            if self.eval_output_layer == 0:
                out = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            else:
                out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
            )
            out["enc_aux_outputs"] = self._set_aux_loss(
                enc_topk_logits_list, enc_topk_bboxes_list
            )
            out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}

            if dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                )
                out["dn_pre_outputs"] = {
                    "pred_logits": dn_pre_logits,
                    "pred_boxes": dn_pre_bboxes,
                }
                out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
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
        teacher_corners=None,
        teacher_logits=None,
    ):
        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_corners": c,
                "ref_points": d,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            for a, b, c, d in zip(
                outputs_class, outputs_coord, outputs_corners, outputs_ref
            )
        ]
