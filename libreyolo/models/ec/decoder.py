"""ECTransformer (D-FINE-style) decoder ported from EdgeCrafter (Apache-2.0).

Inference path is exact-match to upstream. Differences from LibreYOLO's D-FINE
decoder:
  * no ``enc_output`` projection layer (raw memory feeds enc_score_head)
  * TransformerDecoderLayer norms are named norm1/norm2 (no norm3)
  * query_pos_head MLP has 3 layers (vs D-FINE's 2)
  * ``up`` / ``reg_scale`` / ``anchors`` / ``valid_mask`` registered on the
    outer module, not on ``self.decoder``.

Segmentation head is intentionally omitted; this is the detection-only port.
The training/denoising path is included only enough to keep state-dict keys
matching upstream — actual training will be wired in a follow-up.
"""

from __future__ import annotations

import copy
import functools
import math
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ..dfine.denoising import get_contrastive_denoising_training_group
from .utils import (
    bias_init_with_prob,
    deformable_attention_core_func_v2,
    distance2bbox,
    distance2pose,
    get_activation,
    inverse_sigmoid,
    weighting_function,
)


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
        gates = torch.sigmoid(self.gate(torch.cat([x1, x2], dim=-1)))
        g1, g2 = gates.chunk(2, dim=-1)
        return self.norm(g1 * x1 + g2 * x2)


class Integral(nn.Module):
    """Distribution-to-scalar integral.

    Mathematically: ``out[i] = sum_j softmax(x)[i,j] * project[j]``.
    Implemented as elementwise mul + sum-reduce instead of the equivalent
    1-D matmul. The matmul form (``F.linear(softmax_x, project)``) hits a
    PyTorch MPS verifier bug during backward (``mps.matmul op contracting
    dimensions differ 4000 & 33``); the elementwise rewrite avoids it.
    """

    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = (x * project.to(x.device)).sum(dim=-1).reshape(-1, 4)
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
        return scores + self.reg_conf(stat.reshape(B, L, -1))


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
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels
            num_points_list = num_points
        else:
            num_points_list = [num_points] * num_levels
        self.num_points_list = num_points_list

        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer(
            "num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32)
        )

        self.total_points = num_heads * sum(num_points_list)
        self.method = method
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self.ms_deformable_attn_core = functools.partial(
            deformable_attention_core_func_v2, method=method
        )
        self._reset_parameters()
        if method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2 * math.pi / self.num_heads
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

    def forward(self, query, reference_points, value, value_spatial_shapes):
        bs, Len_q = query.shape[:2]
        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list), 2
        )
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list)
        )
        attention_weights = F.softmax(attention_weights, dim=-1)

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
            num_points_scale = self.num_points_scale.to(query.dtype).unsqueeze(-1)
            offset = (
                sampling_offsets
                * num_points_scale
                * reference_points[:, :, None, :, 2:]
                * self.offset_scale
            )
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}"
            )

        return self.ms_deformable_attn_core(
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
            self.num_points_list,
        )


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
    ):
        super().__init__()
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points, method=cross_attn_method
        )
        self.dropout2 = nn.Dropout(dropout)
        self.gateway = Gate(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        target,
        reference_points,
        value,
        spatial_shapes,
        attn_mask=None,
        query_pos_embed=None,
    ):
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = self.norm1(target + self.dropout1(target2))

        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes,
        )
        target = self.gateway(target, self.dropout2(target2))

        target2 = self.linear2(self.dropout3(self.activation(self.linear1(target))))
        target = target + self.dropout4(target2)
        return self.norm2(target.clamp(min=-65504, max=65504))


class TransformerDecoder(nn.Module):
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
        segmentation_head: Optional[nn.Module] = None,
    ):
        super().__init__()
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
        self.lqe_layers = nn.ModuleList(
            [copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)]
        )
        # Optional. Only present in ecseg variants.
        self.segmentation_head = segmentation_head

    def value_op(
        self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes
    ):
        value = value_proj(memory) if value_proj is not None else memory
        value = (
            F.interpolate(memory, size=value_scale)
            if value_scale is not None
            else value
        )
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(
            self.reg_max, self.up, self.reg_scale, deploy=True
        )
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList(
            [nn.Identity()] * self.eval_idx + [self.lqe_layers[self.eval_idx]]
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
        spatial_features=None,
    ):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        (
            dec_out_bboxes,
            dec_out_logits,
            dec_out_pred_corners,
            dec_out_refs,
            dec_out_hs,
        ) = [], [], [], [], []

        project = (
            self.project
            if hasattr(self, "project")
            else weighting_function(self.reg_max, up, reg_scale)
        )

        ref_points_detach = F.sigmoid(ref_points_unact)
        query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

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
            )

            if i == 0:
                pre_bboxes = F.sigmoid(
                    pre_bbox_head(output) + inverse_sigmoid(ref_points_detach)
                )
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(
                ref_points_initial, integral(pred_corners, project), reg_scale
            )

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                dec_out_hs.append(output)
                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        if self.segmentation_head is not None and spatial_features is not None:
            dec_out_segs = self.segmentation_head(
                spatial_features=spatial_features,
                query_features=dec_out_hs,
            )
            return (
                torch.stack(dec_out_bboxes),
                torch.stack(dec_out_logits),
                torch.stack(dec_out_pred_corners),
                torch.stack(dec_out_refs),
                pre_bboxes,
                pre_scores,
                torch.stack(dec_out_segs),
            )

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
        )


class ECTransformer(nn.Module):
    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=(256, 256, 256),
        feat_strides=(8, 16, 32),
        num_levels=3,
        num_points=(3, 6, 3),
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.0,
        activation="silu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=(640, 640),
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        share_bbox_head=False,
        share_score_head=False,
        mask_downsample_ratio: Optional[int] = None,
    ):
        super().__init__()
        feat_channels = list(feat_channels)
        feat_strides = list(feat_strides)
        num_points = list(num_points)
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
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        self._build_input_proj_layer(feat_channels)

        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)

        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
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
        )

        segmentation_head = (
            SegmentationHead(
                hidden_dim,
                num_layers,
                downsample_ratio=mask_downsample_ratio,
                image_size=eval_spatial_size or (640, 640),
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
            eval_idx,
            layer_scale,
            act=activation,
            segmentation_head=segmentation_head,
        )

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )
            init.normal_(self.denoising_class_embed.weight[:-1])

        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        if query_select_method == "agnostic":
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=activation)

        self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3, act=activation)
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=activation)
        self.integral = Integral(self.reg_max)

        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
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

        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)

        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList(
            [nn.Identity()] * self.eval_idx + [self.dec_score_head[self.eval_idx]]
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity()
                for i in range(len(self.dec_bbox_head))
            ]
        )

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
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
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
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

    def _get_encoder_input(self, feats):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                proj_feats.append(
                    self.input_proj[i](feats[-1] if i == len_srcs else proj_feats[-1])
                )

        feat_flatten = []
        spatial_shapes = []
        for feat in proj_feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
        return torch.concat(feat_flatten, 1), spatial_shapes

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
            anchors.append(torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(
            -1, keepdim=True
        )
        anchors = torch.log(anchors / (1 - anchors))
        return torch.where(valid_mask, anchors, torch.inf), valid_mask

    def _get_decoder_input(
        self, memory, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None
    ):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(
                spatial_shapes, device=memory.device
            )
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        memory = valid_mask.to(memory.dtype) * memory
        enc_outputs_logits = self.enc_score_head(memory)

        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            memory, enc_outputs_logits, anchors, self.num_queries
        )

        enc_topk_bbox_unact = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training:
            enc_topk_bboxes_list.append(F.sigmoid(enc_topk_bbox_unact))
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat(
                [denoising_bbox_unact, enc_topk_bbox_unact], dim=1
            )
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory, outputs_logits, outputs_anchors_unact, topk):
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        elif self.query_select_method == "agnostic":
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

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

    @staticmethod
    @torch.jit.unused
    def _set_aux_loss(outputs_class, outputs_coord):
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class, outputs_coord)
        ]

    @staticmethod
    @torch.jit.unused
    def _set_aux_loss2(
        outputs_class,
        outputs_coord,
        outputs_corners,
        outputs_ref,
        teacher_corners=None,
        teacher_logits=None,
    ):
        results = []
        for c, b, corners, ref in zip(
            outputs_class, outputs_coord, outputs_corners, outputs_ref
        ):
            results.append(
                {
                    "pred_logits": c,
                    "pred_boxes": b,
                    "pred_corners": corners,
                    "ref_points": ref,
                    "teacher_corners": teacher_corners,
                    "teacher_logits": teacher_logits,
                }
            )
        return results

    def forward(self, feats, targets=None, spatial_feat=None):
        memory, spatial_shapes = self._get_encoder_input(feats)

        if self.training and self.num_denoising > 0 and targets is not None:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
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
        ) = self._get_decoder_input(
            memory, spatial_shapes, denoising_logits, denoising_bbox_unact
        )

        decoder_out = self.decoder(
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
            spatial_features=spatial_feat,
        )
        if len(decoder_out) == 7:
            out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_scores, out_segs = decoder_out
        else:
            out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_scores = decoder_out
            out_segs = None

        # Split DN vs non-DN halves of every per-layer output.
        if self.training and dn_meta is not None:
            s_idx = dn_meta["dn_num_split"]
            dn_pre_logits, pre_scores = self._split(pre_scores, 1, s_idx)
            dn_pre_bboxes, pre_bboxes = self._split(pre_bboxes, 1, s_idx)
            dn_out_logits, out_logits = self._split(out_logits, 2, s_idx)
            dn_out_bboxes, out_bboxes = self._split(out_bboxes, 2, s_idx)
            dn_out_corners, out_corners = self._split(out_corners, 2, s_idx)
            dn_out_refs, out_refs = self._split(out_refs, 2, s_idx)

        if not self.training:
            result = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
            if out_segs is not None:
                result["pred_masks"] = out_segs[-1]
            return result

        out = {
            "pred_logits": out_logits[-1],
            "pred_boxes": out_bboxes[-1],
            "pred_corners": out_corners[-1],
            "ref_points": out_refs[-1],
            "up": self.up,
            "reg_scale": self.reg_scale,
        }

        if self.aux_loss:
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
            out["pre_outputs"] = {"pred_logits": pre_scores, "pred_boxes": pre_bboxes}
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


# ===========================================================================
# Segmentation head (ECSeg)
#
# Ported from EdgeCrafter's ``ecseg/engine/edgecrafter/segmentation_head.py``,
# which itself is derived from RF-DETR's seg head. State-dict naming preserved
# so released ecseg weights load with strict=False.
# ===========================================================================


class DepthwiseConvBlock(nn.Module):
    """Simplified ConvNeXt block (no MLP subnet)."""

    def __init__(self, dim: int, layer_scale_init_value: float = 0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return x + residual


class MLPBlock(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float = 0):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)]
        )
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        residual = x
        x = self.norm_in(x)
        for layer in self.layers:
            x = layer(x)
        if self.gamma is not None:
            x = self.gamma * x
        return x + residual


class SegmentationHead(nn.Module):
    """Lightweight per-instance mask head.

    Takes ``spatial_features`` from the encoder (typically ``feats[0]``, the
    highest-resolution map) and a list of per-decoder-layer query features.
    Produces one ``(B, N, H, W)`` mask-logit map per decoder layer.

    During eval the decoder breaks at ``eval_idx`` and only the last layer's
    query features are emitted, so ``len(query_features) == 1`` and only
    ``self.blocks[0]`` is actually applied to the spatial map (this matches
    upstream behavior).
    """

    def __init__(
        self,
        in_dim: int,
        num_blocks: int,
        bottleneck_ratio: int = 1,
        downsample_ratio: int = 4,
        image_size=(640, 640),
    ):
        super().__init__()
        self.downsample_ratio = downsample_ratio
        self.interaction_dim = (
            in_dim // bottleneck_ratio if bottleneck_ratio is not None else in_dim
        )
        self.blocks = nn.ModuleList([DepthwiseConvBlock(in_dim) for _ in range(num_blocks)])
        self.spatial_features_proj = (
            nn.Identity()
            if bottleneck_ratio is None
            else nn.Conv2d(in_dim, self.interaction_dim, kernel_size=1)
        )
        self.query_features_block = MLPBlock(in_dim)
        self.query_features_proj = (
            nn.Identity()
            if bottleneck_ratio is None
            else nn.Linear(in_dim, self.interaction_dim)
        )
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.image_size = tuple(image_size)

    def forward(
        self,
        spatial_features: torch.Tensor,
        query_features,
        skip_blocks: bool = False,
    ):
        target_size = (
            self.image_size[0] // self.downsample_ratio,
            self.image_size[1] // self.downsample_ratio,
        )
        spatial_features = F.interpolate(
            spatial_features, size=target_size, mode="bilinear", align_corners=False
        )
        mask_logits = []
        if not skip_blocks:
            for block, qf in zip(self.blocks, query_features):
                spatial_features = block(spatial_features)
                spatial_features_proj = self.spatial_features_proj(spatial_features)
                qf = self.query_features_proj(self.query_features_block(qf))
                mask_logits.append(
                    torch.einsum("bchw,bnc->bnhw", spatial_features_proj, qf) + self.bias
                )
        else:
            assert len(query_features) == 1
            qf = self.query_features_proj(
                self.query_features_block(query_features[0])
            )
            mask_logits.append(
                torch.einsum("bchw,bnc->bnhw", spatial_features, qf) + self.bias
            )
        return mask_logits


# ===========================================================================
# Pose decoder (ECPose / DETRPose)
#
# Sibling of :class:`ECTransformer` that swaps the box head for a per-keypoint
# DFL head and adds a within-instance / across-instance / cross-attention
# decoder layer pattern. Reuses :class:`Gate`, :class:`Integral`, :class:`MLP`
# from above. ``MSDeformAttnPose`` is a separate variant of multi-scale
# deformable attention because the upstream pose code uses a different
# state-dict layout (single ``n_points`` per level vs. EC's per-level
# ``num_points_list``) and pre-splits the value tensor before each layer.
# ===========================================================================


def _ms_deform_attn_core_pytorch_pose(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """Pose-specific deformable attention core.

    Mirrors super-gradients/DETRPose's ``ms_deform_attn_core_pytorch``: ``value``
    is a tuple of pre-split per-level tensors of shape
    ``(bs * n_heads, head_dim, h * w)``.
    """
    _, head_dim, _ = value[0].shape
    bs, num_query, num_heads, num_levels, num_points, _ = sampling_locations.shape

    sampling_grids = 2 * sampling_locations - 1
    sampling_grids = sampling_grids.transpose(1, 2).flatten(0, 1)

    sampling_value_list = []
    for lid_, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[lid_].unflatten(2, (h, w))
        grid_l = sampling_grids[:, :, lid_]
        sampling_value_list.append(
            F.grid_sample(
                value_l, grid_l,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )
        )
    attn = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_query, num_levels * num_points
    )
    out = (torch.cat(sampling_value_list, dim=-1) * attn).sum(-1).view(
        bs, num_heads * head_dim, num_query
    )
    return out.transpose(1, 2)


class MSDeformAttnPose(nn.Module):
    """Multi-scale deformable attention used by the pose decoder.

    Unlike :class:`MSDeformableAttention`, this variant takes a fixed
    ``n_points`` per level (no per-level list) and consumes ``value`` as a
    tuple of per-level tensors that the caller has already split.
    """

    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i % 4 + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.reshape(-1))
        if self.n_points % 4 != 0:
            init.constant_(self.sampling_offsets.bias, 0.0)
        init.constant_(self.attention_weights.weight, 0.0)
        init.constant_(self.attention_weights.bias, 0.0)

    def forward(self, query, reference_points, value, input_spatial_shapes):
        bs, len_q, _ = query.shape

        sampling_offsets = self.sampling_offsets(query).view(
            bs, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attn = self.attention_weights(query).view(
            bs, len_q, self.n_heads, self.n_levels * self.n_points
        )
        attn = F.softmax(attn, -1).view(
            bs, len_q, self.n_heads, self.n_levels, self.n_points
        )

        # reference_points enters as (bs, len_q, num_levels, num_per_kpt, 2);
        # transpose so per-keypoint axis becomes the level-broadcast axis.
        reference_points = torch.transpose(reference_points, 2, 3).flatten(1, 2)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(input_spatial_shapes, device=query.device)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.n_levels, 1, 2
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}"
            )

        return _ms_deform_attn_core_pytorch_pose(
            value, input_spatial_shapes, sampling_locations, attn
        )


class LQEPose(nn.Module):
    """Pose Local Quality Estimation.

    Refines bbox confidence by sampling the encoder's lowest-stride feature map
    at each predicted keypoint location and feeding the top-k of those values
    through a small MLP. Counterpart to detection :class:`LQE` (which derives
    quality from the box-corner distribution stats instead).
    """

    def __init__(self, topk: int, hidden_dim: int, num_layers: int, num_keypoints: int):
        super().__init__()
        self.k = topk
        self.num_keypoints = num_keypoints
        self.reg_conf = MLP(num_keypoints * (topk + 1), hidden_dim, 1, num_layers)
        init.constant_(self.reg_conf.layers[-1].weight, 0)
        init.constant_(self.reg_conf.layers[-1].bias, 0)

    def forward(self, scores, pred_poses, feat):
        b, l = pred_poses.shape[:2]
        pred_poses = pred_poses.reshape(b, l, self.num_keypoints, 2)
        sampling_values = F.grid_sample(
            feat,
            2 * pred_poses - 1,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        prob_topk = sampling_values.topk(self.k, dim=-1)[0]
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        return scores + self.reg_conf(stat.reshape(b, l, -1))


class PoseDeformableTransformerDecoderLayer(nn.Module):
    """Decoder layer with within- and across-instance self-attention + cross.

    Each query carries one global-instance token followed by ``num_keypoints``
    per-keypoint tokens. Within-instance self-attention lets keypoints attend
    to each other inside an instance; across-instance self-attention lets
    keypoints at the same body-index attend across detected people; deformable
    cross-attention pulls features from the encoder memory.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        n_levels: int = 3,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        super().__init__()
        self.within_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.within_dropout = nn.Dropout(dropout)
        self.within_norm = nn.LayerNorm(d_model)
        self.across_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.across_dropout = nn.Dropout(dropout)
        self.across_norm = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformAttnPose(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.gateway = Gate(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        if pos is not None:
            np_ = pos.shape[2]
            tensor[:, :, -np_:] = tensor[:, :, -np_:] + pos
        return tensor

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return self.norm2(tgt.clamp(min=-65504, max=65504))

    def forward(
        self,
        tgt_pose: torch.Tensor,
        tgt_pose_query_pos: torch.Tensor,
        tgt_pose_reference_points: torch.Tensor,
        attn_mask=None,
        memory: torch.Tensor = None,
        memory_spatial_shapes=None,
    ):
        bs, nq, num_kpt, d_model = tgt_pose.shape

        # within-instance self-attention
        q = k = self.with_pos_embed(tgt_pose, tgt_pose_query_pos).flatten(0, 1)
        tgt2 = self.within_attn(q, k, tgt_pose.flatten(0, 1))[0].reshape(
            bs, nq, num_kpt, d_model
        )
        tgt_pose = tgt_pose + self.within_dropout(tgt2)
        tgt_pose = self.within_norm(tgt_pose)

        # across-instance self-attention
        tgt_pose = tgt_pose.transpose(1, 2).flatten(0, 1)  # (bs*num_kpt, nq, d_model)
        q_pose = k_pose = tgt_pose
        tgt2_pose = self.across_attn(q_pose, k_pose, tgt_pose, attn_mask=attn_mask)[0]
        tgt2_pose = tgt2_pose.reshape(bs * num_kpt, nq, d_model)
        tgt_pose = tgt_pose + self.across_dropout(tgt2_pose)
        tgt_pose = self.across_norm(tgt_pose)
        tgt_pose = tgt_pose.reshape(bs, num_kpt, nq, d_model).transpose(1, 2)

        # deformable cross-attention
        tgt2_pose = self.cross_attn(
            self.with_pos_embed(tgt_pose, tgt_pose_query_pos).flatten(1, 2),
            tgt_pose_reference_points,
            memory,
            memory_spatial_shapes,
        ).reshape(bs, nq, num_kpt, d_model)

        tgt_pose = self.gateway(tgt_pose, self.dropout1(tgt2_pose))
        return self.forward_ffn(tgt_pose)


class PoseTransformerDecoder(nn.Module):
    """Iterative-refinement decoder over ``num_decoder_layers`` pose layers."""

    def __init__(
        self,
        decoder_layer: PoseDeformableTransformerDecoderLayer,
        num_layers: int,
        hidden_dim: int = 256,
        num_keypoints: int = 17,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_keypoints = num_keypoints
        self.eval_idx = num_layers - 1
        self.half_pose_ref_point_head = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

        # sine positional embedding tables
        dim_t = torch.arange(hidden_dim // 2, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / (hidden_dim // 2))
        self.register_buffer("dim_t", dim_t)
        self.scale = 2 * math.pi

    def _sine_embedding(self, pos_tensor: torch.Tensor) -> torch.Tensor:
        x_embed = pos_tensor[..., 0:1] * self.scale
        y_embed = pos_tensor[..., 1:2] * self.scale
        pos_x = x_embed / self.dim_t
        pos_y = y_embed / self.dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3)

    def forward(
        self,
        tgt: torch.Tensor,
        memory,
        refpoints_sigmoid: torch.Tensor,
        pre_pose_head: nn.Module,
        pose_head: nn.ModuleList,
        class_head: nn.ModuleList,
        lqe_head: nn.ModuleList,
        feat_lqe: torch.Tensor,
        integral: nn.Module,
        up: torch.Tensor,
        reg_scale: torch.Tensor,
        reg_max: int,
        project: torch.Tensor,
        attn_mask=None,
        spatial_shapes=None,
        eval_aux: bool = False,
    ):
        output = tgt
        refpoint_pose = refpoints_sigmoid
        output_pose_detach = pred_corners_undetach = 0

        dec_out_poses = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []

        pre_poses = None
        pre_scores = None
        for layer_id, layer in enumerate(self.layers):
            refpoint_pose_input = refpoint_pose[:, :, None]
            refpoint_only_pose = refpoint_pose[:, :, 1:]
            pose_query_sine = self._sine_embedding(refpoint_only_pose)
            pose_query_pos = self.half_pose_ref_point_head(pose_query_sine)

            output = layer(
                tgt_pose=output,
                tgt_pose_query_pos=pose_query_pos,
                tgt_pose_reference_points=refpoint_pose_input,
                attn_mask=attn_mask,
                memory=memory,
                memory_spatial_shapes=spatial_shapes,
            )

            output_pose = output[:, :, 1:]
            output_instance = output[:, :, 0]

            if layer_id == 0:
                pre_poses = F.sigmoid(
                    pre_pose_head(output_pose) + inverse_sigmoid(refpoint_only_pose)
                )
                pre_scores = class_head[0](output_instance)
                ref_pose_initial = pre_poses.detach()

            pred_corners = pose_head[layer_id](output_pose + output_pose_detach) + pred_corners_undetach
            refpoint_without_center = distance2pose(
                ref_pose_initial, integral(pred_corners, project), reg_scale
            )
            refpoint_center = torch.mean(refpoint_without_center, dim=2, keepdim=True)
            refpoint_pose = torch.cat([refpoint_center, refpoint_without_center], dim=2)

            if self.training or eval_aux or layer_id == self.eval_idx:
                score = class_head[layer_id](output_instance)
                logit = lqe_head[layer_id](score, refpoint_without_center, feat_lqe)
                dec_out_logits.append(logit)
                dec_out_poses.append(refpoint_without_center)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_pose_initial)
                if (not self.training) and (not eval_aux):
                    break

            pred_corners_undetach = pred_corners
            if self.training:
                refpoint_pose = refpoint_pose.detach()
                output_pose_detach = output_pose.detach()
            else:
                output_pose_detach = output_pose

        return (
            torch.stack(dec_out_poses),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_poses,
            pre_scores,
        )


class ECPoseTransformer(nn.Module):
    """Top-level pose transformer (sibling of :class:`ECTransformer`).

    Takes the same multi-scale encoder features as EC, but produces
    ``(pred_logits, pred_keypoints)`` via a DETR-style per-query keypoint
    decoder with iterative DFL refinement.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        nhead: int = 8,
        num_queries: int = 60,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        num_feature_levels: int = 3,
        dec_n_points: int = 4,
        learnable_tgt_init: bool = True,
        num_classes: int = 2,
        num_keypoints: int = 17,
        feat_strides=(8, 16, 32),
        eval_spatial_size=None,
        reg_max: int = 32,
        reg_scale: float = 4.0,
        cls_no_bias: bool = False,
        dec_pred_class_embed_share: bool = False,
        dec_pred_pose_embed_share: bool = False,
        two_stage_class_embed_share: bool = False,
        two_stage_bbox_embed_share: bool = False,
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = list(feat_strides)
        self.eval_spatial_size = (
            list(eval_spatial_size) if eval_spatial_size is not None else None
        )
        self.reg_max = reg_max
        self.deploy = False
        self.eval_aux = False

        decoder_layer = PoseDeformableTransformerDecoderLayer(
            d_model=hidden_dim,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=dec_n_points,
        )
        self.decoder = PoseTransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            hidden_dim=hidden_dim,
            num_keypoints=num_keypoints,
        )

        self.keypoint_embedding = nn.Embedding(num_keypoints, hidden_dim)
        self.instance_embedding = nn.Embedding(1, hidden_dim)
        self.learnable_tgt_init = learnable_tgt_init
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim) if learnable_tgt_init else None

        self.label_enc = nn.Embedding(80 + 1, hidden_dim)
        self.pose_enc = nn.Embedding(num_keypoints, hidden_dim)

        # two-stage encoder output projection
        self.enc_output = nn.Linear(hidden_dim, hidden_dim)
        self.enc_output_norm = nn.LayerNorm(hidden_dim)

        _class_embed = nn.Linear(hidden_dim, num_classes, bias=(not cls_no_bias))
        if not cls_no_bias:
            init.constant_(_class_embed.bias, bias_init_with_prob(0.01))

        _pre_point_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        init.constant_(_pre_point_embed.layers[-1].weight, 0)
        init.constant_(_pre_point_embed.layers[-1].bias, 0)

        _point_embed = MLP(hidden_dim, hidden_dim, 2 * (reg_max + 1), 3)
        init.constant_(_point_embed.layers[-1].weight, 0)
        init.constant_(_point_embed.layers[-1].bias, 0)

        _lqe_embed = LQEPose(4, 256, 2, num_keypoints)

        self.class_embed = nn.ModuleList(
            [
                _class_embed
                if dec_pred_class_embed_share
                else copy.deepcopy(_class_embed)
                for _ in range(num_decoder_layers)
            ]
        )
        self.lqe_embed = nn.ModuleList(
            [
                _lqe_embed if dec_pred_class_embed_share else copy.deepcopy(_lqe_embed)
                for _ in range(num_decoder_layers)
            ]
        )
        self.pose_embed = nn.ModuleList(
            [
                _point_embed
                if dec_pred_pose_embed_share
                else copy.deepcopy(_point_embed)
                for _ in range(num_decoder_layers)
            ]
        )
        self.pre_pose_embed = _pre_point_embed
        self.integral = Integral(reg_max)

        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)

        # two-stage encoder-output projections
        _keypoint_embed = MLP(hidden_dim, 2 * hidden_dim, 2 * num_keypoints, 4)
        init.constant_(_keypoint_embed.layers[-1].weight, 0)
        init.constant_(_keypoint_embed.layers[-1].bias, 0)
        self.enc_pose_embed = (
            _keypoint_embed if two_stage_bbox_embed_share else copy.deepcopy(_keypoint_embed)
        )
        self.enc_out_class_embed = (
            _class_embed if two_stage_class_embed_share else copy.deepcopy(_class_embed)
        )

        self._reset_parameters()

        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttnPose):
                m._reset_parameters()

    def _generate_anchors(self, spatial_shapes=None, device="cpu"):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])
        anchors = []
        for h, w in spatial_shapes:
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, h - 1, h, dtype=torch.float32, device=device),
                torch.linspace(0, w - 1, w, dtype=torch.float32, device=device),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], -1)
            grid = (grid.unsqueeze(0) + 0.5) / torch.tensor(
                [w, h], dtype=torch.float32, device=device
            )
            anchors.append(grid.view(1, -1, 2))
        anchors = torch.cat(anchors, 1)
        valid = ((anchors > 0.01) & (anchors < 0.99)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        return anchors, ~valid

    def _get_encoder_input(self, feats):
        feat_flatten = []
        spatial_shapes = []
        split_sizes = []
        for feat in feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
            split_sizes.append(h * w)
        return torch.cat(feat_flatten, 1), spatial_shapes, split_sizes

    def convert_to_deploy(self):
        self.project = weighting_function(
            self.reg_max, self.up, self.reg_scale, deploy=True
        )
        self.lqe_embed = nn.ModuleList(
            [nn.Identity()] * (self.num_decoder_layers - 1)
            + [self.lqe_embed[self.num_decoder_layers - 1]]
        )
        self.deploy = True

    def forward(self, feats, targets=None, samples=None):
        memory, spatial_shapes, split_sizes = self._get_encoder_input(feats)

        if self.training:
            output_proposals, valid_mask = self._generate_anchors(
                spatial_shapes, memory.device
            )
            output_memory = memory.masked_fill(valid_mask, 0.0)
            output_proposals = output_proposals.repeat(memory.size(0), 1, 1)
        else:
            output_proposals = self.anchors.repeat(memory.size(0), 1, 1)
            output_memory = memory.masked_fill(self.valid_mask, 0.0)

        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        topk = self.num_queries
        enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
        topk_idx = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]

        topk_memory = output_memory.gather(
            dim=1,
            index=topk_idx.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]),
        )
        topk_anchors = output_proposals.gather(
            dim=1, index=topk_idx.unsqueeze(-1).repeat(1, 1, 2)
        )

        bs, nq = topk_memory.shape[:2]
        delta_unsig_kpt = self.enc_pose_embed(topk_memory).reshape(
            bs, nq, self.num_keypoints, 2
        )
        enc_outputs_pose_coord = F.sigmoid(delta_unsig_kpt + topk_anchors.unsqueeze(-2))
        enc_outputs_center_coord = torch.mean(enc_outputs_pose_coord, dim=2, keepdim=True)
        enc_outputs_pose_coord = torch.cat(
            [enc_outputs_center_coord, enc_outputs_pose_coord], dim=2
        )
        refpoint_pose_sigmoid = enc_outputs_pose_coord.detach()

        if self.learnable_tgt_init:
            tgt = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1).unsqueeze(-2)
        else:
            tgt = topk_memory.detach().unsqueeze(-2)

        tgt_pose = (
            self.keypoint_embedding.weight[None, None]
            .repeat(1, topk, 1, 1)
            .expand(bs, -1, -1, -1)
            + tgt
        )
        tgt_global = (
            self.instance_embedding.weight[None, None]
            .repeat(1, topk, 1, 1)
            .expand(bs, -1, -1, -1)
        )
        tgt_pose = torch.cat([tgt_global, tgt_pose], dim=2)

        # pre-split memory for the deformable attention
        value = memory.unflatten(2, (self.nhead, -1))
        value = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_sizes, dim=-1)

        project = (
            self.project
            if hasattr(self, "project")
            else weighting_function(self.reg_max, self.up, self.reg_scale)
        )

        out_poses, out_logits, _, _, _, _ = self.decoder(
            tgt=tgt_pose,
            memory=value,
            refpoints_sigmoid=refpoint_pose_sigmoid,
            spatial_shapes=spatial_shapes,
            attn_mask=None,
            pre_pose_head=self.pre_pose_embed,
            pose_head=self.pose_embed,
            class_head=self.class_embed,
            lqe_head=self.lqe_embed,
            feat_lqe=feats[0],
            up=self.up,
            reg_max=self.reg_max,
            reg_scale=self.reg_scale,
            integral=self.integral,
            project=project,
            eval_aux=self.eval_aux,
        )

        if not self.deploy:
            # Stack output keypoints into shape (L, bs, nq, 2*K) for evaluation.
            out_poses = out_poses.flatten(-2)

        return {"pred_logits": out_logits[-1], "pred_keypoints": out_poses[-1]}
