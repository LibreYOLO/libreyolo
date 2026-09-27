"""OV-DEIM decoder (vendored, inference-only).

Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Ported from OV-DEIM ``model/decoder/ovdeim_decoder.py``, ``cls_embed.py`` and
``decoder/utils.py`` (https://github.com/wleilei/OV-DEIM, Apache-2.0). Module
and attribute names mirror upstream so released checkpoints load with a plain
key strip. Training-only branches (contrastive denoising, auxiliary losses)
are removed; construct with ``num_denoising=0``.
"""

import copy
import functools
import math
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ....kernels.attention.ms_deform_attn import (
    maybe_ms_deform_attn_v2,
    ms_deform_attn_available,
    spatial_shapes_tensor,
)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    return float(-math.log((1 - prior_prob) / prior_prob))


def deformable_attention_core_func_v2(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: List[int],
    method="default",
):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels * n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels * n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]

    The loop below is the default and the export path. When the optional
    accelerated ``ms_deform_attn`` slot resolves and the flat point layout
    reshapes onto the slot's ``(n_levels, n_points)`` one, it takes over
    (see ``libreyolo/kernels/attention/ms_deform_attn.py``);
    ``method='discrete'`` never does, its integer-index sampling is a
    different equation.
    """
    if method == "default" and ms_deform_attn_available(value):
        accelerated = maybe_ms_deform_attn_v2(
            value,
            spatial_shapes_tensor(value_spatial_shapes, value.device),
            sampling_locations,
            attention_weights,
            num_points_list,
        )
        if accelerated is not None:
            return accelerated
    bs, _, n_head, c = value.shape
    _, Len_q, _, _, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)

    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(tuple(num_points_list), dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value_list[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]

        if method == "default":
            sampling_value_l = F.grid_sample(
                value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        elif method == "discrete":
            sampling_coord = (
                sampling_grid_l * torch.tensor([[w, h]], device=value.device) + 0.5
            ).to(torch.int64)
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 2)

            s_idx = (
                torch.arange(sampling_coord.shape[0], device=value.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l: torch.Tensor = value_l[s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(
                bs * n_head, c, Len_q, num_points_list[level]
            )

        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, sum(num_points_list))
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


def get_activation(act, inplace: bool = True):
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act

    act = act.lower()
    if act in ("silu", "swish"):
        m = nn.SiLU()
    elif act == "relu":
        m = nn.ReLU()
    elif act == "leaky_relu":
        m = nn.LeakyReLU()
    elif act == "gelu":
        m = nn.GELU()
    elif act == "hardsigmoid":
        m = nn.Hardsigmoid()
    else:
        raise RuntimeError(f"Unsupported activation: {act!r}")

    if hasattr(m, "inplace"):
        m.inplace = inplace
    return m


class ClassEmbed(nn.Module):
    """Region-text alignment head: cosine similarity with learned scale/bias."""

    def __init__(self, init_bias: float = 100.0, init_scale: float = 15.0):
        super().__init__()
        bias_value = -math.log(init_bias)
        self.lang_bias = nn.Parameter(torch.full((), bias_value))
        self.lang_scale = nn.Parameter(torch.tensor(init_scale).log())

    def forward(self, image_embeds, lang_embeds, mask=None):
        image_norm = F.normalize(image_embeds, p=2, dim=-1)
        lang_norm = F.normalize(lang_embeds, p=2, dim=-1)

        logits = image_norm @ lang_norm.transpose(2, 1)  # [batch, num_queries, num_classes]
        logits = logits * torch.exp(self.lang_scale) + self.lang_bias

        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
        return logits


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, act: str = "relu") -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points=4,
        method: str = "default",
        offset_scale: float = 0.5,
    ) -> None:
        """Multi-Scale Deformable Attention"""
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        num_points_list = num_points
        assert len(num_points_list) == num_levels, (
            f"Length of num_points_list ({len(num_points_list)}) must match num_levels ({num_levels})"
        )
        self.num_points_list = num_points_list

        num_points_scale = [1.0 / n for n in self.num_points_list for _ in range(n)]
        self.register_buffer("num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)
        self._reset_parameters()

        if method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self) -> None:
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.cat(
            [torch.arange(1, n + 1, dtype=torch.float32) for n in self.num_points_list]
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
        value_spatial_shapes: List[Tuple[int, int]],
        value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bs, query_length = query.shape[:2]
        value_length = value.shape[1]

        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)

        value = value.view(bs, value_length, self.num_heads, self.head_dim)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, query_length, self.num_heads, sum(self.num_points_list), 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, query_length, self.num_heads, sum(self.num_points_list)
        )
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes, device=query.device, dtype=query.dtype)
            offset_normalizer = offset_normalizer.flip([1]).view(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points.view(bs, query_length, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}"
            )

        return self.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list
        )


# Taken from: https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py#L14-L34
class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.w12.weight)
        init.constant_(self.w12.bias, 0)
        init.xavier_uniform_(self.w3.weight)
        init.constant_(self.w3.bias, 0)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        output = output * self.scale
        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        n_levels: int = 4,
        n_points=4,
        cross_attn_method: str = "default",
    ) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(d_model)

        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)

        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)

        self.ffn = SwiGLUFFN(d_model, dim_feedforward, d_model)

    def with_pos_embed(self, tensor: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        target: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        memory_spatial_shapes: List[Tuple[int, int]],
        attn_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        query_pos_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            memory,
            memory_spatial_shapes,
            memory_mask,
        )
        target = target + self.dropout2(target2)
        target = self.norm2(target)

        target2 = self.ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target)
        return target


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim: int, decoder_layer: nn.Module, num_layers: int, eval_idx: int = -1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        target: torch.Tensor,
        ref_points_unact: torch.Tensor,
        memory: torch.Tensor,
        memory_spatial_shapes: List[Tuple[int, int]],
        text_feats: torch.Tensor,
        bbox_head: nn.ModuleList,
        score_head: nn.ModuleList,
        query_pos_head: nn.Module,
        attn_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact.clamp(min=-500, max=500))
        query_pos_embed = query_pos_head(ref_points_detach)
        output = target
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            output = layer(
                output, ref_points_input, memory, memory_spatial_shapes, attn_mask, memory_mask, query_pos_embed
            )

            inter_ref_bbox = F.sigmoid(
                (bbox_head[i](output) + inverse_sigmoid(ref_points_detach)).clamp(min=-500, max=500)
            )

            if i == self.eval_idx:
                dec_out_logits.append(score_head[i](output, text_feats))
                dec_out_bboxes.append(inter_ref_bbox)
                break
            ref_points_detach = inter_ref_bbox.detach()

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


class OVDEIMDecoder(nn.Module):
    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_queries: int = 300,
        feat_channels: List[int] = [512, 1024, 2048],
        feat_strides: List[int] = [8, 16, 32],
        num_levels: int = 3,
        num_points=4,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "silu",
        num_denoising: int = 0,
        learn_query_content: bool = False,
        eval_spatial_size: Optional[Tuple[int, int]] = None,
        eval_idx: int = -1,
        eps: float = 1e-2,
        cross_attn_method: str = "default",
        query_select_method: str = "default",
        num_enc_queries: int = 0,
    ) -> None:
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        assert query_select_method in ("default", "agnostic")
        assert cross_attn_method in ("default", "discrete")

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides + [feat_strides[-1] * 2] * (num_levels - len(feat_strides))
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method
        self.num_enc_queries = num_enc_queries

        self._build_input_proj_layer(feat_channels)

        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_points, cross_attn_method
        )
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx)

        # Inference port: the contrastive-denoising embedding is training-only
        # (upstream's own evaluation script skips loading it).
        self.num_denoising = num_denoising
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3, act=activation)

        self.enc_output = nn.Identity()

        self.enc_score_head = ClassEmbed() if query_select_method != "agnostic" else nn.Linear(hidden_dim, 1)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=activation)

        self.dec_score_head = nn.ModuleList([ClassEmbed() for _ in range(num_layers)])
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4, 3, act=activation) for _ in range(num_layers)]
        )

        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)
        else:
            self.anchors = None
            self.valid_mask = None

        self._reset_parameters(feat_channels)

    def _reset_parameters(self, feat_channels) -> None:
        bias = bias_init_with_prob(0.01)
        if isinstance(self.enc_score_head, nn.Linear):
            init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_head, reg_head in zip(self.dec_score_head, self.dec_bbox_head):
            if isinstance(cls_head, nn.Linear):
                init.constant_(cls_head.bias, bias)
            init.constant_(reg_head.layers[-1].weight, 0)
            init.constant_(reg_head.layers[-1].bias, 0)

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
                                ("conv", nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
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
                                ("conv", nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
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
                proj_feats.append(self.input_proj[i](feats[-1] if i == len_srcs else proj_feats[-1]))

        feat_flatten = []
        spatial_shapes = []
        for feat in proj_feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append((h, w))
        return torch.cat(feat_flatten, 1), spatial_shapes

    def _generate_anchors(
        self,
        spatial_shapes: Optional[List[Tuple[int, int]]] = None,
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if spatial_shapes is None:
            spatial_shapes = [
                (int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s))
                for s in self.feat_strides
            ]

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).to(dtype=dtype)
            grid_xy = (grid_xy + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            lvl_anchors = torch.cat([grid_xy, wh], dim=-1).view(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.cat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_decoder_input(self, memory: torch.Tensor, spatial_shapes, text_feats: torch.Tensor):
        anchors, valid_mask = (
            self._generate_anchors(spatial_shapes, device=memory.device)
            if self.training or self.eval_spatial_size is None
            else (self.anchors, self.valid_mask)
        )
        memory = memory * valid_mask.to(memory.dtype)

        output_memory = self.enc_output(memory)
        enc_outputs_logits = self.enc_score_head(output_memory, text_feats)
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            output_memory, enc_outputs_logits, anchors, self.num_queries
        )
        enc_topk_bbox_unact = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        content = (
            self.tgt_embed.weight.unsqueeze(0).repeat(memory.shape[0], 1, 1)
            if self.learn_query_content
            else enc_topk_memory.detach()
        )
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        return content, enc_topk_bbox_unact

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_coords_unact: torch.Tensor, topk: int):
        if self.query_select_method == "default":
            scores = outputs_logits.max(-1).values
        elif self.query_select_method == "agnostic":
            scores = outputs_logits.squeeze(-1)
        else:
            raise ValueError(f"Unknown query_select_method: {self.query_select_method}")

        _, ind_top = torch.topk(scores, topk, dim=-1)

        coords_top = outputs_coords_unact.gather(
            1, ind_top.unsqueeze(-1).repeat(1, 1, outputs_coords_unact.shape[-1])
        )
        logits_top = outputs_logits.gather(1, ind_top.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))
        memory_top = memory.gather(1, ind_top.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return memory_top, logits_top, coords_top

    def forward(self, feats: List[torch.Tensor], text_feats: torch.Tensor) -> dict:
        memory, spatial_shapes = self._get_encoder_input(feats)

        init_ref_contents, init_ref_points_unact = self._get_decoder_input(memory, spatial_shapes, text_feats)

        out_bboxes, out_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            text_feats,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            None,
            None,
        )

        return {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
