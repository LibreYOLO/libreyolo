"""RiO-DETRv2 decoder: boundary and angle distribution refinement for OBBs.

Adapted from ``RicePasteM/RiO-DETRv2`` at commit
``ab8b22df5d800a927774c8c67e77fd29c0fe6967`` (Apache-2.0), specifically
``engine/rtv4/rtdetrv2_obb_decoder.py`` (``RTDETRTransformerv2OBBFDR``,
``TransformerDecoderOBBFDR``, ``MSDeformableAttention``, ``AngleLQE``,
``QueryPosHeadRiO``), ``engine/rtv4/dfine_utils.py`` (angle decoding and
``distance2obb_local``) and ``engine/rtv4/utils.py``
(``coordinate_to_encoding_4``).  See ``NOTICE`` in this directory.

Only the pieces RiO-DETRv2 adds are defined here.  Input projection, anchor
generation, query selection and the five-coordinate encoder head are inherited
from LibreYOLO's RT-DETRv2 OBB decoder; the FDR weighting function, integral
and box-quality head are LibreYOLO's D-FINE ports.  Inference only: the
denoising and loss outputs of the upstream training path are not implemented.
"""

from __future__ import annotations

import copy
import functools
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from ..dfine.decoder import LQE
from ..dfine.fdr import Integral, weighting_function
from ..rtdetrv2.decoder import MLP, MSDeformableAttention, TransformerDecoderLayer
from ..rtdetrv2.obb_decoder import RTDETRTransformerv2OBB
from ..rtdetrv2.utils import (
    bias_init_with_prob,
    deformable_attention_core_func_v2,
    inverse_sigmoid,
)

__all__ = ["RiODETRv2Transformer"]


def coordinate_to_encoding_4(
    coord_tensor: torch.Tensor,
    num_feats: int = 128,
    temperature: int = 10000,
    scale: float = 2 * math.pi,
) -> torch.Tensor:
    """Sine embedding of ``(cx, cy, w, h)``, concatenated as ``(y, x, w, h)``."""
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=coord_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)

    def _embed(values: torch.Tensor) -> torch.Tensor:
        pos = (values * scale)[..., None] / dim_t
        return torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1).flatten(2)

    pos_x = _embed(coord_tensor[..., 0])
    pos_y = _embed(coord_tensor[..., 1])
    pos_w = _embed(coord_tensor[..., 2])
    pos_h = _embed(coord_tensor[..., 3])
    return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)


def wrap_angle_01(angle: torch.Tensor) -> torch.Tensor:
    """Wrap normalized angles (``angle / pi``) to ``[0, 1)``."""
    wrapped = torch.fmod(angle, 1.0)
    return torch.where(wrapped < 0, wrapped + 1.0, wrapped)


def decode_circular_angle_delta(logits: torch.Tensor) -> torch.Tensor:
    """Circular mean of uniform angle bins -> shortest residual in ``[-0.5, 0.5]``."""
    num_bins = logits.shape[-1]
    probs = F.softmax(logits, dim=-1)
    bin_ids = torch.arange(num_bins, device=logits.device, dtype=logits.dtype)
    phase = bin_ids * (2.0 * math.pi / num_bins)
    sin_mean = (probs * torch.sin(phase)).sum(dim=-1, keepdim=True)
    cos_mean = (probs * torch.cos(phase)).sum(dim=-1, keepdim=True)
    return torch.atan2(sin_mean, cos_mean) / (2.0 * math.pi)


def decode_linear_angle_delta(logits: torch.Tensor, max_delta: float) -> torch.Tensor:
    """Expected value of bins spread linearly over ``[-max_delta, max_delta]``."""
    if max_delta <= 0:
        raise ValueError(f"max_delta must be positive, got {max_delta}")
    values = torch.linspace(
        -max_delta, max_delta, logits.shape[-1], dtype=logits.dtype, device=logits.device
    )
    return (F.softmax(logits, dim=-1) * values).sum(dim=-1, keepdim=True)


def _rotate_points(points: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    x = points[..., 0]
    y = points[..., 1]
    return torch.stack([x * cos_a - y * sin_a, x * sin_a + y * cos_a], dim=-1)


def distance2obb_local(
    ref_obbs: torch.Tensor,
    distance: torch.Tensor,
    target_angle: torch.Tensor,
    reg_scale: torch.Tensor,
) -> torch.Tensor:
    """Decode four edge distances, measured in the predicted box's frame.

    The reference box provides the center anchor and the width/height
    normalization; the distances are rotated by the predicted angle.
    Returns normalized ``(cx, cy, w, h)``.
    """
    reg_scale = abs(reg_scale)
    ref_xy = ref_obbs[..., :2]
    ref_wh = ref_obbs[..., 2:4].clamp(min=1e-16)
    target_angle = target_angle[..., 0] * math.pi

    left = (0.5 * reg_scale + distance[..., 0]) * (ref_wh[..., 0] / reg_scale)
    top = (0.5 * reg_scale + distance[..., 1]) * (ref_wh[..., 1] / reg_scale)
    right = (0.5 * reg_scale + distance[..., 2]) * (ref_wh[..., 0] / reg_scale)
    bottom = (0.5 * reg_scale + distance[..., 3]) * (ref_wh[..., 1] / reg_scale)

    local_center = torch.stack([(right - left) * 0.5, (bottom - top) * 0.5], dim=-1)
    global_center = ref_xy + _rotate_points(local_center, target_angle)
    wh = torch.stack([left + right, top + bottom], dim=-1)
    return torch.cat([global_center, wh], dim=-1)


class AngleLQE(nn.Module):
    """Score refinement from per-layer angle entropy and box anisotropy."""

    def __init__(self, num_angle_layers: int, hidden_dim: int = 16, num_mlp_layers: int = 2, act="relu"):
        super().__init__()
        self.num_angle_layers = num_angle_layers
        self.angle_conf = MLP(num_angle_layers + 1, hidden_dim, 1, num_mlp_layers, act=act)
        init.constant_(self.angle_conf.layers[-1].bias, 0)
        init.constant_(self.angle_conf.layers[-1].weight, 0)

    def forward(self, scores, angle_entropies, pred_boxes):
        if len(angle_entropies) != self.num_angle_layers:
            raise ValueError(
                f"Expected {self.num_angle_layers} angle entropy tensors, got {len(angle_entropies)}"
            )
        wh = pred_boxes[..., 2:4].detach().abs().clamp_min(1e-6)
        anisotropy = 1.0 - wh.min(dim=-1, keepdim=True).values / wh.max(dim=-1, keepdim=True).values
        stat = torch.cat([*(entropy.detach() for entropy in angle_entropies), anisotropy], dim=-1)
        return scores + self.angle_conf(stat)


class QueryPosHeadRiO(nn.Module):
    """Query positional embedding from a sine encoding of ``(cx, cy, w, h)``."""

    def __init__(self, hidden_dim: int, num_feats: int, act="relu"):
        super().__init__()
        self.num_feats = num_feats
        self.ref_point_head = MLP(hidden_dim * 2, hidden_dim, hidden_dim, 2, act=act)

    def forward(self, reference_points: torch.Tensor) -> torch.Tensor:
        embed = coordinate_to_encoding_4(reference_points[..., :4], num_feats=self.num_feats)
        return self.ref_point_head(embed)


class RiOMSDeformableAttention(MSDeformableAttention):
    """Rotated deformable attention; the second half of the heads samples at +90 degrees."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The released checkpoints use the upstream pure-PyTorch sampling
        # equation.  LibreYOLO's optional fused kernel reduces in a different
        # order, so keep the reference path for exact raw-output parity.
        self.ms_deformable_attn_core = functools.partial(
            deformable_attention_core_func_v2,
            method=self.method,
            allow_acceleration=False,
        )

    def forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        if reference_points.shape[-1] != 5:
            raise ValueError(
                "RiO-DETRv2 reference_points must hold (cx, cy, w, h, angle/pi), "
                f"got last dimension {reference_points.shape[-1]}"
            )
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        num_points = sum(self.num_points_list)

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(bs, len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(bs, len_q, self.num_heads, num_points, 2)
        attention_weights = F.softmax(
            self.attention_weights(query).reshape(bs, len_q, self.num_heads, num_points), dim=-1
        )

        references = reference_points.expand(-1, -1, self.num_levels, -1)
        expanded = []
        for level, n in enumerate(self.num_points_list):
            ref = references[:, :, level : level + 1, :]
            expanded.append(ref.unsqueeze(2).repeat(1, 1, 1, n, 1))
        expanded = torch.cat(expanded, dim=3)

        point_scale = self.num_points_scale.to(device=query.device, dtype=query.dtype).view(1, 1, 1, -1, 1)
        angle = (expanded[..., 4:] * math.pi).expand(-1, -1, self.num_heads, -1, -1)
        head_angle_offset = torch.zeros(self.num_heads, device=angle.device, dtype=angle.dtype)
        head_angle_offset[self.num_heads // 2 :] = math.pi / 2
        angle = angle + head_angle_offset.view(1, 1, self.num_heads, 1, 1)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        rotation = torch.cat([cos_a, -sin_a, sin_a, cos_a], dim=-1)
        rotation = rotation.reshape(*rotation.shape[:-1], 2, 2)
        wh = (expanded[..., 2:4] * self.offset_scale).expand(-1, -1, self.num_heads, -1, -1)
        rotated = torch.einsum("...ij,...j->...i", rotation, wh)
        sampling_locations = expanded[..., :2] + sampling_offsets * point_scale * rotated

        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list
        )
        return self.output_proj(output)


class RiOTransformerDecoderLayer(TransformerDecoderLayer):
    """RT-DETRv2 decoder layer with the RiO rotated cross-attention."""

    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, dropout=0.0,
                 activation="relu", n_levels=4, n_points=4, cross_attn_method="default"):
        super().__init__(d_model, n_head, dim_feedforward, dropout, activation,
                         n_levels, n_points, cross_attn_method=cross_attn_method)
        self.cross_attn = RiOMSDeformableAttention(
            d_model, n_head, n_levels, n_points, method=cross_attn_method
        )


class RiOFDRDecoder(nn.Module):
    """Decoder stack with four-side boundary and coarse-to-fine angle refinement."""

    def __init__(self, hidden_dim, decoder_layer, num_layers, reg_max, reg_scale, up,
                 angle_bins, eval_idx=-1, angle_fdr_scale_factor=1.5, act="relu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.reg_max = reg_max
        # Registered here as well as on the parent so the upstream state-dict
        # keys ``decoder.decoder.reg_scale`` / ``decoder.decoder.up`` load.
        self.reg_scale = reg_scale
        self.up = up
        self.angle_bins = angle_bins
        # coarse_to_fine: layer 0 decodes a full circular residual, later
        # layers a bounded linear residual that shrinks by ``scale_factor``.
        if angle_fdr_scale_factor < 1.0:
            raise ValueError("angle_fdr_scale_factor must be at least 1.0")
        self.angle_max_residuals = tuple(0.5 / (angle_fdr_scale_factor ** i) for i in range(num_layers))
        self.angle_circular_layers = (True,) + (False,) * (num_layers - 1)
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.lqe_layers = nn.ModuleList([LQE(4, 64, 2, reg_max, act=act) for _ in range(num_layers)])
        self.angle_lqe = AngleLQE(self.eval_idx + 1, act=act)

    def _update_angle(self, angle_logits, ref_angle, layer_idx):
        if self.angle_circular_layers[layer_idx]:
            delta = decode_circular_angle_delta(angle_logits)
        else:
            delta = decode_linear_angle_delta(angle_logits, self.angle_max_residuals[layer_idx])
        return wrap_angle_01(ref_angle + delta)

    def _normalized_angle_entropy(self, angle_logits):
        log_prob = F.log_softmax(angle_logits.detach(), dim=-1)
        entropy = -(log_prob.exp() * log_prob).sum(dim=-1, keepdim=True)
        return entropy / math.log(self.angle_bins)

    @staticmethod
    def _sanitize_xywh(boxes):
        return boxes.clamp(min=1e-6, max=1.0 - 1e-6)

    def forward(self, target, ref_points_unact, memory, memory_spatial_shapes, bbox_head,
                angle_head, score_head, query_pos_head, pre_bbox_head, integral, up,
                reg_scale, attn_mask=None, memory_mask=None):
        output = target
        output_detach = pred_corners_undetach = 0
        ref_points_detach = F.sigmoid(ref_points_unact)
        project = weighting_function(self.reg_max, up, reg_scale)
        ref_obb_initial = None
        angle_entropies = []

        for i, layer in enumerate(self.layers):
            query_pos_embed = query_pos_head(ref_points_detach)
            output = layer(output, ref_points_detach.unsqueeze(2), memory, memory_spatial_shapes,
                           attn_mask, memory_mask, query_pos_embed)

            bbox_logits = bbox_head[i](output + output_detach)
            angle_logits = angle_head[i](output)
            if i <= self.eval_idx:
                angle_entropies.append(self._normalized_angle_entropy(angle_logits))

            if i == 0:
                # The first-layer prediction is the fixed reference frame for
                # every later layer's edge distances.
                pre_xywh = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach[..., :4]))
                pre_angle = self._update_angle(angle_logits, ref_points_detach[..., 4:5], i)
                ref_obb_initial = torch.cat([self._sanitize_xywh(pre_xywh), pre_angle], dim=-1).detach()

            inter_angle = self._update_angle(angle_logits, ref_points_detach[..., 4:5], i)
            pred_corners = bbox_logits + pred_corners_undetach
            inter_xywh = distance2obb_local(
                ref_obb_initial, integral(pred_corners, project), inter_angle, reg_scale
            )
            inter_ref_bbox = torch.cat([inter_xywh, inter_angle], dim=-1)

            if i == self.eval_idx:
                scores = self.lqe_layers[i](score_head[i](output), pred_corners)
                scores = self.angle_lqe(scores, angle_entropies, inter_ref_bbox)
                return inter_ref_bbox, scores

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        raise RuntimeError("eval_idx is outside the decoder stack")


class RiODETRv2Transformer(RTDETRTransformerv2OBB):
    """RiO-DETRv2 decoder head emitting normalized ``(cx, cy, w, h, angle/pi)``.

    Configuration fixed to the released checkpoints: ``query_pos_method='rio'``,
    plain RT-DETRv2 decoder attention (``use_dfine_decoder_attention=False``),
    separate box and angle heads, ``angle_fdr_mode='coarse_to_fine'`` and the
    angle-aware quality head.
    """

    def __init__(self, *args, reg_max: int = 32, reg_scale: float = 4.0, angle_bins: int = 36,
                 angle_fdr_scale_factor: float = 1.5, **kwargs):
        kwargs.setdefault("learn_query_content", True)
        super().__init__(*args, **kwargs)
        num_points = kwargs.get("num_points", 4)
        hidden_dim = self.hidden_dim

        self.reg_max = reg_max
        self.angle_bins = angle_bins
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)

        decoder_layer = RiOTransformerDecoderLayer(
            hidden_dim,
            self.nhead,
            kwargs.get("dim_feedforward", 1024),
            kwargs.get("dropout", 0.0),
            kwargs.get("activation", "relu"),
            self.num_levels,
            num_points,
            cross_attn_method=self.cross_attn_method,
        )
        self.decoder = RiOFDRDecoder(
            hidden_dim,
            decoder_layer,
            self.num_layers,
            reg_max,
            self.reg_scale,
            self.up,
            angle_bins,
            eval_idx=kwargs.get("eval_idx", -1),
            angle_fdr_scale_factor=angle_fdr_scale_factor,
        )
        self.eval_idx = self.decoder.eval_idx

        self.query_pos_head = QueryPosHeadRiO(hidden_dim, hidden_dim // 2)
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4 * (reg_max + 1), 3) for _ in range(self.num_layers)]
        )
        self.dec_angle_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, angle_bins, 3) for _ in range(self.num_layers)]
        )
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.integral = Integral(reg_max)
        self._reset_rio_parameters()

    def _reset_rio_parameters(self) -> None:
        bias = bias_init_with_prob(0.01)
        for cls_, reg, ang in zip(self.dec_score_head, self.dec_bbox_head, self.dec_angle_head):
            init.constant_(cls_.bias, bias)
            for head in (reg, ang):
                init.constant_(head.layers[-1].weight, 0)
                init.constant_(head.layers[-1].bias, 0)
        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)
        init.xavier_uniform_(self.query_pos_head.ref_point_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.ref_point_head.layers[1].weight)

    def set_eval_spatial_size(self, eval_spatial_size) -> None:
        """Re-target the fixed-shape anchor buffers to another native canvas.

        Released checkpoints carry ``anchors`` / ``valid_mask`` buffers saved
        at their training canvas (1024 for DOTA, 800 for DIOR-R).  The saved
        anchors differ from freshly generated ones by up to ~6e-4, so the
        buffers are resized here and then overwritten by ``load_state_dict``.
        """
        self.eval_spatial_size = tuple(eval_spatial_size)
        anchors, valid_mask = self._generate_anchors()
        self.register_buffer("anchors", anchors.to(self.anchors.device))
        self.register_buffer("valid_mask", valid_mask.to(self.valid_mask.device))
        self._eval_spatial_shape_key = self._spatial_shape_key(
            [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
             for s in self.feat_strides]
        )
        self._anchor_cache.clear()

    def _get_anchors_for_spatial_shapes(self, spatial_shapes, memory):
        if torch.jit.is_tracing() and self._spatial_shape_key(spatial_shapes) != self._eval_spatial_shape_key:
            # Fixed-shape export at a non-native canvas: generated anchors are
            # traced as constants for that canvas.
            return self._generate_anchors(spatial_shapes, dtype=memory.dtype, device=memory.device)
        return super()._get_anchors_for_spatial_shapes(spatial_shapes, memory)

    def forward(self, feats, targets=None):
        if self.training and targets is not None:
            raise NotImplementedError("RiO-DETRv2 training is not implemented in LibreYOLO")
        memory, spatial_shapes = self._get_encoder_input(feats)
        content, ref_points_unact, _, _ = self._get_decoder_input(memory, spatial_shapes)
        boxes, logits = self.decoder(
            content,
            ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_angle_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
        )
        return {"pred_logits": logits, "pred_boxes": boxes}
