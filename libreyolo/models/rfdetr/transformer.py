"""Transformer (encoder + decoder) and multi-scale deformable attention for RF-DETR.

Ported from RF-DETR (https://github.com/roboflow/rf-detr).
Copyright (c) 2025 Roboflow, Inc. All Rights Reserved.
Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR).
Copyright (c) 2024 Baidu. All Rights Reserved.
Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR).
Copyright (c) 2021 Microsoft. All Rights Reserved.
Modified from DETR (https://github.com/facebookresearch/detr).
Copyright (c) Facebook, Inc. and its affiliates.
Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR).
Copyright (c) 2020 SenseTime. All Rights Reserved.
"""

import copy
import logging
import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.init import constant_, xavier_uniform_

from ...kernels.attention.ms_deform_attn import maybe_ms_deform_attn
from .keypoints import KEYPOINT_PRED_DIM, ConditionalQueryInitializer
from .tensors import _bilinear_grid_sample

logger = logging.getLogger(__name__)


def _safe_multinormalize(dim: int) -> int:
    """Clamp a MultiheadAttention head count to at least one.

    Ported from RF-DETR v1.8.0 (GroupPose keypoint additions).
    """
    return max(1, dim)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor, dim=128):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim = int(dim)
    dim_t = pos_tensor.new_ones((dim,), dtype=pos_tensor.dtype).cumsum(0) - 1
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def gen_encoder_output_proposals(memory, memory_padding_mask=None, spatial_shapes=None, unsigmoid=True):
    r"""
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    proposals = []
    _cur = 0
    for lvl, (height, width) in enumerate(spatial_shapes):
        if memory_padding_mask is not None:
            # reshape(-1, ...) infers batch dynamically in ONNX instead of constant N_
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + height * width)].reshape(-1, height, width, 1)

            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        else:
            # Derive batch-sized tensors from memory so ONNX traces them as symbolic
            # (torch.full((N_,), ...) bakes N_=8 as a constant; zeros_like is dynamic)
            valid_height = torch.zeros_like(memory[:, 0, 0]).long() + height
            valid_width = torch.zeros_like(memory[:, 0, 0]).long() + width

        grid_y = memory.new_ones((height, width), dtype=torch.float32).cumsum(0) - 1
        grid_x = memory.new_ones((height, width), dtype=torch.float32).cumsum(1) - 1
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # height, width, 2

        # reshape(-1, ...) and unsqueeze(0) broadcasting avoid hardcoding N_ in ONNX
        scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).reshape(-1, 1, 1, 2)
        grid = (grid.unsqueeze(0) + 0.5) / scale.float()  # [1, H_, W_, 2] / [N_, 1, 1, 2] → [N_, H_, W_, 2]

        wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)

        proposal = torch.cat((grid, wh), -1).reshape(-1, height * width, 4)  # -1 infers N_ dynamically
        proposals.append(proposal)
        _cur += height * width

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)

    if unsigmoid:
        output_proposals = torch.log(output_proposals / (1 - output_proposals))  # unsigmoid
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))
    else:
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))

    output_memory = memory
    if memory_padding_mask is not None:
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


# ---------------------------------------------------------------------------
# Multi-scale deformable attention (originally Deformable DETR / SenseTime).
# ---------------------------------------------------------------------------


def ms_deform_attn_core_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    value_spatial_shapes_hw: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Portable deformable-attention core, used on every device/backend.

    This is the upstream-parity ``grid_sample`` port and the only path
    exports ever see. The accelerated ``ms_deform_attn`` kernel slot is
    consulted in :meth:`MSDeformAttn.forward` *before* the layout transpose
    that produces this function's ``(bs, heads, c, Len_in)`` value tensor,
    so the fast path pays no extra copies.
    """
    batch_size, n_heads, head_dim, _ = value.shape
    _, len_query, n_heads, num_levels, num_points, _ = sampling_locations.shape
    # Use Python int pairs when available (required for torch.export compatibility,
    # since iterating over a tensor and using scalar elements as split/view sizes
    # fails during FakeTensor tracing).
    shapes = value_spatial_shapes_hw if value_spatial_shapes_hw is not None else value_spatial_shapes
    value_list = value.split([height * width for height, width in shapes], dim=3)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_index, (height, width) in enumerate(shapes):
        value_l_ = value_list[level_index].view(batch_size * n_heads, head_dim, height, width)
        sampling_grid_l_ = sampling_grids[:, :, :, level_index].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = _bilinear_grid_sample(value_l_, sampling_grid_l_, padding_mode="zeros", align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * n_heads, 1, len_query, num_levels * num_points
    )
    sampling_value_list = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (sampling_value_list * attention_weights).sum(-1).view(batch_size, n_heads * head_dim, len_query)
    return output.transpose(1, 2).contiguous()


def _is_power_of_2(n: int) -> bool:
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    """Multi-scale deformable attention (Deformable DETR)."""

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads, but got {} and {}".format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the"
                " dimension of each attention head a power of 2"
                " which is more efficient in our CUDA implementation."
            )

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

        self._export = False

    def export(self):
        self._export = True

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
        input_spatial_shapes_hw: list[tuple[int, int]] | None = None,
    ):
        """Forward pass of MSDeformAttn.

        Args:
            query: (N, Length_{query}, C)
            reference_points: (N, Length_{query}, n_levels, 2) with range in [0, 1],
                top-left (0,0), bottom-right (1, 1), including padding area; or
                (N, Length_{query}, n_levels, 4) adding additional (w, h) to form reference boxes.
            input_flatten: (N, sum_{l=0}^{L-1} H_l * W_l, C)
            input_spatial_shapes: (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            input_level_start_index: (n_levels,), [0, H_0*W_0, H_0*W_0+H_1*W_1, ...,
                H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
            input_padding_mask: (N, sum_{l=0}^{L-1} H_l * W_l), True for padding elements,
                False for non-padding elements.
            input_spatial_shapes_hw: List of (H, W) int pairs, same ordering as
                input_spatial_shapes. When provided, these Python ints are used for tensor
                split/view operations inside ms_deform_attn_core_pytorch so that the function
                is compatible with torch.export.export (FakeTensor tracing cannot extract
                concrete values from a tensor).

        Returns:
            Output tensor of shape (N, Length_{query}, C).
        """
        batch_size, len_query, _ = query.shape
        batch_size, len_input, _ = input_flatten.shape
        error_msg = "input_spatial_shapes must match the flattened input length"
        # torch.export captures torch._assert on a tensor comparison as an
        # aten.item call, which produces an unbacked symbol and makes the
        # graph unguardable ("Could not guard on data-dependent expression
        # Eq(u0, 1)"). The check is a developer sanity check over spatial
        # shapes that are constant for a fixed export canvas, so evaluate it
        # in Python when the shapes are concrete and skip the tensor path.
        if input_spatial_shapes_hw is not None and not isinstance(
            len_input, torch.Tensor
        ):
            # Callers that carry the geometry as Python integer pairs (the
            # RF-DETR decoder always does) validate against those values: it
            # avoids the unbacked symbol under strict torch.export capture,
            # and it avoids the device readback below, which is a full GPU
            # pipeline drain per decoder layer in eager training
            # (Tensor.__int__ syncs, invisibly to Python-level profiling).
            expected_len_in = sum(h * w for h, w in input_spatial_shapes_hw)
            assert expected_len_in == len_input, error_msg
        # The int() readback below is a host sync, which CUDA graph capture
        # forbids; the same values were already validated during the graph
        # warm-up iterations, so the check is skipped only while capturing.
        elif (
            not torch.jit.is_tracing()
            and not isinstance(input_spatial_shapes, torch.fx.Proxy)
            and not (
                input_spatial_shapes.is_cuda
                and torch.cuda.is_current_stream_capturing()
            )
        ):
            try:
                expected_len_in = int(
                    (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum()
                )
            except Exception:  # noqa: BLE001 - symbolic shapes under export
                expected_len_in = None
            if expected_len_in is not None and not isinstance(len_input, torch.Tensor):
                assert expected_len_in == len_input, error_msg

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, len_query, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            batch_size, len_query, self.n_heads, self.n_levels * self.n_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1])
            )
        attention_weights = F.softmax(attention_weights, -1)

        if not self._export:
            # Consult the accelerated slot with the classic Deformable-DETR
            # layout while ``value`` is still (bs, Len_in, d_model): one
            # zero-copy view away from the slot's (bs, Len_in, heads, c).
            # Export mode always takes the portable core below, whose
            # ``value_spatial_shapes_hw`` int pairs keep it traceable.
            accelerated = maybe_ms_deform_attn(
                value.view(
                    batch_size, len_input, self.n_heads, self.d_model // self.n_heads
                ),
                input_spatial_shapes,
                sampling_locations,
                attention_weights.unflatten(-1, (self.n_levels, self.n_points)),
            )
            if accelerated is not None:
                return self.output_proj(accelerated)

        value = (
            value.transpose(1, 2).contiguous().view(batch_size, self.n_heads, self.d_model // self.n_heads, len_input)
        )
        output = ms_deform_attn_core_pytorch(
            value,
            input_spatial_shapes,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=input_spatial_shapes_hw,
        )
        output = self.output_proj(output)
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        sa_nhead=8,
        ca_nhead=8,
        num_queries=300,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        group_detr=1,
        two_stage=False,
        num_feature_levels=4,
        dec_n_points=4,
        lite_refpoint_refine=False,
        decoder_norm_type="LN",
        bbox_reparam=False,
        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        # All keypoint flags default to off so the detection-only construction
        # and forward path remain byte-identical to the original.
        use_grouppose_keypoints=False,
        num_keypoints_per_class=None,
        grouppose_keypoint_dim_downscale=1,
        keypoint_cross_attn=True,
        inter_instance_kp_attn=False,
        num_registers=0,
        dual_projector_kp_only=False,
    ):
        super().__init__()
        self.encoder = None

        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        self.use_grouppose_keypoints = use_grouppose_keypoints
        self.dual_projector_kp_only = dual_projector_kp_only
        self.num_keypoints_per_class = num_keypoints_per_class or []
        self.num_registers = num_registers

        decoder_layer = TransformerDecoderLayer(
            d_model,
            sa_nhead,
            ca_nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            group_detr=group_detr,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points,
            skip_self_attn=False,
            enable_keypoint_processing=use_grouppose_keypoints,
            grouppose_keypoint_dim_downscale=grouppose_keypoint_dim_downscale,
            keypoint_cross_attn=keypoint_cross_attn,
            inter_instance_kp_attn=inter_instance_kp_attn,
        )
        assert decoder_norm_type in ["LN", "Identity"]
        norm = {
            "LN": lambda channels: nn.LayerNorm(channels),
            "Identity": lambda channels: nn.Identity(),
        }
        decoder_norm = norm[decoder_norm_type](d_model)

        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            lite_refpoint_refine=lite_refpoint_refine,
            bbox_reparam=bbox_reparam,
            enable_keypoint_processing=use_grouppose_keypoints,
            num_keypoints_per_class=self.num_keypoints_per_class,
            grouppose_keypoint_dim_downscale=grouppose_keypoint_dim_downscale,
        )

        self.two_stage = two_stage
        if two_stage:
            self.enc_output = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(group_detr)])
            self.enc_output_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(group_detr)])

            # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
            if use_grouppose_keypoints and self.num_keypoints_per_class:
                total_keypoints = sum(self.num_keypoints_per_class)
                if total_keypoints > 0:
                    keypoint_dim = d_model // grouppose_keypoint_dim_downscale
                    self.keypoint_query_initializer = ConditionalQueryInitializer(
                        d_model, total_keypoints, out_dim=keypoint_dim
                    )
                    self.keypoint_query_initializer_enc = ConditionalQueryInitializer(
                        d_model, total_keypoints, out_dim=keypoint_dim
                    )
                    # NOTE: the released rf-detr-keypoint-preview checkpoint stores a
                    # 3-layer MLP ending in KEYPOINT_PRED_DIM (=8) channels here -- the
                    # same module RF-DETR's LWDETR builds as ``keypoint_embed`` and
                    # deep-copies into ``enc_out_keypoint_embed`` (lwdetr.py). The stale
                    # ``MLP(keypoint_dim, d_model, keypoint_dim, 2)`` placeholder in the
                    # official transformer.py is overwritten before weights load, so we
                    # build the checkpoint-faithful shape directly for strict loading.
                    self.enc_out_keypoint_embed = nn.ModuleList(
                        [MLP(keypoint_dim, keypoint_dim, KEYPOINT_PRED_DIM, 3) for _ in range(group_detr)]
                    )

        self._reset_parameters()

        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        # Register tokens used by the GroupPose path (num_registers=0 -> no params).
        if num_registers > 0:
            self.register_tokens = nn.Parameter(torch.empty(num_registers, d_model).normal_())
            self.register_ref_points = nn.Parameter(torch.zeros(num_registers, 4))

        self.num_queries = num_queries
        self.d_model = d_model
        self.dec_layers = num_decoder_layers
        self.group_detr = group_detr
        self.num_feature_levels = num_feature_levels
        self.bbox_reparam = bbox_reparam

        self._export = False

    def export(self):
        self._export = True

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def get_valid_ratio(self, mask):
        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_height.float() / height
        valid_ratio_w = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def _cached_spatial_shapes(
        self, spatial_shapes_hw: "list[tuple[int, int]]", device
    ) -> torch.Tensor:
        """Constant (n_levels, 2) level-shape tensor, cached per shape set.

        Writing Python ints element-wise into a CUDA tensor issues one tiny
        unpinned host-to-device copy per level on every forward, which is
        wasted work eagerly and illegal inside CUDA graph capture. The
        values only depend on the input resolution, so build the tensor
        once per (shapes, device) and reuse it; the cached tensor is
        read-only downstream.

        Under ``torch.compile`` the tensor is built in-graph instead: the
        cache's attribute write and key comparison become Dynamo guards that
        pin every input resolution, so each multi-scale size would recompile.
        ``torch.full`` keeps symbolic sizes symbolic (``torch.tensor`` of
        symbolic ints specializes them). Export keeps the cached path.
        """
        if torch.compiler.is_compiling() and not getattr(
            torch.compiler, "is_exporting", lambda: False
        )():
            return torch.stack(
                [
                    torch.stack(
                        [
                            torch.full((), h, dtype=torch.long, device=device),
                            torch.full((), w, dtype=torch.long, device=device),
                        ]
                    )
                    for h, w in spatial_shapes_hw
                ]
            )
        key = (tuple(spatial_shapes_hw), str(device))
        cached = getattr(self, "_spatial_shapes_cache", None)
        if cached is None or cached[0] != key:
            tensor = torch.tensor(
                spatial_shapes_hw, device=device, dtype=torch.long
            )
            self._spatial_shapes_cache = (key, tensor)
        return self._spatial_shapes_cache[1]

    def forward(self, srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None):
        src_flatten = []
        mask_flatten = [] if masks is not None else None
        lvl_pos_embed_flatten = []
        # Under tracing, build spatial_shapes as a tensor directly so that
        # the ONNX tracer can track h/w symbolically instead of baking them
        # in as constants. Outside tracing the tensor is a per-resolution
        # constant and comes from _cached_spatial_shapes after the loop.
        tracing = torch.jit.is_tracing()
        if tracing:
            spatial_shapes = torch.empty((len(srcs), 2), device=srcs[0].device, dtype=torch.long)
        # Keep Python int pairs for gen_encoder_output_proposals — its loop uses h/w
        # as slice indices and linspace steps, which require Python ints, not tensors.
        spatial_shapes_hw: list[tuple[int, int]] = []
        valid_ratios = [] if masks is not None else None
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            _, c, h, w = src.shape
            if tracing:
                spatial_shapes[lvl, 0] = h
                spatial_shapes[lvl, 1] = w
            spatial_shapes_hw.append((h, w))

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed_flatten.append(pos_embed)
            src_flatten.append(src)
            if masks is not None:
                mask = masks[lvl].flatten(1)  # bs, hw
                mask_flatten.append(mask)
        if not tracing:
            spatial_shapes = self._cached_spatial_shapes(
                spatial_shapes_hw, srcs[0].device
            )
        memory = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        if masks is not None:
            mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        # Flatten optional dual-projector features for keypoint-specific cross-attention.
        cross_attn_memory = None
        if cross_attn_srcs is not None:
            ca_flatten = []
            for src in cross_attn_srcs:
                tensor = getattr(src, "tensors", src)
                ca_flatten.append(tensor.flatten(2).transpose(1, 2))
            cross_attn_memory = torch.cat(ca_flatten, 1)

        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes_hw, unsigmoid=not self.bbox_reparam
            )
            # group detr for first stage
            refpoint_embed_ts, memory_ts, boxes_ts = [], [], []
            group_detr = self.group_detr if self.training else 1
            for g_idx in range(group_detr):
                output_memory_gidx = self.enc_output_norm[g_idx](self.enc_output[g_idx](output_memory))

                enc_outputs_class_unselected_gidx = self.enc_out_class_embed[g_idx](output_memory_gidx)
                if self.bbox_reparam:
                    enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed[g_idx](output_memory_gidx)
                    enc_outputs_coord_cxcy_gidx = (
                        enc_outputs_coord_delta_gidx[..., :2] * output_proposals[..., 2:] + output_proposals[..., :2]
                    )
                    enc_outputs_coord_wh_gidx = enc_outputs_coord_delta_gidx[..., 2:].exp() * output_proposals[..., 2:]
                    enc_outputs_coord_unselected_gidx = torch.concat(
                        [enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx], dim=-1
                    )
                else:
                    enc_outputs_coord_unselected_gidx = (
                        self.enc_out_bbox_embed[g_idx](output_memory_gidx) + output_proposals
                    )  # (bs, \sum{hw}, 4) unsigmoid

                topk = min(self.num_queries, enc_outputs_class_unselected_gidx.shape[-2])
                topk_proposals_gidx = torch.topk(enc_outputs_class_unselected_gidx.max(-1)[0], topk, dim=1)[1]  # bs, nq

                refpoint_embed_gidx_undetach = torch.gather(
                    enc_outputs_coord_unselected_gidx, 1, topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, 4)
                )  # unsigmoid
                # for decoder layer, detached as initial ones, (bs, nq, 4)
                refpoint_embed_gidx = refpoint_embed_gidx_undetach.detach()

                # get memory tgt
                tgt_undetach_gidx = torch.gather(
                    output_memory_gidx, 1, topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, self.d_model)
                )

                refpoint_embed_ts.append(refpoint_embed_gidx)
                memory_ts.append(tgt_undetach_gidx)
                boxes_ts.append(refpoint_embed_gidx_undetach)
            # concat on dim=1, the nq dimension, (bs, nq, d) --> (bs, nq, d)
            refpoint_embed_ts = torch.cat(refpoint_embed_ts, dim=1)
            # (bs, nq, d)
            memory_ts = torch.cat(memory_ts, dim=1)  # .transpose(0, 1)
            boxes_ts = torch.cat(boxes_ts, dim=1)  # .transpose(0, 1)

        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        enc_kp_predictions = None
        init_kp_ref_xy = None
        keypoint_memory_ts = None
        if self.two_stage and self.use_grouppose_keypoints and hasattr(self, "keypoint_query_initializer"):
            keypoint_memory_ts = self.keypoint_query_initializer_enc(memory_ts)
            boxes_ref = boxes_ts if self.bbox_reparam else boxes_ts.sigmoid()
            group_detr = len(self.enc_out_keypoint_embed)

            kp_mem_chunks = keypoint_memory_ts.chunk(group_detr, dim=1)
            boxes_chunks = boxes_ref.chunk(group_detr, dim=1)
            kp_pred_chunks = []
            for g_idx in range(group_detr):
                kp_delta = self.enc_out_keypoint_embed[g_idx](kp_mem_chunks[g_idx])
                ref_wh = boxes_chunks[g_idx][..., 2:].unsqueeze(-2)
                ref_xy = boxes_chunks[g_idx][..., :2].unsqueeze(-2)
                kp_xy = kp_delta[..., :2] * ref_wh + ref_xy
                kp_pred_chunks.append(torch.cat([kp_xy, kp_delta[..., 2:]], dim=-1))

            enc_kp_predictions = torch.cat(kp_pred_chunks, dim=1)
            init_kp_ref_xy = enc_kp_predictions[..., :2].detach()

        if self.dec_layers > 0:
            # Use memory.shape[0] (traced as a symbolic Shape+Gather node in ONNX)
            # instead of the Python-int `bs` (which bakes batch=8 as a constant Tile op).
            # expand().contiguous() is functionally identical to repeat() but produces
            # a dynamic Expand op that TRT can handle with variable batch sizes.
            tgt = query_feat.unsqueeze(0).expand(memory.shape[0], -1, -1).contiguous()
            refpoint_embed = refpoint_embed.unsqueeze(0).expand(memory.shape[0], -1, -1).contiguous()
            if self.two_stage:
                ts_len = refpoint_embed_ts.shape[-2]
                refpoint_embed_ts_subset = refpoint_embed[..., :ts_len, :]
                refpoint_embed_subset = refpoint_embed[..., ts_len:, :]

                if self.bbox_reparam:
                    refpoint_embed_cxcy = refpoint_embed_ts_subset[..., :2] * refpoint_embed_ts[..., 2:]
                    refpoint_embed_cxcy = refpoint_embed_cxcy + refpoint_embed_ts[..., :2]
                    refpoint_embed_wh = refpoint_embed_ts_subset[..., 2:].exp() * refpoint_embed_ts[..., 2:]
                    refpoint_embed_ts_subset = torch.concat([refpoint_embed_cxcy, refpoint_embed_wh], dim=-1)
                else:
                    refpoint_embed_ts_subset = refpoint_embed_ts_subset + refpoint_embed_ts

                refpoint_embed = torch.concat([refpoint_embed_ts_subset, refpoint_embed_subset], dim=-2)

            # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
            # Insert register tokens per group (num_registers=0 -> no-op).
            original_num_queries_per_group = None
            if self.num_registers > 0:
                bs = memory.shape[0]
                group_count = self.group_detr if self.training else 1
                original_num_queries_per_group = tgt.shape[1] // group_count
                reg_tgt = self.register_tokens.unsqueeze(0).expand(bs, -1, -1)
                reg_ref = self.register_ref_points.unsqueeze(0).expand(bs, -1, -1)
                tgt_chunks = list(tgt.split(original_num_queries_per_group, dim=1))
                ref_chunks = list(refpoint_embed.split(original_num_queries_per_group, dim=1))
                tgt = torch.cat([torch.cat([chunk, reg_tgt], dim=1) for chunk in tgt_chunks], dim=1)
                refpoint_embed = torch.cat([torch.cat([chunk, reg_ref], dim=1) for chunk in ref_chunks], dim=1)
                if init_kp_ref_xy is not None:
                    num_keypoints = init_kp_ref_xy.shape[2]
                    reg_kp_xy = self.register_ref_points[:, :2].sigmoid()
                    reg_kp_xy = reg_kp_xy.unsqueeze(0).unsqueeze(2).expand(bs, -1, num_keypoints, -1)
                    kp_ref_chunks = list(init_kp_ref_xy.split(original_num_queries_per_group, dim=1))
                    init_kp_ref_xy = torch.cat([torch.cat([chunk, reg_kp_xy], dim=1) for chunk in kp_ref_chunks], dim=1)

            tgt_keypoints = None
            if self.use_grouppose_keypoints:
                if not hasattr(self, "keypoint_query_initializer"):
                    raise ValueError(
                        "use_grouppose_keypoints=True requires keypoint initializers "
                        "(ensure two_stage=True and num_keypoints_per_class is set)"
                    )
                tgt_keypoints = self.keypoint_query_initializer(tgt)

            # Route memories: kp_only mode keeps main features for detection and
            # second projector memory for keypoint cross-attention.
            if self.dual_projector_kp_only and cross_attn_memory is not None:
                decoder_memory = memory
                kp_cross_attn_memory = cross_attn_memory
            else:
                decoder_memory = cross_attn_memory if cross_attn_memory is not None else memory
                kp_cross_attn_memory = None

            decoder_outputs = self.decoder(
                tgt,
                decoder_memory,
                memory_key_padding_mask=mask_flatten,
                pos=lvl_pos_embed_flatten,
                refpoints_unsigmoid=refpoint_embed,
                level_start_index=level_start_index,
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios.to(decoder_memory.dtype) if valid_ratios is not None else valid_ratios,
                spatial_shapes_hw=spatial_shapes_hw,
                tgt_keypoints=tgt_keypoints,
                init_kp_ref_xy=init_kp_ref_xy,
                kp_cross_attn_memory=kp_cross_attn_memory,
            )

            if self.use_grouppose_keypoints and len(decoder_outputs) > 2:
                hs, references, keypoint_hs = decoder_outputs[:3]
            else:
                hs, references = decoder_outputs[:2]
                keypoint_hs = None

            # Remove register tokens from decoder outputs.
            if self.num_registers > 0 and original_num_queries_per_group is not None:
                group_count = self.group_detr if self.training else 1
                n_with_reg = hs.shape[2] // group_count
                hs = torch.cat(
                    [c[:, :, :original_num_queries_per_group, :] for c in hs.split(n_with_reg, dim=2)],
                    dim=2,
                )
                references = torch.cat(
                    [c[:, :, :original_num_queries_per_group, :] for c in references.split(n_with_reg, dim=2)],
                    dim=2,
                )
                if keypoint_hs is not None:
                    keypoint_hs = torch.cat(
                        [c[:, :, :original_num_queries_per_group] for c in keypoint_hs.split(n_with_reg, dim=2)],
                        dim=2,
                    )
        else:
            assert self.two_stage, "if not using decoder, two_stage must be True"
            hs = None
            references = None
            keypoint_hs = None

        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        # When keypoints are disabled the return tuple is unchanged from detection.
        if self.use_grouppose_keypoints:
            return_values = [hs, references]
            if self.two_stage:
                return_values.append(memory_ts)
                if self.bbox_reparam:
                    return_values.append(boxes_ts)
                else:
                    return_values.append(boxes_ts.sigmoid())
            else:
                return_values.extend([None, None])
            return_values.append(keypoint_hs)
            return_values.append(enc_kp_predictions)
            return_values.append(keypoint_memory_ts if self.two_stage else None)
            return tuple(return_values)

        if self.two_stage:
            if self.bbox_reparam:
                return hs, references, memory_ts, boxes_ts
            else:
                return hs, references, memory_ts, boxes_ts.sigmoid()
        return hs, references, None, None


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        lite_refpoint_refine=False,
        bbox_reparam=False,
        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        enable_keypoint_processing=False,
        num_keypoints_per_class=None,
        grouppose_keypoint_dim_downscale=1,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.lite_refpoint_refine = lite_refpoint_refine
        self.bbox_reparam = bbox_reparam

        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        self.enable_keypoint_processing = enable_keypoint_processing
        self.num_keypoints_per_class = num_keypoints_per_class
        self.grouppose_keypoint_dim_downscale = grouppose_keypoint_dim_downscale

        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)

        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        self.keypoint_pos_embed = None
        if enable_keypoint_processing and num_keypoints_per_class:
            kp_dim = d_model // grouppose_keypoint_dim_downscale
            self.keypoint_pos_embed = nn.Parameter(torch.randn(sum(num_keypoints_per_class), kp_dim))
            self._create_keypoint_class_mask()

        # Populated externally (e.g. by LWDETR) when iterative bbox refinement is active.
        # Declared here so that ``hasattr(self, "bbox_embed")``/``is not None`` short-circuits
        # even without an external injection (e.g. a standalone-constructed Transformer).
        self.bbox_embed = None

        self._export = False

    def export(self):
        self._export = True

    def _create_keypoint_class_mask(self) -> Tensor:
        """Create an attention mask that blocks cross-class keypoint interactions.

        Ported from RF-DETR v1.8.0 (GroupPose keypoint additions).
        """
        # Schema changes can rebuild this buffer after the decoder moved to GPU.
        # Preserve its device; on first creation use an existing decoder parameter.
        current_mask = self._buffers.get("keypoint_class_mask")
        device = (
            current_mask.device if current_mask is not None
            else self.ref_point_head.layers[0].weight.device
        )
        if not self.num_keypoints_per_class:
            mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
        else:
            total_kp = sum(self.num_keypoints_per_class)
            mask = torch.zeros(1 + total_kp, 1 + total_kp, dtype=torch.bool, device=device)
            offset = 1
            for class_idx_i, num_kp_i in enumerate(self.num_keypoints_per_class):
                if num_kp_i == 0:
                    continue
                start_i = offset + sum(self.num_keypoints_per_class[:class_idx_i])
                end_i = start_i + num_kp_i
                for class_idx_j, num_kp_j in enumerate(self.num_keypoints_per_class):
                    if num_kp_j == 0 or class_idx_i == class_idx_j:
                        continue
                    start_j = offset + sum(self.num_keypoints_per_class[:class_idx_j])
                    end_j = start_j + num_kp_j
                    mask[start_i:end_i, start_j:end_j] = True

        if "keypoint_class_mask" in self._buffers:
            self._buffers["keypoint_class_mask"] = mask
        else:
            self.register_buffer("keypoint_class_mask", mask, persistent=True)
        return self.keypoint_class_mask

    def refpoints_refine(self, refpoints_unsigmoid, new_refpoints_delta):
        if self.bbox_reparam:
            new_refpoints_cxcy = (
                new_refpoints_delta[..., :2] * refpoints_unsigmoid[..., 2:] + refpoints_unsigmoid[..., :2]
            )
            new_refpoints_wh = new_refpoints_delta[..., 2:].exp() * refpoints_unsigmoid[..., 2:]
            new_refpoints_unsigmoid = torch.concat([new_refpoints_cxcy, new_refpoints_wh], dim=-1)
        else:
            new_refpoints_unsigmoid = refpoints_unsigmoid + new_refpoints_delta
        return new_refpoints_unsigmoid

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None,
        # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
        spatial_shapes: Optional[Tensor] = None,  # num_levels, 2
        valid_ratios: Optional[Tensor] = None,
        spatial_shapes_hw: list[tuple[int, int]] | None = None,
        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        tgt_keypoints: Optional[Tensor] = None,
        init_kp_ref_xy: Optional[Tensor] = None,
        kp_cross_attn_memory: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []
        hs_refpoints_unsigmoid = [refpoints_unsigmoid]

        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        keypoint_tgt = None
        kp_query_pos = None
        intermediate_keypoints = []

        if self.enable_keypoint_processing:
            if not self.lite_refpoint_refine:
                raise ValueError("Keypoint processing requires lite_refpoint_refine=True")
            if tgt_keypoints is None:
                raise ValueError("Keypoint processing is enabled but tgt_keypoints was not provided")
            if init_kp_ref_xy is None:
                raise ValueError("Keypoint processing is enabled but init_kp_ref_xy was not provided")
            keypoint_tgt = tgt_keypoints
            assert self.keypoint_pos_embed is not None, (
                "keypoint_pos_embed must be initialized for keypoint processing"
            )
            kp_query_pos = (
                self.keypoint_pos_embed.unsqueeze(0)
                .unsqueeze(0)
                .expand(keypoint_tgt.shape[0], keypoint_tgt.shape[1], -1, -1)
            )

        def get_reference(refpoints):
            # [num_queries, batch_size, 4]
            obj_center = refpoints[..., :4]

            if self._export:
                query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model / 2)  # bs, nq, 256*2
                refpoints_input = obj_center[:, :, None]  # bs, nq, 1, 4
            else:
                refpoints_input = (
                    obj_center[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )  # bs, nq, nlevel, 4
                query_sine_embed = gen_sineembed_for_position(
                    refpoints_input[:, :, 0, :], self.d_model / 2
                )  # bs, nq, 256*2
            query_pos = self.ref_point_head(query_sine_embed)
            return obj_center, refpoints_input, query_pos, query_sine_embed

        # always use init refpoints
        if self.lite_refpoint_refine:
            if self.bbox_reparam:
                obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid)
            else:
                obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid())

        for layer_id, layer in enumerate(self.layers):
            # iter refine each layer
            if not self.lite_refpoint_refine:
                if self.bbox_reparam:
                    obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid)
                else:
                    obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(
                        refpoints_unsigmoid.sigmoid()
                    )

            # For the first decoder layer, we do not apply transformation over p_s
            pos_transformation = 1

            query_pos = query_pos * pos_transformation

            # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
            if self.enable_keypoint_processing and keypoint_tgt is not None:
                layer_outputs = layer(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    pos=pos,
                    query_pos=query_pos,
                    query_sine_embed=query_sine_embed,
                    is_first=(layer_id == 0),
                    reference_points=refpoints_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    spatial_shapes_hw=spatial_shapes_hw,
                    keypoint_tgt=keypoint_tgt,
                    keypoint_pos=kp_query_pos,
                    keypoint_class_mask=self.keypoint_class_mask,
                    kp_cross_attn_memory=kp_cross_attn_memory,
                )
                output, keypoint_tgt = layer_outputs
                intermediate_keypoints.append(keypoint_tgt)
            else:
                output = layer(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    pos=pos,
                    query_pos=query_pos,
                    query_sine_embed=query_sine_embed,
                    is_first=(layer_id == 0),
                    reference_points=refpoints_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    spatial_shapes_hw=spatial_shapes_hw,
                )

            if not self.lite_refpoint_refine:
                # box iterative update
                new_refpoints_delta = self.bbox_embed(output)
                new_refpoints_unsigmoid = self.refpoints_refine(refpoints_unsigmoid, new_refpoints_delta)
                if layer_id != self.num_layers - 1:
                    hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)
                refpoints_unsigmoid = new_refpoints_unsigmoid.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self._export:
                # to shape: B, N, C
                hs = intermediate[-1]
                if self.bbox_embed is not None:
                    ref = hs_refpoints_unsigmoid[-1]
                else:
                    ref = refpoints_unsigmoid
                # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
                if self.enable_keypoint_processing:
                    return hs, ref, intermediate_keypoints[-1]
                return hs, ref
            # box iterative update
            if self.bbox_embed is not None:
                results = [
                    torch.stack(intermediate),
                    torch.stack(hs_refpoints_unsigmoid),
                ]
            else:
                results = [torch.stack(intermediate), refpoints_unsigmoid.unsqueeze(0)]

            # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
            if self.enable_keypoint_processing:
                results.append(torch.stack(intermediate_keypoints))

            return results

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        sa_nhead,
        ca_nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        group_detr=1,
        num_feature_levels=4,
        dec_n_points=4,
        skip_self_attn=False,
        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        enable_keypoint_processing=False,
        grouppose_keypoint_dim_downscale=1,
        keypoint_cross_attn=True,
        inter_instance_kp_attn=False,
    ):
        super().__init__()
        # Decoder Self-Attention
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=sa_nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Decoder Cross-Attention
        self.cross_attn = MSDeformAttn(d_model, n_levels=num_feature_levels, n_heads=ca_nhead, n_points=dec_n_points)

        self.nhead = ca_nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.group_detr = group_detr

        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        self.enable_keypoint_processing = enable_keypoint_processing
        self.inter_instance_kp_attn = inter_instance_kp_attn and enable_keypoint_processing
        self.keypoint_cross_attn = keypoint_cross_attn and enable_keypoint_processing

        if enable_keypoint_processing:
            kp_dim = d_model // grouppose_keypoint_dim_downscale
            # When downscale == 1 these projections are parameter-free Identity ops,
            # so the official checkpoint contains no weights for them.
            self.inst_in_proj = nn.Linear(d_model, kp_dim) if grouppose_keypoint_dim_downscale > 1 else nn.Identity()
            self.inst_pos_in_proj = (
                nn.Linear(d_model, kp_dim) if grouppose_keypoint_dim_downscale > 1 else nn.Identity()
            )
            self.inst_out_proj = nn.Linear(kp_dim, d_model) if grouppose_keypoint_dim_downscale > 1 else nn.Identity()
            self.memory_in_proj = nn.Linear(d_model, kp_dim) if grouppose_keypoint_dim_downscale > 1 else nn.Identity()
            self.kp_inst_self_attn = nn.MultiheadAttention(
                embed_dim=kp_dim,
                num_heads=_safe_multinormalize(sa_nhead // grouppose_keypoint_dim_downscale),
                dropout=dropout,
                batch_first=True,
            )
            self.kp_inst_dropout = nn.Dropout(dropout)
            self.kp_inst_norm = nn.LayerNorm(d_model)
            self.kp_norm = nn.LayerNorm(kp_dim)
            self.kp_dropout = nn.Dropout(dropout)

            if self.inter_instance_kp_attn:
                self.inter_inst_kp_attn = nn.MultiheadAttention(
                    embed_dim=kp_dim,
                    num_heads=_safe_multinormalize(ca_nhead // grouppose_keypoint_dim_downscale),
                    dropout=dropout,
                    batch_first=True,
                )
                self.inter_inst_kp_dropout = nn.Dropout(dropout)
                self.inter_inst_kp_norm = nn.LayerNorm(kp_dim)

            if self.keypoint_cross_attn:
                self.kp_cross_attn = MSDeformAttn(
                    kp_dim,
                    n_levels=num_feature_levels,
                    n_heads=_safe_multinormalize(ca_nhead // grouppose_keypoint_dim_downscale),
                    n_points=dec_n_points,
                )
                self.kp_cross_attn_dropout = nn.Dropout(dropout)
                self.kp_cross_attn_norm = nn.LayerNorm(kp_dim)

            self.kp_linear1 = nn.Linear(kp_dim, d_model * 4 // grouppose_keypoint_dim_downscale)
            self.kp_dropout2 = nn.Dropout(dropout)
            self.kp_linear3 = nn.Linear(d_model * 4 // grouppose_keypoint_dim_downscale, kp_dim)
            self.kp_dropout4 = nn.Dropout(dropout)
            self.kp_norm5 = nn.LayerNorm(kp_dim)

            self.instance_kp_layer_scale = nn.Parameter(torch.ones(1) * 1e-6)

        self._export = False

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        query_sine_embed=None,
        is_first=False,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        spatial_shapes_hw: list[tuple[int, int]] | None = None,
        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        keypoint_tgt: Optional[Tensor] = None,  # [B, N, total_kp_per_instance, C]
        keypoint_pos: Optional[Tensor] = None,  # [B, N, total_kp_per_instance, C]
        keypoint_class_mask: Optional[Tensor] = None,  # [1 + K, 1 + K]
        kp_cross_attn_memory: Optional[Tensor] = None,
    ):
        bs, num_queries, _ = tgt.shape

        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: batch_size x num_queries x 256
        q = k = tgt + query_pos
        v = tgt
        if self.training:
            q = torch.cat(q.split(num_queries // self.group_detr, dim=1), dim=0)
            k = torch.cat(k.split(num_queries // self.group_detr, dim=1), dim=0)
            v = torch.cat(v.split(num_queries // self.group_detr, dim=1), dim=0)

        tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False)[0]

        if self.training:
            tgt2 = torch.cat(tgt2.split(bs, dim=0), dim=1)
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            memory_key_padding_mask,
            input_spatial_shapes_hw=spatial_shapes_hw,
        )
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        if self.enable_keypoint_processing:
            if keypoint_tgt is None or keypoint_pos is None:
                raise ValueError("Keypoint processing is enabled but keypoint_tgt/keypoint_pos missing")

            tgt_for_kp = self.inst_in_proj(tgt)
            tgt_for_kp_pos = self.inst_pos_in_proj(query_pos)

            # ========== Begin of Keypoint-Instance Self-Attention =============
            _, _n_queries, num_kp, kp_dim = keypoint_tgt.shape

            tgt_expanded = tgt_for_kp.unsqueeze(2)  # [B, N, 1, C]
            query_expanded = torch.zeros_like(tgt_for_kp).unsqueeze(2)  # [B, N, 1, C]

            combined_feat = torch.cat([tgt_expanded, keypoint_tgt], dim=2)  # [B, N, 1 + K, C]
            combined_pos = torch.cat([query_expanded, keypoint_pos], dim=2)  # [B, N, 1 + K, C]

            combined_feat = combined_feat.reshape(bs * num_queries, 1 + num_kp, kp_dim)
            combined_pos = combined_pos.reshape(bs * num_queries, 1 + num_kp, kp_dim)
            q = k = combined_feat + combined_pos
            v = combined_feat

            combined_out = self.kp_inst_self_attn(q, k, v, attn_mask=keypoint_class_mask, need_weights=False)[0]
            combined_out = combined_out.reshape(bs, num_queries, 1 + num_kp, kp_dim)
            tgt2 = combined_out[:, :, 0, :]
            keypoint_tgt2 = combined_out[:, :, 1:, :]

            tgt = tgt + self.kp_inst_dropout(self.inst_out_proj(tgt2)) * self.instance_kp_layer_scale
            tgt = self.kp_inst_norm(tgt)
            keypoint_tgt = keypoint_tgt + self.kp_dropout(keypoint_tgt2)
            keypoint_tgt = self.kp_norm(keypoint_tgt)
            # ========== End of Keypoint-Instance Self-Attention =============

            # ========== Begin of Cross-Keypoint Attention =============
            if self.inter_instance_kp_attn:
                swapped_keypoint_tgt = keypoint_tgt.transpose(1, 2).reshape(bs * num_kp, num_queries, kp_dim)
                swapped_keypoint_pos = (
                    tgt_for_kp_pos.unsqueeze(1)
                    .expand(bs, num_kp, num_queries, kp_dim)
                    .reshape(
                        bs * num_kp,
                        num_queries,
                        kp_dim,
                    )
                )
                q = swapped_keypoint_tgt + swapped_keypoint_pos
                v = swapped_keypoint_tgt
                swapped_out = self.inter_inst_kp_attn(q, key=q, value=v, need_weights=False)[0]
                swapped_out = swapped_out.view(bs, num_kp, num_queries, kp_dim).transpose(1, 2)
                keypoint_tgt = keypoint_tgt + self.inter_inst_kp_dropout(swapped_out)
                keypoint_tgt = self.inter_inst_kp_norm(keypoint_tgt)
            # ========== End of Cross-Keypoint Attention =============

            # ========== Begin of Keypoint-Specific Cross-Attention =============
            if self.keypoint_cross_attn:
                keypoint_query = self.with_pos_embed(
                    keypoint_tgt, tgt_for_kp_pos.unsqueeze(2).expand(bs, num_queries, num_kp, kp_dim)
                )
                keypoint_query = keypoint_query.reshape(bs, num_queries * num_kp, kp_dim)
                bbox_ref_for_kp = (
                    reference_points.unsqueeze(2)
                    .expand(
                        bs,
                        num_queries,
                        num_kp,
                        reference_points.shape[2],
                        reference_points.shape[3],
                    )
                    .reshape(bs, num_queries * num_kp, reference_points.shape[2], reference_points.shape[3])
                )
                kp_memory = kp_cross_attn_memory if kp_cross_attn_memory is not None else memory
                keypoint_tgt = keypoint_tgt + self.kp_cross_attn_dropout(
                    self.kp_cross_attn(
                        keypoint_query,
                        bbox_ref_for_kp,
                        self.memory_in_proj(kp_memory),
                        spatial_shapes,
                        level_start_index,
                        memory_key_padding_mask,
                        input_spatial_shapes_hw=spatial_shapes_hw,
                    ).reshape(bs, num_queries, num_kp, kp_dim)
                )
                keypoint_tgt = self.kp_cross_attn_norm(keypoint_tgt)
            # ========== End of Keypoint-Specific Cross-Attention =============

            # ========== Begin of Keypoint-Specific FFN =============
            keypoint_tgt = keypoint_tgt + self.kp_dropout4(
                self.kp_linear3(self.kp_dropout2(self.activation(self.kp_linear1(keypoint_tgt))))
            )
            keypoint_tgt = self.kp_norm5(keypoint_tgt)
            # ========== End of Keypoint-Specific FFN =============

            return tgt, keypoint_tgt

        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        query_sine_embed=None,
        is_first=False,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        spatial_shapes_hw: list[tuple[int, int]] | None = None,
        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        keypoint_tgt: Optional[Tensor] = None,
        keypoint_pos: Optional[Tensor] = None,
        keypoint_class_mask: Optional[Tensor] = None,
        kp_cross_attn_memory: Optional[Tensor] = None,
    ):
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            query_sine_embed,
            is_first,
            reference_points,
            spatial_shapes,
            level_start_index,
            spatial_shapes_hw=spatial_shapes_hw,
            keypoint_tgt=keypoint_tgt,
            keypoint_pos=keypoint_pos,
            keypoint_class_mask=keypoint_class_mask,
            kp_cross_attn_memory=kp_cross_attn_memory,
        )


def _get_clones(module, num_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(num_layers)])


def build_transformer(args):

    two_stage = getattr(args, "two_stage", False)

    return Transformer(
        d_model=args.hidden_dim,
        sa_nhead=args.sa_nheads,
        ca_nhead=args.ca_nheads,
        num_queries=args.num_queries,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
        group_detr=args.group_detr,
        two_stage=two_stage,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        lite_refpoint_refine=args.lite_refpoint_refine,
        decoder_norm_type=args.decoder_norm,
        bbox_reparam=args.bbox_reparam,
        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        # Detection-only builder args may omit keypoint-only fields; default to the
        # non-keypoint path so existing detection builds are unchanged.
        use_grouppose_keypoints=getattr(args, "use_grouppose_keypoints", False),
        num_keypoints_per_class=getattr(args, "num_keypoints_per_class", []),
        grouppose_keypoint_dim_downscale=getattr(args, "grouppose_keypoint_dim_downscale", 1),
        keypoint_cross_attn=getattr(args, "keypoint_cross_attn", True),
        inter_instance_kp_attn=getattr(args, "inter_instance_kp_attn", False),
        num_registers=getattr(args, "num_decoder_registers", 0),
        dual_projector_kp_only=getattr(args, "dual_projector_kp_only", False),
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
