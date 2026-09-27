"""Triton multi-scale deformable attention (``ms_deform_attn`` / ``triton``).

Implements the published Deformable DETR sampling equation (Zhu et al.,
ICLR 2021): for each query, head, level and point, bilinear-sample the
value map at ``sampling_locations`` (``align_corners=False``, zeros
padding) and reduce with ``attention_weights``. Written from that
equation; not derived from any compiled CUDA or third-party Triton kernel.

Inference only. Inputs that require grad return ``None`` so the caller
keeps its portable ``grid_sample`` path (and the Hub kernel, when that
provider is the one ``resolve`` selected). Accumulates in fp32 so fp16
and bf16 match the portable core to the usual half-precision tolerance.

Eligible inputs are CUDA fp32 / fp16 / bf16. The provider is on whenever
Triton imports and CUDA is visible; ``LIBREYOLO_TRITON_MSDA=0`` disables
it. Hub stays preferred: this module registers first, the Hub provider
registers on top.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Optional

import torch

from .. import clear_cache, register

logger = logging.getLogger(__name__)


_SLOT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_triton_failed = False


def _env_enabled() -> bool:
    return os.environ.get("LIBREYOLO_TRITON_MSDA", "").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def _eligible() -> bool:
    return (
        _env_enabled()
        and not _triton_failed
        and importlib.util.find_spec("triton") is not None
        and torch.cuda.is_available()
    )


def _disable(exc: BaseException) -> None:
    """One-shot disable after a compile/launch failure. Matches the Hub path."""
    global _triton_failed
    if _triton_failed:
        return
    _triton_failed = True
    clear_cache()
    logger.warning("Triton ms_deform_attn failed, falling back: %s", exc)


def _supported_inputs(
    value: torch.Tensor,
    spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> bool:
    if not isinstance(spatial_shapes, torch.Tensor):
        return False
    if value.requires_grad or sampling_locations.requires_grad or attention_weights.requires_grad:
        return False
    device = value.device
    if not (
        value.is_cuda
        and sampling_locations.device == device
        and attention_weights.device == device
        and spatial_shapes.device == device
    ):
        return False
    if not (
        value.dtype in _SLOT_DTYPES
        and sampling_locations.dtype in _SLOT_DTYPES
        and attention_weights.dtype in _SLOT_DTYPES
    ):
        return False
    if value.dim() != 4 or sampling_locations.dim() != 6 or attention_weights.dim() != 5:
        return False
    if spatial_shapes.dim() != 2 or spatial_shapes.shape[1] != 2:
        return False
    batch, len_in, n_heads, channels = value.shape
    n_queries = sampling_locations.shape[1]
    n_levels = sampling_locations.shape[3]
    n_points = sampling_locations.shape[4]
    if batch == 0 or n_queries == 0:
        return False
    if sampling_locations.shape[5] != 2:
        return False
    if (
        sampling_locations.shape[0] != batch
        or attention_weights.shape[0] != batch
        or sampling_locations.shape[2] != n_heads
        or attention_weights.shape[2] != n_heads
        or attention_weights.shape[1] != n_queries
        or spatial_shapes.shape[0] != n_levels
        or attention_weights.shape[3] != n_levels
        or attention_weights.shape[4] != n_points
    ):
        return False
    if not (0 < channels <= 128):
        return False
    # Area must match Len_in or the kernel indexes off the value buffer.
    # Skip the host readback while CUDA-graph capturing: warmup already
    # validated this shape, and .item() would abort capture.
    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    if not capturing:
        areas = spatial_shapes[:, 0] * spatial_shapes[:, 1]
        if int(areas.sum().item()) != int(len_in):
            return False
        if bool((spatial_shapes <= 0).any().item()):
            return False
    return True


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _level_starts(shapes: torch.Tensor) -> torch.Tensor:
    areas = shapes[:, 0] * shapes[:, 1]
    return torch.cat([areas.new_zeros(1), areas.cumsum(0)[:-1]]).contiguous()


def _msda_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _msda_fwd(
        value_ptr,
        loc_ptr,
        attn_ptr,
        shapes_ptr,
        starts_ptr,
        out_ptr,
        batch,
        n_queries,
        n_heads,
        n_channels,
        n_levels,
        n_points,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vc,
        stride_lb,
        stride_lq,
        stride_lh,
        stride_ll,
        stride_lp,
        stride_ab,
        stride_aq,
        stride_ah,
        stride_al,
        stride_ap,
        stride_ob,
        stride_oq,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n_head_query = n_heads * n_queries
        batch_idx = pid // n_head_query
        rem = pid - batch_idx * n_head_query
        query_idx = rem // n_heads
        head_idx = rem - query_idx * n_heads
        if batch_idx >= batch:
            return

        offs_c = tl.arange(0, BLOCK_C)
        mask_c = offs_c < n_channels
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

        for level in range(n_levels):
            height = tl.load(shapes_ptr + level * 2 + 0)
            width = tl.load(shapes_ptr + level * 2 + 1)
            start = tl.load(starts_ptr + level)
            height_f = height.to(tl.float32)
            width_f = width.to(tl.float32)
            for point in range(n_points):
                loc_off = (
                    batch_idx * stride_lb
                    + query_idx * stride_lq
                    + head_idx * stride_lh
                    + level * stride_ll
                    + point * stride_lp
                )
                loc_x = tl.load(loc_ptr + loc_off + 0).to(tl.float32)
                loc_y = tl.load(loc_ptr + loc_off + 1).to(tl.float32)
                weight = tl.load(
                    attn_ptr
                    + batch_idx * stride_ab
                    + query_idx * stride_aq
                    + head_idx * stride_ah
                    + level * stride_al
                    + point * stride_ap
                ).to(tl.float32)

                # grid_sample, align_corners=False, locations in [0, 1].
                sample_x = loc_x * width_f - 0.5
                sample_y = loc_y * height_f - 0.5
                x0 = tl.floor(sample_x)
                y0 = tl.floor(sample_y)
                wx = sample_x - x0
                wy = sample_y - y0
                ix0 = x0.to(tl.int64)
                iy0 = y0.to(tl.int64)
                ix1 = ix0 + 1
                iy1 = iy0 + 1
                in00 = (iy0 >= 0) & (iy0 < height) & (ix0 >= 0) & (ix0 < width)
                in01 = (iy0 >= 0) & (iy0 < height) & (ix1 >= 0) & (ix1 < width)
                in10 = (iy1 >= 0) & (iy1 < height) & (ix0 >= 0) & (ix0 < width)
                in11 = (iy1 >= 0) & (iy1 < height) & (ix1 >= 0) & (ix1 < width)
                iy0c = tl.minimum(tl.maximum(iy0, 0), height - 1)
                ix0c = tl.minimum(tl.maximum(ix0, 0), width - 1)
                iy1c = tl.minimum(tl.maximum(iy1, 0), height - 1)
                ix1c = tl.minimum(tl.maximum(ix1, 0), width - 1)
                base = value_ptr + batch_idx * stride_vb + head_idx * stride_vh
                acc += (
                    tl.load(
                        base + (start + iy0c * width + ix0c) * stride_vs + offs_c * stride_vc,
                        mask=mask_c & in00,
                        other=0.0,
                    ).to(tl.float32)
                    * ((1.0 - wy) * (1.0 - wx) * weight)
                )
                acc += (
                    tl.load(
                        base + (start + iy0c * width + ix1c) * stride_vs + offs_c * stride_vc,
                        mask=mask_c & in01,
                        other=0.0,
                    ).to(tl.float32)
                    * ((1.0 - wy) * wx * weight)
                )
                acc += (
                    tl.load(
                        base + (start + iy1c * width + ix0c) * stride_vs + offs_c * stride_vc,
                        mask=mask_c & in10,
                        other=0.0,
                    ).to(tl.float32)
                    * (wy * (1.0 - wx) * weight)
                )
                acc += (
                    tl.load(
                        base + (start + iy1c * width + ix1c) * stride_vs + offs_c * stride_vc,
                        mask=mask_c & in11,
                        other=0.0,
                    ).to(tl.float32)
                    * (wy * wx * weight)
                )

        out_off = (
            batch_idx * stride_ob
            + query_idx * stride_oq
            + head_idx * n_channels
            + offs_c
        )
        tl.store(out_ptr + out_off, acc, mask=mask_c)

    return _msda_fwd


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _msda_kernel()
    return _KERNEL


def triton_ms_deform_attn(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Run MSDA on the in-tree Triton kernel, or return None to fall back."""
    if _triton_failed:
        return None
    if not _supported_inputs(
        value, spatial_shapes, sampling_locations, attention_weights
    ):
        return None
    try:
        import triton  # noqa: F401
    except Exception:
        return None

    value = value.contiguous()
    sampling_locations = sampling_locations.contiguous()
    attention_weights = attention_weights.contiguous()
    shapes = spatial_shapes.to(dtype=torch.int64, device=value.device).contiguous()
    starts = _level_starts(shapes)

    batch, _, n_heads, n_channels = value.shape
    _, n_queries, _, n_levels, n_points, _ = sampling_locations.shape
    output = torch.empty(
        batch, n_queries, n_heads * n_channels, device=value.device, dtype=torch.float32
    )
    block_c = _next_power_of_2(int(n_channels))
    grid = (batch * n_queries * n_heads,)
    try:
        with torch.cuda.device(value.device):
            _kernel()[grid](
                value,
                sampling_locations,
                attention_weights,
                shapes,
                starts,
                output,
                batch,
                n_queries,
                n_heads,
                n_channels,
                n_levels,
                n_points,
                value.stride(0),
                value.stride(1),
                value.stride(2),
                value.stride(3),
                sampling_locations.stride(0),
                sampling_locations.stride(1),
                sampling_locations.stride(2),
                sampling_locations.stride(3),
                sampling_locations.stride(4),
                attention_weights.stride(0),
                attention_weights.stride(1),
                attention_weights.stride(2),
                attention_weights.stride(3),
                attention_weights.stride(4),
                output.stride(0),
                output.stride(1),
                BLOCK_C=block_c,
            )
    except Exception as exc:
        _disable(exc)
        return None
    if value.dtype != torch.float32:
        return output.to(dtype=value.dtype)
    return output


register("ms_deform_attn", triton_ms_deform_attn, name="triton", predicate=_eligible)


__all__ = ["triton_ms_deform_attn"]
