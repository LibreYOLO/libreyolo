"""Automatic batch-size estimation for training workloads.

Probes the model at small batch sizes in *training mode with a backward pass*,
fits a line to the (batch → GPU memory) curve, and extrapolates to find the
largest global batch that stays within a target fraction of total VRAM.

Why train mode + backward (not eval + no_grad like Ultralytics):
  Eval+no_grad only measures inference memory.  During training PyTorch retains
  all intermediate activations for backward, and gradient tensors are allocated.
  For deep CNNs like YOLO9 this adds 5-10× the memory seen at inference.
  Probing with a backward pass captures the true training memory footprint.

Result is always a power of 2 strictly below the extrapolated value, so it
composes cleanly with power-of-2 nbs values for gradient accumulation.

Global batch semantics (Ultralytics-mirror): the returned value is the GLOBAL
batch. Under DDP the trainer divides it by world_size for per-rank loaders.
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn

from libreyolo.training.distributed import (
    is_distributed,
    is_main_process,
)

logger = logging.getLogger(__name__)

_DEFAULT_FRACTION: float = 0.70
_DEFAULT_MAX_PROBE: int = 64
_BATCH_SAFE_MAX: int = 1024


# =============================================================================
# Helpers
# =============================================================================


def _floor_pow2_strict(x: float) -> int:
    """Return the largest power of 2 *strictly less than* x.

    This is intentionally conservative: the probe targets 70 % of VRAM but
    the fit is an approximation, so we never bet on the exact extrapolated
    value.  Returns 1 for x ≤ 2.

    Examples:  33.0 → 32,  32.0 → 16,  65.0 → 64,  17.5 → 16
    """
    if x <= 2:
        return 1
    p = int(math.log2(x))   # floor(log2(x))
    result = 1 << p          # 2^p  ≤  x
    if result >= x:          # exact power of 2 — go one step lower
        result >>= 1
    return max(1, result)


def _find_grad_tensor(obj) -> torch.Tensor | None:
    """Return the first tensor with requires_grad=True found anywhere in obj.

    Handles tensors, tuples/lists (nested), and dicts. Returns None when no
    differentiable tensor is present (e.g. model output is None or detached).
    """
    if isinstance(obj, torch.Tensor):
        return obj if obj.requires_grad else None
    if isinstance(obj, (tuple, list)):
        for item in obj:
            t = _find_grad_tensor(item)
            if t is not None:
                return t
    if isinstance(obj, dict):
        for v in obj.values():
            t = _find_grad_tensor(v)
            if t is not None:
                return t
    return None


# =============================================================================
# Core probe
# =============================================================================


def autobatch(
    model: nn.Module,
    imgsz: int = 640,
    amp: bool = True,
    fraction: float = _DEFAULT_FRACTION,
    default: int = 16,
    max_probe: int = _DEFAULT_MAX_PROBE,
) -> int:
    """Estimate the optimal *global* batch size for the given model and image size.

    Probes in training mode with a backward pass at powers-of-2 batch sizes up
    to *max_probe*, fits a line to the (batch → peak memory) curve, extrapolates
    to *fraction* of total VRAM, then returns the largest power of 2 strictly
    below that value.  Returns *default* when CUDA is unavailable or probing
    fails.

    Args:
        model: Model already resident on the target CUDA device.
        imgsz: Square input size (height == width).
        amp: Whether AMP (autocast) will be used during training.
        fraction: Target fraction of *total* VRAM to occupy (default 0.70).
        default: Fallback batch size for non-CUDA devices or probe failures.
        max_probe: Largest batch size to probe (default 64; set to nbs).

    Returns:
        Estimated optimal global batch size — a power of 2, always ≥ 1.
    """
    device = next(model.parameters()).device

    if device.type != "cuda":
        logger.info("AutoBatch: non-CUDA device (%s) — keeping batch=%d", device.type, default)
        return default

    props = torch.cuda.get_device_properties(device)
    total_gib = props.total_memory / 1024**3
    target_gib = total_gib * fraction

    logger.info(
        "AutoBatch: %s  total=%.1f GiB  target=%.0f%%",
        props.name,
        total_gib,
        fraction * 100,
    )

    # Powers of 2 from 1 up to max_probe
    probe_batches: List[int] = []
    b = 1
    while b <= max_probe:
        probe_batches.append(b)
        b *= 2

    probe_sizes: List[int] = []
    probe_mem: List[float] = []

    was_training = model.training
    model.train()
    try:
        ctx = torch.autocast("cuda") if amp else contextlib.nullcontext()
        for b in probe_batches:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                x = torch.zeros(b, 3, imgsz, imgsz, dtype=torch.float32, device=device)
                with ctx:
                    out = model(x)
                t = _find_grad_tensor(out)
                if t is not None:
                    t.float().sum().backward()
                model.zero_grad(set_to_none=True)
                mem = torch.cuda.max_memory_allocated(device) / 1024**3
                probe_sizes.append(b)
                probe_mem.append(mem)
                logger.debug("AutoBatch: batch=%d  mem=%.3f GiB", b, mem)
                if mem > total_gib * 0.90:
                    break
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    model.zero_grad(set_to_none=True)
                    break
                raise
    finally:
        model.train(was_training)
        torch.cuda.empty_cache()

    if len(probe_sizes) < 2:
        logger.warning("AutoBatch: too few probe points — keeping batch=%d", default)
        return default

    slope, intercept = np.polyfit(probe_sizes, probe_mem, 1)
    if slope <= 0:
        logger.warning("AutoBatch: non-positive slope — keeping batch=%d", default)
        return default

    raw = min((target_gib - intercept) / slope, max_probe)
    result = _floor_pow2_strict(raw)
    result = max(1, min(result, _BATCH_SAFE_MAX))

    logger.info(
        "AutoBatch: estimated optimal global batch = %d  (raw=%.1f, target=%.1f GiB)",
        result,
        raw,
        target_gib,
    )
    return result


def _fit_batch_size(
    probe_sizes: List[int],
    probe_mem: List[float],
    target_gib: float,
) -> int | None:
    """Fit a line to (batch → memory) samples and extrapolate to *target_gib*.

    Returns the rounded optimal batch size clamped to [1, _BATCH_SAFE_MAX],
    or None when the slope is non-positive (degenerate fit).
    """
    slope, intercept = np.polyfit(probe_sizes, probe_mem, 1)
    if slope <= 0:
        return None
    optimal = int(round((target_gib - intercept) / slope))
    return max(1, min(optimal, _BATCH_SAFE_MAX))


# =============================================================================
# DDP-aware resolver
# =============================================================================


def resolve_auto_batch(
    model: nn.Module,
    imgsz: int = 640,
    amp: bool = True,
    fraction: float = _DEFAULT_FRACTION,
    world_size: int = 1,
    default: int = 16,
    nbs: int | None = None,
) -> int:
    """Run ``autobatch`` on rank 0 and broadcast the result to all ranks.

    The probe runs on a single GPU and returns the per-GPU capacity.  Under
    DDP that capacity is scaled by *world_size* to form the global batch,
    capped at *nbs* so the effective batch (global x accumulation steps) never
    exceeds the nominal batch size.  This means adding GPUs reduces the number
    of accumulation steps rather than shrinking per-GPU batch size.

    When *nbs* is not provided the global batch is simply rounded down to the
    nearest multiple of *world_size*.

    Args:
        model: Model on the target device (not yet DDP-wrapped).
        imgsz: Square input size.
        amp: Whether AMP is active.
        fraction: Target fraction of total VRAM (default 0.70).
        world_size: Number of DDP ranks (1 for single-GPU).
        default: Fallback when CUDA is unavailable.
        nbs: Nominal batch size — caps the global batch and sets the probe
            limit so per-GPU capacity never exceeds nbs.

    Returns:
        Global batch size, divisible by *world_size* and ≥ 1.
    """
    ws = max(1, world_size)
    max_probe = nbs if (nbs is not None and nbs > 0) else _DEFAULT_MAX_PROBE

    if is_main_process():
        try:
            per_gpu = autobatch(
                model, imgsz=imgsz, amp=amp, fraction=fraction,
                default=default, max_probe=max_probe,
            )
        except Exception as exc:
            logger.warning("AutoBatch: probe failed (%s) — using default %d", exc, default)
            per_gpu = default
    else:
        per_gpu = 0

    if is_distributed():
        import torch.distributed as dist

        # broadcast_object_list relies on pickle serialisation which can fail
        # silently on NCCL when rank 0 is delayed; a long tensor broadcast is
        # a simpler and more reliable primitive for a single integer.
        device = next(model.parameters()).device
        t = torch.tensor([per_gpu], dtype=torch.long, device=device)
        dist.broadcast(t, src=0)
        per_gpu = int(t.item())

    if nbs is not None and nbs > 0:
        # Scale per-GPU capacity to global, capped at nbs so that more GPUs
        # reduce accumulation steps rather than shrinking per-GPU batch.
        global_batch = min(per_gpu * ws, nbs)
        # Round down to a multiple of world_size (each rank gets equal share).
        global_batch = max(ws, (global_batch // ws) * ws)
        if global_batch > nbs:
            logger.warning(
                "AutoBatch: world_size=%d exceeds nbs=%d — global batch is %d. "
                "Gradient accumulation will not reach the intended effective batch size.",
                ws, nbs, global_batch,
            )
    else:
        global_batch = max(ws, (per_gpu // ws) * ws)

    logger.info("AutoBatch: per-GPU=%d  world_size=%d  global=%d", per_gpu, ws, global_batch)
    return global_batch


__all__ = ["autobatch", "resolve_auto_batch", "_fit_batch_size", "_floor_pow2_strict"]
