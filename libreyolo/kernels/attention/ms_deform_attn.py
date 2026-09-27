"""Multi-scale deformable attention op slot (``ms_deform_attn``).

Slot signature (the classic Deformable-DETR layout):

- ``value``: ``(bs, Len_in, n_heads, c)``
- ``spatial_shapes``: ``(n_levels, 2)`` int64 tensor of ``(H, W)`` per level
- ``sampling_locations``: ``(bs, Len_q, n_heads, n_levels, n_points, 2)``
- ``attention_weights``: ``(bs, Len_q, n_heads, n_levels, n_points)``
- returns ``(bs, Len_q, n_heads * c)``, or None when the input is not
  eligible so the caller falls back to its portable path.

Like the GEMM slots, no reference implementation is registered: every model
family keeps its own upstream-parity ``grid_sample`` port as the default and
only consults this slot through :func:`maybe_ms_deform_attn`.

Two accelerated providers share the slot. The in-tree Triton kernel
(``ms_deform_attn_triton``) needs no extra package and covers CUDA
fp32/fp16/bf16 inference; it registers first. The Hub provider loads the
compiled CUDA kernel published at
``kernels-community/deformable-detr`` on the Hugging Face Hub (Apache-2.0)
via the optional ``kernels`` package and registers on top, so it wins
when the extra is installed. Nothing is vendored: the artifact is
fetched at runtime, pinned to the audited revision in ``_HUB_REVISION`` so
a moved branch can never swap the binary that runs in-process. When the
installed ``kernels`` release cannot resolve the pin (its resolver rejects
commit SHAs and validates a newer metadata schema), the provider fetches
the pinned snapshot directly and imports the matching build variant itself
— same binary, same pin, no resolver in between. Installing
the ``libreyolo[hub-kernels]`` extra is the opt-in; once the ``kernels``
package is present the provider is on by default and
``LIBREYOLO_HUB_KERNELS=0`` disables it. The autograd bridge
below follows the ``MSDeformAttnFunction`` interface of Deformable-DETR
(https://github.com/fundamentalvision/Deformable-DETR, Apache-2.0,
Copyright (c) 2020 SenseTime).
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
import platform as _platform
import sys
from pathlib import Path
from typing import Optional

import torch

from .. import clear_cache, iter_impls, register, resolve

logger = logging.getLogger(__name__)

_HUB_REPO = "kernels-community/deformable-detr"
# Commit pin: native code fetched from the Hub executes in-process (forward
# and backward), so selection by repo name alone would let a moved branch
# change the binary under users. Bump deliberately, with a GPU parity run
# (tests/unit/kernels/test_ms_deform_attn.py::test_hub_matches_portable_on_cuda).
_HUB_REVISION = "4d2393e5d7879f7cf68db04cc7c9c7342272bc05"
_MAX_IM2COL_STEP = 64
_SLOT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

_hub_kernel = None
_hub_failed = False
_missing_hub_hint_emitted = False


def _hub_enabled() -> bool:
    """Hub kernels are on by default; installing the extra is the opt-in.

    The runtime fetch only ever happens when the optional ``kernels``
    package is installed (see :func:`_eligible`). Without it, an eager CUDA
    call that falls back emits one install hint.
    ``LIBREYOLO_HUB_KERNELS=0`` disables the provider and the hint.
    """
    return os.environ.get("LIBREYOLO_HUB_KERNELS", "").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def _kernel_selection_disables_acceleration() -> bool:
    """Whether the global selector explicitly requests the portable path."""
    forced = os.environ.get("LIBREYOLO_KERNELS", "").strip().lower()
    if not forced:
        forced = os.environ.get("LIBREYOLO_QUANT_KERNELS", "").strip().lower()
    return forced in ("off", "reference")


def _hub_client_installed_for_hint() -> bool:
    """Check discovery without letting an importer error break fallback."""
    try:
        return importlib.util.find_spec("kernels") is not None
    except Exception:
        # The registry already logs predicate failures. Suppress this optional
        # install hint rather than raising again or misdiagnosing the package.
        return True


def _warn_missing_hub_kernels_once(value: Optional[torch.Tensor]) -> None:
    """Hint once when a real CUDA call has no Hub client to accelerate it."""
    global _missing_hub_hint_emitted
    if (
        _missing_hub_hint_emitted
        or not getattr(value, "is_cuda", False)
        or not _hub_enabled()
        or _kernel_selection_disables_acceleration()
        or _hub_client_installed_for_hint()
    ):
        return
    _missing_hub_hint_emitted = True
    logger.warning(
        "No accelerated MSDA provider accepted this CUDA call; using the portable "
        "path. Install `libreyolo[hub-kernels]` for the compiled kernel, or set "
        "LIBREYOLO_HUB_KERNELS=0 to silence this hint."
    )


def _eligible() -> bool:
    """Cheap predicate: the expensive Hub fetch is deferred to the first call."""
    return (
        _hub_enabled()
        and not _hub_failed
        and importlib.util.find_spec("kernels") is not None
        and torch.cuda.is_available()
    )


def _pinned_variant_name() -> Optional[str]:
    """The build-variant directory the running torch/CUDA/OS maps to.

    Mirrors the ``kernels`` package's naming (``torch<maj><min>-[cxx11-]
    cu<ver>-<arch>-<os>``) strictly: an exact match or nothing, because a
    compiled extension is only ABI-safe against the torch minor and CUDA
    build it was compiled for.
    """
    cuda = torch.version.cuda
    if not cuda:
        return None
    torch_tag = "".join(torch.__version__.split(".")[:2])
    cuda_tag = cuda.replace(".", "")
    os_name = _platform.system().lower()
    machine = _platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    if os_name == "windows":
        return f"torch{torch_tag}-cu{cuda_tag}-{machine}-windows"
    if os_name == "linux":
        return f"torch{torch_tag}-cxx11-cu{cuda_tag}-{machine}-linux"
    return None


def _load_pinned_snapshot():
    """Load the pinned revision without the ``kernels`` resolver.

    Current ``kernels`` releases resolve revisions through the Hub's kernels
    API, which rejects commit SHAs, and validate a metadata schema this
    repo's artifacts predate — either one silently kills the provider even
    though the pinned binaries themselves load and pass parity. The pin is
    the contract, so fetch it directly (``huggingface_hub`` resolves SHA
    revisions fine) and import the matching build variant ourselves.
    """
    variant = _pinned_variant_name()
    if variant is None:
        return None
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            _HUB_REPO,
            revision=_HUB_REVISION,
            allow_patterns=[f"build/{variant}/*"],
        )
    )
    package = root / "build" / variant
    if not (package / "__init__.py").exists():
        logger.warning(
            "Hub kernel %s has no build variant %s; using the portable path",
            _HUB_REPO,
            variant,
        )
        return None
    module_name = f"_libreyolo_hub_{variant.replace('-', '_')}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name,
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_hub_kernel():
    """Fetch and cache the Hub kernel; a failure disables the provider."""
    global _hub_kernel, _hub_failed
    if _hub_kernel is not None or _hub_failed:
        return _hub_kernel
    try:
        from kernels import get_kernel

        _hub_kernel = get_kernel(_HUB_REPO, revision=_HUB_REVISION)
    except Exception as exc:
        # ``get_kernel`` is version-sensitive (SHA pins and this repo's
        # metadata schema stopped resolving in newer releases); the direct
        # snapshot loader below is the compatibility path.
        logger.debug("kernels.get_kernel could not load %s: %s", _HUB_REPO, exc)
        try:
            _hub_kernel = _load_pinned_snapshot()
        except Exception as exc2:
            logger.warning("Hub kernel %s unavailable: %s", _HUB_REPO, exc2)
    if _hub_kernel is None:
        _hub_failed = True
        clear_cache()
    return _hub_kernel


class _MSDeformAttnFunction(torch.autograd.Function):
    """Autograd bridge over the compiled forward/backward pair."""

    @staticmethod
    def forward(
        ctx,
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        ctx.im2col_step = im2col_step
        output = _hub_kernel.ms_deform_attn_forward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )
        ctx.save_for_backward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = (
            _hub_kernel.ms_deform_attn_backward(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                grad_output.contiguous(),
                ctx.im2col_step,
            )
        )
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def level_start_index(spatial_shapes: torch.Tensor) -> torch.Tensor:
    """Per-level start offsets into the flattened value, from (H, W) pairs."""
    areas = spatial_shapes[:, 0] * spatial_shapes[:, 1]
    return torch.cat([areas.new_zeros(1), areas.cumsum(0)[:-1]])


def _supported_inputs(
    value: torch.Tensor,
    spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> bool:
    if not isinstance(spatial_shapes, torch.Tensor):
        return False
    if not (
        value.is_cuda
        and sampling_locations.is_cuda
        and attention_weights.is_cuda
        and spatial_shapes.is_cuda
    ):
        return False
    # Half inputs are eligible: the provider upcasts them to fp32 at the
    # call boundary so autocast runs share the fp32 fused path's numerics
    # (the compiled kernel also has native fp16/bf16 paths, but autocast
    # hands the slot a mix — fp16 value with fp32 softmax weights — and a
    # common fp32 dtype is both required and the more precise choice).
    if not (
        value.dtype in _SLOT_DTYPES
        and sampling_locations.dtype in _SLOT_DTYPES
        and attention_weights.dtype in _SLOT_DTYPES
    ):
        return False
    if value.dim() != 4 or sampling_locations.dim() != 6 or attention_weights.dim() != 5:
        return False
    batch = value.shape[0]
    if batch == 0:
        return False
    step = batch if batch < _MAX_IM2COL_STEP else _MAX_IM2COL_STEP
    return batch % step == 0


def hub_ms_deform_attn(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Run MSDA on the compiled Hub kernel, or return None to fall back."""
    global _hub_failed
    if not _supported_inputs(
        value, spatial_shapes, sampling_locations, attention_weights
    ):
        return None
    if _hub_kernel is None and torch.cuda.is_current_stream_capturing():
        # Never fetch/load the kernel module inside CUDA graph capture; the
        # load performs I/O and module initialization that would poison the
        # capture. Graph warmup iterations load it, after which captures
        # record the compiled op like any other.
        return None
    if _load_hub_kernel() is None:
        return None
    batch = value.shape[0]
    step = batch if batch < _MAX_IM2COL_STEP else _MAX_IM2COL_STEP
    shapes = spatial_shapes.to(dtype=torch.int64)
    try:
        # ``float()`` is a no-op view of an fp32 tensor and an autograd-tracked
        # cast otherwise, so half inputs (autocast) run the fp32 kernel and
        # their gradients flow back in the input dtype.
        output = _MSDeformAttnFunction.apply(
            value.float().contiguous(),
            shapes,
            level_start_index(shapes),
            sampling_locations.float().contiguous(),
            attention_weights.float().contiguous(),
            step,
        )
        return output if value.dtype == torch.float32 else output.to(value.dtype)
    except Exception as exc:
        # A kernel that loads but rejects this torch/GPU combination must
        # never break inference: disable the provider and fall back.
        _hub_failed = True
        clear_cache()
        logger.warning("Hub kernel %s failed, falling back: %s", _HUB_REPO, exc)
        return None


def _not_exporting() -> bool:
    """Fallback for ``torch.compiler.is_exporting`` on torch versions without it."""
    return False


def ms_deform_attn_available(value: Optional[torch.Tensor] = None) -> bool:
    """Whether the slot could run here, checked before adapting layouts.

    Families whose native layout differs from the slot's ask this first so
    the adaptation work (and any ``tolist()`` that would bake constants into
    a trace) never happens on the portable path. Tracing and export always
    report unavailable: exported graphs must not capture a runtime-fetched
    kernel. ``is_exporting`` covers non-strict ``torch.export``, which traces
    with FakeTensors without setting ``is_compiling``.

    Pass the original value tensor so an actual CUDA fallback can emit the
    one-time Hub-kernel install hint without warning for CPU model calls.
    """
    if (
        torch.jit.is_tracing()
        or torch.compiler.is_compiling()
        or getattr(torch.compiler, "is_exporting", _not_exporting)()
        or torch.onnx.is_in_onnx_export()
    ):
        return False
    available = resolve("ms_deform_attn") is not None
    if not available:
        _warn_missing_hub_kernels_once(value)
    return available


@functools.lru_cache(maxsize=32)
def _cached_shapes(pairs: tuple, device: torch.device) -> torch.Tensor:
    return torch.tensor(pairs, dtype=torch.int64, device=device)


def spatial_shapes_tensor(value_spatial_shapes, device) -> torch.Tensor:
    """Normalize per-level ``(H, W)`` pairs to the int64 tensor the slot wants.

    Several families carry their spatial shapes as Python pairs; the compiled
    kernel reads them from device memory. The small results are cached so the
    host-to-device copy happens once per shape set rather than per call. Only
    valid under :func:`ms_deform_attn_available`, which rules out tracing.
    """
    if isinstance(value_spatial_shapes, torch.Tensor):
        if (
            value_spatial_shapes.dtype == torch.int64
            and value_spatial_shapes.device == device
        ):
            return value_spatial_shapes
        value_spatial_shapes = value_spatial_shapes.tolist()
    pairs = tuple((int(height), int(width)) for height, width in value_spatial_shapes)
    return _cached_shapes(pairs, device)


def maybe_ms_deform_attn(
    value: torch.Tensor,
    spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Resolve the ``ms_deform_attn`` slot and run it, or return None.

    Callers keep their portable grid_sample port as the fallback. Tracing
    and export always take the portable path: exported graphs must not
    capture a runtime-fetched kernel. ``is_exporting`` covers non-strict
    ``torch.export``, which traces with FakeTensors without setting
    ``is_compiling``.
    """
    if not ms_deform_attn_available(value):
        return None
    # Walk eligible providers. Hub is preferred but may return None for an
    # input or disable itself after a launch failure; Triton must still run.
    for _name, impl in iter_impls("ms_deform_attn"):
        output = impl(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        if output is not None:
            return output
    _warn_missing_hub_kernels_once(value)
    return None


def maybe_ms_deform_attn_v2(
    value: torch.Tensor,
    spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list,
) -> Optional[torch.Tensor]:
    """Slot adapter for the RT-DETRv2-style flat-points layout.

    v2 cores carry ``sum(num_points_list)`` sampling points flattened across
    levels and support a different point count per level, while the slot's
    layout is ``(n_levels, n_points)``. Only a uniform count reshapes onto
    it, so a ragged ``num_points_list`` returns None and the caller keeps its
    portable path. ``value`` must already be in the slot's
    ``(bs, Len_in, n_heads, c)`` layout.
    """
    if not ms_deform_attn_available(value):
        return None
    levels = len(num_points_list)
    if levels == 0 or len(spatial_shapes) != levels:
        return None
    points = int(num_points_list[0])
    if any(int(count) != points for count in num_points_list):
        return None
    return maybe_ms_deform_attn(
        value,
        spatial_shapes,
        sampling_locations.unflatten(3, (levels, points)),
        attention_weights.unflatten(3, (levels, points)),
    )


# Triton registers first (older); Hub stays preferred when its extra is
# installed. A missing Triton install must not hide the Hub provider.
try:
    from .ms_deform_attn_triton import triton_ms_deform_attn  # noqa: F401
except Exception as exc:
    logger.debug("Triton ms_deform_attn unavailable: %s", exc)

register("ms_deform_attn", hub_ms_deform_attn, name="hub", predicate=_eligible)


__all__ = [
    "hub_ms_deform_attn",
    "level_start_index",
    "maybe_ms_deform_attn",
    "maybe_ms_deform_attn_v2",
    "ms_deform_attn_available",
    "spatial_shapes_tensor",
]
