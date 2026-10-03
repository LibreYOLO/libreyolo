"""Core AI (``.aimodel``) export.

Core AI is Apple's on-device inference stack, the successor generation to
Core ML. Unlike the Core ML path, which records a single trace through
``torch.jit.trace``, this converter goes through ``torch.export``: a real
graph capture with guards. That is stricter (host scalar reads and
data-dependent control flow are rejected rather than silently baked in) and
removes the need for the static-eval monkey patches the Core ML path carries.

Pipeline::

    torch.export.export -> run_decompositions -> TorchConverter -> optimize
                        -> AIProgram.save_asset(.aimodel)

Artifacts declare ``minimum_os = OSVersion.v27``; that is the only value the
Core AI toolchain offers, and it gates *deployment*, not conversion. Both
conversion and Python-side execution work on earlier macOS through the
runtime shipped inside the ``coreai`` wheel, but numerics differ between OS
versions, so parity must be recorded on macOS 27.

SCOPE: EXPORT, NOT INFERENCE
----------------------------
There is no Core AI entry in ``libreyolo/backends``. This module converts a
model and the support matrix records numeric parity against a reference
artifact, but nothing in the library loads an ``.aimodel`` back for inference
the way ``backends/onnx.py`` loads an ONNX graph. A ``validated`` row here is
a claim that the exported graph computes the same numbers as the reference,
not that ``predict`` will run it. Consumers use the Core AI runtime directly.

OUTPUT ORDERING CONTRACT
------------------------
Core AI returns a **named dict**, and its key order matches neither the eager
forward's tuple order nor anything a caller can guess. Consumers must map by
name. The exported output names are recorded in the artifact metadata under
``coreai_output_names`` so a backend can rebuild the canonical ordering
without re-deriving it. Never pair Core AI outputs with eager outputs
positionally.
"""

from __future__ import annotations

import json
import logging
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .coreml import _prepare_rtdetr_static_eval, _wrap_for_family

logger = logging.getLogger(__name__)

# Only value the toolchain exposes; kept named so the reason is greppable.
_MINIMUM_OS = "v27"
_ANCHOR_FREEZE_FAMILIES = {
    "yolo9",
    "yolo9_e2e",
    "yolo9_p2",
    "yolox",
    "yolo1",
    "yolo2",
    "yolo3",
    "yolo4",
    "yolo7",
    "yolonas",
    "picodet",
    "rtmdet",
}
_DARKNET_FAMILIES = {"yolo1", "yolo2", "yolo3", "yolo4"}
_RTDETR_STATIC_FAMILIES = {
    "rtdetr",
    "rtdetrv2",
    "rtdetrv4",
    "dfine",
    "deim",
    "deimv2",
    "ec",
}


def _fuse_darknet_batchnorm(nn_model: nn.Module):
    """Fold exact Darknet inference BN into its preceding convolution.

    Core AI 0.4.1 does not preserve Darknet's normalization formula,
    ``(x - mean) / (sqrt(var) + eps) * weight + bias``. In particular,
    Darknet adds epsilon after the square root, unlike ``nn.BatchNorm2d``.
    Folding the frozen inference parameters into the convolution is
    algebraically equivalent and removes the converter-sensitive expression.

    The swap is scoped to Core AI graph preparation and restored afterwards.
    """
    from ..models.darknet.blocks import DarknetConv

    prepared: list[tuple[DarknetConv, nn.Conv2d, nn.Conv2d, nn.Module]] = []
    for module in nn_model.modules():
        if not isinstance(module, DarknetConv) or module.bn is None:
            continue

        conv = module.conv
        bn = module.bn
        replacement = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        with torch.no_grad():
            scale = bn.weight / (torch.sqrt(bn.running_var) + bn.eps)
            replacement.weight.copy_(conv.weight * scale[:, None, None, None])
            conv_bias = (
                conv.bias
                if conv.bias is not None
                else torch.zeros_like(bn.running_mean)
            )
            replacement.bias.copy_((conv_bias - bn.running_mean) * scale + bn.bias)

        prepared.append((module, replacement, conv, bn))

    # Do not mutate the live graph until every replacement has been allocated
    # and populated. A failure on a later layer must leave earlier layers
    # untouched, before the caller has had a chance to register our restore
    # callback.
    for module, replacement, _, _ in prepared:
        module.conv = replacement
        module.bn = None

    if prepared:
        logger.info(
            "Folded %d exact Darknet batch-normalization layers into their "
            "convolutions for Core AI conversion.",
            len(prepared),
        )

    def _restore():
        for module, _, conv, bn in prepared:
            module.conv = conv
            module.bn = bn

    return _restore


def _freeze_anchor_grid(nn_model: nn.Module, dummy: torch.Tensor):
    """Bake a detection head's anchor grid as constants for the export canvas.

    The shared exporter sets ``head.export = True`` before handing the model
    over. In that mode the head rebuilds its anchor grid from live feature
    shapes every forward, and ``torch.export`` turns the ``h * w`` products
    into unbacked symbols and refuses the graph. Core AI artifacts are
    fixed-canvas, so the grid is a constant: heads that support it
    (``YOLO9Head.freeze_anchor_grid``) pin the grid for ``dummy``'s H/W.
    Wrapped models are searched through their ``.model`` attributes.

    Returns a callable that unfreezes the grid and restores ``head.export``.
    """
    target = nn_model
    head = getattr(target, "head", None)
    while head is None and isinstance(getattr(target, "model", None), nn.Module):
        target = target.model
        head = getattr(target, "head", None)
    if head is None or not hasattr(head, "freeze_anchor_grid"):
        return lambda: None

    was_export = getattr(head, "export", False)
    head.freeze_anchor_grid((int(dummy.shape[-2]), int(dummy.shape[-1])))

    def _restore():
        head.unfreeze_anchor_grid()
        head.export = was_export

    return _restore


class _YoloNASDecodedOnly(nn.Module):
    """Expose only YOLO-NAS's decoded predictions, matching the ONNX contract.

    In export mode the head returns ``(decoded_predictions, raw_predictions)``
    (see models/yolonas/nn.py). The ONNX path ships the decoded pair alone, so
    an artifact that also carries the raw per-level maps disagrees on arity
    with every other backend and cannot be parity-checked against them.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            decoded = out[0]
            if isinstance(decoded, (list, tuple)):
                return tuple(decoded)
        return out


class _FOMOPreprocess(nn.Module):
    """Map canonical RGB[0,1] input to FOMO's RGB[-1,1] contract."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model((x - 0.5) / 0.5)


def _wrap_coreai_contract(
    nn_model: nn.Module,
    model_family: str | None,
) -> nn.Module:
    """Apply the complete per-family Core AI input/output contract."""
    family = (model_family or "").lower()
    wrapped = _wrap_for_family(nn_model, family)
    if family == "fomo":
        wrapped = _FOMOPreprocess(wrapped)
    if family == "yolonas":
        wrapped = _YoloNASDecodedOnly(wrapped)
    return wrapped.eval()


def _rebake_rfdetr_pos_embed(nn_model: nn.Module, dummy: torch.Tensor):
    """Re-bake RF-DETR's position embedding for the actual export canvas.

    The RF-DETR backbone already carries the right idea: ``export()``
    interpolates the pretrained position-embedding grid once, eagerly, stores
    the result as a Parameter, and swaps ``interpolate_pos_encoding`` for a
    version that returns that Parameter whenever the token count matches.

    It bakes for the backbone's CONFIGURED shape, though, which is 384 for the
    nano size. Export at any other canvas and the token counts no longer match
    (640 needs 40x40 = 1600 patches against the baked 24x24 = 576), so the fast
    path is skipped and the fallback runs a bicubic *with antialiasing* inside
    the traced graph. The converter has no lowering for
    ``aten._upsample_bicubic2d_aa`` and refuses the program.

    Pointing the backbone at the export canvas and re-running its own bake
    moves that interpolation back out of the graph. The numbers are unchanged:
    the eager fallback resizes the baked grid to the canvas with an
    antialiased bicubic, and re-baking performs exactly that resize, just
    ahead of capture instead of during it.

    This is deliberately NOT a general interception of ``F.interpolate``. An
    earlier attempt did that, replaying results by call order, and it silently
    changed model outputs by up to 9.5e-01 whenever the call sequence did not
    line up. Reusing the model's own baking path keeps the correctness
    argument local and checkable.

    Returns a callable restoring the original state.
    """
    h, w = int(dummy.shape[-2]), int(dummy.shape[-1])

    targets = [
        mod
        for mod in nn_model.modules()
        if hasattr(mod, "_export")
        and callable(getattr(mod, "export", None))
        and isinstance(getattr(mod, "shape", None), (tuple, list))
    ]
    if not targets:
        return lambda: None

    undo = []
    for mod in targets:
        if tuple(mod.shape)[:2] == (h, w):
            continue  # already baked for this canvas
        embeddings = getattr(getattr(mod, "encoder", None), "embeddings", None)
        if embeddings is None:
            continue
        prev_shape = mod.shape
        prev_export = mod._export
        prev_pe = embeddings.position_embeddings
        prev_interp = embeddings.__dict__.get("interpolate_pos_encoding")

        mod.shape = (h, w)
        mod._export = False
        try:
            mod.export()
        except Exception as exc:  # noqa: BLE001 - leave the graph to fail loudly
            logger.warning(
                "Could not re-bake the RF-DETR position embedding for %dx%d "
                "(%s); the antialiased bicubic will stay in the graph.",
                h,
                w,
                exc,
            )
            mod.shape = prev_shape
            mod._export = prev_export
            embeddings.position_embeddings = prev_pe
            if prev_interp is None:
                embeddings.__dict__.pop("interpolate_pos_encoding", None)
            else:
                embeddings.interpolate_pos_encoding = prev_interp
            continue
        logger.info(
            "Re-baked the RF-DETR position embedding for the %dx%d export "
            "canvas (was %s).",
            h,
            w,
            tuple(prev_shape)[:2],
        )
        undo.append((mod, embeddings, prev_shape, prev_export, prev_pe, prev_interp))

    def _restore():
        for mod, embeddings, prev_shape, prev_export, prev_pe, prev_interp in undo:
            mod.shape = prev_shape
            mod._export = prev_export
            embeddings.position_embeddings = prev_pe
            if prev_interp is None:
                embeddings.__dict__.pop("interpolate_pos_encoding", None)
            else:
                embeddings.interpolate_pos_encoding = prev_interp

    return _restore


def _replace_adaptive_avg_pool(nn_model: nn.Module):
    """Replace ``AdaptiveAvgPool2d(1)`` with an exact spatial mean.

    ``AdaptiveAvgPool2d`` decomposes to ``aten.as_strided``, which the Core AI
    converter cannot lower. ``as_strided`` is a primitive with no
    decomposition, so it cannot be expanded away.

    When the target output is 1x1 the operation is exactly a mean over the
    spatial dimensions, so substituting one changes nothing numerically while
    removing the operator entirely. Any other output size is left alone: the
    equivalence does not hold there, and a conversion error is better than a
    silent approximation.

    Returns a callable restoring the original modules.
    """

    class _SpatialMean(nn.Module):
        def forward(self, x):
            return x.mean(dim=(-2, -1), keepdim=True)

    def _is_global(mod):
        size = getattr(mod, "output_size", None)
        return size == 1 or size == (1, 1) or size == [1, 1]

    swapped = []
    for parent in nn_model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.AdaptiveAvgPool2d) and _is_global(child):
                setattr(parent, name, _SpatialMean())
                swapped.append((parent, name, child))
    if swapped:
        logger.info(
            "Replaced %d AdaptiveAvgPool2d(1) with an exact spatial mean "
            "(avoids aten.as_strided).",
            len(swapped),
        )

    def _restore():
        for parent, name, child in swapped:
            setattr(parent, name, child)

    return _restore


def _require_coreai():
    """Import the Core AI toolchain with an actionable message if absent."""
    try:
        from coreai_torch import TorchConverter, get_decomp_table
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Core AI export requires the 'coreai-torch' package. Install it "
            "with `pip install libreyolo[coreai]`. Note that coreai-torch "
            "pins torch to 2.11.x, and that Core AI artifacts can only be "
            "produced and executed on macOS."
        ) from exc
    return TorchConverter, get_decomp_table


def _minimum_os():
    """The OS version the asset declares.

    Passed explicitly rather than left to the toolchain default, so the value
    the module docstring promises is the value the artifact carries even if
    that default moves.
    """
    from coreai.authoring import OSVersion

    return getattr(OSVersion, _MINIMUM_OS)


def _stringify_metadata_value(value: Any) -> str:
    """Encode structured metadata without Python-repr-only values."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _asset_metadata(values: dict[str, Any]):
    """Build the asset metadata record, or ``None`` when there is nothing to say.

    Everything the exporter knows goes in as creator-defined metadata, keyed
    and stringified, because the asset format stores custom entries as strings.
    """
    if not values:
        return None
    from coreai.authoring.asset import AIModelAssetMetadata

    record = AIModelAssetMetadata()
    for key, value in values.items():
        if value is None:
            continue
        record.set_custom(str(key), _stringify_metadata_value(value))
    return record


def _snapshot_rtdetr_static_eval(nn_model: nn.Module):
    """Snapshot what ``_prepare_rtdetr_static_eval`` is about to overwrite.

    That helper bakes resolution-specific tensors straight onto the live
    encoder and decoder (``eval_spatial_size``, one ``pos_embed{idx}`` per
    encoder index, and the ``anchors`` / ``valid_mask`` buffers) and returns
    nothing, so without this the caller's model keeps export-only state
    afterwards and quietly misbehaves at any other resolution.

    Returns a callable restoring the previous state.
    """
    target = nn_model
    while (
        getattr(target, "encoder", None) is None
        and getattr(target, "decoder", None) is None
        and getattr(target, "model", None) is not None
    ):
        target = target.model

    missing = object()
    saved: list = []
    encoder = getattr(target, "encoder", None)
    if encoder is not None and hasattr(encoder, "build_2d_sincos_position_embedding"):
        saved.append(
            (
                encoder,
                "eval_spatial_size",
                getattr(encoder, "eval_spatial_size", missing),
            )
        )
        for idx in getattr(encoder, "use_encoder_idx", []):
            name = f"pos_embed{idx}"
            saved.append((encoder, name, getattr(encoder, name, missing)))

    decoder = getattr(target, "decoder", None)
    if decoder is not None and hasattr(decoder, "_generate_anchors"):
        saved.append(
            (
                decoder,
                "eval_spatial_size",
                getattr(decoder, "eval_spatial_size", missing),
            )
        )
        # Buffers are restored through _buffers so a name that was absent
        # before export does not linger as a registered buffer afterwards.
        for name in ("anchors", "valid_mask"):
            value = decoder._buffers.get(name, missing)
            saved.append((decoder, f"_buffer:{name}", value))

    def _restore():
        for owner, name, value in saved:
            if name.startswith("_buffer:"):
                key = name.split(":", 1)[1]
                if value is missing:
                    owner._buffers.pop(key, None)
                else:
                    owner._buffers[key] = value
            else:
                if value is missing:
                    try:
                        delattr(owner, name)
                    except AttributeError:
                        pass
                else:
                    setattr(owner, name, value)

    return _restore


def _force_manual_grid_sample():
    """Enable the repository's gather-based deformable sampler temporarily."""
    tokens = []
    try:
        from ..models.dfine import ms_deform as _dfine_ms

        tokens.append(
            (
                _dfine_ms._FORCE_MANUAL_GRID_SAMPLE,
                _dfine_ms._FORCE_MANUAL_GRID_SAMPLE.set(True),
            )
        )
    except ImportError:
        pass

    def _restore():
        for flag, token in reversed(tokens):
            flag.reset(token)

    return _restore


@contextmanager
def _prepare_coreai_graph(
    nn_model: nn.Module,
    dummy: torch.Tensor,
    model_family: str | None,
):
    """Apply fixed-canvas graph preparation and always restore live state."""
    family = (model_family or "").lower()
    with ExitStack() as stack:
        if family in _DARKNET_FAMILIES:
            stack.callback(_fuse_darknet_batchnorm(nn_model))

        if family in _ANCHOR_FREEZE_FAMILIES:
            stack.callback(_freeze_anchor_grid(nn_model, dummy))

        if family in _RTDETR_STATIC_FAMILIES:
            # Snapshot first: preparation mutates resolution-specific buffers
            # and can itself fail.
            stack.callback(_snapshot_rtdetr_static_eval(nn_model))
            _prepare_rtdetr_static_eval(nn_model, dummy.shape[-2], dummy.shape[-1])

        stack.callback(_force_manual_grid_sample())
        stack.callback(_rebake_rfdetr_pos_embed(nn_model, dummy))
        stack.callback(_replace_adaptive_avg_pool(nn_model))
        yield nn_model


def _exported_output_names(exported_program) -> list[str]:
    """Declared output names of the exported graph, in graph order.

    These are the keys the runtime hands back. They are graph node names
    (``cat_33``, ``silu_120``, ...) and carry no semantic meaning, which is
    exactly why they have to be recorded rather than inferred.
    """
    try:
        return [str(name) for name in exported_program.graph_signature.user_outputs]
    except AttributeError:  # pragma: no cover - signature shape changed upstream
        logger.warning(
            "Could not read output names from the exported program; the "
            "Core AI artifact will not carry an output-name mapping."
        )
    return []


def prepare_frozen_classifier_export(
    model,
    kwargs: dict[str, Any],
    *,
    default_output: str,
) -> tuple[int, str, dict[str, Any]]:
    """Validate a frozen CLIP-style export and build standard metadata."""
    from .exporter import CoreAIExporter

    options = dict(kwargs)
    imgsz = options.pop("imgsz", None)
    output_path = options.pop("output_path", None) or default_output
    half = bool(options.pop("half", False))
    int8 = bool(options.pop("int8", False))
    data = options.pop("data", None)
    dynamic = bool(options.pop("dynamic", False))
    batch = int(options.pop("batch", 1))
    nms = bool(options.pop("nms", False))
    device = options.pop("device", None)

    # Accepted by the shared public export signature but irrelevant to the
    # direct torch.export path used by frozen vision-language classifiers.
    for name in (
        "opset",
        "simplify",
        "verbose",
        "fraction",
        "allow_download_scripts",
        "_pre_trace_hook",
    ):
        options.pop(name, None)
    if options:
        names = ", ".join(sorted(options))
        raise TypeError(f"Unsupported Core AI export options: {names}")
    if dynamic:
        raise NotImplementedError(
            "Core AI frozen-class export uses a fixed input shape; "
            "dynamic=True is not supported."
        )
    if batch != 1:
        raise NotImplementedError(
            "Core AI frozen-class export currently requires batch=1."
        )
    if device not in (None, "cpu", torch.device("cpu")):
        raise NotImplementedError(
            "Core AI conversion runs on CPU; pass device='cpu' or omit device."
        )

    if imgsz is None:
        height = width = int(model.input_size)
    elif isinstance(imgsz, (tuple, list)):
        if len(imgsz) != 2:
            raise ValueError(f"imgsz must be an int or (height, width), got {imgsz}")
        height, width = (int(imgsz[0]), int(imgsz[1]))
    else:
        height = width = int(imgsz)
    if height <= 0 or width <= 0:
        raise ValueError(f"imgsz values must be positive, got {(height, width)}")
    if height != width:
        raise NotImplementedError(
            "Frozen CLIP-style Core AI export requires a square input."
        )

    exporter = CoreAIExporter(model)
    half, int8 = exporter._validate(half, int8, data)
    exporter._preflight(
        half=half,
        int8=int8,
        data=data,
        nms=nms,
    )
    metadata = exporter._build_metadata("fp32", False, None, imgsz=(height, width))
    return height, str(output_path), metadata


def export_coreai(
    nn_model: nn.Module,
    dummy: torch.Tensor,
    *,
    output_path: str | Path,
    precision: str = "fp32",
    metadata: dict[str, Any] | None = None,
    model_family: str | None = None,
    dynamic: bool = False,
    **kwargs: Any,
) -> str:
    """Convert *nn_model* to a Core AI ``.aimodel`` asset and return its path.

    ``dummy`` fixes the export resolution. Core AI artifacts are static-shape
    in v1, matching the Core ML precedent, so the canvas the caller passes is
    the canvas the artifact runs at.
    """
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported Core AI conversion options: {names}")
    if dynamic:
        raise NotImplementedError(
            "Core AI export uses a fixed input shape; dynamic=True is not supported."
        )
    if precision != "fp32":
        raise NotImplementedError("Core AI export currently supports FP32 only.")
    TorchConverter, get_decomp_table = _require_coreai()

    # Third-party defect workarounds; see coreai_compat for what and why.
    from .coreai_compat import apply as _apply_coreai_shims

    _apply_coreai_shims()

    output_path = Path(output_path)
    if output_path.suffix != ".aimodel":
        output_path = output_path.with_suffix(".aimodel")

    # Reuse the Core ML family wrappers verbatim: they ARE the per-family
    # output contract, and a Core AI artifact must emit the same tensors as
    # the other backends for the same family or downstream consumers cannot
    # swap formats.
    # Reuse one wrapper function in conversion and parity tests so the
    # reference cannot accidentally retain YOLO-NAS's raw training maps.
    wrapped = _wrap_coreai_contract(nn_model, model_family)

    # Fixed-canvas preparation is scoped so both successful conversion and any
    # capture/lowering failure leave the caller's live model unchanged.
    with _prepare_coreai_graph(wrapped, dummy, model_family):
        logger.info("Step 1/3: Capturing the graph with torch.export")
        exported = torch.export.export(wrapped, args=(dummy,))

        logger.info("Step 2/3: Lowering to Core AI IR")
        table = dict(get_decomp_table())
        from torch._decomp import get_decompositions

        # The Core AI converter has no lowering for aten.grid_sampler_2d
        # (the deformable-attention sampler in the DETR families).
        # PyTorch core ships a reference decomposition; folding it into the
        # table lowers the op inside the exported graph into primitives the
        # converter already supports.
        table.update(get_decompositions([torch.ops.aten.grid_sampler_2d]))
        exported = exported.run_decompositions(table)
        program = TorchConverter().add_exported_program(exported).to_coreai()

    logger.info("Step 3/3: Optimizing and writing the asset")
    program.optimize()

    output_names = _exported_output_names(exported)
    meta = dict(metadata or {})
    if output_names:
        # Recorded so a backend can restore canonical ordering. See the
        # OUTPUT ORDERING CONTRACT note in the module docstring. This has to
        # reach the asset: the ordering contract is the one thing a consumer
        # cannot re-derive, and an earlier version of this function built the
        # dictionary and then dropped it, documenting a guarantee the artifact
        # did not carry.
        meta["coreai_output_names"] = output_names

    output_path.parent.mkdir(parents=True, exist_ok=True)
    program.save_asset(
        output_path, metadata=_asset_metadata(meta), minimum_os=_minimum_os()
    )

    logger.info("Core AI export complete: %s", output_path)
    if output_names:
        logger.info("Core AI output names (in graph order): %s", output_names)
    return str(output_path)


__all__ = ["export_coreai", "prepare_frozen_classifier_export"]
