"""Runtime auto-conversion of upstream checkpoints to LibreYOLO format.

LibreYOLO's model families are ported from MIT/Apache upstream projects whose
released checkpoints are *almost* loadable but do not carry LibreYOLO v1.0
metadata (family, size, task, class count, class names). When the factory
meets such a file it calls :func:`autoconvert_upstream_checkpoint`, which:

1. unwraps the tensor dict from the common upstream layouts
   (``ema.module`` / ``ema_state_dict`` / ``ema_net`` / ``net`` / ``model`` /
   ``state_dict`` / plain),
2. asks every registered family — via
   :meth:`BaseModel.convert_upstream_state_dict` — whether it recognizes the
   layout, remapping keys where the upstream naming differs from the native
   port (YOLO9, RT-DETR/v2/v4, PicoDet, RTMDet),
3. wraps the winner in a strict v1.0 metadata checkpoint (size, task and class
   count read from the tensors themselves, so fine-tuned checkpoints convert
   correctly), and
4. writes it beside the source as ``<source>-<Prefix><size>[-task].pt`` and
   returns the new path so the factory can load it normally.

RF-DETR keeps a bespoke recognizer because it needs the full checkpoint (not
just the tensor dict) for size detection and COCO class remapping, and is only
lazily registered when its optional dependencies are installed.

Sibling families that share architecture keys are disambiguated the same way
the factory does: upstream-style filename hints first, and a hard refusal for
the DEIM/D-FINE tie that nothing can break.
"""

from __future__ import annotations

import argparse
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from ..tasks import task_to_suffix
from ..utils.serialization import (
    REQUIRED_CHECKPOINT_METADATA_KEYS,
    CheckpointMetadataError,
    load_untrusted_torch_file,
    validate_checkpoint_metadata,
    wrap_libreyolo_checkpoint,
)

logger = logging.getLogger(__name__)

_UPSTREAM_SAFE_GLOBALS = (argparse.Namespace,)

# Families the generic recognizer never claims. L2CS is inference-only with
# redistribution-restricted weights; RF-DETR has its own recognizer below.
_SKIP_FAMILIES = frozenset({"l2cs", "rfdetr"})


# ---------------------------------------------------------------------------
# Checkpoint unwrapping and metadata extraction
# ---------------------------------------------------------------------------


def _candidate_tensor_dicts(loaded: Any):
    """Yield possible weight dicts in EMA-first preference order.

    Each candidate is tried until one actually holds tensors, so an empty or
    metadata-only ``ema`` block does not mask valid weights under ``model``.
    """
    if not isinstance(loaded, dict):
        return
    ema = loaded.get("ema")
    if isinstance(ema, dict):
        if isinstance(ema.get("module"), dict):
            yield ema["module"]
        else:
            # Legacy flat EMA wrappers store the tensors directly.
            yield ema
    ema_state = loaded.get("ema_state_dict")
    if isinstance(ema_state, dict):
        # mmengine ExpMomentumEMA prefixes module params with "module.".
        yield {
            k[len("module."):]: v
            for k, v in ema_state.items()
            if k.startswith("module.")
        } or ema_state
    for key in ("ema_net", "net", "model", "state_dict"):
        if isinstance(loaded.get(key), dict):
            yield loaded[key]
    yield loaded


def _plain_state_dict(loaded: Any) -> dict[str, torch.Tensor]:
    """Return the tensor-only state dict across common upstream layouts."""
    for candidate in _candidate_tensor_dicts(loaded):
        state = {k: v for k, v in candidate.items() if isinstance(v, torch.Tensor)}
        if not state:
            continue
        # Strip DDP / torch.compile wrappers some redistributions keep.
        for prefix in ("module.", "_orig_mod."):
            if any(k.startswith(prefix) for k in state):
                state = {
                    (k[len(prefix):] if k.startswith(prefix) else k): v
                    for k, v in state.items()
                }
        # Some redistributions nest weights under a ``model.model.`` prefix.
        if all(k.startswith("model.model.") for k in state):
            state = {k[len("model.model."):]: v for k, v in state.items()}
        return state
    return {}


def _has_partial_metadata(loaded: Any) -> bool:
    """True when the checkpoint carries any LibreYOLO metadata key."""
    if not isinstance(loaded, dict):
        return False
    metadata_keys = set(REQUIRED_CHECKPOINT_METADATA_KEYS) - {"model"}
    return bool(metadata_keys & set(loaded))


def _checkpoint_names(loaded: Any, nc: int | None = None) -> Any | None:
    """Extract class names from common upstream checkpoint metadata."""
    if not isinstance(loaded, dict):
        return None
    names = loaded.get("names")
    if names is not None:
        return names

    args = loaded.get("args") or loaded.get("hyper_parameters") or {}
    class_names = (
        args.get("class_names")
        if isinstance(args, dict)
        else getattr(args, "class_names", None)
    )
    if class_names is None:
        return None
    if isinstance(class_names, dict):
        names = {int(key): str(value) for key, value in class_names.items()}
        if nc is not None:
            return {key: value for key, value in names.items() if key < nc}
        return names

    names = list(class_names)
    return names[:nc] if nc is not None else names


def _safe_metadata_value(value: Any) -> Any | None:
    """Return a safe-loader-compatible metadata value, or ``None`` if unsafe."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            safe_item = _safe_metadata_value(item)
            if safe_item is not None:
                safe[str(key)] = safe_item
        return safe
    if isinstance(value, (list, tuple)):
        safe_items = []
        for item in value:
            safe_item = _safe_metadata_value(item)
            if safe_item is not None:
                safe_items.append(safe_item)
        return safe_items
    if isinstance(value, argparse.Namespace):
        return _safe_metadata_value(vars(value))
    return None


def _checkpoint_args(loaded: Any) -> dict[str, Any] | None:
    """Extract upstream args as plain metadata safe for weights-only loading."""
    if not isinstance(loaded, dict):
        return None
    raw_args = loaded.get("args") or loaded.get("hyper_parameters")
    safe_args = _safe_metadata_value(raw_args)
    if isinstance(safe_args, dict) and safe_args:
        class_names = safe_args.get("class_names")
        if isinstance(class_names, dict):
            indexed_names = []
            for key, value in class_names.items():
                try:
                    indexed_names.append((int(key), str(value)))
                except (TypeError, ValueError):
                    indexed_names = []
                    break
            indexes = [index for index, _value in sorted(indexed_names)]
            if indexes and indexes == list(range(indexes[-1] + 1)):
                safe_args["class_names"] = [
                    value for _index, value in sorted(indexed_names)
                ]
            else:
                safe_args.pop("class_names", None)
        return safe_args
    return None


def _metadata_value(loaded: Any, name: str) -> Any:
    if not isinstance(loaded, dict):
        return None
    if name in loaded:
        return loaded[name]
    args = loaded.get("args") or loaded.get("hyper_parameters") or {}
    if isinstance(args, dict):
        return args.get(name)
    return getattr(args, name, None)


def _name_count(names: Any) -> int | None:
    if isinstance(names, (dict, list, tuple)):
        return len(names)
    return None


# ---------------------------------------------------------------------------
# Loading upstream files
# ---------------------------------------------------------------------------


class _InertStub:
    """Inert stand-in for a third-party class pickled into a checkpoint.

    Construction, ``__setstate__`` and attribute state are all no-ops, so the
    unpickler can materialize the object graph without executing third-party
    code. Only tensors survive into the converted checkpoint; stub instances
    (training metadata such as mm-series config/log objects) are discarded.
    """

    def __new__(cls, *args, **kwargs):  # noqa: D102 — pickle may pass args
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


_BLOCKED_GLOBAL_RE = re.compile(r"GLOBAL ([A-Za-z_][\w.]*) was not an allowed global")


def _stub_for_blocked_global(exc: Exception) -> type | None:
    """Fabricate an inert stand-in for the global a safe-load error names.

    The stub shadows the real class in the ``weights_only`` allowlist (torch
    resolves allowlisted globals by ``module.qualname`` to the object we
    provide), so the pickle can never reach real third-party or builtin
    callables. ``builtins`` globals are never stubbed.
    """
    message = str(exc)
    if exc.__cause__ is not None:
        message += "\n" + str(exc.__cause__)
    match = _BLOCKED_GLOBAL_RE.search(message)
    if match is None:
        return None
    module, _, qualname = match.group(1).rpartition(".")
    if not module or module == "builtins":
        return None
    stub = type(qualname, (_InertStub,), {})
    stub.__module__ = module
    stub.__qualname__ = qualname
    return stub


def _load_upstream_file(model_path: str) -> Any:
    """Safe-load an upstream checkpoint, tolerating pickled config objects.

    Some upstream training checkpoints (e.g. mm-series RTMDet) embed library
    objects the ``weights_only`` loader rejects. Those objects are metadata we
    do not need, so each blocked global is retried with an inert stub class
    that satisfies the unpickler without executing anything.
    """
    stubs: list[type] = []
    for _attempt in range(32):
        try:
            return load_untrusted_torch_file(
                model_path,
                map_location="cpu",
                context="upstream weights",
                safe_globals=_UPSTREAM_SAFE_GLOBALS + tuple(stubs),
            )
        except Exception as exc:
            stub = _stub_for_blocked_global(exc)
            if stub is None:
                raise
            stubs.append(stub)
    raise RuntimeError(
        f"Gave up stubbing pickled globals in {model_path} after 32 attempts."
    )


# ---------------------------------------------------------------------------
# Family recognition
# ---------------------------------------------------------------------------


def _candidate_classes() -> list:
    from .base import BaseModel

    return [cls for cls in BaseModel._registry if cls.FAMILY not in _SKIP_FAMILIES]


def _claim_upstream_state(
    state: dict[str, torch.Tensor],
    *,
    partial_metadata: bool,
) -> list[tuple[type, dict]]:
    """Collect ``(model_class, native_state_dict)`` claims in registry order.

    Files that already carry partial LibreYOLO metadata (``names``, ``nc``…)
    may be old LibreYOLO trainer checkpoints that belong to the factory's
    legacy compatibility path. Those always hold native keys, so when partial
    metadata is present only claims whose conversion actually changed the
    keyset — proof of an upstream layout — are accepted.
    """
    claims = []
    for cls in _candidate_classes():
        try:
            converted = cls.convert_upstream_state_dict(state)
        except Exception as exc:  # noqa: BLE001 — one family must not block the rest
            logger.debug("%s upstream recognition failed: %s", cls.FAMILY, exc)
            continue
        if not converted:
            continue
        if partial_metadata and converted.keys() == state.keys():
            continue
        claims.append((cls, converted))
    return claims


def _resolve_claim(
    claims: list[tuple[type, dict]],
    source: Path,
) -> Optional[tuple[type, dict]]:
    """Pick one claim, mirroring the factory's dispatch rules.

    Upstream-style filename hints win first. A subclass claim then beats its
    base class — registration order follows class creation, so a derived
    family (RT-DETRv4) registers *after* the base (D-FINE) it refines, and
    its positive markers must not lose to the base's broader passthrough.
    The DEIM/D-FINE tie is unresolvable from tensors alone (identical
    architecture keys), so it is refused unless EC, DEIMv2 or RT-DETRv4 —
    whose more-specific detectors legitimately also match those decoder
    keys — claimed the file.
    """
    if not claims:
        return None

    for cls, converted in claims:
        if cls.detect_size_from_filename(source.name):
            return cls, converted

    claims = [
        (cls, converted)
        for cls, converted in claims
        if not any(
            other is not cls and issubclass(other, cls) for other, _state in claims
        )
    ]

    families = {cls.FAMILY for cls, _converted in claims}
    # EC, DEIMv2 and RT-DETRv4 legitimately match D-FINE/DEIM-ish decoder keys
    # but are positively identified by their own markers, so they break the tie.
    if {"dfine", "deim"}.issubset(families) and not (
        families & {"ec", "deimv2", "rtdetrv4"}
    ):
        logger.warning(
            "Ambiguous D-FINE/DEIM upstream checkpoint %s: both families share "
            "the same architecture keys; skipping auto-conversion. Use an "
            "upstream-style filename such as dfine_hgnetv2_n_coco.pth or "
            "deim_hgnetv2_n_coco.pth, or instantiate LibreDFINE/LibreDEIM "
            "directly.",
            source.name,
        )
        return None

    return claims[0]


def _wrap_claim(
    cls: type,
    converted: dict[str, torch.Tensor],
    loaded: Any,
    source: Path,
) -> Optional[Tuple[dict, str, str, str, str]]:
    """Build ``(wrapped, family, prefix, size, task)`` for a resolved claim."""
    size = cls.detect_size(converted) or cls.detect_size_from_filename(source.name)
    if size is None:
        logger.warning(
            "Upstream %s checkpoint recognized but its size could not be "
            "inferred; skipping auto-conversion.",
            cls.FAMILY,
        )
        return None

    task = (
        cls.detect_checkpoint_task(converted)
        or cls.detect_task_from_filename(source.name)
        or cls.DEFAULT_TASK
    )
    nc = cls.detect_nb_classes(converted) or 80
    names = _checkpoint_names(loaded, nc)
    extra_metadata: dict[str, Any] = {}
    if task == "pose":
        num_keypoints = None
        detect_keypoints = getattr(cls, "detect_num_keypoints", None)
        if callable(detect_keypoints):
            num_keypoints = detect_keypoints(converted)
        if num_keypoints is None:
            num_keypoints = getattr(cls, "POSE_NUM_KEYPOINTS", None)
        if num_keypoints:
            extra_metadata["num_keypoints"] = int(num_keypoints)
            # Upstream pose releases are COCO-trained: x,y,visibility labels.
            extra_metadata["keypoint_dim"] = 3
    converted = {
        k: (v.float() if v.is_floating_point() else v) for k, v in converted.items()
    }
    try:
        wrapped = wrap_libreyolo_checkpoint(
            converted,
            model_family=cls.FAMILY,
            size=size,
            task=task,
            nc=nc,
            names=names,
            **extra_metadata,
        )
    except CheckpointMetadataError as exc:
        logger.warning(
            "Upstream %s checkpoint recognized but could not be wrapped "
            "(size=%s, task=%s): %s",
            cls.FAMILY,
            size,
            task,
            exc,
        )
        return None
    return wrapped, cls.FAMILY, cls.FILENAME_PREFIX, size, task


# ---------------------------------------------------------------------------
# RF-DETR — bespoke recognizer (lazy registration, checkpoint-level metadata)
# ---------------------------------------------------------------------------


def _is_coco_rfdetr_checkpoint(loaded: Any) -> bool:
    """Return True only when metadata supports RF-DETR COCO remapping."""
    names = _checkpoint_names(loaded)
    if _name_count(names) == 80:
        return True

    for field in ("dataset", "dataset_file", "dataset_name", "data"):
        value = _metadata_value(loaded, field)
        if isinstance(value, str) and "coco" in value.lower():
            return True
    return False


def _rfdetr_class_metadata(
    loaded: Any,
    raw_nc: int | None,
) -> tuple[int, Any | None]:
    """Resolve RF-DETR public class metadata without guessing custom 90-class heads."""
    if raw_nc == 90 and _is_coco_rfdetr_checkpoint(loaded):
        # COCO arch-classes (91 outputs incl. background) -> LibreYOLO's COCO-80.
        return 80, _checkpoint_names(loaded, 80)

    nc = raw_nc if raw_nc else 80
    return nc, _checkpoint_names(loaded, nc)


def _try_rfdetr(loaded: Any) -> Optional[Tuple[dict, str, str, str, str]]:
    """Return ``(wrapped, family, prefix, size, task)`` for an upstream RF-DETR file."""
    from . import try_ensure_rfdetr

    rfdetr_cls = try_ensure_rfdetr()
    if rfdetr_cls is None:
        return None

    from .rfdetr.model import _checkpoint_model_state

    state = _checkpoint_model_state(loaded)
    if not state:
        return None

    # Require RF-DETR-specific markers so RT-DETR/D-FINE checkpoints (which share
    # encoder/decoder-ish keys) are not misclaimed.
    keys_lower = [k.lower() for k in state]
    is_rfdetr = any(
        "dinov2" in k or "query_embed" in k or "enc_out_class_embed" in k for k in keys_lower
    ) or ("class_embed.bias" in state and any(k.startswith("backbone.0") for k in state))
    if not is_rfdetr:
        return None

    size = rfdetr_cls.detect_size(state, state_dict=loaded)
    if size is None:
        logger.warning(
            "Upstream RF-DETR checkpoint recognized but its size could not be "
            "inferred; skipping auto-conversion."
        )
        return None

    task = "segment" if any(k.startswith("segmentation_head") for k in state) else "detect"

    raw_nc = rfdetr_cls.detect_nb_classes(state)
    nc, names = _rfdetr_class_metadata(loaded, raw_nc)
    extra_metadata: dict[str, Any] = {}
    args = _checkpoint_args(loaded)
    if args is not None:
        extra_metadata["args"] = args

    wrapped = wrap_libreyolo_checkpoint(
        state,
        model_family="rfdetr",
        size=size,
        task=task,
        nc=nc,
        names=names,
        **extra_metadata,
    )
    return wrapped, "rfdetr", rfdetr_cls.FILENAME_PREFIX, size, task


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _canonical_path(source: Path, prefix: str, size: str, task: str) -> Path:
    """Build a source-specific converted checkpoint path beside source."""
    suffix = task_to_suffix(task)
    task_part = f"-{suffix}" if suffix else ""
    return source.parent / f"{source.stem}-{prefix}{size}{task_part}.pt"


def autoconvert_upstream_checkpoint(
    model_path: str,
    *,
    loaded: Any | None = None,
) -> Optional[str]:
    """Convert an upstream checkpoint to a LibreYOLO v1.0 ``.pt``.

    Args:
        model_path: Path to the (possibly upstream) checkpoint file.
        loaded: Pre-loaded checkpoint object, when the caller already has it
            from a safe load. When ``None`` the file is loaded through the safe
            loader with the minimal upstream allowlist (plus inert stubs for
            pickled third-party config objects).

    Returns:
        Path to the converted file written beside the source, or ``None`` if
        the file is not a recognized upstream checkpoint of any registered
        family.
    """
    path = Path(model_path)
    if not path.exists():
        return None

    if loaded is None:
        try:
            loaded = _load_upstream_file(model_path)
        except Exception as exc:  # noqa: BLE001 — any load failure means we can't help
            logger.debug("Auto-conversion could not load %s: %s", model_path, exc)
            return None

    # Already a complete LibreYOLO v1.0 checkpoint — nothing to convert.
    if isinstance(loaded, dict) and not validate_checkpoint_metadata(loaded, strict=False):
        return None

    result = None
    state = _plain_state_dict(loaded)
    if state:
        claims = _claim_upstream_state(
            state, partial_metadata=_has_partial_metadata(loaded)
        )
        chosen = _resolve_claim(claims, path)
        if chosen is not None:
            result = _wrap_claim(chosen[0], chosen[1], loaded, path)
    if result is None:
        result = _try_rfdetr(loaded)
    if result is None:
        return None

    wrapped, family, prefix, size, task = result
    out_path = _canonical_path(path, prefix, size, task)

    # Always (re)write the source-specific conversion. This keeps repeated loads
    # of the same source fresh while avoiding collisions with official weights
    # or other fine-tunes of the same family/size/task in the directory.
    try:
        torch.save(wrapped, out_path)
    except OSError as exc:
        # Read-only source directory (e.g. a mounted cache). The converted
        # checkpoint is the only loadable form for remapped families, so fall
        # back to the temp dir rather than dropping the conversion.
        try:
            fallback_dir = Path(tempfile.gettempdir()) / "libreyolo-autoconvert"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            fallback_path = fallback_dir / out_path.name
            torch.save(wrapped, fallback_path)
        except OSError as fallback_exc:
            logger.warning(
                "Recognized upstream %s checkpoint but could not write %s (%s) "
                "or the temp-dir fallback (%s).",
                family,
                out_path,
                exc,
                fallback_exc,
            )
            return None
        logger.info(
            "Could not write %s (%s); wrote the converted checkpoint to %s instead.",
            out_path,
            exc,
            fallback_path,
        )
        out_path = fallback_path
    logger.info(
        "Converted upstream %s weights (%s) -> %s in LibreYOLO format (nc=%d).",
        family,
        path.name,
        out_path.name,
        wrapped["nc"],
    )
    return str(out_path)
