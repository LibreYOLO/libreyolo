"""Runtime auto-conversion of upstream flagship weights to LibreYOLO format.

The two LibreYOLO flagships, YOLO9 (CNN) and RF-DETR (transformer), are ported
from MIT/Apache upstream projects whose released checkpoints are *almost*
loadable but do not carry LibreYOLO v1.0 metadata:

- **YOLO9** (MultimediaTechLab/YOLO): plain ``state_dict`` with numbered layer
  indices. The keys need structural remapping (see
  :mod:`libreyolo.models.yolo9.convert`).
- **RF-DETR** (roboflow/rf-detr): training checkpoint whose keys already match
  LibreYOLO's native port, but which embeds an ``argparse.Namespace`` (so the
  factory's safe inspection load rejects it) and carries optimizer/EMA cruft.

This module detects those upstream layouts, converts them to a strict v1.0
metadata-wrapped checkpoint, writes it next to the source under the canonical
``Libre<FAMILY><size>[-task].pt`` name, and returns the new path so the factory
can load it normally. Class count is taken from the upstream head, so
fine-tuned (non-COCO) checkpoints convert correctly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from ..tasks import task_to_suffix
from ..utils.serialization import (
    load_trusted_torch_file,
    validate_checkpoint_metadata,
    wrap_libreyolo_checkpoint,
)

logger = logging.getLogger(__name__)

# Canonical filename prefixes per flagship family (see docs/nomenclature.md).
_PREFIX = {"yolo9": "LibreYOLO9", "rfdetr": "LibreRFDETR"}


def _plain_state_dict(loaded: Any) -> dict[str, torch.Tensor]:
    """Return the tensor-only state dict across common upstream layouts."""
    obj = loaded
    if isinstance(obj, dict):
        if isinstance(obj.get("state_dict"), dict):
            obj = obj["state_dict"]
        elif isinstance(obj.get("model"), dict):
            obj = obj["model"]
    if not isinstance(obj, dict):
        return {}
    state = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    # Some redistributions nest weights under a ``model.model.`` prefix.
    if state and all(k.startswith("model.model.") for k in state):
        state = {k[len("model.model."):]: v for k, v in state.items()}
    return state


def _canonical_path(source: Path, prefix: str, size: str, task: str) -> Path:
    """Build the canonical ``Libre<FAMILY><size>[-task].pt`` path beside source."""
    suffix = task_to_suffix(task)
    task_part = f"-{suffix}" if suffix else ""
    return source.parent / f"{prefix}{size}{task_part}.pt"


def _try_yolo9(loaded: Any) -> Optional[Tuple[dict, str, str, str]]:
    """Return ``(wrapped, family, size, task)`` for an upstream YOLO9 file."""
    from .yolo9.convert import (
        convert_state_dict,
        infer_config,
        infer_nb_classes,
        is_upstream_state_dict,
    )

    state = _plain_state_dict(loaded)
    if not state or not is_upstream_state_dict(state):
        return None

    config = infer_config(state)
    if config is None:
        logger.warning(
            "Upstream YOLO9 checkpoint recognized but its size could not be "
            "inferred; skipping auto-conversion."
        )
        return None

    converted, _stats = convert_state_dict(state, config)
    nc = infer_nb_classes(state) or 80
    wrapped = wrap_libreyolo_checkpoint(
        converted, model_family="yolo9", size=config, task="detect", nc=nc,
    )
    return wrapped, "yolo9", config, "detect"


def _try_rfdetr(loaded: Any) -> Optional[Tuple[dict, str, str, str]]:
    """Return ``(wrapped, family, size, task)`` for an upstream RF-DETR file."""
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
    if raw_nc == 90:
        # COCO arch-classes (91 outputs incl. background) -> LibreYOLO's COCO-80.
        nc, names = 80, None
    else:
        nc = raw_nc if raw_nc else 80
        names = loaded.get("names") if isinstance(loaded, dict) else None

    wrapped = wrap_libreyolo_checkpoint(
        state, model_family="rfdetr", size=size, task=task, nc=nc, names=names,
    )
    return wrapped, "rfdetr", size, task


def autoconvert_upstream_checkpoint(
    model_path: str,
    *,
    loaded: Any | None = None,
) -> Optional[str]:
    """Convert an upstream flagship checkpoint to a LibreYOLO v1.0 ``.pt``.

    Args:
        model_path: Path to the (possibly upstream) checkpoint file.
        loaded: Pre-loaded checkpoint object, when the caller already has it
            from a safe load. When ``None`` the file is loaded with the trusted
            loader (needed for RF-DETR, whose ``argparse.Namespace`` breaks the
            safe loader).

    Returns:
        Path to the converted file written beside the source, or ``None`` if the
        file is not a recognized upstream YOLO9 / RF-DETR checkpoint.
    """
    path = Path(model_path)
    if not path.exists():
        return None

    if loaded is None:
        try:
            loaded = load_trusted_torch_file(
                model_path, map_location="cpu", context="upstream weights"
            )
        except Exception as exc:  # noqa: BLE001 — any load failure means we can't help
            logger.debug("Auto-conversion could not load %s: %s", model_path, exc)
            return None

    # Already a complete LibreYOLO v1.0 checkpoint — nothing to convert.
    if isinstance(loaded, dict) and not validate_checkpoint_metadata(loaded, strict=False):
        return None

    result = _try_yolo9(loaded) or _try_rfdetr(loaded)
    if result is None:
        return None

    wrapped, family, size, task = result
    out_path = _canonical_path(path, _PREFIX[family], size, task)

    # Always (re)write from the current source. The canonical name encodes only
    # family + size + task, so two different source files in one directory (e.g.
    # COCO and a custom fine-tune of the same size) share a name; reusing a
    # stale conversion would silently load the wrong weights. Conversion is
    # cheap relative to the source load that already happened.
    torch.save(wrapped, out_path)
    logger.info(
        "Converted upstream %s weights (%s) -> %s in LibreYOLO format (nc=%d).",
        family,
        path.name,
        out_path.name,
        wrapped["nc"],
    )
    return str(out_path)
