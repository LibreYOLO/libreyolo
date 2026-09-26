"""Predict keyword compatibility policy."""

from __future__ import annotations

import warnings
from numbers import Integral


NOOP_PREDICT_KWARGS = {
    "agnostic_nms",
    "boxes",
    "dnn",
    "half",
    "line_width",
    "retina_masks",
    "show_conf",
    "show_labels",
    "verbose",
}
REJECTED_PREDICT_KWARGS = {"visualize", "embed"}
ACCEPTED_PREDICT_KWARGS = {
    "classes",
    "conf",
    "device",
    "imgsz",
    "iou",
    "max_det",
    "augment",
    "save",
    "stream",
    "stream_buffer",
    "vid_stride",
}


def normalize_classes(classes):
    """Accept a single class id (``classes=0``) as a one-element list."""
    if isinstance(classes, Integral) and not isinstance(classes, bool):
        return [int(classes)]
    return classes


def normalize_predict_kwargs(kwargs: dict, passthrough: set[str] | None = None) -> dict:
    """Warn or fail for predict kwargs LibreYOLO does not implement."""
    passthrough = passthrough or set()
    remaining = dict(kwargs)

    rejected = sorted(k for k in remaining if k in REJECTED_PREDICT_KWARGS)
    if rejected:
        raise NotImplementedError(
            f"LibreYOLO does not support these predict options: {', '.join(rejected)}."
        )

    noops = sorted(k for k in remaining if k in NOOP_PREDICT_KWARGS)
    for key in noops:
        warnings.warn(
            f"Predict option {key!r} is accepted for CLI compatibility but is "
            "currently a no-op in LibreYOLO.",
            stacklevel=3,
        )
        remaining.pop(key, None)

    for key in ACCEPTED_PREDICT_KWARGS:
        remaining.pop(key, None)

    forwarded = {}
    for key in sorted(passthrough):
        if key in remaining:
            forwarded[key] = remaining.pop(key)

    if remaining:
        raise TypeError(
            "Unsupported predict option(s): "
            f"{', '.join(sorted(remaining))}. "
            "Supported options include conf, iou, imgsz, device, classes, "
            "max_det, save, stream, and vid_stride."
        )

    return forwarded
