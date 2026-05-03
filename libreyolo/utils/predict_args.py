"""Predict keyword compatibility policy."""

from __future__ import annotations

import warnings


NOOP_PREDICT_KWARGS = {
    "agnostic_nms",
    "augment",
    "boxes",
    "dnn",
    "half",
    "line_width",
    "retina_masks",
    "show_conf",
    "show_labels",
    "stream_buffer",
    "verbose",
}
REJECTED_PREDICT_KWARGS = {"visualize", "embed"}
ACCEPTED_PREDICT_KWARGS = {"device"}


def normalize_predict_kwargs(kwargs: dict, passthrough: set[str] | None = None) -> dict:
    """Warn or fail for Ultralytics kwargs LibreYOLO does not implement."""
    passthrough = passthrough or set()
    remaining = dict(kwargs)

    rejected = sorted(k for k in remaining if k in REJECTED_PREDICT_KWARGS)
    if rejected:
        raise NotImplementedError(
            "LibreYOLO does not support these predict options: "
            f"{', '.join(rejected)}."
        )

    noops = sorted(k for k in remaining if k in NOOP_PREDICT_KWARGS)
    for key in noops:
        warnings.warn(
            f"Predict option {key!r} is accepted for Ultralytics compatibility "
            "but is currently a no-op in LibreYOLO.",
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
