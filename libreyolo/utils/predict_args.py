"""Predict keyword compatibility policy."""

from __future__ import annotations

import warnings
from numbers import Integral


NOOP_PREDICT_KWARGS = {
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
#: Tasks ``predict(agnostic_nms=True)`` covers: the ones that return boxes.
AGNOSTIC_NMS_TASKS = ("detect", "segment", "pose", "obb")
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


# predict()'s default max_det, and the candidate budget every family's
# postprocess supports.
DEFAULT_MAX_DET = 300


def postprocess_max_det(max_det: int, classes) -> int:
    """Return the ``max_det`` to pass to a postprocess that runs before the
    ``classes`` filter.

    Postprocess keeps the top ``max_det`` detections over every class, so with
    a class filter a small ``max_det`` could keep only other classes. Keep at
    least the default budget there and cut to ``max_det`` after filtering.
    """
    if classes is None:
        return max_det
    return max(int(max_det), DEFAULT_MAX_DET)


def check_agnostic_nms(agnostic_nms, task) -> bool:
    """Validate ``agnostic_nms`` for a task and return it as a bool.

    Class-agnostic NMS suppresses overlapping boxes, so it applies to the
    tasks that return boxes; asking for it elsewhere is an error rather than
    a silent no-op.
    """
    if not agnostic_nms:
        return False
    if task not in AGNOSTIC_NMS_TASKS:
        raise ValueError(
            f"agnostic_nms=True is not supported for task '{task}'; it covers "
            f"{', '.join(AGNOSTIC_NMS_TASKS)}"
        )
    return True


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

    # A predict path that implements agnostic_nms takes it as a named
    # argument, so it only reaches here on a path that does not: fine when
    # off, an error when asked for.
    if remaining.pop("agnostic_nms", False):
        raise NotImplementedError(
            "agnostic_nms=True is not supported by this model's predict path."
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
