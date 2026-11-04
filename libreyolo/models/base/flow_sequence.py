"""Sliding frame windows for the optical flow task.

Optical flow is defined between two frames, so a flow model never sees one
image: every source is read as an ordered sequence, and the runner hands the
family one *window* per consecutive pair. ``FLOW_CONTEXT = (before, after)``
on the family says how many extra frames the window carries on each side of
the pair; a plain two-frame model keeps the default ``(0, 0)``, a three-frame
model such as MEMFOF declares ``(1, 1)``.

Windows are edge padded: at the start of a sequence the first frame stands in
for the frames before it, at the end the last frame stands in for the frames
after it. That keeps the result count at ``N - 1`` for ``N`` frames whatever
the context, so callers never have to special-case sequence boundaries.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Iterator, List, Optional, Tuple, TypeVar

import numpy as np

T = TypeVar("T")

# Default ``imgsz`` of the flow families that run at native resolution (RAFT,
# WAFT, FlowSeek, DPFlow). For them ``imgsz`` is a cap on the long side: frames
# above it are downscaled before the network and the flow is rescaled back,
# smaller frames run untouched. 8192 keeps every realistic frame, 8K included,
# at its native size while giving the shared runner, the CLI defaults, the
# validator and the model inventory a plain integer to work with.
NATIVE_FLOW_IMGSZ = 8192


def flow_long_side_cap(max_side) -> Optional[int]:
    """Normalise an ``imgsz`` for a native-resolution flow family.

    Returns the cap as an int, or ``None`` when there is none (``None`` or a
    non-positive value). A rectangular ``(height, width)`` is rejected: for
    these families ``imgsz`` is one number, the maximum long side.
    """
    if max_side is None:
        return None
    if isinstance(max_side, (tuple, list)):
        raise ValueError(
            "imgsz for a native-resolution optical flow family is a single int, "
            f"the maximum long side in pixels; got {tuple(max_side)!r}."
        )
    cap = int(max_side)
    return cap if cap > 0 else None


def capped_hw(height: int, width: int, max_side) -> Tuple[int, int]:
    """Working canvas of a frame under a long-side cap (aspect kept, never upscaled)."""
    cap = flow_long_side_cap(max_side)
    height, width = int(height), int(width)
    if cap is None or max(height, width) <= cap:
        return height, width
    scale = cap / float(max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def downscale_rgb(img_hwc, max_side) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Return a frame at its working canvas under ``max_side`` plus ``(sx, sy)``.

    ``(sx, sy)`` are the factors that map vectors predicted on the working
    canvas back to the original canvas (``1.0`` each when nothing changed).
    Downscaling uses area interpolation; frames at or below the cap are
    returned untouched so the default path stays bit-exact.
    """
    img = np.asarray(img_hwc)
    orig_h, orig_w = img.shape[:2]
    work_h, work_w = capped_hw(orig_h, orig_w, max_side)
    if (work_h, work_w) != (orig_h, orig_w):
        import cv2

        img = cv2.resize(img, (work_w, work_h), interpolation=cv2.INTER_AREA)
    return img, (orig_w / float(work_w), orig_h / float(work_h))

FLOW_MIN_FRAMES_MESSAGE = (
    "optical flow needs at least two frames: pass a list of frames, a "
    "directory of frames, a video file, or a live source with stream=True."
)


def validate_flow_context(context: object) -> Tuple[int, int]:
    """Return ``(before, after)`` from a family's ``FLOW_CONTEXT`` or raise."""
    try:
        before, after = context  # type: ignore[misc]
    except (TypeError, ValueError):
        raise TypeError(
            "FLOW_CONTEXT must be a (before, after) pair of non-negative ints, "
            f"got {context!r}."
        ) from None
    if (
        isinstance(before, bool)
        or isinstance(after, bool)
        or not isinstance(before, int)
        or not isinstance(after, int)
        or before < 0
        or after < 0
    ):
        raise ValueError(
            "FLOW_CONTEXT must be a (before, after) pair of non-negative ints, "
            f"got {context!r}."
        )
    return int(before), int(after)


def iter_flow_windows(
    frames: Iterable[T], before: int = 0, after: int = 0
) -> Iterator[Tuple[int, List[T]]]:
    """Yield ``(t, window)`` for every consecutive pair of *frames*.

    ``window`` has ``before + 2 + after`` entries: ``window[before]`` is frame
    ``t`` and ``window[before + 1]`` is frame ``t + 1``; the entries around
    them are the neighbouring frames, edge padded at both ends of the
    sequence. Pair ``t`` is emitted as soon as frame ``t + 1 + after`` has
    been read, so live sources stream with a fixed lag of ``after`` frames and
    nothing is buffered beyond one window.

    Raises ``ValueError`` once the source turns out to hold fewer than two
    frames. For a lazy source that error surfaces while iterating, which is
    the earliest point at which the frame count is known.
    """
    before, after = validate_flow_context((before, after))
    length = before + 2 + after
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError(FLOW_MIN_FRAMES_MESSAGE) from None

    # Left padding: ``before`` copies of the first frame ahead of frame 0.
    window: deque = deque([first] * (before + 1), maxlen=length)
    count = 1
    emitted = 0
    for frame in iterator:
        window.append(frame)
        count += 1
        if len(window) == length:
            yield emitted, list(window)
            emitted += 1
    if count < 2:
        raise ValueError(FLOW_MIN_FRAMES_MESSAGE)

    # Right padding: repeat the last frame until the final pair is out.
    last = window[-1]
    while emitted < count - 1:
        window.append(last)
        if len(window) == length:
            yield emitted, list(window)
            emitted += 1


__all__ = ["FLOW_MIN_FRAMES_MESSAGE", "iter_flow_windows", "validate_flow_context"]
