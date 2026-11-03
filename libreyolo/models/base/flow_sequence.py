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
from typing import Iterable, Iterator, List, Tuple, TypeVar

T = TypeVar("T")

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
