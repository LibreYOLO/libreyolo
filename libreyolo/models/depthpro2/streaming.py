#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from .encoder import KVCache
from .fov import median_focal_length_px

if TYPE_CHECKING:
    from .nn import DepthPro2Net, VideoMode

Prediction = Mapping[str, torch.Tensor | None]


def blend_ramp(length: int, like: torch.Tensor) -> torch.Tensor:
    """Weights from 1 to 0 across ``length`` frames, endpoints excluded so the seam has no step."""
    return torch.linspace(1, 0, length + 2, device=like.device, dtype=like.dtype)[
        1:-1
    ].view(-1, 1, 1)


class DepthStream:
    """A session over one video, returning each frame's prediction once, in order, never revised.

    Open one stream per video: its recent frames and focal length carry over after ``flush``.

    Buffered mode predicts clips of ``window`` frames overlapping by ``overlap`` and cross-fades the
    overlaps. Unbuffered mode predicts clips of one frame, each also attending the ``window - 1``
    frames before it through a key-value cache.
    """

    def __init__(
        self,
        model: DepthPro2Net,
        window: int,
        overlap: int,
        f_px: float | torch.Tensor | None,
        *,
        mode: VideoMode,
    ):
        """Open a stream over ``model``, in eval mode; see :meth:`DepthPro2Net.stream`."""
        if mode not in ("buffered", "unbuffered"):
            raise ValueError(
                f"mode must be 'buffered' or 'unbuffered', got {mode!r}."
            )
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}.")
        if mode == "buffered" and not 0 <= overlap < window:
            raise ValueError(
                f"overlap must be from 0 to window - 1 = {window - 1}, got {overlap}."
            )
        if mode == "unbuffered" and model.training:
            raise RuntimeError(
                "Unbuffered mode is inference only; call .eval() on the model."
            )
        self.model = model
        self.window = window
        self._f_px = (
            None
            if f_px is None
            else torch.as_tensor(f_px, dtype=torch.float32).squeeze()
        )
        self._fovs: list[torch.Tensor] = []
        self._cache = KVCache(window) if mode == "unbuffered" else None
        self._clip = window if self._cache is None else 1
        self._overlap = overlap if self._cache is None else 0
        # The frames not yet returned are the last len(_held) + _new; the _held ones are predicted.
        self._frames: deque[torch.Tensor] = deque(maxlen=self._clip)
        self._held: list[torch.Tensor] = []
        self._new = 0

    @torch.no_grad()
    def step(self, frame: torch.Tensor) -> list[Prediction]:
        """Add the next normalized frame ``(3, H, W)``; return the predictions it finishes."""
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError(
                f"step takes one frame (3, H, W), got {tuple(frame.shape)}."
            )
        if self._frames and frame.shape != self._frames[-1].shape:
            raise ValueError(
                f"step takes frames of one size, got {tuple(frame.shape)} after "
                f"{tuple(self._frames[-1].shape)}; open a new stream for frames of another size."
            )
        self._frames.append(frame)
        self._new += 1
        if len(self._held) + self._new < self._clip:
            return []
        return self._predict(len(self._frames), keep=self._overlap)

    @torch.no_grad()
    def flush(self) -> list[Prediction]:
        """End the video; return the predictions not yet returned."""
        if not self._new:  # the frames held come from a full clip, so the focal length is set
            return self._emit(len(self._held), self._f_px)
        # With overlap, the last clip is pulled back to a full window over frames already returned.
        return self._predict(
            len(self._frames) if self._overlap else self._new, keep=0
        )

    def _predict(self, length: int, keep: int) -> list[Prediction]:
        """Predict the last ``length`` frames; return those finished, all but the last ``keep``."""
        frames = list(self._frames)[-length:]
        # Stacked in the call and not kept, so padding frees the clip before the forward.
        inverse_depth = self.model.predict_inverse_depth(
            torch.stack(frames), self._cache
        )
        f_px = self._focal_length(frames)
        # Drop the frames already returned.
        inverse_depth = inverse_depth[length - len(self._held) - self._new :]
        held, self._new = len(self._held), 0
        if held:  # cross-fade, with no scale fit: its error would accumulate over a long video
            previous = torch.stack(self._held)
            ramp = blend_ramp(held, previous)
            self._held = list(ramp * previous + (1 - ramp) * inverse_depth[:held])
        self._held += list(inverse_depth[held:])
        return self._emit(len(self._held) - keep, f_px)

    def _focal_length(self, frames: list[torch.Tensor]) -> torch.Tensor | None:
        """The focal length given, else the median estimate so far, kept once a window is in."""
        if self._f_px is not None or self.model.fov is None:
            return self._f_px
        self._fovs.append(self.model.fov(torch.stack(frames)))
        fovs = torch.cat(self._fovs)
        f_px = median_focal_length_px(fovs, *frames[0].shape[-2:])
        if len(fovs) >= self.window:
            self._f_px = f_px
        return f_px

    def _emit(self, count: int, f_px: torch.Tensor | None) -> list[Prediction]:
        """Return the first ``count`` predictions held, with depth from focal length ``f_px``."""
        done, self._held = self._held[:count], self._held[count:]
        if f_px is not None and done:
            f_px = f_px.to(done[0].device)
        return [
            {
                "inverse_depth": inverse_depth,
                "depth": (
                    None
                    if f_px is None
                    else f_px / (inverse_depth.shape[-1] * inverse_depth)
                ),
                "focallength_px": f_px,
            }
            for inverse_depth in done
        ]
