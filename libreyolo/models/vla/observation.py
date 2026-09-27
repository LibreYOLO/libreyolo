"""Observation assembly for the LibreVLA tier (pure, unit-tested offline).

A VLA observation is a set of named camera frames, a proprioceptive state
vector and an instruction. This module owns the user-facing rules from ADR
0028: how a single frame or a ``{name: frame}`` dict maps onto a family's
camera slots, how ``state`` is coerced (array, callable, or missing), and
how a PIL frame becomes the ``(3, H, W)`` float tensor policies expect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
from PIL import Image

from ...utils.image_loader import ImageLoader

logger = logging.getLogger(__name__)

__all__ = [
    "Observation",
    "coerce_state",
    "frame_to_tensor",
    "load_frames",
    "map_cameras",
]


@dataclass
class Observation:
    """One policy input: frames keyed by camera slot, a state, an instruction."""

    frames: Dict[str, Image.Image]
    state: np.ndarray
    instruction: str
    path: Optional[str] = None
    frame_idx: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> Image.Image:
        """The first camera frame; ``Results.orig_shape`` comes from it."""
        return next(iter(self.frames.values()))


def map_cameras(
    frames: Mapping[str, Any],
    slots: Sequence[str],
    cameras: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Map user camera names onto a family's ordered camera slots.

    Rules (ADR 0028):

    - With ``cameras=[...]`` the i-th user name maps to the i-th slot. A key
      outside that list raises.
    - Without it, a key equal to a slot name takes that slot; every other
      key is assigned to the free slots in order.
    - More frames than slots raises; fewer is legal (families decide).
    """
    if not slots:
        raise ValueError("This family declares no camera slots.")
    if not frames:
        raise ValueError("At least one camera frame is required.")
    keys = [str(k) for k in frames]
    if len(keys) > len(slots):
        raise ValueError(
            f"{len(keys)} camera frames given but the model has "
            f"{len(slots)} camera slot(s): {list(slots)}."
        )
    mapping: Dict[str, str] = {}
    if cameras is not None:
        cameras = [str(c) for c in cameras]
        if len(cameras) > len(slots):
            raise ValueError(
                f"cameras lists {len(cameras)} names but the model has "
                f"{len(slots)} camera slot(s)."
            )
        for key in keys:
            if key not in cameras:
                raise ValueError(
                    f"Camera {key!r} is not in cameras={cameras}. Name every "
                    "camera you pass, in slot order."
                )
            mapping[key] = slots[cameras.index(key)]
    else:
        free = [s for s in slots if s not in keys]
        for key in keys:
            if key in slots:
                mapping[key] = key
            else:
                mapping[key] = free.pop(0)
    out: Dict[str, Any] = {}
    for slot in slots:
        for key, target in mapping.items():
            if target == slot:
                out[slot] = frames[key]
    return out


def load_frames(
    frames: Mapping[str, Any], color_format: str = "auto"
) -> Dict[str, Image.Image]:
    """Load every value of a slot-keyed mapping into a PIL RGB image."""
    return {
        slot: ImageLoader.load(value, color_format=color_format)
        for slot, value in frames.items()
    }


def frame_to_tensor(image: Image.Image):
    """PIL RGB frame to a ``(3, H, W)`` float32 tensor in ``[0, 1]``."""
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def coerce_state(
    state: Any,
    dim: int,
    *,
    warn: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """Turn the ``state=`` argument into a float32 ``(dim,)`` vector.

    A callable is invoked (the live-loop hook). ``None`` becomes zeros and
    reports through ``warn`` once per caller, because a policy fed a zero
    state still runs but its actions are not meaningful.
    """
    if callable(state):
        state = state()
    if state is None:
        if warn is not None:
            warn(
                "state=None: using a zero proprioceptive vector. The policy "
                "runs, but its actions are not meaningful without the real "
                "robot state."
            )
        return np.zeros(int(dim), dtype=np.float32)
    array = np.asarray(state, dtype=np.float32).reshape(-1)
    if array.shape[0] != int(dim):
        raise ValueError(
            f"state has {array.shape[0]} values but the model expects {int(dim)}."
        )
    if not np.isfinite(array).all():
        raise ValueError("state must be finite.")
    return array


def action_names_from_features(features: Mapping[str, Any]) -> Optional[List[str]]:
    """Read per-dimension action names from a LeRobot ``features`` mapping."""
    action = features.get("action") if isinstance(features, Mapping) else None
    names = action.get("names") if isinstance(action, Mapping) else None
    if isinstance(names, (list, tuple)) and all(isinstance(n, str) for n in names):
        return list(names)
    if isinstance(names, Mapping):
        # Some datasets nest names as {"motors": [...]}.
        for value in names.values():
            if isinstance(value, (list, tuple)):
                return [str(v) for v in value]
    return None
