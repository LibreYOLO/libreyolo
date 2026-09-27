"""Clip sampling and preprocessing for V-JEPA 2.

V-JEPA 2 consumes a *clip*, not a frame. Everything in this module exists to
turn a finite video, an image, or an explicit tensor into the public 5D layout
``(B, F, C, H, W)`` with the geometry a given checkpoint was trained for.

Preprocessing follows the pinned upstream video processor: RGB, resize the
short side, center crop to the checkpoint crop size, scale to [0, 1] and
normalize with the ImageNet statistics.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

# ImageNet statistics, matching the pinned upstream processor config.
PIXEL_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
PIXEL_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# The released 64-frame checkpoints pair with a stride of 2, which spans an
# inclusive window of 127 source frames -- the common official fixture.
DEFAULT_FRAME_STRIDE = 2


def clip_frame_indices(
    total_frames: int,
    clip_frames: int,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
) -> List[int]:
    """Deterministic centered indices for one clip.

    The span covered is ``(clip_frames - 1) * frame_stride + 1`` source frames.
    A longer video is centered on that span. A shorter one uses every real
    frame it has first and only then repeats the final frame -- repeating
    early would throw away real motion.
    """
    if total_frames <= 0:
        raise ValueError("cannot sample a clip from a video with no frames")
    if clip_frames <= 0:
        raise ValueError(f"clip_frames must be positive, got {clip_frames}")
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")

    span = (clip_frames - 1) * frame_stride + 1
    if total_frames >= span:
        start = (total_frames - span) // 2
        return [start + i * frame_stride for i in range(clip_frames)]

    # Too short for the full strided span: walk real frames, then hold the last.
    indices = [i * frame_stride for i in range(clip_frames)]
    return [min(i, total_frames - 1) for i in indices]


def resize_short_side(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize so the short side equals ``size``, preserving aspect ratio."""
    import cv2

    height, width = frame.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("cannot resize a zero-sized frame")
    scale = size / min(height, width)
    new_w = max(size, int(round(width * scale)))
    new_h = max(size, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(frame, (new_w, new_h), interpolation=interpolation)


def center_crop(frame: np.ndarray, size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if height < size or width < size:
        raise ValueError(
            f"frame {width}x{height} is smaller than the crop size {size}"
        )
    top = (height - size) // 2
    left = (width - size) // 2
    return frame[top : top + size, left : left + size]


def preprocess_frames(
    frames: Sequence[np.ndarray],
    crop_size: int,
) -> torch.Tensor:
    """RGB uint8 HWC frames -> normalized float32 ``(1, F, C, H, W)``."""
    if len(frames) == 0:
        raise ValueError("cannot build a clip from zero frames")

    processed = []
    for frame in frames:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"expected an RGB HWC frame, got shape {tuple(frame.shape)}"
            )
        resized = center_crop(resize_short_side(frame, crop_size), crop_size)
        processed.append(resized)

    clip = np.stack(processed).astype(np.float32) / 255.0  # (F, H, W, C)
    tensor = torch.from_numpy(clip).permute(0, 3, 1, 2)    # (F, C, H, W)

    mean = torch.tensor(PIXEL_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(PIXEL_STD, dtype=torch.float32).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)  # (1, F, C, H, W)


def image_to_clip(frame: np.ndarray, crop_size: int, clip_frames: int) -> torch.Tensor:
    """Represent a still image as a static clip.

    This is a compatibility behaviour, not a motion representation: every
    frame is identical, so the model sees no motion whatsoever. Only as many
    frames as the tubelet requires are produced.
    """
    repeats = max(1, clip_frames)
    return preprocess_frames([frame] * repeats, crop_size)


def validate_clip_tensor(tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Validate an explicit in-memory clip in public ``(B, F, C, H, W)`` layout."""
    if tensor.ndim != 5:
        raise ValueError(
            "an explicit V-JEPA 2 clip must be a 5D tensor in public layout "
            f"(B, F, C, H, W); got {tensor.ndim}D shape {tuple(tensor.shape)}"
        )
    if tensor.shape[2] != 3:
        raise ValueError(
            "public clip layout is (B, F, C, H, W) with C=3 at dim 2; got shape "
            f"{tuple(tensor.shape)}. A (B, C, F, H, W) tensor is a different video."
        )
    if tensor.shape[3] != crop_size or tensor.shape[4] != crop_size:
        raise ValueError(
            f"this checkpoint requires {crop_size}x{crop_size} frames; got "
            f"{tensor.shape[3]}x{tensor.shape[4]}"
        )
    return tensor.float()
