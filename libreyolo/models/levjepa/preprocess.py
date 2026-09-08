"""Clip sampling and ImageNet preprocessing for LeVJEPA inference."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
TARGET_FPS = 7.5
DEFAULT_SOURCE_FPS = 15.0


def clip_frame_indices(
    total_frames: int,
    clip_frames: int = 16,
    *,
    source_fps: float = DEFAULT_SOURCE_FPS,
    target_fps: float = TARGET_FPS,
) -> list[int]:
    """Return a deterministic centered window sampled near the training FPS."""

    if total_frames <= 0:
        raise ValueError("cannot sample a clip from a video with no frames")
    if clip_frames <= 0:
        raise ValueError("clip_frames must be positive")
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = DEFAULT_SOURCE_FPS
    stride = max(1, int(round(source_fps / target_fps)))
    span = (clip_frames - 1) * stride + 1
    start = max(0, (total_frames - span) // 2)
    return [
        min(start + index * stride, total_frames - 1) for index in range(clip_frames)
    ]


def _resize_short_side(frame: np.ndarray, size: int) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("cannot resize a zero-sized frame")
    scale = size / min(height, width)
    output_size = (max(size, round(width * scale)), max(size, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(frame, output_size, interpolation=interpolation)


def _center_crop(frame: np.ndarray, size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    top = (height - size) // 2
    left = (width - size) // 2
    return frame[top : top + size, left : left + size]


def preprocess_frames(frames: Sequence[np.ndarray], size: int = 224) -> torch.Tensor:
    """Convert RGB uint8 HWC frames to normalized ``(1, F, C, H, W)``."""

    if not frames:
        raise ValueError("cannot build a clip from zero frames")
    processed = []
    for frame in frames:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected an RGB HWC frame, got {tuple(frame.shape)}")
        processed.append(_center_crop(_resize_short_side(frame, size), size))
    array = np.stack(processed).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2)
    mean = torch.tensor(PIXEL_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(PIXEL_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


def validate_clip_tensor(
    tensor: torch.Tensor,
    *,
    frames: int = 16,
    size: int = 224,
) -> torch.Tensor:
    if tensor.ndim != 5 or tensor.shape[2] != 3:
        raise ValueError(
            "an explicit LeVJEPA clip must use (B, F, C, H, W) with C=3; "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.shape[1] != frames or tuple(tensor.shape[-2:]) != (size, size):
        raise ValueError(
            f"LeVJEPA requires {frames} frames at {size}x{size}; got "
            f"{tuple(tensor.shape)}"
        )
    return tensor.float()
