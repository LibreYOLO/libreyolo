"""Original-canvas transforms for Marigold V2 dense predictions.

Apache-2.0 input recipe adapted from huawei-bayerlab/marigold-v2
cc6a7031abcd59fd9e1ceff7fdd0d9687d389bc5, Copyright 2026 Huawei Technologies.
"""

import cv2
import numpy as np
import torch


def preprocess_numpy(image, input_size=0):
    height, width = image.shape[:2]
    if input_size is None or input_size == 0:
        target_h, target_w = (height + 15) // 16 * 16, (width + 15) // 16 * 16
        native = True
    elif isinstance(input_size, (list, tuple)) and len(input_size) == 2:
        target_h, target_w = map(int, input_size)
        native = False
    elif isinstance(input_size, int):
        target_h = target_w = input_size
        native = False
    else:
        raise ValueError(
            "Marigold V2 imgsz must be 0, a positive int, or (height, width)."
        )
    if min(target_h, target_w) <= 0 or target_h % 16 or target_w % 16:
        raise ValueError(
            "Marigold V2 image dimensions must be positive multiples of 16."
        )
    # The upstream reader normalizes in float64, then its resize casts to
    # float32. Resizing 8-bit RGB first has different rounding and sharp edges.
    normalized = np.asarray(image, np.uint8) / 255.0 * 2.0 - 1.0
    if not native or (target_h, target_w) != (height, width):
        normalized = cv2.resize(
            normalized.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4,
        )
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)), 1.0


def output_map(decoded, task):
    """Convert the native BCHW decoder output, retaining its model precision."""
    if task == "depth":
        return decoded.mean(dim=1, keepdim=True)
    if task == "normal":
        raw = decoded.float()
        norm = torch.linalg.vector_norm(raw, dim=1, keepdim=True)
        normal = torch.where(norm > 1e-6, raw / norm.clamp_min(1e-6), 0.0)
        return normal.to(decoded.dtype)
    if task == "albedo":
        return ((decoded.float() + 1.0) * 0.5).to(decoded.dtype)
    raise ValueError(f"Unsupported Marigold V2 task: {task}")


def resize_output(value, original_size):
    """Resize CHW floats with the upstream OpenCV bilinear convention."""
    array = value.detach().float().cpu().numpy()
    width, height = original_size
    if array.shape[-2:] == (height, width):
        return array
    return np.stack(
        [cv2.resize(c, (width, height), interpolation=cv2.INTER_LINEAR) for c in array]
    )
