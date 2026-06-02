"""LibreFOMO training dataset.

Supports YOLO data.yaml loading: reads images and YOLO-format box annotations
(``class cx cy w h`` normalized to [0, 1]) from a standard dataset directory,
then encodes box centers as FOMO grid targets on-the-fly.

The loader returns ``(img_tensor, grid_target, img_info, img_id)`` 4-tuples so
the base trainer's dataloader collate and batch loop work without changes.

Dataset format:

    dataset/
    ├── data.yaml          # nc, names, path, train/val/test keys
    ├── train/
    │   ├── images/        # .jpg / .png files
    │   └── labels/        # .txt files: one row per box: "class cx cy w h"
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from ...models.librefomo.utils import MEAN, STD


# ---------------------------------------------------------------------------
# Image preprocessing (mirrors reference FOMODataset.image_transform)
# ---------------------------------------------------------------------------

def _transform_image(pil_image: Image.Image, input_size: int) -> torch.Tensor:
    """Resize to input_size × input_size and normalize to [-1, 1] (RGB)."""
    img = pil_image.convert("RGB").resize((input_size, input_size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32))


# ---------------------------------------------------------------------------
# Grid-target encoding
# ---------------------------------------------------------------------------

def _boxes_xyxy_to_grid(
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    input_size: int,
    grid_size: int,
) -> torch.Tensor:
    """Encode xyxy-pixel boxes as a FOMO grid target tensor.

    Args:
        boxes_xyxy: (N, 4) float array — x1, y1, x2, y2 in *input_size* pixel coords.
        classes: (N,) int array — class index (0-based foreground).
        input_size: Model input resolution (square).
        grid_size: Output grid side length (= input_size // 8).

    Returns:
        LongTensor of shape (grid_size, grid_size).
        Value 0 = background, 1..nc = foreground class index.
    """
    grid = torch.zeros((grid_size, grid_size), dtype=torch.long)
    for (x1, y1, x2, y2), cls in zip(boxes_xyxy, classes):
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        gx = int((cx / input_size) * grid_size)
        gy = int((cy / input_size) * grid_size)
        gx = min(max(gx, 0), grid_size - 1)
        gy = min(max(gy, 0), grid_size - 1)
        grid[gy, gx] = int(cls) + 1  # +1 because 0 = background
    return grid


# ---------------------------------------------------------------------------
# Dataset (data.yaml path)
# ---------------------------------------------------------------------------

class FOMOYOLODataset(Dataset):
    """FOMO dataset backed by a YOLO-format image directory.

    Expects YOLO-format box annotations (class cx cy w h in [0,1] coords).
    The boxes are first converted to pixel coords in ``input_size`` space,
    then encoded as FOMO grid targets (box size is discarded — only the
    center cell matters for point localization).
    """

    def __init__(
        self,
        img_files: list,
        label_files: list,
        input_size: int,
        grid_size: int,
    ) -> None:
        self.img_files = img_files
        self.label_files = label_files
        self.input_size = input_size
        self.grid_size = grid_size

    def __len__(self) -> int:
        return len(self.img_files)

    def _load_labels(self, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (boxes_xyxy_pixel, classes_int) or empty arrays."""
        try:
            with open(label_path) as f:
                lines = f.read().strip().split("\n")
        except (FileNotFoundError, OSError):
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        boxes, classes = [], []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # Normalised → pixel in input_size space
            x1 = (cx - w / 2) * self.input_size
            y1 = (cy - h / 2) * self.input_size
            x2 = (cx + w / 2) * self.input_size
            y2 = (cy + h / 2) * self.input_size
            boxes.append([x1, y1, x2, y2])
            classes.append(cls)
        if not boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)

    def __getitem__(self, idx: int):
        img_path = self.img_files[idx]
        try:
            pil_img = Image.open(img_path)
        except Exception:
            pil_img = Image.fromarray(np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8))

        img_tensor = _transform_image(pil_img, self.input_size)

        label_path = self.label_files[idx] if idx < len(self.label_files) else ""
        boxes_xyxy, classes = self._load_labels(label_path)
        grid = _boxes_xyxy_to_grid(boxes_xyxy, classes, self.input_size, self.grid_size)

        img_info = (self.input_size, self.input_size)
        return img_tensor, grid, img_info, idx
