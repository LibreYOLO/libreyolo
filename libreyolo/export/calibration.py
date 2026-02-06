"""
INT8 calibration utilities for TensorRT export.

Provides a calibration data loader that feeds representative images
to TensorRT for computing quantization scales.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, List, Optional

from libreyolo.data.utils import load_data_config, get_img_files


class CalibrationDataLoader:
    """
    Calibration data provider for INT8 quantization.

    Loads images from a dataset config and provides batches of preprocessed
    numpy arrays suitable for TensorRT calibration.

    Example::

        calib_loader = CalibrationDataLoader(
            data="coco8.yaml",
            imgsz=640,
            batch=8,
            fraction=0.5,
        )
        for batch in calib_loader:
            # batch is np.ndarray of shape (B, 3, H, W), dtype float32
            ...
    """

    def __init__(
        self,
        data: str,
        imgsz: int = 640,
        batch: int = 8,
        fraction: float = 1.0,
        model_type: str = "yolov9",
    ):
        """
        Initialize calibration data loader.

        Args:
            data: Path to data.yaml configuration file or built-in dataset name.
            imgsz: Input image size (square).
            batch: Batch size for calibration.
            fraction: Fraction of dataset to use (0.0-1.0). Use smaller values
                     for faster calibration with slight accuracy tradeoff.
            model_type: Model type for preprocessing ("yolox" or "yolov9").
                       YOLOX uses BGR 0-255, YOLOv9 uses RGB 0-1.
        """
        self.imgsz = imgsz
        self.batch = batch
        self.fraction = max(0.0, min(1.0, fraction))
        self.model_type = model_type

        # Load dataset config (handles resolve, download, path resolution)
        data_config = load_data_config(data, autodownload=True)

        # Get train images (preferred for calibration - more diverse) or val
        root = Path(data_config.get("path", "."))

        # Check for pre-resolved image files (from .txt format datasets)
        if "train_img_files" in data_config:
            self.img_files = [Path(f) for f in data_config["train_img_files"]]
        elif "val_img_files" in data_config:
            self.img_files = [Path(f) for f in data_config["val_img_files"]]
        else:
            # Resolve from directory/file path - prefer train for more diversity
            train_path = data_config.get("train") or data_config.get("val")
            if train_path is None:
                raise ValueError(f"Dataset config must have 'train' or 'val' key")

            # Get image file list
            self.img_files = get_img_files(train_path, prefix=str(root))

        if len(self.img_files) == 0:
            raise ValueError(f"No images found in dataset: {data}")

        # Apply fraction
        total = len(self.img_files)
        self.num_samples = max(1, int(total * self.fraction))
        self.img_files = self.img_files[:self.num_samples]

        # Compute number of batches
        self._num_batches = (self.num_samples + self.batch - 1) // self.batch

    def _preprocess(self, img_path: Path) -> np.ndarray:
        """
        Preprocess a single image for calibration.

        Performs letterbox resize and normalization matching inference preprocessing.
        Uses model-specific preprocessing:
        - YOLOX: BGR color, 0-255 range
        - YOLOv9: RGB color, 0-1 range

        Args:
            img_path: Path to image file.

        Returns:
            Preprocessed image as CHW float32 array.
        """
        # Read image (cv2 loads as BGR)
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        # Color space: YOLOX uses BGR, YOLOv9 uses RGB
        if self.model_type != "yolox":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Letterbox resize to target size
        h, w = img.shape[:2]
        scale = min(self.imgsz / h, self.imgsz / w)
        new_h, new_w = int(h * scale), int(w * scale)

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image (gray padding like YOLO)
        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_h = (self.imgsz - new_h) // 2
        pad_w = (self.imgsz - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img

        # Convert to CHW float32
        # YOLOX: 0-255 range, YOLOv9: 0-1 range
        if self.model_type == "yolox":
            padded = padded.transpose(2, 0, 1).astype(np.float32)
        else:
            padded = padded.transpose(2, 0, 1).astype(np.float32) / 255.0

        return padded

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield batches of calibration data as numpy arrays."""
        batch_data = []

        for img_path in self.img_files:
            try:
                img = self._preprocess(img_path)
                batch_data.append(img)
            except Exception as e:
                # Skip problematic images
                print(f"Warning: Skipping {img_path}: {e}")
                continue

            if len(batch_data) == self.batch:
                yield np.stack(batch_data, axis=0)
                batch_data = []

        # Yield remaining samples (pad if needed for TensorRT)
        if batch_data:
            # Pad last batch to full size if needed
            while len(batch_data) < self.batch:
                batch_data.append(batch_data[-1].copy())
            yield np.stack(batch_data, axis=0)

    def __len__(self) -> int:
        """Return number of calibration batches."""
        return self._num_batches

    @property
    def shape(self) -> tuple:
        """Return shape of a single batch: (batch, channels, height, width)."""
        return (self.batch, 3, self.imgsz, self.imgsz)

    @property
    def dtype(self) -> np.dtype:
        """Return data type of calibration batches."""
        return np.float32


def get_calibration_dataloader(
    data: str,
    imgsz: int = 640,
    batch: int = 8,
    fraction: float = 1.0,
    model_type: str = "yolov9",
) -> CalibrationDataLoader:
    """
    Factory function for calibration data loader.

    Args:
        data: Path to data.yaml or built-in dataset name (e.g., "coco8").
        imgsz: Input image size.
        batch: Batch size for calibration.
        fraction: Fraction of dataset to use.
        model_type: Model type for preprocessing ("yolox" or "yolov9").

    Returns:
        CalibrationDataLoader instance.
    """
    return CalibrationDataLoader(data, imgsz, batch, fraction, model_type)
