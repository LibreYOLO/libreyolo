"""FOMO training dataset."""

from __future__ import annotations

import random
from typing import Any, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .utils import MEAN, STD
from ...training.augment import augment_hsv, mirror, xyxy2cxcywh
from ...data.utils import polygon_to_cxcywh


def _transform_image(pil_image: Image.Image, input_size: int) -> torch.Tensor:
    """Resize to input_size × input_size and normalize to [-1, 1] (RGB)."""
    img = pil_image.convert("RGB").resize((input_size, input_size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32))


def _boxes_xyxy_to_grid(
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    input_size: int,
    grid_size: int,
) -> torch.Tensor:
    """Encode xyxy-pixel boxes as a FOMO grid target tensor."""
    grid = torch.zeros((grid_size, grid_size), dtype=torch.long)
    for (x1, y1, x2, y2), cls in zip(boxes_xyxy, classes):
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        gx = int((cx / input_size) * grid_size)
        gy = int((cy / input_size) * grid_size)
        gx = min(max(gx, 0), grid_size - 1)
        gy = min(max(gy, 0), grid_size - 1)
        grid[gy, gx] = int(cls) + 1
    return grid


class FOMOYOLODataset(Dataset):
    """FOMO dataset backed by a YOLO-format image directory."""

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
        from ...utils.input_channels import active_dataset_channels

        self.input_channels = active_dataset_channels()

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
            if len(parts) > 5:
                coords = [float(p) for p in parts[1:]]
                cx, cy, w, h = polygon_to_cxcywh(coords)
            else:
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
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
            with Image.open(img_path) as pil_img:
                if getattr(self, "input_channels", 3) == 1:
                    pil_img = pil_img.convert("L").convert("RGB")
                img_tensor = _transform_image(pil_img, self.input_size)
        except (FileNotFoundError, OSError) as e:
            raise FileNotFoundError(f"Cannot read image: {img_path}") from e

        label_path = self.label_files[idx] if idx < len(self.label_files) else ""
        boxes_xyxy, classes = self._load_labels(label_path)
        grid = _boxes_xyxy_to_grid(boxes_xyxy, classes, self.input_size, self.grid_size)

        img_info = (self.input_size, self.input_size)
        return img_tensor, grid, img_info, idx


def _boxes_cxcy_to_grid(
    boxes_cxcy: np.ndarray,
    classes: np.ndarray,
    input_size: int,
    grid_size: int,
) -> torch.Tensor:
    """Encode cxcy-pixel box centers as a FOMO grid target tensor."""
    grid = torch.zeros((grid_size, grid_size), dtype=torch.long)
    for (cx, cy), cls in zip(boxes_cxcy, classes):
        gx = int((cx / input_size) * grid_size)
        gy = int((cy / input_size) * grid_size)
        gx = min(max(gx, 0), grid_size - 1)
        gy = min(max(gy, 0), grid_size - 1)
        grid[gy, gx] = int(cls) + 1
    return grid


class FOMOAugmentedDataset(Dataset):
    """Dataset wrapper for FOMO that consumes YOLOX-style augmented targets."""

    def __init__(
        self,
        augmented_dataset: Dataset,
        input_size: int,
        grid_size: int,
    ) -> None:
        self.dataset = augmented_dataset
        self.input_size = input_size
        self.grid_size = grid_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, targets, img_info, img_id = self.dataset[idx]

        img_rgb = img[::-1, :, :]  # BGR -> RGB
        img_normalized = (img_rgb / 255.0 - MEAN[:, None, None]) / STD[:, None, None]
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_normalized, dtype=np.float32))

        valid_mask = targets[:, 3] > 0
        valid_targets = targets[valid_mask]

        if len(valid_targets) > 0:
            classes = valid_targets[:, 0].astype(np.int64)
            boxes_cxcy = valid_targets[:, 1:3]
            grid = _boxes_cxcy_to_grid(boxes_cxcy, classes, self.input_size, self.grid_size)
        else:
            grid = torch.zeros((self.grid_size, self.grid_size), dtype=torch.long)

        return img_tensor, grid, img_info, img_id

    def close_mosaic(self) -> None:
        if hasattr(self.dataset, "close_mosaic"):
            self.dataset.close_mosaic()


class FOMOTrainTransform:
    """Transform for FOMO training data using direct stretch-resizing."""

    def __init__(self, max_labels: int = 50, flip_prob: float = 0.5, hsv_prob: float = 1.0) -> None:
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    @property
    def wants_unresized_image(self) -> bool:
        return True

    def __call__(
        self, image: np.ndarray, targets: np.ndarray, input_dim: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()

        if len(boxes) == 0:
            targets_t = np.zeros((self.max_labels, 5), dtype=np.float32)
            image_resized = cv2.resize(
                image,
                (input_dim[1], input_dim[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            image_chw = image_resized.transpose((2, 0, 1))
            image_chw = np.ascontiguousarray(image_chw, dtype=np.float32)
            return image_chw, targets_t

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4].copy()
        labels_o = targets_o[:, 4].copy()

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        image_t, boxes = mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape

        image_resized = cv2.resize(
            image_t,
            (input_dim[1], input_dim[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        image_chw = image_resized.transpose((2, 0, 1))
        image_chw = np.ascontiguousarray(image_chw, dtype=np.float32)

        scale_x = input_dim[1] / width
        scale_y = input_dim[0] / height

        boxes = xyxy2cxcywh(boxes)
        boxes[:, 0::2] *= scale_x
        boxes[:, 1::2] *= scale_y

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_resized = cv2.resize(
                image_o,
                (input_dim[1], input_dim[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            image_chw = image_resized.transpose((2, 0, 1))
            image_chw = np.ascontiguousarray(image_chw, dtype=np.float32)

            scale_x_o = input_dim[1] / width_o
            scale_y_o = input_dim[0] / height_o
            boxes_o = xyxy2cxcywh(boxes_o)
            boxes_o[:, 0::2] *= scale_x_o
            boxes_o[:, 1::2] *= scale_y_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_chw, padded_labels


def build_fomo_datasets(
    config: Any,
    model: Any,
    wrapper_model: Any,
    device: torch.device,
    input_size: int,
    grid_size: int,
) -> Tuple[Dataset, Dataset, Any, Any]:
    from ...data import load_data_config, get_img_files, img2label_paths
    from .loss import FOMOLoss
    from ...training.distributed import is_main_process
    import logging
    logger = logging.getLogger("libreyolo.models.fomo.dataset")

    data_cfg = load_data_config(
        config.data,
        allow_scripts=config.allow_download_scripts,
    )
    data_dir = data_cfg["root"]

    train_img_files = data_cfg.get("train_img_files")
    train_label_files = data_cfg.get("train_label_files")
    if not train_img_files:
        train_path = data_cfg.get("train", "images/train")
        train_img_files = get_img_files(train_path, prefix=data_dir)
        train_label_files = img2label_paths(train_img_files)
    elif train_label_files is None:
        train_label_files = img2label_paths(train_img_files)

    val_img_files = data_cfg.get("val_img_files")
    val_label_files = data_cfg.get("val_label_files")
    if not val_img_files:
        val_path = data_cfg.get("val", "images/val")
        try:
            val_img_files = get_img_files(val_path, prefix=data_dir)
            val_label_files = img2label_paths(val_img_files)
        except (FileNotFoundError, ValueError):
            val_img_files, val_label_files = [], []
    elif val_label_files is None:
        val_label_files = img2label_paths(val_img_files)

    dataset_nc = data_cfg.get("nc")
    if dataset_nc is None and data_cfg.get("names") is not None:
        dataset_nc = len(data_cfg["names"])
    dataset_nc = int(dataset_nc) if dataset_nc is not None else config.num_classes

    if dataset_nc != getattr(model, "nc", config.num_classes):
        logger.info(
            "Dataset nc=%d differs from model nc=%d — rebuilding head.",
            dataset_nc,
            getattr(model, "nc", config.num_classes),
        )
        if wrapper_model is not None:
            wrapper_model._rebuild_for_new_classes(dataset_nc)
            model = wrapper_model.model
        else:
            raise ValueError(
                f"Dataset nc={dataset_nc} differs from model nc={getattr(model, 'nc', config.num_classes)}, "
                "but wrapper_model is None — cannot rebuild head safely."
            )
    config.num_classes = dataset_nc

    fg_weight = getattr(config, "fg_weight", 100.0)
    loss_fn = FOMOLoss(
        num_classes=dataset_nc,
        fg_weight=fg_weight,
        device=device,
    ).to(device)
    if is_main_process():
        logger.info(
            "FOMOLoss rebuilt with resolved dataset nc=%d", dataset_nc
        )

    raw_names = data_cfg.get("names")
    if raw_names is not None and wrapper_model is not None:
        from ..base import BaseModel
        wrapper_model.names = BaseModel._sanitize_names(
            raw_names if isinstance(raw_names, dict)
            else {i: n for i, n in enumerate(raw_names)},
            dataset_nc,
        )

    any_aug = (
        config.mosaic_prob > 0
        or config.mixup_prob > 0
        or config.hsv_prob > 0
        or config.flip_prob > 0
        or config.degrees > 0
        or config.translate > 0
        or config.shear > 0
    )

    if any_aug:
        from ...data.dataset import YOLODataset
        from ...training.augment import MosaicMixupDataset

        if is_main_process():
            from ...data.augment.spec import mixup_gating_warning

            gating_msg = mixup_gating_warning(
                "fomo", config.mosaic_prob, config.mixup_prob
            )
            if gating_msg:
                logger.warning(gating_msg)
            logger.info("FOMO Training: Data augmentation enabled.")
            logger.info(f"  mosaic_prob: {config.mosaic_prob}")
            logger.info(f"  mixup_prob: {config.mixup_prob}")
            logger.info(f"  flip_prob: {config.flip_prob}")
            logger.info(f"  hsv_prob: {config.hsv_prob}")

        base_train_ds = YOLODataset(
            img_files=train_img_files,
            label_files=train_label_files,
            img_size=(input_size, input_size),
            preproc=None,
        )

        preproc = FOMOTrainTransform(
            max_labels=50,
            flip_prob=config.flip_prob,
            hsv_prob=config.hsv_prob,
        )

        augmented_train_ds = MosaicMixupDataset(
            dataset=base_train_ds,
            img_size=(input_size, input_size),
            mosaic=config.mosaic_prob > 0,
            preproc=preproc,
            degrees=config.degrees,
            translate=config.translate,
            mosaic_scale=config.mosaic_scale,
            mixup_scale=config.mixup_scale,
            shear=config.shear,
            enable_mixup=config.mixup_prob > 0,
            mosaic_prob=config.mosaic_prob,
            mixup_prob=config.mixup_prob,
        )

        train_ds = FOMOAugmentedDataset(
            augmented_dataset=augmented_train_ds,
            input_size=input_size,
            grid_size=grid_size,
        )
    else:
        train_ds = FOMOYOLODataset(train_img_files, train_label_files, input_size, grid_size)

    val_ds = FOMOYOLODataset(val_img_files or [], val_label_files or [], input_size, grid_size)
    return train_ds, val_ds, model, loss_fn

