"""Semantic-segmentation dataset for LibreYOLO.

Semantic datasets pair each image with a dense single-channel mask whose pixel
values are class IDs; ``255`` marks ignored pixels excluded from loss and
metrics::

    dataset/
        images/train/*.jpg
        images/val/*.jpg
        masks/train/*.png      # same stem as the paired image
        masks/val/*.png

The dataset YAML follows the common contract (``path``/``train``/``val``/
``test``/``names``) plus two semantic keys:

- ``masks_dir``: mask directory name substituted for ``images`` in each image
  path (default ``masks``). When the YAML omits ``masks_dir``, masks are
  rasterized at load time from YOLO ``segment`` polygon labels and a
  ``background`` class is appended after the object classes.
- ``label_mapping``: optional ``{source_id: train_id}`` remap applied to mask
  pixel values; unmapped source values become ignore.

Masks must be lossless single-channel images (PNG). Palette-mode PNGs are
read as palette indices, which is the conventional encoding for class maps.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .utils import get_img_files, img2label_paths, load_data_config

logger = logging.getLogger(__name__)

IGNORE_INDEX = 255

# Letterbox padding color for the image canvas (matches detection letterbox).
_PAD_COLOR = 114

MASK_FORMATS = (".png", ".tif", ".tiff", ".bmp")


def img2mask_paths(img_paths: List[Path], masks_dir: str = "masks") -> List[Path]:
    """Convert image paths to mask paths.

    Convention mirrors ``img2label_paths``: replace ``images`` with
    ``masks_dir`` in the path and look for a lossless mask extension with the
    same stem.
    """
    mask_paths = []
    for img_path in img_paths:
        path_str = str(img_path)
        for sep in (os.sep, "/", "\\"):
            path_str = path_str.replace(f"{sep}images{sep}", f"{sep}{masks_dir}{sep}")
            path_str = path_str.replace(f"{sep}images", f"{sep}{masks_dir}")
        base = Path(path_str)
        for suffix in MASK_FORMATS:
            candidate = base.with_suffix(suffix)
            if candidate.exists():
                base = candidate
                break
        else:
            base = base.with_suffix(".png")
        mask_paths.append(base)
    return mask_paths


def _load_mask_image(mask_path: Path) -> np.ndarray:
    """Read a single-channel mask file into an integer ``(H, W)`` array."""
    with Image.open(mask_path) as mask_img:
        if mask_img.mode in ("L", "P", "I", "I;16"):
            mask = np.asarray(mask_img)
        else:
            raise ValueError(
                f"Semantic mask {mask_path} has mode '{mask_img.mode}'. "
                "Masks must be single-channel class-ID images "
                "(PNG modes L, P, I, or I;16)."
            )
    if mask.ndim != 2:
        raise ValueError(
            f"Semantic mask {mask_path} has shape {mask.shape}; expected (H, W)."
        )
    return mask.astype(np.int64, copy=False)


def _apply_label_mapping(
    mask: np.ndarray, label_mapping: Dict[int, int], ignore_index: int
) -> np.ndarray:
    """Remap source mask values to train IDs; unmapped values become ignore."""
    lut_size = max(256, int(mask.max()) + 1) if mask.size else 256
    lut = np.full(lut_size, ignore_index, dtype=np.int64)
    for source_id, train_id in label_mapping.items():
        lut[int(source_id)] = int(train_id)
    return lut[mask]


def _rasterize_polygon_labels(
    label_file: Path,
    orig_shape: Tuple[int, int],
    background_id: int,
) -> np.ndarray:
    """Rasterize YOLO segment polygon labels into a dense class map.

    The canvas starts as ``background_id``; polygons are filled with their
    class ID in label-file order. Box-only 5-field rows are filled as
    rectangles, matching the segment contract where a detect row is a
    rectangular segment.
    """
    height, width = orig_shape
    canvas = Image.new("I", (width, height), background_id)
    draw = ImageDraw.Draw(canvas)
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
            if len(coords) == 4:
                cx, cy, w, h = coords
                points = [
                    ((cx - w / 2) * width, (cy - h / 2) * height),
                    ((cx + w / 2) * width, (cy - h / 2) * height),
                    ((cx + w / 2) * width, (cy + h / 2) * height),
                    ((cx - w / 2) * width, (cy + h / 2) * height),
                ]
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                points = [
                    (coords[i] * width, coords[i + 1] * height)
                    for i in range(0, len(coords), 2)
                ]
            else:
                raise ValueError(
                    f"Invalid segment label row in {label_file}: {line!r}"
                )
            draw.polygon(points, fill=class_id)
    return np.asarray(canvas).astype(np.int64)


def resolve_semantic_data(data: str | Path) -> Dict:
    """Load and sanity-check a semantic dataset YAML config.

    Returns the ``load_data_config`` dict. ``names``/``nc`` describe the mask
    label space; when the config has no ``masks_dir`` the polygon fallback
    appends a ``background`` class, which `SemanticDataset` reflects in its
    ``nc``/``names`` attributes.
    """
    config = load_data_config(str(data))
    if not config.get("names"):
        raise ValueError(
            f"Semantic dataset config {data!r} must define class names."
        )
    return config


class SemanticDataset(Dataset):
    """Dense semantic-segmentation dataset returning ``(img, mask, info, id)``.

    Images are letterboxed (default) or stretched to ``imgsz``; masks follow
    with nearest-neighbor geometry and ignore-valued padding. Training
    augmentation applies horizontal flips and scale jitter with random crops.
    """

    def __init__(
        self,
        data_config: Dict,
        split: str,
        imgsz: int,
        augment: bool = False,
        resize_mode: str = "letterbox",
        ignore_index: int = IGNORE_INDEX,
        scale_jitter: Tuple[float, float] = (0.5, 1.5),
    ):
        if resize_mode not in ("letterbox", "stretch"):
            raise ValueError(
                f"resize_mode must be 'letterbox' or 'stretch', got {resize_mode!r}"
            )
        self.split = split
        self.imgsz = int(imgsz)
        self.augment = augment
        self.resize_mode = resize_mode
        self.ignore_index = int(ignore_index)
        self.scale_jitter = scale_jitter

        split_value = data_config.get(split)
        if not split_value:
            raise ValueError(f"Semantic dataset config has no '{split}' split.")
        self.img_files = data_config.get(f"{split}_img_files") or get_img_files(
            split_value
        )
        if not self.img_files:
            raise FileNotFoundError(
                f"No images found for semantic split '{split}' at {split_value}."
            )

        names = data_config.get("names") or {}
        if isinstance(names, list):
            names = {index: name for index, name in enumerate(names)}
        self.names: Dict[int, str] = {int(k): str(v) for k, v in names.items()}
        base_nc = int(data_config.get("nc") or len(self.names))

        raw_mapping = data_config.get("label_mapping") or None
        self.label_mapping = (
            {int(k): int(v) for k, v in raw_mapping.items()} if raw_mapping else None
        )
        if self.label_mapping:
            invalid = sorted(
                v for v in self.label_mapping.values() if not 0 <= v < base_nc
            )
            if invalid:
                raise ValueError(
                    f"label_mapping train IDs {invalid} fall outside 0..{base_nc - 1}."
                )

        masks_dir = data_config.get("masks_dir")
        self.masks_dir = str(masks_dir) if masks_dir else None
        if self.masks_dir:
            # Dense-mask layout: nc is the mask label space as configured.
            self.nc = base_nc
            self.mask_files = img2mask_paths(self.img_files, self.masks_dir)
            missing = [str(p) for p in self.mask_files if not p.exists()]
            if missing:
                preview = ", ".join(missing[:3])
                raise FileNotFoundError(
                    f"{len(missing)} semantic mask file(s) missing for split "
                    f"'{split}' (e.g. {preview}). Expected masks under "
                    f"'{self.masks_dir}' mirroring the images tree."
                )
            self.label_files = None
        else:
            # Polygon fallback: object classes plus an appended background.
            self.nc = base_nc + 1
            self.background_id = base_nc
            self.names = dict(self.names)
            self.names[self.background_id] = "background"
            self.label_files = img2label_paths(self.img_files)
            self.mask_files = None

    def __len__(self) -> int:
        return len(self.img_files)

    def _load_target_mask(self, index: int, orig_shape: Tuple[int, int]) -> np.ndarray:
        if self.mask_files is not None:
            mask = _load_mask_image(self.mask_files[index])
            if mask.shape != orig_shape:
                raise ValueError(
                    f"Semantic mask {self.mask_files[index]} shape {mask.shape} "
                    f"does not match image shape {orig_shape}."
                )
            if self.label_mapping:
                mask = _apply_label_mapping(mask, self.label_mapping, self.ignore_index)
            invalid = (mask != self.ignore_index) & ((mask < 0) | (mask >= self.nc))
            if bool(invalid.any()):
                bad = sorted(np.unique(mask[invalid]).tolist())[:5]
                raise ValueError(
                    f"Semantic mask {self.mask_files[index]} contains class IDs "
                    f"{bad} outside 0..{self.nc - 1} (ignore={self.ignore_index}). "
                    "Use label_mapping to remap source IDs."
                )
            return mask
        return _rasterize_polygon_labels(
            self.label_files[index], orig_shape, self.background_id
        )

    def _resize(
        self, img: np.ndarray, mask: np.ndarray, scale: float
    ) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
        """Resize image (bilinear) and mask (nearest) and pad/crop to imgsz."""
        h0, w0 = img.shape[:2]
        if self.resize_mode == "stretch":
            new_w = new_h = self.imgsz
            ratio = 1.0
        else:
            ratio = min(self.imgsz / h0, self.imgsz / w0) * scale
            new_w = max(1, int(round(w0 * ratio)))
            new_h = max(1, int(round(h0 * ratio)))

        img_pil = Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
        mask_pil = Image.fromarray(mask.astype(np.int32), mode="I").resize(
            (new_w, new_h), Image.NEAREST
        )
        img = np.array(img_pil)
        mask = np.asarray(mask_pil).astype(np.int64)

        # Random crop any overflow (scale jitter can exceed imgsz).
        if new_h > self.imgsz or new_w > self.imgsz:
            top = random.randint(0, max(0, new_h - self.imgsz))
            left = random.randint(0, max(0, new_w - self.imgsz))
            img = img[top : top + self.imgsz, left : left + self.imgsz]
            mask = mask[top : top + self.imgsz, left : left + self.imgsz]
            new_h, new_w = img.shape[:2]

        pad_h = self.imgsz - new_h
        pad_w = self.imgsz - new_w
        top = pad_h // 2
        left = pad_w // 2
        if pad_h or pad_w:
            img = np.pad(
                img,
                ((top, pad_h - top), (left, pad_w - left), (0, 0)),
                constant_values=_PAD_COLOR,
            )
            mask = np.pad(
                mask,
                ((top, pad_h - top), (left, pad_w - left)),
                constant_values=self.ignore_index,
            )
        return img, mask, ratio, (top, left)

    def __getitem__(self, index: int):
        img_path = self.img_files[index]
        with Image.open(img_path) as img_pil:
            img = np.array(img_pil.convert("RGB"))
        orig_shape = img.shape[:2]
        mask = self._load_target_mask(index, orig_shape)

        scale = 1.0
        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if self.resize_mode == "letterbox":
                scale = random.uniform(*self.scale_jitter)

        img, mask, ratio, pad = self._resize(img, mask, scale)

        img_tensor = (
            torch.from_numpy(np.ascontiguousarray(img))
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()
        img_info = {
            "orig_shape": (int(orig_shape[0]), int(orig_shape[1])),
            "ratio": float(ratio),
            "pad": (int(pad[0]), int(pad[1])),
            "resize_mode": self.resize_mode,
            "img_path": str(img_path),
        }
        return img_tensor, mask_tensor, img_info, index


def semantic_collate_fn(batch):
    """Collate semantic samples into the trainer's 4-tuple batch shape.

    Returns ``(imgs, masks, img_infos, img_ids)``: ``imgs`` is ``[B,3,H,W]``
    float in ``[0, 1]`` and ``masks`` is ``[B,H,W]`` long with ignore pixels
    set to the dataset ignore index.
    """
    imgs = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([item[1] for item in batch], dim=0)
    img_infos = [item[2] for item in batch]
    img_ids = [item[3] for item in batch]
    return imgs, masks, img_infos, img_ids
