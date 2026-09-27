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

from .augment.color import augment_hsv
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
            if not 0 <= class_id < background_id:
                raise ValueError(
                    f"Polygon label class {class_id} in {label_file} falls "
                    f"outside 0..{background_id - 1}."
                )
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
                raise ValueError(f"Invalid segment label row in {label_file}: {line!r}")
            draw.polygon(points, fill=class_id)
    return np.asarray(canvas).astype(np.int64)


def _canvas_hw(imgsz: int | Tuple[int, int]) -> Tuple[int, int]:
    """Normalize a scalar or ``(height, width)`` canvas to explicit H and W."""
    if isinstance(imgsz, int):
        return int(imgsz), int(imgsz)
    if len(imgsz) != 2:
        raise ValueError(f"imgsz must be an int or (height, width), got {imgsz!r}")
    return int(imgsz[0]), int(imgsz[1])


def valid_content_hw(
    orig_shape: Tuple[int, int], ratio: float, canvas_hw: Tuple[int, int]
) -> Tuple[int, int]:
    """Size of the real (unpadded) content region inside a letterboxed canvas.

    This is the content size *after* :meth:`SemanticDataset._resize` finishes,
    so callers (e.g. flip-TTA validation, which must flip only the real content
    and leave letterbox padding in place) can locate the top-left-anchored valid
    region without re-deriving the math.

    The clamp to the canvas is what makes it the *final* content size rather
    than the intermediate resize size: ``_resize`` may scale past ``imgsz`` when
    ``scale > 1`` (training jitter) and then random-crop the overflow back down,
    so the stored content is capped at the canvas either way. Do not use this to
    compute ``_resize``'s resize step itself — clamping there would squash the
    aspect ratio and skip the crop.
    """
    orig_h, orig_w = orig_shape
    canvas_h, canvas_w = canvas_hw
    new_h = min(canvas_h, max(1, int(round(orig_h * ratio))))
    new_w = min(canvas_w, max(1, int(round(orig_w * ratio))))
    return new_h, new_w


def resolve_semantic_data(data: str | Path, allow_scripts: bool = False) -> Dict:
    """Load and sanity-check a semantic dataset YAML config.

    Returns the ``load_data_config`` dict. ``names``/``nc`` describe the mask
    label space; when the config has no ``masks_dir`` the polygon fallback
    appends a ``background`` class, which `SemanticDataset` reflects in its
    ``nc``/``names`` attributes. ``allow_scripts`` is forwarded to
    ``load_data_config`` so explicitly allowed Python download scripts run.
    """
    config = load_data_config(str(data), allow_scripts=allow_scripts)
    if not config.get("names"):
        raise ValueError(f"Semantic dataset config {data!r} must define class names.")
    return config


class SemanticDataset(Dataset):
    """Dense semantic-segmentation dataset returning ``(img, mask, info, id)``.

    Images are letterboxed (default), stretched, or (``resize_crop``) short-side-
    resized and random-cropped to ``imgsz``; masks follow with nearest-neighbor
    geometry and ignore-valued padding. Training augmentation applies horizontal
    flips, scale jitter with random crops, and HSV photometric jitter on the
    image only (masks are never recolored). ``resize_crop`` uses dense full-res
    crops (mmseg-style) for training and letterbox geometry for validation.

    Images are always scaled to ``[0, 1]`` (``/255``) and nothing else. Any
    per-channel standardization a family needs (e.g. SegFormer's ImageNet
    mean/std for its MiT backbone) is applied inside that family's ``forward``
    on the raw ``[0, 1]`` tensor, so the dataset stays ``/255``-only for every
    family and train / val / inference share one input contract.
    """

    def __init__(
        self,
        data_config: Dict,
        split: str,
        imgsz: int | Tuple[int, int],
        augment: bool = False,
        resize_mode: str = "letterbox",
        ignore_index: int = IGNORE_INDEX,
        scale_jitter: Tuple[float, float] = (0.5, 1.5),
        hsv_prob: float = 0.5,
        crop_cat_max_ratio: float = 0.75,
        photometric=None,
    ):
        if resize_mode not in ("letterbox", "stretch", "resize_crop", "rescale_crop"):
            raise ValueError(
                "resize_mode must be 'letterbox', 'stretch', 'resize_crop', or "
                f"'rescale_crop', got {resize_mode!r}"
            )
        self.split = split
        # imgsz may be a scalar (square canvas, the historical contract) or an
        # explicit (height, width). A scalar is exactly equivalent to the equal
        # pair: every geometry decision below reads canvas_h/canvas_w.
        self.imgsz = imgsz if isinstance(imgsz, int) else tuple(int(v) for v in imgsz)
        self.canvas_h, self.canvas_w = _canvas_hw(imgsz)
        self.augment = augment
        self.resize_mode = resize_mode
        self.ignore_index = int(data_config.get("ignore_index", ignore_index))
        self.scale_jitter = scale_jitter
        self.hsv_prob = float(hsv_prob)
        # Optional family-local photometric transform (RGB HWC in, RGB HWC
        # out). When set it replaces the shared HSV-gain jitter; masks are
        # never handed to it, so class IDs cannot be recolored.
        self.photometric = photometric
        # resize_crop-only: reject a random crop whose most-common (non-ignore)
        # class exceeds this fraction (mmseg cat_max_ratio; 1.0 disables).
        self.crop_cat_max_ratio = float(crop_cat_max_ratio)

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
        # rescale_crop only defines a *training* sampler; without augmentation
        # it validates by direct resize to the canvas, which is exactly what
        # its family's inference preprocessor does.
        if self.resize_mode in ("stretch", "rescale_crop"):
            new_h, new_w = self.canvas_h, self.canvas_w
            ratio = 1.0
        else:
            ratio = min(self.canvas_h / h0, self.canvas_w / w0) * scale
            # Deliberately NOT clamped to imgsz: with scale > 1 (training
            # jitter) this overshoots the canvas on purpose, and the random-crop
            # below takes the overflow back out. Clamping here instead would
            # squash the aspect ratio and make that crop dead code, turning
            # zoom-in jitter into a plain resize. valid_content_hw() reports the
            # post-crop size and is the right thing for readers of the canvas.
            new_w = max(1, int(round(w0 * ratio)))
            new_h = max(1, int(round(h0 * ratio)))

        img_pil = Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
        mask_pil = Image.fromarray(mask.astype(np.int32), mode="I").resize(
            (new_w, new_h), Image.NEAREST
        )
        img = np.array(img_pil)
        mask = np.asarray(mask_pil).astype(np.int64)

        # Random crop any overflow (scale jitter can exceed the canvas).
        if new_h > self.canvas_h or new_w > self.canvas_w:
            top = random.randint(0, max(0, new_h - self.canvas_h))
            left = random.randint(0, max(0, new_w - self.canvas_w))
            img = img[top : top + self.canvas_h, left : left + self.canvas_w]
            mask = mask[top : top + self.canvas_h, left : left + self.canvas_w]
            new_h, new_w = img.shape[:2]

        # Top-left anchored padding, matching the family inference letterbox
        # (content at the origin, pad at bottom/right).
        pad_h = self.canvas_h - new_h
        pad_w = self.canvas_w - new_w
        if pad_h or pad_w:
            img = np.pad(
                img,
                ((0, pad_h), (0, pad_w), (0, 0)),
                constant_values=_PAD_COLOR,
            )
            mask = np.pad(
                mask,
                ((0, pad_h), (0, pad_w)),
                constant_values=self.ignore_index,
            )
        return img, mask, ratio, (0, 0)

    def _resize_and_crop(
        self, img: np.ndarray, mask: np.ndarray, scale: float
    ) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
        """mmseg-style train sampling: resize the SHORT side to ``imgsz * scale``
        (keep aspect), pad up to ``imgsz`` if a side falls short, then random-crop
        ``imgsz x imgsz``. Unlike ``_resize`` (which fits the LONG side and pads),
        this yields dense full-resolution crops. When ``crop_cat_max_ratio < 1``
        the crop is re-sampled (up to 10x) to avoid a patch dominated by a single
        class (mmseg ``cat_max_ratio``).

        The short-side reference for a rectangular canvas is its shorter side,
        so a scalar canvas keeps its exact historical behavior."""
        h0, w0 = img.shape[:2]
        reference = min(self.canvas_h, self.canvas_w)
        ratio = (reference / min(h0, w0)) * scale
        new_w = max(1, int(round(w0 * ratio)))
        new_h = max(1, int(round(h0 * ratio)))

        img_pil = Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
        mask_pil = Image.fromarray(mask.astype(np.int32), mode="I").resize(
            (new_w, new_h), Image.NEAREST
        )
        img = np.array(img_pil)
        mask = np.asarray(mask_pil).astype(np.int64)

        return self._pad_and_random_crop(img, mask, ratio)

    def _rescale_and_crop(
        self, img: np.ndarray, mask: np.ndarray, scale: float
    ) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
        """Absolute-rescale sampling: the PP-LiteSeg / STDC Cityscapes recipe.

        Scales the *source* image by ``scale`` (both sides, aspect preserved),
        ignore-pads any side that falls short of the crop, then takes a random
        ``canvas_h x canvas_w`` crop. Unlike ``resize_crop`` the scale is not
        relative to the canvas: the recipe's 0.125..1.5 range is defined
        against the dataset's own resolution, which is what makes its extreme
        low end meaningful."""
        h0, w0 = img.shape[:2]
        ratio = float(scale)
        new_w = max(1, int(round(w0 * ratio)))
        new_h = max(1, int(round(h0 * ratio)))

        img_pil = Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
        mask_pil = Image.fromarray(mask.astype(np.int32), mode="I").resize(
            (new_w, new_h), Image.NEAREST
        )
        img = np.array(img_pil)
        mask = np.asarray(mask_pil).astype(np.int64)
        return self._pad_and_random_crop(img, mask, ratio)

    def _pad_and_random_crop(
        self, img: np.ndarray, mask: np.ndarray, ratio: float
    ) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
        """Ignore-pad each side up to the canvas, then random-crop the canvas."""
        new_h, new_w = img.shape[:2]
        pad_h = max(0, self.canvas_h - new_h)
        pad_w = max(0, self.canvas_w - new_w)
        if pad_h or pad_w:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=_PAD_COLOR)
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=self.ignore_index)
        H, W = img.shape[:2]

        top, left = 0, 0
        for _ in range(10):
            top = random.randint(0, H - self.canvas_h)
            left = random.randint(0, W - self.canvas_w)
            if self.crop_cat_max_ratio >= 1.0:
                break
            cand = mask[top : top + self.canvas_h, left : left + self.canvas_w]
            labels = cand[cand != self.ignore_index]
            if labels.size == 0:
                break
            _, counts = np.unique(labels, return_counts=True)
            if counts.max() / counts.sum() <= self.crop_cat_max_ratio:
                break
        img = img[top : top + self.canvas_h, left : left + self.canvas_w]
        mask = mask[top : top + self.canvas_h, left : left + self.canvas_w]
        return img, mask, ratio, (0, 0)

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
            if self.photometric is not None:
                img = np.ascontiguousarray(self.photometric(img))
            elif self.hsv_prob and random.random() < self.hsv_prob:
                # augment_hsv follows the shared BGR contract; images here are
                # RGB, so flip to BGR for the jitter and back to RGB. The mask
                # is never passed in, so class IDs are untouched.
                img_bgr = np.ascontiguousarray(img[..., ::-1])
                augment_hsv(img_bgr)
                img = np.ascontiguousarray(img_bgr[..., ::-1])
            if self.resize_mode in ("letterbox", "resize_crop", "rescale_crop"):
                scale = random.uniform(*self.scale_jitter)

        if self.resize_mode == "resize_crop" and self.augment:
            img, mask, ratio, pad = self._resize_and_crop(img, mask, scale)
        elif self.resize_mode == "rescale_crop" and self.augment:
            img, mask, ratio, pad = self._rescale_and_crop(img, mask, scale)
        else:
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
