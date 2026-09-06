"""
Dataset classes for YOLOX training.

Supports both COCO JSON format and YOLO txt format.
"""

import copy
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, Sampler
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from .cache import ImageCacheMixin
from .yolo_coco_api import parse_yolo_label_line
from .obb import (
    canonicalize_xywhr,
    corners_to_xywhr,
    parse_yolo_obb_label_line,
    xywhr_to_proxy_xyxy,
)
from libreyolo.training.distributed import is_main_process
from libreyolo.utils.image_size import imgsz_to_hw

logger = logging.getLogger(__name__)


def _xyxy_to_ring(x1: float, y1: float, x2: float, y2: float) -> List[np.ndarray]:
    """Convert one pixel-space box to the shared ring contract."""
    ring = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )
    return [ring]


class DenseMaskRing(np.ndarray):
    """Polygon ring carrying a dense mask for mask-aware transforms.

    Used when the polygon ring is a lossy approximation of the true mask
    (e.g., a contour extracted from an RLE-decoded mask). For polygon-sourced
    annotations the ring is itself exact, so a plain ndarray is stored instead
    and consumers that need crop-fidelity materialize the mask on demand.
    """

    def __new__(cls, ring: np.ndarray, mask: np.ndarray):
        obj = np.asarray(ring, dtype=np.float32).view(cls)
        obj.dense_mask = np.ascontiguousarray(mask.astype(np.uint8))
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.dense_mask = getattr(obj, "dense_mask", None)

    def copy(self, order="C"):
        copied = super().copy(order).view(type(self))
        copied.dense_mask = None if self.dense_mask is None else self.dense_mask.copy()
        return copied


def _mask_to_rings(mask: np.ndarray) -> List[np.ndarray]:
    """Convert a binary mask to polygon rings using OpenCV contours."""
    mask_u8 = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rings = []
    for contour in contours:
        ring = contour.reshape(-1, 2)
        if ring.shape[0] >= 3:
            rings.append(ring.astype(np.float32))
    if rings or mask_u8.sum() == 0:
        return rings

    ys, xs = np.where(mask_u8 > 0)
    x1, x2 = float(xs.min()), float(xs.max() + 1)
    y1, y2 = float(ys.min()), float(ys.max() + 1)
    return [
        np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
    ]


def _coco_segmentation_to_rings(
    segmentation,
    *,
    height: int | None = None,
    width: int | None = None,
) -> List[np.ndarray]:
    """Convert COCO polygon or RLE segmentation to pixel-space rings."""
    if isinstance(segmentation, list):
        rings = []
        for polygon in segmentation:
            if polygon is None or len(polygon) < 6:
                continue
            ring = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            rings.append(ring)
        return rings

    if not isinstance(segmentation, dict):
        return []
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        return []

    rle = segmentation
    if isinstance(rle.get("counts"), list):
        if height is None or width is None:
            return []
        rle = mask_utils.frPyObjects(rle, height, width)
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    rings = _mask_to_rings(decoded)
    if rings:
        rings[0] = DenseMaskRing(rings[0], decoded)
    return rings


def _points_to_xywhr(points: np.ndarray) -> np.ndarray:
    """Fit canonical ``xywhr`` to pixel-space points."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
        raise ValueError("COCO OBB source must provide at least 3 points")
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(points)
    return canonicalize_xywhr((cx, cy, w, h, math.radians(angle_deg)))


def _clip_points_to_image(points: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0], 0.0, float(width))
    points[:, 1] = np.clip(points[:, 1], 0.0, float(height))
    return points


def _coco_obb_to_xywhr(obj: dict, width: int, height: int) -> np.ndarray | None:
    """Return pixel-space ``xywhr`` for a COCO-style OBB annotation.

    Preferred source is an ``obb`` field. Segmentation polygons/RLEs are
    refitted to their minimum-area rectangle. Plain COCO ``bbox`` remains a
    useful axis-aligned fallback with zero rotation.
    """
    obb = obj.get("obb")
    if isinstance(obb, dict):
        obb = obb.get("corners", obb.get("xywhr"))
    if obb is not None:
        try:
            values = np.asarray(obb, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            values = None
        if values is not None and np.isfinite(values).all():
            if values.size == 8:
                corners = _clip_points_to_image(values.reshape(4, 2), width, height)
                try:
                    return corners_to_xywhr(corners)
                except ValueError:
                    pass
            elif values.size == 5:
                cx, cy, box_w, box_h, angle = map(float, values)
                if box_w > 0.0 and box_h > 0.0:
                    return canonicalize_xywhr((cx, cy, box_w, box_h, angle))

    segmentation = obj.get("segmentation")
    if segmentation:
        rings = _coco_segmentation_to_rings(
            segmentation,
            height=height,
            width=width,
        )
        points = [
            np.asarray(ring, dtype=np.float32).reshape(-1, 2)
            for ring in rings
            if ring is not None and len(ring) >= 3
        ]
        if points:
            try:
                return _points_to_xywhr(
                    _clip_points_to_image(
                        np.concatenate(points, axis=0),
                        width,
                        height,
                    )
                )
            except ValueError:
                return None

    bbox = obj.get("bbox")
    if bbox is None:
        return None
    values = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if values.size != 4 or not np.isfinite(values).all():
        return None
    x1 = max(0.0, float(values[0]))
    y1 = max(0.0, float(values[1]))
    x2 = min(float(width), x1 + max(0.0, float(values[2])))
    y2 = min(float(height), y1 + max(0.0, float(values[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return canonicalize_xywhr(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1, 0.0))


class YOLODataset(ImageCacheMixin, Dataset):
    """
    YOLO format dataset supporting both directory and file list modes.

    Mode 1 (Directory): Traditional structure
        dataset/images/{split}/*.jpg
        dataset/labels/{split}/*.txt

    Mode 2 (File List): .txt file format
        Provide img_files list directly, labels inferred via img2label_paths()

    Each label file contains one object per line:
    class_id center_x center_y width height  (all normalized 0-1)
    """

    def __init__(
        self,
        data_dir: str | None = None,
        split: str = "train",
        img_size: int | Tuple[int, int] | List[int] = (640, 640),
        preproc=None,
        img_files: List[Path] | None = None,
        label_files: List[Path] | None = None,
        load_segments: bool = False,
        load_obb: bool = False,
        num_classes: int | None = None,
        single_cls: bool = False,
    ):
        """
        Initialize YOLO dataset.

        Args:
            data_dir: Path to dataset root (for directory mode).
            split: "train" or "val" (for directory mode).
            img_size: Target image size (height, width).
            preproc: Preprocessing transform.
            img_files: List of image paths (for file list mode).
            label_files: List of label paths (optional, inferred if not provided).
            num_classes: Optional class-count bound used for OBB label validation.
            single_cls: Remap every non-negative class id to class 0.
        """
        self.img_size = imgsz_to_hw(img_size, name="img_size")
        self.preproc = preproc
        self._input_dim = self.img_size
        self.load_segments = load_segments
        self.load_obb = load_obb
        self.num_classes = num_classes
        self.single_cls = bool(single_cls)
        if self.load_segments and self.load_obb:
            raise ValueError("YOLODataset cannot load segmentation and OBB labels together")

        if img_files is not None:
            # File list mode (.txt format)
            self.img_files = [Path(f) for f in img_files]
            if label_files is not None:
                self.label_files = [Path(f) for f in label_files]
            else:
                # Infer label paths from image paths
                from libreyolo.data import img2label_paths

                self.label_files = img2label_paths(self.img_files)

            self.data_dir = None
            self.split = None
            self.img_dir = None
            self.label_dir = None
        else:
            # Directory mode (original behavior)
            if data_dir is None:
                raise ValueError("Either data_dir or img_files must be provided")

            self.data_dir = Path(data_dir)
            self.split = split
            self.img_dir = self.data_dir / "images" / split
            self.label_dir = self.data_dir / "labels" / split

            if not self.img_dir.exists():
                raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

            # Collect image files from directory
            self.img_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                self.img_files.extend(self.img_dir.glob(ext))
                self.img_files.extend(self.img_dir.glob(ext.upper()))
            self.img_files = sorted(set(self.img_files))

            # Generate corresponding label file paths
            self.label_files = [
                self.label_dir / (f.stem + ".txt") for f in self.img_files
            ]

        self.num_imgs = len(self.img_files)

        if self.num_imgs == 0:
            raise ValueError("No images found")

        # Pre-load annotations
        self.annotations = self._load_annotations()

    def _load_annotations(self) -> List:
        """Load all annotations."""
        total = len(self.img_files)
        source = self._annotation_source()
        main = is_main_process()
        if main:
            logger.info("Loading %d YOLO annotations from %s...", total, source)
        start = time.perf_counter()

        pairs = list(zip(self.img_files, self.label_files))
        max_workers = min(8, os.cpu_count() or 1, total)

        def load_one(pair):
            img_file, label_file = pair
            return self._load_label(label_file, img_file)

        tqdm_disable = not (main and sys.stderr.isatty())
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                annotations = list(
                    tqdm(
                        executor.map(load_one, pairs),
                        total=total,
                        desc=f"Loading YOLO annotations ({source})",
                        file=sys.stderr,
                        disable=tqdm_disable,
                    )
                )
        else:
            annotations = [
                load_one(pair)
                for pair in tqdm(
                    pairs,
                    total=total,
                    desc=f"Loading YOLO annotations ({source})",
                    file=sys.stderr,
                    disable=tqdm_disable,
                )
            ]

        if main:
            logger.info(
                "Loaded %d YOLO annotations from %s in %.2fs",
                total,
                source,
                time.perf_counter() - start,
            )
        if self.load_obb:
            invalid_obb_rows = 0
            invalid_obb_files = 0
            first_invalid_obb = None
            normalized_annotations = []
            for annotation, skipped_count, first_error in annotations:
                normalized_annotations.append(annotation)
                if skipped_count:
                    invalid_obb_rows += skipped_count
                    invalid_obb_files += 1
                    first_invalid_obb = first_invalid_obb or first_error
            annotations = normalized_annotations
            if invalid_obb_rows and main:
                logger.warning(
                    "Skipped %d invalid YOLO OBB label rows across %d files from %s. "
                    "First invalid row: %s",
                    invalid_obb_rows,
                    invalid_obb_files,
                    source,
                    first_invalid_obb,
                )
        if self.load_segments:
            self.segments = [item[1] for item in annotations]
            annotations = [item[0] for item in annotations]
        else:
            self.segments = None

        if sum(a[0].shape[0] for a in annotations) == 0:
            logger.warning("No labels found in %d files from %s.", total, source)
        return annotations

    def _annotation_source(self) -> str:
        """Return a compact source label for annotation loading progress."""
        if self.split is not None:
            return str(self.split)
        if self.label_files:
            label_dir = self.label_files[0].parent
            if label_dir.parent.name:
                return f"{label_dir.parent.name}/{label_dir.name}"
            return str(label_dir)
        return "dataset"

    def _load_label(self, label_file: Path, img_file: Path) -> Tuple:
        """Load annotation for a single image."""
        # Read image to get dimensions
        try:
            with Image.open(img_file) as im:
                # Use the stored (non-EXIF-rotated) dimensions so label-space
                # dims match the pixels from cv2.imdecode below, which is called
                # with IMREAD_IGNORE_ORIENTATION. Both stay in stored orientation
                # on every OpenCV build (imdecode's native EXIF handling is
                # build-dependent, so relying on it would mismatch dims vs pixels
                # on builds that ignore EXIF).
                width, height = im.size
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            raise FileNotFoundError(f"Cannot read image: {img_file}") from e

        # Load labels
        labels = []
        segments = []
        skipped_obb_rows = 0
        first_obb_error = None
        if label_file.exists():
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if self.load_obb:
                        try:
                            cls_id, corners = parse_yolo_obb_label_line(
                                parts,
                                # single_cls discards the source class id, so the
                                # class-count bound does not apply to it. The YOLO
                                # box path already works this way: parse_yolo_label_line
                                # remaps to 0 and only then checks the range. Passing
                                # the bound here instead dropped every row whose
                                # source id was >= num_classes.
                                num_classes=None if self.single_cls else self.num_classes,
                                clip=True,
                            )
                            pixel_corners = corners.copy()
                            pixel_corners[:, 0] *= width
                            pixel_corners[:, 1] *= height
                            xywhr = corners_to_xywhr(pixel_corners)
                            proxy = xywhr_to_proxy_xyxy(xywhr)
                        except ValueError as exc:
                            skipped_obb_rows += 1
                            first_obb_error = first_obb_error or f"{label_file.name}: {exc}"
                            continue
                        if self.single_cls:
                            cls_id = 0
                        labels.append([*proxy.tolist(), cls_id, float(xywhr[4])])
                    else:
                        # Use the validation parser as the single source of
                        # truth for YOLO boxes and polygons. It derives polygon
                        # extents, rejects malformed/non-finite rows, clips both
                        # endpoints to the image, and drops boxes with no
                        # visible area. Training and scoring therefore consume
                        # identical geometry (#814).
                        parsed = parse_yolo_label_line(
                            line,
                            width,
                            height,
                            self.num_classes,
                            label_file,
                            return_segment=self.load_segments,
                            single_cls=self.single_cls,
                        )
                        if parsed is None:
                            continue

                        if self.load_segments:
                            cls_id, x1, y1, x2, y2, _, segment = parsed
                            if segment:
                                ring = np.asarray(segment, dtype=np.float32).reshape(
                                    -1, 2
                                )
                                segments.append([ring])
                            else:
                                segments.append(_xyxy_to_ring(x1, y1, x2, y2))
                        else:
                            cls_id, x1, y1, x2, y2, _ = parsed

                        labels.append([x1, y1, x2, y2, cls_id])

        # Create annotation array
        if labels:
            res = np.array(labels, dtype=np.float32)
        else:
            ncol = 6 if self.load_obb else 5
            res = np.zeros((0, ncol), dtype=np.float32)

        # Scale to target image size
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        if len(res) > 0:
            res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))
        file_name = img_file.name

        annotation = (res, img_info, resized_info, file_name)
        if self.load_segments:
            return annotation, segments
        if self.load_obb:
            return annotation, skipped_obb_rows, first_obb_error
        return annotation

    def __len__(self):
        return self.num_imgs

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value):
        self._input_dim = value

    def load_anno(self, index: int) -> np.ndarray:
        """Load annotation for given index."""
        return self.annotations[index][0]

    def _image_path(self, index: int) -> Path:
        """Source image path for given index (used for disk caching)."""
        return self.img_files[index]

    def _decode_image(self, index: int) -> np.ndarray:
        """Decode image from disk for given index."""
        img_file = self.img_files[index]
        img = cv2.imdecode(
            np.fromfile(str(img_file), dtype=np.uint8),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        if img is None:
            raise ValueError(f"Failed to load {img_file}")
        return img

    # load_resized_img comes from ImageCacheMixin: the deterministic resize is
    # the post-resize cache point, so the mixin owns both the math and the cache.

    def _load_segments(self, index: int):
        if self.segments is None:
            return None
        return copy.deepcopy(self.segments[index])

    def pull_item(self, index: int):
        """Get item without preprocessing."""
        label, origin_image_size, _, _ = self.annotations[index]
        segments = self._load_segments(index)
        if getattr(self.preproc, "wants_unresized_image", False):
            img = self.load_image(index)
            label = copy.deepcopy(label)
            if label.shape[0] > 0:
                target_h, target_w = self.img_size
                r = min(target_h / origin_image_size[0], target_w / origin_image_size[1])
                if r > 0:
                    label[:, :4] = label[:, :4] / r
            if self.load_segments:
                return img, label, origin_image_size, index, segments
            return img, label, origin_image_size, index
        img = self.load_resized_img(index)
        if self.load_segments:
            if segments:
                # Boxes were scaled by ``r`` at load time to match the resized
                # canvas; scale the sibling segments by the same ratio so masks
                # stay aligned with boxes when imgsz != native.
                r = min(
                    self.img_size[0] / origin_image_size[0],
                    self.img_size[1] / origin_image_size[1],
                )
                for rings in segments:
                    for ring in rings:
                        ring *= r
            return img, copy.deepcopy(label), origin_image_size, index, segments
        return img, copy.deepcopy(label), origin_image_size, index

    def __getitem__(self, index: int):
        """Get preprocessed item."""
        item = self.pull_item(index)
        if len(item) == 5:
            img, target, img_info, img_id, segments = item
        else:
            img, target, img_info, img_id = item
            segments = None

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        if self.load_segments:
            return img, target, img_info, img_id, segments
        return img, target, img_info, img_id


class COCODataset(ImageCacheMixin, Dataset):
    """
    COCO format dataset for YOLOX training.

    Directory structure:
    dataset/
    ├── annotations/
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    ├── train2017/
    │   ├── img1.jpg
    │   └── ...
    └── val2017/
    """

    def __init__(
        self,
        data_dir: str,
        json_file: str = "instances_train2017.json",
        name: str = "train2017",
        img_size: int | Tuple[int, int] | List[int] = (640, 640),
        preproc=None,
        load_segments: bool = False,
        load_obb: bool = False,
        num_classes: int | None = None,
        names=None,
        single_cls: bool = False,
    ):
        """
        Initialize COCO dataset.

        Args:
            data_dir: Path to dataset root
            json_file: COCO annotation JSON file name
            name: Image folder name (e.g., 'train2017')
            img_size: Target image size (height, width)
            preproc: Preprocessing transform
            single_cls: Remap emitted annotations to class 0 after resolving
                the original COCO category mapping.
        """
        if load_segments and load_obb:
            raise ValueError("COCODataset cannot load segmentation and OBB labels together")
        try:
            from pycocotools.coco import COCO
        except ImportError:
            raise ImportError(
                "pycocotools is required for COCO format. "
                "Install with: pip install pycocotools"
            )

        self.data_dir = Path(data_dir)
        self.json_file = json_file
        self.name = name
        self.img_size = imgsz_to_hw(img_size, name="img_size")
        self._input_dim = self.img_size
        self.preproc = preproc
        self.load_segments = load_segments
        self.load_obb = load_obb
        self.num_classes = num_classes
        self.names = names
        self.single_cls = bool(single_cls)

        # Load COCO annotations
        ann_file = self._annotation_path()
        self.coco = COCO(str(ann_file))

        # Remove useless info to save memory
        self._remove_useless_info()

        self.ids = self.coco.getImgIds()
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.class_ids)
        self.category_id_to_label, self.label_to_category_id = (
            self._build_category_mappings()
        )
        if self.single_cls:
            self._classes = ("object",)
        elif self.names is None:
            self._classes = tuple([c["name"] for c in self.cats])
        else:
            class_names = self._normalized_class_names()
            class_count = self.num_classes or max(class_names, default=-1) + 1
            self._classes = tuple(
                class_names.get(i, f"class_{i}") for i in range(class_count)
            )

        # Pre-load annotations
        self.annotations = self._load_coco_annotations()

    def _annotation_path(self) -> Path:
        path = Path(self.json_file)
        if path.is_absolute():
            return path
        if path.parent != Path("."):
            return self.data_dir / path
        return self.data_dir / "annotations" / path

    def _normalized_class_names(self) -> dict[int, str]:
        if self.names is None:
            return {}
        if isinstance(self.names, dict):
            out = {}
            for key, value in self.names.items():
                try:
                    index = int(key)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"COCO dataset class name key must be an integer: {key!r}") from exc
                if index < 0:
                    raise ValueError(f"COCO dataset class name key must be non-negative: {key!r}")
                out[index] = str(value)
            return out
        return {i: str(name) for i, name in enumerate(self.names)}

    def _build_category_mappings(self) -> tuple[dict[int, int], dict[int, int]]:
        if self.names is None:
            category_to_label = {
                category_id: label for label, category_id in enumerate(self.class_ids)
            }
        else:
            class_names = self._normalized_class_names()
            name_to_label: dict[str, int] = {}
            for label, name in class_names.items():
                if name in name_to_label:
                    raise ValueError(f"Duplicate class name in dataset YAML: {name!r}")
                name_to_label[name] = label

            category_to_label = {}
            for category in self.cats:
                category_name = str(category.get("name", ""))
                if category_name not in name_to_label:
                    raise ValueError(
                        "COCO category name not found in dataset YAML names: "
                        f"{category_name!r}"
                    )
                category_to_label[int(category["id"])] = name_to_label[category_name]

        if self.num_classes is not None and not self.single_cls:
            for category_id, label in category_to_label.items():
                if label < 0 or label >= self.num_classes:
                    raise ValueError(
                        f"COCO category id {category_id} maps to class {label}, "
                        f"outside configured num_classes={self.num_classes}."
                    )

        label_to_category = {}
        for category_id, label in category_to_label.items():
            if label in label_to_category:
                raise ValueError(
                    f"Multiple COCO categories map to class {label}; "
                    "dataset YAML names must be unique."
                )
            label_to_category[label] = category_id
        if self.single_cls:
            collapsed_category_id = self.class_ids[0] if self.class_ids else 0
            label_to_category = {0: collapsed_category_id}
        return category_to_label, label_to_category

    def _remove_useless_info(self):
        """Remove useless info from COCO to save memory."""
        dataset = self.coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset.get("images", []):
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if not self.load_segments and not self.load_obb:
            for anno in dataset.get("annotations", []):
                anno.pop("segmentation", None)

    def _load_coco_annotations(self) -> List:
        """Load all annotations."""
        total = len(self.ids)
        source = f"{self.name}/{self.json_file}"
        logger.info("Loading %d COCO annotations from %s...", total, source)
        start = time.perf_counter()
        annotations = [
            self._load_anno_from_id(id_)
            for id_ in tqdm(
                self.ids,
                total=total,
                desc=f"Loading COCO annotations ({self.name})",
                file=sys.stderr,
                disable=not sys.stderr.isatty(),
            )
        ]
        logger.info(
            "Loaded %d COCO annotations from %s in %.2fs",
            total,
            source,
            time.perf_counter() - start,
        )
        if self.load_segments:
            self.segments = [item[1] for item in annotations]
            annotations = [item[0] for item in annotations]
        else:
            self.segments = None

        if sum(a[0].shape[0] for a in annotations) == 0:
            logger.warning("No labels found in %d files from %s.", total, source)
        return annotations

    def _load_anno_from_id(self, id_: int) -> Tuple:
        """Load annotation for a single image ID."""
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)

        objs = []
        segments = []
        for obj in annotations:
            try:
                area = float(obj.get("area", 1.0))
            except (TypeError, ValueError):
                area = 0.0
            if area <= 0.0:
                continue
            if self.load_obb:
                xywhr = _coco_obb_to_xywhr(obj, width, height)
                if xywhr is None:
                    continue
                proxy = xywhr_to_proxy_xyxy(xywhr)
                objs.append((obj, proxy, float(xywhr[4])))
                continue
            x1 = max(0, obj["bbox"][0])
            y1 = max(0, obj["bbox"][1])
            x2 = min(width, x1 + max(0, obj["bbox"][2]))
            y2 = min(height, y1 + max(0, obj["bbox"][3]))
            if x2 > x1 and y2 > y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)
                if self.load_segments:
                    segments.append(
                        _coco_segmentation_to_rings(
                            obj.get("segmentation", []),
                            height=height,
                            width=width,
                        )
                    )

        num_objs = len(objs)
        width_out = 6 if self.load_obb else 5
        res = np.zeros((num_objs, width_out), dtype=np.float32)
        for ix, obj in enumerate(objs):
            if self.load_obb:
                coco_obj, proxy, angle = obj
                cls = self.category_id_to_label[coco_obj["category_id"]]
                if self.single_cls:
                    cls = 0
                res[ix, 0:4] = proxy
                res[ix, 4] = cls
                res[ix, 5] = angle
            else:
                cls = self.category_id_to_label[obj["category_id"]]
                if self.single_cls:
                    cls = 0
                res[ix, 0:4] = obj["clean_bbox"]
                res[ix, 4] = cls

        # Scale to target size
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))
        file_name = im_ann.get("file_name", f"{id_:012}.jpg")

        annotation = (res, img_info, resized_info, file_name)
        if self.load_segments:
            return annotation, segments
        return annotation

    def __len__(self):
        return self.num_imgs

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value):
        self._input_dim = value

    def load_anno(self, index: int) -> np.ndarray:
        """Load annotation for given index."""
        return self.annotations[index][0]

    def _image_path(self, index: int) -> Path:
        """Source image path for given index (used for disk caching)."""
        file_name = self.annotations[index][3]
        image_root = Path(self.name)
        if image_root.is_absolute():
            return image_root / file_name
        return self.data_dir / image_root / file_name

    def _decode_image(self, index: int) -> np.ndarray:
        """Decode image from disk for given index."""
        img_file = str(self._image_path(index))
        img = cv2.imdecode(
            np.fromfile(img_file, dtype=np.uint8),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        if img is None:
            raise ValueError(f"Failed to load {img_file}")
        return img

    # load_resized_img comes from ImageCacheMixin: the deterministic resize is
    # the post-resize cache point, so the mixin owns both the math and the cache.

    def _load_segments(self, index: int):
        if self.segments is None:
            return None
        return copy.deepcopy(self.segments[index])

    def pull_item(self, index: int):
        """Get item without preprocessing."""
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        segments = self._load_segments(index)
        if getattr(self.preproc, "wants_unresized_image", False):
            # Preprocessor handles all resizing in one pass (avoids the
            # letterbox-then-stretch double-resize). Targets are already
            # scaled by the dataset's letterbox ratio; we undo that here so
            # the preprocessor sees them in original-image coords matching
            # the original-image pixels we hand over.
            img = self.load_image(index)
            label = copy.deepcopy(label)
            if label.shape[0] > 0:
                target_h, target_w = self.img_size
                r = min(target_h / origin_image_size[0], target_w / origin_image_size[1])
                if r > 0:
                    label[:, :4] = label[:, :4] / r
            if self.load_segments:
                return img, label, origin_image_size, id_, segments
            return img, label, origin_image_size, id_
        img = self.load_resized_img(index)
        if self.load_segments:
            return img, copy.deepcopy(label), origin_image_size, id_, segments
        return img, copy.deepcopy(label), origin_image_size, id_

    def __getitem__(self, index: int):
        """Get preprocessed item."""
        item = self.pull_item(index)
        if len(item) == 5:
            img, target, img_info, img_id, segments = item
        else:
            img, target, img_info, img_id = item
            segments = None

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        if self.load_segments:
            return img, target, img_info, img_id, segments
        return img, target, img_info, img_id


def yolox_collate_fn(batch):
    """
    Collate function for YOLOX training.

    Returns:
        imgs: (B, C, H, W) tensor
        targets: (B, max_labels, 5) tensor
        img_infos: tuple of image info
        img_ids: tuple of image ids
    """
    has_segments = len(batch[0]) == 5
    if has_segments:
        imgs, targets, img_infos, img_ids, segments = zip(*batch)
    else:
        imgs, targets, img_infos, img_ids = zip(*batch)

    # Stack images
    imgs = torch.from_numpy(np.stack(imgs))

    # Stack targets (already padded to max_labels)
    targets = torch.from_numpy(np.stack(targets))

    if has_segments:
        if all(isinstance(s, np.ndarray) for s in segments):
            return imgs, targets, img_infos, img_ids, _pad_stack_masks(segments)
        if all(isinstance(s, torch.Tensor) for s in segments):
            return imgs, targets, img_infos, img_ids, _pad_stack_mask_tensors(segments)
        return imgs, targets, img_infos, img_ids, list(segments)
    return imgs, targets, img_infos, img_ids


def _pad_stack_masks(masks_list):
    """Stack per-image ``(n_i, H, W)`` mask arrays into ``(B, max_n, H, W)``.

    Transforms emit only as many mask rows as an image has instances (mask
    row ``i`` aligns with label row ``i``); padding to a fixed ``max_labels``
    slot count made the seg label buffer dwarf the image batch and blow up
    host RAM with multiple dataloader workers on COCO-scale datasets
    (issue #527). Pad rows are zero masks, matching the old padded contract.
    """
    max_n = max((m.shape[0] for m in masks_list), default=0)
    h, w = masks_list[0].shape[-2:]
    out = np.zeros((len(masks_list), max_n, h, w), dtype=masks_list[0].dtype)
    for i, m in enumerate(masks_list):
        if m.shape[0]:
            out[i, : m.shape[0]] = m
    return torch.from_numpy(out)


def _pad_stack_mask_tensors(masks_list):
    """Tensor twin of :func:`_pad_stack_masks`, preserving device and dtype."""
    max_n = max((m.shape[0] for m in masks_list), default=0)
    h, w = masks_list[0].shape[-2:]
    out = masks_list[0].new_zeros((len(masks_list), max_n, h, w))
    for i, m in enumerate(masks_list):
        if m.shape[0]:
            out[i, : m.shape[0]] = m
    return out


class DistributedWithReplacementSampler(Sampler):
    """With-replacement, DDP-aware sampler for the ``min_samples`` epoch floor.

    Mirrors ``DistributedSampler`` semantics: every rank makes the same
    per-epoch draw of ``total_size`` indices with replacement, seeded with
    ``seed + epoch`` (call ``set_epoch`` each epoch, exactly like
    ``DistributedSampler``, so draws differ per epoch while staying
    deterministic for resume), then keeps its ``rank::num_replicas`` slice.
    ``num_samples`` is rounded up to ``ceil(num_samples / num_replicas)`` per
    rank so every rank yields the same count and no rank runs out of batches
    before the others.
    """

    def __init__(
        self,
        dataset,
        num_samples: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ):
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if not 0 <= rank < num_replicas:
            raise ValueError(
                f"rank must be in [0, {num_replicas - 1}], got {rank}"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.num_samples = math.ceil(num_samples / num_replicas)
        self.total_size = self.num_samples * num_replicas
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randint(
            len(self.dataset), (self.total_size,), generator=g
        ).tolist()
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def create_dataloader(
    dataset,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    sampler=None,
    min_samples: int = 0,
    class_balanced: bool = False,
):
    """
    Create a DataLoader for YOLOX training.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Shuffle data (ignored when ``sampler`` is given — PyTorch
            forbids passing both)
        pin_memory: Pin memory for faster GPU transfer
        sampler: Optional sampler (e.g. ``DistributedSampler`` for DDP). When
            provided, the sampler's own shuffling takes over and ``shuffle``
            is forced to False to satisfy PyTorch's mutual-exclusion check.
        min_samples: Opt-in epoch-length floor for tiny datasets. When > 0
            and ``len(dataset) < min_samples``, each epoch draws
            ``min_samples`` samples with replacement instead of one short
            pass over the dataset: a with-replacement ``RandomSampler``
            replaces shuffling, a ``DistributedSampler`` passed via
            ``sampler`` is swapped for its with-replacement equivalent (any
            other custom sampler is respected as-is), and ``num_workers`` is
            clamped to ``len(dataset)``. 0 (the default) leaves the loader
            exactly as before the knob existed.
        class_balanced: Opt-in LVIS-style repeat-factor sampling. Off by
            default. When on, replaces the shuffle / DistributedSampler with
            a weighted draw; ``min_samples`` still floors the draw length.
            Cannot be combined with a custom non-distributed sampler.
    """
    if class_balanced:
        from torch.utils.data.distributed import DistributedSampler

        from .class_balanced import build_class_balanced_sampler

        if sampler is not None and not isinstance(sampler, DistributedSampler):
            raise ValueError(
                "class_balanced=True cannot be combined with a custom "
                f"sampler ({type(sampler).__name__}). Pass sampler=None or "
                "a DistributedSampler."
            )
        draw = len(dataset)
        if min_samples > 0 and 0 < len(dataset) < min_samples:
            draw = min_samples
            num_workers = min(num_workers, len(dataset))
        sampler = build_class_balanced_sampler(
            dataset, num_samples=draw, distributed_sampler=sampler
        )
        if is_main_process():
            logger.info(
                "class_balanced sampling active: %d images, drawing %d "
                "samples per epoch%s",
                len(dataset),
                len(sampler),
                " (DDP shard)" if hasattr(sampler, "num_replicas") else "",
            )
    elif min_samples > 0 and 0 < len(dataset) < min_samples:
        num_workers = min(num_workers, len(dataset))
        if sampler is None:
            # Draws from the global torch RNG, the same source the default
            # ``shuffle=True`` RandomSampler uses, so sampling stays
            # reproducible under torch.manual_seed and differs epoch to
            # epoch.
            sampler = RandomSampler(
                dataset, replacement=True, num_samples=min_samples
            )
        else:
            from torch.utils.data.distributed import DistributedSampler

            if isinstance(sampler, DistributedSampler):
                sampler = DistributedWithReplacementSampler(
                    dataset,
                    num_samples=min_samples,
                    num_replicas=sampler.num_replicas,
                    rank=sampler.rank,
                    seed=sampler.seed,
                )
        if is_main_process():
            logger.info(
                f"min_samples={min_samples} epoch floor active: dataset has "
                f"{len(dataset)} images; drawing {len(sampler)} samples per "
                f"epoch per rank with replacement"
            )
    try:
        visible_samples = len(sampler) if sampler is not None else len(dataset)
    except TypeError:
        visible_samples = len(dataset)
    drop_last = visible_samples >= batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=yolox_collate_fn,
        drop_last=drop_last,
    )
