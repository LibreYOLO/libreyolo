"""
YOLOCocoAPI - COCO-compatible evaluation interface for YOLO text-label datasets.

Ported from RF-DETR to enable COCO evaluation on YOLO format datasets.
This provides the pycocotools-compatible interface needed for standard COCO metrics.

Copyright (c) 2025 Roboflow (RF-DETR). Licensed under Apache 2.0.
Adapted for LibreYOLO.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .utils import img2label_paths, polygon_to_cxcywh

logger = logging.getLogger(__name__)


def parse_yolo_label_line(
    line: str,
    img_w: int,
    img_h: int,
    num_classes: Optional[int],
    label_path: Optional[Path] = None,
    return_segment: bool = False,
    single_cls: bool = False,
    class_remap: Optional[Dict[int, int]] = None,
) -> Optional[Tuple]:
    """
    Parse a single line from a YOLO label file.

    Args:
        line: Label line from .txt file
        img_w: Image width in pixels
        img_h: Image height in pixels
        num_classes: Total number of classes, or ``None`` when the caller does
            not have the dataset class count available.
        label_path: Path to label file (for warnings)
        single_cls: Remap non-negative class ids to class 0 before validation.
            Ignored when ``class_remap`` is given.
        class_remap: Optional ``{orig_id: new_id}`` mapping for training on a
            class subset (see ``load_data_config(classes=...)``). A class id
            not present as a key is silently dropped (excluded by design, not
            a data error); one present is rewritten to its mapped id before
            the bounds check below. Takes precedence over ``single_cls``.

    Returns:
        Tuple of (class_id, x1, y1, x2, y2, area) in pixel coordinates,
        or None if invalid/skipped.

    YOLO format: class_id center_x center_y width height
    (normalized coordinates in [0, 1])
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) < 5:
        return None

    # Parse values with error handling
    try:
        class_id = int(parts[0])

        segment = None
        if len(parts) > 5:
            # Segmentation format: derive bbox from polygon vertices.
            coords = [float(p) for p in parts[1:]]
            if len(coords) < 6 or len(coords) % 2:
                raise ValueError("polygon rows require at least 3 coordinate pairs")
            if not np.isfinite(coords).all():
                raise ValueError("label coordinates must be finite")
            cx, cy, bw, bh = polygon_to_cxcywh(coords)
            if return_segment:
                segment = []
                for x, y in zip(coords[0::2], coords[1::2]):
                    segment.extend(
                        [
                            float(max(0, min(img_w, x * img_w))),
                            float(max(0, min(img_h, y * img_h))),
                        ]
                    )
        else:
            # Detection format: class_id cx cy w h
            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
            if not np.isfinite((cx, cy, bw, bh)).all():
                raise ValueError("label coordinates must be finite")
    except ValueError as e:
        if label_path:
            warnings.warn(
                f"Invalid label format in {label_path}: '{line[:50]}...' - {e}"
            )
        return None

    if class_remap is not None:
        if class_id not in class_remap:
            return None
        class_id = class_remap[class_id]
    elif single_cls and class_id >= 0:
        class_id = 0

    # Validate class ID
    if class_id < 0 or (num_classes is not None and class_id >= num_classes):
        valid_range = (
            f"[0, {num_classes - 1}]" if num_classes is not None else "non-negative"
        )
        warnings.warn(
            f"Class ID {class_id} out of range {valid_range} in {label_path}. Skipping."
        )
        return None

    # Convert from normalized cxcywh to pixel xyxy
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h

    # Clamp to image boundaries
    x1 = max(0, min(img_w, x1))
    y1 = max(0, min(img_h, y1))
    x2 = max(0, min(img_w, x2))
    y2 = max(0, min(img_h, y2))

    # Skip invalid boxes (zero or negative area after clamping)
    if x2 <= x1 or y2 <= y1:
        return None

    # Compute area from clamped box
    area = (x2 - x1) * (y2 - y1)

    result = (class_id, x1, y1, x2, y2, area)
    if return_segment:
        return (*result, segment)
    return result


class YOLOCocoAPI:
    """
    Minimal COCO-compatible API for YOLO text-label datasets.

    Provides the interface needed by pycocotools.cocoeval.COCOeval for
    evaluating predictions on YOLO format datasets.

    This enables using COCO evaluation metrics (AP, AR, AP50, AP75, etc.)
    on YOLO-format datasets without converting to COCO JSON.

    Example:
        >>> # For validation dataset
        >>> coco_api = YOLOCocoAPI(
        ...     images_dir="path/to/images/val",
        ...     labels_dir="path/to/labels/val",
        ...     class_names=["person", "car", "dog"],
        ... )
        >>> # Use with COCOeval
        >>> from pycocotools.cocoeval import COCOeval
        >>> coco_eval = COCOeval(coco_api, coco_dt, 'bbox')
        >>> coco_eval.evaluate()
    """

    def __init__(
        self,
        images_dir: Path | None,
        labels_dir: Path | None,
        class_names: List[str],
        load_segments: bool = False,
        image_files: List[Path] | None = None,
        label_files: List[Path] | None = None,
        single_cls: bool = False,
        class_remap: Dict[int, int] | None = None,
    ):
        """
        Initialize COCO API for a YOLO dataset.

        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing .txt label files
            class_names: List of class names (from data.yaml)
            single_cls: Remap all non-negative ground-truth classes to class 0.
                Ignored when ``class_remap`` is given.
            class_remap: Optional ``{orig_id: new_id}`` mapping for scoring a
                class subset (see ``load_data_config(classes=...)``). A class
                id not present as a key is dropped from the ground truth,
                matching the dataloader side of ``classes=`` so mAP is
                computed over the requested subset, not the full dataset.
        """
        self.images_dir = Path(images_dir) if images_dir is not None else None
        self.labels_dir = Path(labels_dir) if labels_dir is not None else None
        self.class_names = class_names
        self.load_segments = load_segments
        self.single_cls = bool(single_cls)
        self.class_remap = class_remap
        num_classes = len(class_names)

        # Build COCO-style data structures
        self.imgs = {}
        self.anns = {}
        self.cats = {}
        self.imgToAnns = {}

        if image_files is not None:
            image_files = [Path(p) for p in image_files]
            if label_files is not None:
                label_files = [Path(p) for p in label_files]
            elif self.labels_dir is not None:
                label_files = [self.labels_dir / f"{p.stem}.txt" for p in image_files]
            else:
                label_files = img2label_paths(image_files)
        else:
            if self.images_dir is None:
                raise ValueError("images_dir or image_files must be provided")
            image_files = []
            extensions = [
                "*.jpg",
                "*.jpeg",
                "*.png",
                "*.bmp",
                "*.webp",
                "*.tiff",
                "*.tif",
                "*.gif",
            ]
            for ext in extensions:
                image_files.extend(list(self.images_dir.glob(ext)))
                image_files.extend(list(self.images_dir.glob(ext.upper())))

            image_files = sorted(set(image_files))
            if self.labels_dir is None:
                raise ValueError("labels_dir must be provided when image_files is omitted")
            label_files = [self.labels_dir / f"{p.stem}.txt" for p in image_files]

        if len(image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir or 'image_files'}")

        logger.info("Building COCO API for %d images...", len(image_files))

        ann_id = 1
        for idx, img_path in enumerate(image_files):
            # Get image size
            if img_path.suffix.lower() == ".npy":
                from ..utils.event_histogram import load_histogram
                h, w = load_histogram(img_path).shape[:2]
            else:
                with Image.open(img_path) as img:
                    w, h = img.size

            img_id = idx
            self.imgs[img_id] = {
                "id": img_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
            }
            self.imgToAnns[img_id] = []

            # Parse labels using shared function (same filtering as dataset __getitem__)
            label_path = label_files[idx]
            if label_path.exists():
                with open(label_path, "r") as f:
                    for line in f:
                        parsed = parse_yolo_label_line(
                            line,
                            w,
                            h,
                            num_classes,
                            label_path,
                            return_segment=load_segments,
                            single_cls=self.single_cls,
                            class_remap=self.class_remap,
                        )
                        if parsed is None:
                            continue

                        if load_segments:
                            class_id, x1, y1, x2, y2, area, segment = parsed
                        else:
                            class_id, x1, y1, x2, y2, area = parsed
                            segment = None

                        # Convert to COCO bbox format [x, y, width, height]
                        box_w = x2 - x1
                        box_h = y2 - y1

                        ann = {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": class_id,
                            "bbox": [x1, y1, box_w, box_h],
                            "area": area,
                            "iscrowd": 0,
                        }
                        if load_segments:
                            if segment:
                                ann["segmentation"] = [segment]
                            else:
                                ann["segmentation"] = []
                        self.anns[ann_id] = ann
                        self.imgToAnns[img_id].append(ann)
                        ann_id += 1

        # Build categories. class_remap's keys are original dataset ids, which
        # only line up with this label-index domain for plain classes=
        # filtering -- single_cls (with or without classes=) has already
        # collapsed class_names to one "object" entry upstream, so leave that
        # single category alone rather than checking it against class_remap.
        for i, name in enumerate(class_names):
            if (
                self.class_remap is not None
                and not self.single_cls
                and i not in self.class_remap
            ):
                continue
            self.cats[i] = {"id": i, "name": name, "supercategory": "object"}

        logger.info(
            "  Loaded %d annotations across %d images", len(self.anns), len(self.imgs)
        )

    def getAnnIds(self, imgIds=None, catIds=None, areaRng=None, iscrowd=None):
        """
        Get annotation IDs matching criteria.

        Args:
            imgIds: Image IDs to filter by (None = all)
            catIds: Category IDs to filter by (None = all)
            areaRng: Area range [min, max] to filter by (not implemented)
            iscrowd: Filter by iscrowd flag (not implemented)

        Returns:
            List of annotation IDs
        """
        if imgIds is None:
            imgIds = list(self.imgs.keys())
        if not isinstance(imgIds, list):
            imgIds = [imgIds]

        ann_ids = []
        for img_id in imgIds:
            for ann in self.imgToAnns.get(img_id, []):
                if catIds is not None and ann["category_id"] not in catIds:
                    continue
                ann_ids.append(ann["id"])
        return ann_ids

    def getCatIds(self, catNms=None, supNms=None, catIds=None):
        """
        Get category IDs.

        Args:
            catNms: Category names (not implemented)
            supNms: Supercategory names (not implemented)
            catIds: Category IDs (returns these if provided)

        Returns:
            List of category IDs
        """
        if catIds is not None:
            return catIds
        return list(self.cats.keys())

    def getImgIds(self, imgIds=None, catIds=None):
        """
        Get image IDs.

        Args:
            imgIds: Image IDs (returns these if provided)
            catIds: Category IDs (not implemented)

        Returns:
            List of image IDs
        """
        if imgIds is not None:
            return imgIds if isinstance(imgIds, list) else [imgIds]
        return list(self.imgs.keys())

    def loadAnns(self, ids=None):
        """
        Load annotations by IDs.

        Args:
            ids: Annotation IDs (None = all)

        Returns:
            List of annotation dicts
        """
        if ids is None:
            return list(self.anns.values())
        if not isinstance(ids, list):
            ids = [ids]
        return [self.anns[i] for i in ids if i in self.anns]

    def loadCats(self, ids=None):
        """
        Load categories by IDs.

        Args:
            ids: Category IDs (None = all)

        Returns:
            List of category dicts
        """
        if ids is None:
            return list(self.cats.values())
        if not isinstance(ids, list):
            ids = [ids]
        return [self.cats[i] for i in ids if i in self.cats]

    def loadImgs(self, ids=None):
        """
        Load images by IDs.

        Args:
            ids: Image IDs (None = all)

        Returns:
            List of image dicts
        """
        if ids is None:
            return list(self.imgs.values())
        if not isinstance(ids, list):
            ids = [ids]
        return [self.imgs[i] for i in ids if i in self.imgs]

    def loadRes(self, results):
        """
        Load detection results and create a result COCO object.

        This mimics pycocotools.coco.COCO.loadRes() to enable COCO evaluation.

        Args:
            results: List of detection dicts in COCO format:
                [{'image_id': int, 'category_id': int, 'bbox': [x,y,w,h], 'score': float}, ...]

        Returns:
            YOLOCocoAPI instance with result annotations
        """
        # Create a new YOLOCocoAPI instance for results
        res_coco = YOLOCocoAPI.__new__(YOLOCocoAPI)

        # Copy ground truth data structures
        res_coco.dataset = self.dataset.copy() if hasattr(self, "dataset") else {}
        res_coco.imgs = self.imgs.copy()
        res_coco.cats = self.cats.copy()
        res_coco.imgToAnns = {img_id: [] for img_id in self.imgs.keys()}

        # Add result annotations
        res_coco.anns = {}
        # COCOeval stores detection ids in gtMatches and treats zero as
        # unmatched. A zero-id prediction lets later duplicates reuse its GT.
        for ann_id, result in enumerate(results, start=1):
            ann = {
                "id": ann_id,
                "image_id": result["image_id"],
                "category_id": result["category_id"],
                "bbox": result["bbox"],  # [x, y, w, h]
                "area": result["bbox"][2] * result["bbox"][3],  # w * h
                "iscrowd": 0,
                "score": result.get("score", 1.0),
            }
            if "segmentation" in result:
                ann["segmentation"] = result["segmentation"]
                # Prefer true mask area over bbox w*h so the COCO small/medium/
                # large AP bucketing matches the segmentation, not the box.
                try:
                    from pycocotools import mask as mask_utils

                    img_meta = res_coco.imgs.get(result["image_id"], {})
                    h, w = img_meta.get("height"), img_meta.get("width")
                    segm = result["segmentation"]
                    rle = None
                    if isinstance(segm, list) and len(segm) > 0 and h and w:
                        rle = mask_utils.merge(mask_utils.frPyObjects(segm, h, w))
                    elif isinstance(segm, dict):
                        rle = dict(segm)
                        if isinstance(rle.get("counts"), list) and h and w:
                            rle = mask_utils.frPyObjects(rle, h, w)
                        elif isinstance(rle.get("counts"), str):
                            rle["counts"] = rle["counts"].encode("ascii")
                    if rle is not None:
                        ann["area"] = float(mask_utils.area(rle))
                except ImportError:
                    pass
            res_coco.anns[ann_id] = ann

            img_id = result["image_id"]
            if img_id in res_coco.imgToAnns:
                res_coco.imgToAnns[img_id].append(ann)

        return res_coco

    def annToRLE(self, ann):
        """Convert an annotation segmentation to pycocotools RLE."""
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            raise ImportError(
                "pycocotools not installed. Install with: pip install pycocotools"
            )

        img = self.imgs[ann["image_id"]]
        h, w = img["height"], img["width"]
        segm = ann.get("segmentation")

        if isinstance(segm, list) and len(segm) > 0:
            rles = mask_utils.frPyObjects(segm, h, w)
            return mask_utils.merge(rles)
        if isinstance(segm, dict):
            rle = dict(segm)
            if isinstance(rle.get("counts"), list):
                rle = mask_utils.frPyObjects(rle, h, w)
            elif isinstance(rle.get("counts"), str):
                rle["counts"] = rle["counts"].encode("ascii")
            return rle

        empty = np.zeros((h, w), dtype=np.uint8)
        return mask_utils.encode(np.asfortranarray(empty))

    def annToMask(self, ann):
        """Convert an annotation segmentation to a binary mask."""
        from pycocotools import mask as mask_utils

        return mask_utils.decode(self.annToRLE(ann))


def create_yolo_coco_api(
    data_yaml_path: str,
    split: str = "val",
    load_segments: bool = False,
    *,
    single_cls: bool = False,
) -> YOLOCocoAPI:
    """
    Create YOLOCocoAPI from a data.yaml file.

    Args:
        data_yaml_path: Path to data.yaml file
        split: Dataset split ('train', 'val', or 'test')
        single_cls: Build a one-class ground-truth view.

    Returns:
        YOLOCocoAPI instance

    Example:
        >>> coco_api = create_yolo_coco_api("path/to/data.yaml", split="val")
        >>> # Use with COCOeval
    """
    from .utils import load_data_config

    # Reuse the same YAML alias and dataset-root resolution logic as the main
    # loader so COCO evaluation works for dataset names like "coco128.yaml".
    data = load_data_config(
        data_yaml_path,
        autodownload=False,
        single_cls=single_cls,
    )
    root = Path(data["path"])

    # Get class names
    names = data.get("names", [])
    if isinstance(names, dict):
        # Format: {0: 'class0', 1: 'class1', ...}
        max_idx = max(names.keys()) if names else -1
        class_names = [names.get(i, f"class_{i}") for i in range(max_idx + 1)]
    else:
        # Format: ['class0', 'class1', ...]
        class_names = list(names)

    # Get split paths
    split_key = split.split("_")[0]  # Handle 'val_speed' etc.
    img_files = data.get(f"{split_key}_img_files")
    if img_files:
        return YOLOCocoAPI(
            images_dir=None,
            labels_dir=None,
            class_names=class_names,
            load_segments=load_segments,
            image_files=img_files,
            label_files=data.get(f"{split_key}_label_files"),
            single_cls=single_cls,
        )

    images_subpath = data.get(split_key, f"images/{split_key}")
    images_dir = Path(images_subpath)
    if not images_dir.is_absolute():
        images_dir = root / images_subpath
    if not images_dir.exists():
        # Try with 'images/' prefix
        images_dir = root / "images" / split_key
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")

    # Resolve labels directory (parallel to images)
    # Handle both /images/ in middle and /images at end of path
    images_str = str(images_dir)
    if images_str.endswith("/images") or images_str.endswith("\\images"):
        # Path ends with /images (e.g., test/images)
        labels_dir = images_dir.parent / "labels"
    else:
        # Path contains /images/ in middle (e.g., images/test)
        labels_dir = Path(
            images_str.replace("/images/", "/labels/").replace(
                "\\images\\", "\\labels\\"
            )
        )

    if not labels_dir.exists():
        # Alternative: labels next to images dir
        labels_dir = images_dir.parent.parent / "labels" / images_dir.name
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")

    return YOLOCocoAPI(
        images_dir=images_dir,
        labels_dir=labels_dir,
        class_names=class_names,
        load_segments=load_segments,
        single_cls=single_cls,
    )
