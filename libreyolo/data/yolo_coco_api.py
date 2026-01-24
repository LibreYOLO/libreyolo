"""
YOLOCocoAPI - COCO-compatible evaluation interface for Ultralytics YOLO format datasets.

Ported from RF-DETR to enable COCO evaluation on YOLO format datasets.
This provides the pycocotools-compatible interface needed for standard COCO metrics.

Copyright (c) 2025 Roboflow (RF-DETR). Licensed under Apache 2.0.
Adapted for LibreYOLO.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import warnings

from PIL import Image


def parse_yolo_label_line(
    line: str,
    img_w: int,
    img_h: int,
    num_classes: int,
    label_path: Optional[Path] = None
) -> Optional[Tuple[int, float, float, float, float, float]]:
    """
    Parse a single line from a YOLO label file.

    Args:
        line: Label line from .txt file
        img_w: Image width in pixels
        img_h: Image height in pixels
        num_classes: Total number of classes
        label_path: Path to label file (for warnings)

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

    # Warn about segmentation format (more than 5 columns)
    if len(parts) > 5:
        warnings.warn(
            f"Label line has {len(parts)} columns (expected 5 for detection). "
            f"This may be a segmentation dataset which is not supported. "
            f"File: {label_path}, Line: '{line[:50]}...'"
        )

    # Parse values with error handling
    try:
        class_id = int(parts[0])
        cx = float(parts[1])
        cy = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])
    except ValueError as e:
        if label_path:
            warnings.warn(f"Invalid label format in {label_path}: '{line[:50]}...' - {e}")
        return None

    # Validate class ID
    if class_id < 0 or class_id >= num_classes:
        warnings.warn(
            f"Class ID {class_id} out of range [0, {num_classes-1}] in {label_path}. Skipping."
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

    return (class_id, x1, y1, x2, y2, area)


class YOLOCocoAPI:
    """
    Minimal COCO-compatible API for Ultralytics YOLO format datasets.

    Provides the interface needed by pycocotools.cocoeval.COCOeval for
    evaluating predictions on YOLO format datasets.

    This enables using COCO evaluation metrics (AP, AR, AP50, AP75, etc.)
    on datasets in Ultralytics format without converting to COCO JSON.

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
        images_dir: Path,
        labels_dir: Path,
        class_names: List[str],
    ):
        """
        Initialize COCO API for a YOLO dataset.

        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing .txt label files
            class_names: List of class names (from data.yaml)
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names = class_names
        num_classes = len(class_names)

        # Build COCO-style data structures
        self.imgs = {}
        self.anns = {}
        self.cats = {}
        self.imgToAnns = {}

        # Find all images
        image_files = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.tiff', '*.tif', '*.gif']
        for ext in extensions:
            image_files.extend(list(self.images_dir.glob(ext)))
            image_files.extend(list(self.images_dir.glob(ext.upper())))

        image_files = sorted(image_files)

        if len(image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

        print(f"Building COCO API for {len(image_files)} images...")

        ann_id = 1
        for idx, img_path in enumerate(image_files):
            # Get image size
            with Image.open(img_path) as img:
                w, h = img.size

            img_id = idx
            self.imgs[img_id] = {
                'id': img_id,
                'file_name': img_path.name,
                'width': w,
                'height': h,
            }
            self.imgToAnns[img_id] = []

            # Parse labels using shared function (same filtering as dataset __getitem__)
            label_path = self.labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parsed = parse_yolo_label_line(line, w, h, num_classes, label_path)
                        if parsed is None:
                            continue

                        class_id, x1, y1, x2, y2, area = parsed

                        # Convert to COCO bbox format [x, y, width, height]
                        box_w = x2 - x1
                        box_h = y2 - y1

                        ann = {
                            'id': ann_id,
                            'image_id': img_id,
                            'category_id': class_id,
                            'bbox': [x1, y1, box_w, box_h],
                            'area': area,
                            'iscrowd': 0,
                        }
                        self.anns[ann_id] = ann
                        self.imgToAnns[img_id].append(ann)
                        ann_id += 1

        # Build categories
        for i, name in enumerate(class_names):
            self.cats[i] = {'id': i, 'name': name, 'supercategory': 'object'}

        print(f"  Loaded {len(self.anns)} annotations across {len(self.imgs)} images")

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
                if catIds is not None and ann['category_id'] not in catIds:
                    continue
                ann_ids.append(ann['id'])
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
        res_coco.dataset = self.dataset.copy() if hasattr(self, 'dataset') else {}
        res_coco.imgs = self.imgs.copy()
        res_coco.cats = self.cats.copy()
        res_coco.imgToAnns = {img_id: [] for img_id in self.imgs.keys()}

        # Add result annotations
        res_coco.anns = {}
        for ann_id, result in enumerate(results):
            ann = {
                'id': ann_id,
                'image_id': result['image_id'],
                'category_id': result['category_id'],
                'bbox': result['bbox'],  # [x, y, w, h]
                'area': result['bbox'][2] * result['bbox'][3],  # w * h
                'iscrowd': 0,
                'score': result.get('score', 1.0),
            }
            res_coco.anns[ann_id] = ann

            img_id = result['image_id']
            if img_id in res_coco.imgToAnns:
                res_coco.imgToAnns[img_id].append(ann)

        return res_coco


def create_yolo_coco_api(data_yaml_path: str, split: str = 'val') -> YOLOCocoAPI:
    """
    Create YOLOCocoAPI from a data.yaml file.

    Args:
        data_yaml_path: Path to data.yaml file
        split: Dataset split ('train', 'val', or 'test')

    Returns:
        YOLOCocoAPI instance

    Example:
        >>> coco_api = create_yolo_coco_api("path/to/data.yaml", split="val")
        >>> # Use with COCOeval
    """
    import yaml

    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Get root directory
    root = Path(data_yaml_path).parent
    if 'path' in data:
        root_override = Path(data['path'])
        if not root_override.is_absolute():
            root = root / root_override
        else:
            root = root_override

    # Get class names
    names = data.get('names', [])
    if isinstance(names, dict):
        # Format: {0: 'class0', 1: 'class1', ...}
        max_idx = max(names.keys()) if names else -1
        class_names = [names.get(i, f'class_{i}') for i in range(max_idx + 1)]
    else:
        # Format: ['class0', 'class1', ...]
        class_names = list(names)

    # Get split paths
    split_key = split.split("_")[0]  # Handle 'val_speed' etc.
    images_subpath = data.get(split_key, f'images/{split_key}')

    # Resolve images directory
    images_dir = root / images_subpath
    if not images_dir.exists():
        # Try with 'images/' prefix
        images_dir = root / 'images' / split_key
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")

    # Resolve labels directory (parallel to images)
    labels_dir = Path(str(images_dir).replace('/images/', '/labels/').replace('\\images\\', '\\labels\\'))
    if not labels_dir.exists():
        # Alternative: labels next to images dir
        labels_dir = images_dir.parent.parent / 'labels' / images_dir.name
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")

    return YOLOCocoAPI(
        images_dir=images_dir,
        labels_dir=labels_dir,
        class_names=class_names,
    )
