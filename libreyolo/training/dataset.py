"""
Dataset classes for YOLOX training.

Supports both COCO JSON format and YOLO txt format.
"""

import copy
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset, DataLoader


def load_data_config(data_path: str) -> Dict:
    """
    Load data configuration from YAML file.

    Expected format:
    ```yaml
    path: /path/to/dataset  # dataset root
    train: images/train     # train images (relative to path)
    val: images/val         # val images (relative to path)
    nc: 80                  # number of classes
    names: ['person', 'bicycle', ...]  # class names
    ```
    """
    with open(data_path, 'r') as f:
        data = yaml.safe_load(f)

    # Resolve paths
    if 'path' in data:
        root = Path(data['path'])
        if not root.is_absolute():
            # Relative to yaml file location
            root = Path(data_path).parent / root
        data['root'] = str(root.resolve())
    else:
        data['root'] = str(Path(data_path).parent)

    return data


class YOLODataset(Dataset):
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
        data_dir: str = None,
        split: str = "train",
        img_size: Tuple[int, int] = (640, 640),
        preproc=None,
        img_files: List[Path] = None,
        label_files: List[Path] = None,
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
        """
        self.img_size = img_size
        self.preproc = preproc
        self._input_dim = img_size

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
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                self.img_files.extend(self.img_dir.glob(ext))
                self.img_files.extend(self.img_dir.glob(ext.upper()))
            self.img_files = sorted(self.img_files)

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
        annotations = []
        for img_file, label_file in zip(self.img_files, self.label_files):
            anno = self._load_label(label_file, img_file)
            annotations.append(anno)
        return annotations

    def _load_label(self, label_file: Path, img_file: Path) -> Tuple:
        """Load annotation for a single image."""
        # Read image to get dimensions
        img = cv2.imread(str(img_file))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_file}")

        height, width = img.shape[:2]

        # Load labels
        labels = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])

                        # Convert normalized xywh to pixel xyxy
                        x1 = (cx - w / 2) * width
                        y1 = (cy - h / 2) * height
                        x2 = (cx + w / 2) * width
                        y2 = (cy + h / 2) * height

                        labels.append([x1, y1, x2, y2, cls_id])

        # Create annotation array
        if labels:
            res = np.array(labels, dtype=np.float32)
        else:
            res = np.zeros((0, 5), dtype=np.float32)

        # Scale to target image size
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        if len(res) > 0:
            res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))
        file_name = img_file.name

        return (res, img_info, resized_info, file_name)

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

    def load_image(self, index: int) -> np.ndarray:
        """Load image for given index."""
        img_file = self.img_files[index]
        img = cv2.imread(str(img_file))
        assert img is not None, f"Failed to load {img_file}"
        return img

    def load_resized_img(self, index: int) -> np.ndarray:
        """Load and resize image."""
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def pull_item(self, index: int):
        """Get item without preprocessing."""
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.load_resized_img(index)
        return img, copy.deepcopy(label), origin_image_size, np.array([index])

    def __getitem__(self, index: int):
        """Get preprocessed item."""
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id


class COCODataset(Dataset):
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
        img_size: Tuple[int, int] = (640, 640),
        preproc=None,
    ):
        """
        Initialize COCO dataset.

        Args:
            data_dir: Path to dataset root
            json_file: COCO annotation JSON file name
            name: Image folder name (e.g., 'train2017')
            img_size: Target image size (height, width)
            preproc: Preprocessing transform
        """
        try:
            from pycocotools.coco import COCO
        except ImportError:
            raise ImportError(
                "pycocotools is required for COCO format. "
                "Install with: pip install pycocotools"
            )

        self.data_dir = data_dir
        self.json_file = json_file
        self.name = name
        self.img_size = img_size
        self._input_dim = img_size
        self.preproc = preproc

        # Load COCO annotations
        ann_file = os.path.join(data_dir, "annotations", json_file)
        self.coco = COCO(ann_file)

        # Remove useless info to save memory
        self._remove_useless_info()

        self.ids = self.coco.getImgIds()
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])

        # Pre-load annotations
        self.annotations = self._load_coco_annotations()

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
        for anno in dataset.get("annotations", []):
            anno.pop("segmentation", None)

    def _load_coco_annotations(self) -> List:
        """Load all annotations."""
        return [self._load_anno_from_id(id_) for id_ in self.ids]

    def _load_anno_from_id(self, id_: int) -> Tuple:
        """Load annotation for a single image ID."""
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)

        objs = []
        for obj in annotations:
            x1 = max(0, obj["bbox"][0])
            y1 = max(0, obj["bbox"][1])
            x2 = min(width, x1 + max(0, obj["bbox"][2]))
            y2 = min(height, y1 + max(0, obj["bbox"][3]))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)
        res = np.zeros((num_objs, 5), dtype=np.float32)
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        # Scale to target size
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))
        file_name = im_ann.get("file_name", f"{id_:012}.jpg")

        return (res, img_info, resized_info, file_name)

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

    def load_image(self, index: int) -> np.ndarray:
        """Load image for given index."""
        file_name = self.annotations[index][3]
        img_file = os.path.join(self.data_dir, self.name, file_name)
        img = cv2.imread(img_file)
        assert img is not None, f"Failed to load {img_file}"
        return img

    def load_resized_img(self, index: int) -> np.ndarray:
        """Load and resize image."""
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def pull_item(self, index: int):
        """Get item without preprocessing."""
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.load_resized_img(index)
        return img, copy.deepcopy(label), origin_image_size, np.array([id_])

    def __getitem__(self, index: int):
        """Get preprocessed item."""
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

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
    imgs, targets, img_infos, img_ids = zip(*batch)

    # Stack images
    imgs = torch.from_numpy(np.stack(imgs))

    # Stack targets (already padded to max_labels)
    targets = torch.from_numpy(np.stack(targets))

    return imgs, targets, img_infos, img_ids


def create_dataloader(
    dataset,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
):
    """
    Create a DataLoader for YOLOX training.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Shuffle data
        pin_memory: Pin memory for faster GPU transfer
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=yolox_collate_fn,
        drop_last=True,
    )
