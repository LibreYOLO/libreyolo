"""
Data utilities for LibreYOLO.

Provides dataset configuration loading, auto-download, and path resolution.
Supports YAML configs with .txt file paths.
"""

from .utils import (
    DATASETS_DIR,
    IMG_FORMATS,
    check_dataset,
    get_img_files,
    img2label_paths,
    list_builtin_datasets,
    load_data_config,
    resolve_dataset_yaml,
    safe_download,
)
from .yolo_coco_api import YOLOCocoAPI, create_yolo_coco_api, parse_yolo_label_line

__all__ = [
    "DATASETS_DIR",
    "IMG_FORMATS",
    "check_dataset",
    "get_img_files",
    "img2label_paths",
    "list_builtin_datasets",
    "load_data_config",
    "resolve_dataset_yaml",
    "safe_download",
    "YOLOCocoAPI",
    "create_yolo_coco_api",
    "parse_yolo_label_line",
]
