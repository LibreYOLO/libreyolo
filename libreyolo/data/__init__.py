"""
Data utilities for LibreYOLO.

Provides dataset configuration loading, auto-download, and path resolution.
Supports YAML configs with .txt file paths.
"""

from .classify_dataset import (
    ClassifyDataset,
    build_classify_transforms,
    classify_collate_fn,
    get_class_names,
    resolve_classify_data,
)
from .obb import parse_yolo_obb_label_line
from .coco_pose import (
    convert_coco_keypoints_json_to_yolo_pose,
    convert_coco_keypoints_splits,
)
from .pose_metadata import (
    COCO17_FLIP_IDX,
    COCO17_KEYPOINT_NAMES,
    COCO17_OKS_SIGMAS,
    COCO17_SKELETON,
    default_oks_sigmas,
)
from .pose_dataset import YOLOPoseDataset, parse_yolo_pose_label_line, pose_collate_fn
from .semantic_dataset import (
    SemanticDataset,
    img2mask_paths,
    resolve_semantic_data,
    semantic_collate_fn,
)
from .utils import (
    DATASETS_DIR,
    check_dataset,
    get_img_files,
    img2label_paths,
    load_data_config,
)
from .yolo_coco_api import YOLOCocoAPI, create_yolo_coco_api, parse_yolo_label_line

__all__ = [
    "DATASETS_DIR",
    "check_dataset",
    "get_img_files",
    "img2label_paths",
    "load_data_config",
    "YOLOCocoAPI",
    "create_yolo_coco_api",
    "parse_yolo_label_line",
    "parse_yolo_obb_label_line",
    "convert_coco_keypoints_json_to_yolo_pose",
    "convert_coco_keypoints_splits",
    "YOLOPoseDataset",
    "parse_yolo_pose_label_line",
    "pose_collate_fn",
    "COCO17_FLIP_IDX",
    "COCO17_KEYPOINT_NAMES",
    "COCO17_OKS_SIGMAS",
    "COCO17_SKELETON",
    "default_oks_sigmas",
    "ClassifyDataset",
    "build_classify_transforms",
    "classify_collate_fn",
    "get_class_names",
    "resolve_classify_data",
    "SemanticDataset",
    "img2mask_paths",
    "resolve_semantic_data",
    "semantic_collate_fn",
]
