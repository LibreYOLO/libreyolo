"""
Data utilities for LibreYOLO.

Provides dataset configuration loading, auto-download, and path resolution.
Supports YAML configs with .txt file paths.
"""

from .classify_dataset import (
    ClassifyDataset,
    build_classify_collate,
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
from .depth_dataset import (
    DepthDataset,
    depth_collate_fn,
    img2depth_paths,
    resolve_depth_data,
)
from .normal_dataset import (
    NormalDataset,
    img2normal_mask_paths,
    img2normal_paths,
    normal_collate_fn,
    normalize_normal_vectors,
    resolve_normal_data,
)
from .edge_dataset import (
    EdgeDataset,
    edge_collate_fn,
    img2edge_mask_paths,
    img2edge_paths,
    load_edge_map,
    load_edge_mask,
    resolve_edge_data,
)
from .restore_dataset import (
    RestoreDataset,
    img2restore_target_paths,
    resolve_restore_data,
    restore_collate_fn,
)
from .panoptic_dataset import (
    PanopticDataset,
    panoptic_collate_fn,
    resolve_panoptic_data,
)
from .semantic_dataset import (
    SemanticDataset,
    img2mask_paths,
    resolve_semantic_data,
    semantic_collate_fn,
    valid_content_hw,
)
from .utils import (
    DATASETS_DIR,
    build_class_remap,
    check_dataset,
    get_coco_annotation_file,
    get_coco_image_dir,
    get_img_files,
    img2label_paths,
    load_data_config,
    resolve_default_coco_image_dir,
)
from .yolo_coco_api import YOLOCocoAPI, create_yolo_coco_api, parse_yolo_label_line

__all__ = [
    "DATASETS_DIR",
    "build_class_remap",
    "check_dataset",
    "get_coco_annotation_file",
    "get_coco_image_dir",
    "get_img_files",
    "img2label_paths",
    "load_data_config",
    "resolve_default_coco_image_dir",
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
    "build_classify_collate",
    "build_classify_transforms",
    "classify_collate_fn",
    "get_class_names",
    "resolve_classify_data",
    "DepthDataset",
    "depth_collate_fn",
    "img2depth_paths",
    "resolve_depth_data",
    "NormalDataset",
    "img2normal_mask_paths",
    "img2normal_paths",
    "normal_collate_fn",
    "normalize_normal_vectors",
    "resolve_normal_data",
    "EdgeDataset",
    "edge_collate_fn",
    "img2edge_mask_paths",
    "img2edge_paths",
    "load_edge_map",
    "load_edge_mask",
    "resolve_edge_data",
    "RestoreDataset",
    "img2restore_target_paths",
    "resolve_restore_data",
    "restore_collate_fn",
    "SemanticDataset",
    "PanopticDataset",
    "panoptic_collate_fn",
    "resolve_panoptic_data",
    "img2mask_paths",
    "resolve_semantic_data",
    "semantic_collate_fn",
    "valid_content_hw",
]
