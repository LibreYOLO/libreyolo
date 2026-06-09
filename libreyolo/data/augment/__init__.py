"""Shared data augmentation library.

All train-time augmentation logic lives in this package; model families
import their recipe from here (and may keep thin re-export shims at their
historical ``models/<family>/transforms.py`` paths for compatibility).

Module map:

- ``color``     — HSV jitter, brightness/contrast
- ``boxes``     — box format conversions and adjustments
- ``geometry``  — random affine, horizontal flip, letterbox variants
- ``mosaic``    — 2x2 mosaic tile geometry, paste-style mixup
- ``segments``  — polygon + dense-mask transforms and rasterization
- ``yolox``     — YOLOX recipe (also used by RTMDet, DAMO-YOLO, PicoDet)
- ``yolo9``     — YOLO9 recipe: normalized-xyxy targets, segments, OBB
- ``yolonas``   — YOLO-NAS recipe: affine + paste mixup, RGB/255 outputs
- ``detr``      — torchvision-v2 DETR recipe (DEIM, D-FINE, DEIMv2, RT-DETRv4, EC)
- ``rfdetr``    — RF-DETR square-resize recipe: det, seg, OBB, pose
- ``pose``      — keypoint-aware recipe shared by YOLO9/YOLO-NAS/EC pose

Only numpy/cv2 primitives are exported here; recipe modules that require
torch/torchvision (``detr``) must be imported explicitly so that importing
this package never pulls heavyweight dependencies.
"""

from .boxes import adjust_box_anns, cxcywh2xyxy, xyxy2cxcywh
from .color import augment_hsv, brightness_contrast
from .geometry import (
    apply_affine_to_bboxes,
    get_affine_matrix,
    get_aug_params,
    letterbox,
    letterbox_rgb01,
    mirror,
    preproc,
    random_affine,
)
from .mosaic import get_mosaic_coordinate, mixup_paste
from .segments import (
    copy_segments,
    crop_segments,
    filter_segments,
    flip_segments_lr,
    flip_segments_ud,
    materialize_dense_masks_for_crop,
    rasterize_segments,
    scale_segments_xy,
    transform_segments,
)

__all__ = [
    "adjust_box_anns",
    "apply_affine_to_bboxes",
    "augment_hsv",
    "brightness_contrast",
    "copy_segments",
    "crop_segments",
    "cxcywh2xyxy",
    "filter_segments",
    "flip_segments_lr",
    "flip_segments_ud",
    "get_affine_matrix",
    "get_aug_params",
    "get_mosaic_coordinate",
    "letterbox",
    "letterbox_rgb01",
    "materialize_dense_masks_for_crop",
    "mirror",
    "mixup_paste",
    "preproc",
    "random_affine",
    "rasterize_segments",
    "scale_segments_xy",
    "transform_segments",
    "xyxy2cxcywh",
]
