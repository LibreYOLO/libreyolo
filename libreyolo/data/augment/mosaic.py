"""Mosaic tile geometry and paste-style mixup shared across YOLO families."""

import random

import cv2
import numpy as np

from .boxes import adjust_box_anns


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    """Return (large_coords, small_coords) for placing an image in the 2x2 mosaic."""
    # Top-left
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # Top-right
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # Bottom-left
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # Bottom-right
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    else:
        raise ValueError(f"mosaic_index must be in [0, 3], got {mosaic_index}")
    return (x1, y1, x2, y2), small_coord


def mixup_paste(dataset, origin_img, origin_labels, input_dim, mixup_scale):
    """YOLOX-style paste mixup: blend a random letterboxed sample at 50/50.

    ``dataset`` must expose ``load_anno(idx)`` and ``pull_item(idx)``; labels
    are pixel ``[x1, y1, x2, y2, class]`` rows. Returns the blended uint8 image
    and the concatenated label rows.
    """
    jit_factor = random.uniform(*mixup_scale)
    flip = random.uniform(0, 1) > 0.5
    cp_labels = []
    while len(cp_labels) == 0:
        cp_index = random.randint(0, len(dataset) - 1)
        cp_labels = dataset.load_anno(cp_index)
    img, cp_labels, _, _ = dataset.pull_item(cp_index)

    if len(img.shape) == 3:
        cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
    else:
        cp_img = np.ones(input_dim, dtype=np.uint8) * 114

    cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
        interpolation=cv2.INTER_LINEAR,
    )

    cp_img[
        : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
    ] = resized_img

    cp_img = cv2.resize(
        cp_img,
        (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
    )
    cp_scale_ratio *= jit_factor

    if flip:
        cp_img = cp_img[:, ::-1, :]

    origin_h, origin_w = cp_img.shape[:2]
    target_h, target_w = origin_img.shape[:2]
    padded_img = np.zeros(
        (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
    )
    padded_img[:origin_h, :origin_w] = cp_img

    x_offset, y_offset = 0, 0
    if padded_img.shape[0] > target_h:
        y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
    if padded_img.shape[1] > target_w:
        x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
    padded_cropped_img = padded_img[
        y_offset : y_offset + target_h, x_offset : x_offset + target_w
    ]

    cp_bboxes_origin_np = adjust_box_anns(
        cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
    )
    if flip:
        cp_bboxes_origin_np[:, 0::2] = (
            origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
        )
    cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
    cp_bboxes_transformed_np[:, 0::2] = np.clip(
        cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
    )
    cp_bboxes_transformed_np[:, 1::2] = np.clip(
        cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
    )

    cls_labels = cp_labels[:, 4:5].copy()
    box_labels = cp_bboxes_transformed_np
    labels = np.hstack((box_labels, cls_labels))
    origin_labels = np.vstack((origin_labels, labels))
    origin_img = origin_img.astype(np.float32)
    origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

    return origin_img.astype(np.uint8), origin_labels
