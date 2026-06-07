"""Shared YOLO-style augmentation primitives.

These helpers are intentionally low-level. Model families should compose them
in their own ``models/<family>/transforms.py`` recipe instead of moving recipe
logic into this shared package.
"""

import math
import random

import cv2
import numpy as np


def xyxy2cxcywh(bboxes):
    """Convert bboxes from xyxy to (cx, cy, w, h) in-place."""
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # w
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # h
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5  # cx
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5  # cy
    return bboxes


def cxcywh2xyxy(bboxes):
    """Convert bboxes from (cx, cy, w, h) to xyxy in-place."""
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5  # x1
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5  # y1
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # x2
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # y2
    return bboxes


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    """Scale + offset boxes, then clip to (w_max, h_max)."""
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    """Random HSV jitter (in-place)."""
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]
    hsv_augs *= np.random.randint(0, 2, 3)  # randomly zero-out each channel
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)


def get_aug_params(value, center=0):
    """Sample a random value from a float or (min, max) range."""
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            f"Affine params should be either a sequence containing two values "
            f"or single float values. Got {value}"
        )


def get_affine_matrix(target_size, degrees=10, translate=0.1, scales=0.1, shear=10):
    """Build a random 2x3 affine matrix (rotation + scale + shear + translation)."""
    twidth, theight = target_size

    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    translation_x = get_aug_params(translate) * twidth
    translation_y = get_aug_params(translate) * theight

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    """Warp box corners through M, then recompute axis-aligned bounds."""
    num_gts = len(targets)

    # Warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # Create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # Clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    """Random affine on image + box targets."""
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)

    return img, targets


def mirror(image, boxes, prob=0.5):
    """Random horizontal flip."""
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc(img, input_size, swap=(2, 0, 1)):
    """Letterbox resize + pad (114) + HWC->CHW transpose."""
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


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
