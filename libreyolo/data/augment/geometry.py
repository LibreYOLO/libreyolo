"""Geometric augmentations and letterbox preprocessing shared by the numpy
pipelines.

Moved verbatim from ``libreyolo/training/augment.py`` (originally adapted
from the official YOLOX repository, Apache-2.0). The projective branch of
:func:`get_affine_matrix` is not from YOLOX: it follows the corner-displacement
design of torchvision's ``RandomPerspective`` (BSD-3-Clause) and solves the
homography with OpenCV's ``cv2.getPerspectiveTransform``.
"""

import math
import random

import cv2
import numpy as np

# ``perspective`` -> torchvision-style ``distortion_scale`` conversion, and the
# largest distortion_scale the projective branch will use. See
# :func:`get_affine_matrix`.
PERSPECTIVE_TO_DISTORTION = 100.0
MAX_PERSPECTIVE_DISTORTION = 0.2
# Smallest homogeneous coordinate used when dividing warped box corners.
_MIN_HOMOGENEOUS_W = 1e-6


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


def get_affine_matrix(
    target_size, degrees=10, translate=0.1, scales=0.1, shear=10, perspective=0.0
):
    """Build a random affine (or projective) warp matrix.

    With ``perspective == 0.0`` this returns the historical 2x3 affine matrix
    (rotation + scale + shear + translation, adapted from YOLOX), byte for
    byte, and draws no extra random numbers.

    With ``perspective > 0.0`` it returns a 3x3 homography ``M = P @ A`` that
    maps source pixels to the ``target_size`` canvas: ``A`` is that same affine
    (built from the same six random draws) and ``P`` is a random projective
    distortion of the canvas applied on top of it.

    ``P`` follows the design of ``torchvision.transforms.RandomPerspective``
    (BSD-3-Clause): the four canvas corners are each pulled inward by a random
    amount and ``P`` is the unique homography sending the canvas rectangle to
    that quadrilateral (solved with ``cv2.getPerspectiveTransform``, i.e. the
    standard four-point linear system). ``perspective`` sets the strength::

        distortion_scale = min(perspective * 100, 0.2)

    where ``distortion_scale`` has torchvision's meaning: every corner moves
    toward the canvas centre by ``uniform(0, distortion_scale * width / 2)``
    pixels in x and ``uniform(0, distortion_scale * height / 2)`` in y, all
    eight offsets independent. So ``perspective=0.001`` displaces each corner
    by at most 5% of the canvas side per axis (32 px at 640), independent of
    the canvas resolution. Unlike torchvision the offsets are continuous
    (``random.uniform``, eight draws made after the six affine draws) and the
    corners are the canvas extents ``(0, 0)..(width, height)``.

    Conditioning: because every corner stays inside its own corner region of
    size ``distortion_scale / 2`` of the canvas, the quadrilateral is always
    convex and ``P`` never folds over. With ``P`` scaled so ``P[2, 2] == 1``
    its denominator on the canvas stays within ``[0.81, 1.24]`` at
    ``perspective=0.001`` and above ``0.63`` at the 0.2 cap
    (``perspective >= 0.002``); larger values are clamped to the cap.
    """
    twidth, theight = target_size

    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    translation_x = get_aug_params(translate) * twidth
    translation_y = get_aug_params(translate) * theight

    if perspective == 0.0:
        M = np.ones([2, 3])
        M[0] = R[0] + shear_y * R[1]
        M[1] = R[1] + shear_x * R[0]
        M[0, 2] = translation_x
        M[1, 2] = translation_y
        return M, scale

    if perspective < 0.0:
        raise ValueError(f"perspective must be non-negative, got {perspective}")

    A = np.eye(3)
    A[0, :2] = R[0, :2] + shear_y * R[1, :2]
    A[1, :2] = R[1, :2] + shear_x * R[0, :2]
    A[0, 2] = translation_x
    A[1, 2] = translation_y

    distortion = min(perspective * PERSPECTIVE_TO_DISTORTION, MAX_PERSPECTIVE_DISTORTION)
    max_dx = distortion * twidth / 2
    max_dy = distortion * theight / 2
    # Corner order: top-left, top-right, bottom-right, bottom-left.
    src = np.array(
        [[0, 0], [twidth, 0], [twidth, theight], [0, theight]], dtype=np.float32
    )
    dst = np.array(
        [
            [random.uniform(0, max_dx), random.uniform(0, max_dy)],
            [twidth - random.uniform(0, max_dx), random.uniform(0, max_dy)],
            [twidth - random.uniform(0, max_dx), theight - random.uniform(0, max_dy)],
            [random.uniform(0, max_dx), theight - random.uniform(0, max_dy)],
        ],
        dtype=np.float32,
    )
    P = cv2.getPerspectiveTransform(src, dst)  # float64, P[2, 2] == 1

    return P @ A, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    """Warp box corners through M, then recompute axis-aligned bounds.

    ``M`` may be a 2x3 affine or a 3x3 homography; for the projective case the
    warped corners are divided by their homogeneous coordinate. A corner far
    outside the canvas can reach the homography's horizon (homogeneous
    coordinate <= 0); it is treated as a point at infinity in its direction, so
    the box extends to the canvas edge instead of folding over.
    """
    num_gts = len(targets)

    # Warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine / projective transform
    if M.shape[0] == 3:
        w = np.maximum(corner_points[:, 2:3], _MIN_HOMOGENEOUS_W)
        corner_points = corner_points[:, :2] / w
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
    perspective=0.0,
):
    """Random affine (or projective, when ``perspective != 0``) on image + boxes."""
    M, scale = get_affine_matrix(
        target_size, degrees, translate, scales, shear, perspective
    )

    if M.shape[0] == 3:
        img = cv2.warpPerspective(
            img, M, dsize=target_size, borderValue=(114, 114, 114)
        )
    else:
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


def mirror_vertical(image, boxes, prob=0.5):
    """Random vertical flip, mirroring :func:`mirror` across the y axis.

    ``boxes`` are xyxy in pixel coordinates: the y columns are reflected about
    the image height, matching the horizontal flip's ``x`` reflection.
    """
    height = image.shape[0]
    if random.random() < prob:
        image = image[::-1]
        boxes[:, 1::2] = height - boxes[:, 3::-2]
    return image, boxes


def rot90_image_boxes(image, boxes, k):
    """Rotate an image by ``k`` quarter turns (``np.rot90``) and remap xyxy boxes.

    ``k`` counter-clockwise quarter turns are applied to the image. Boxes are
    the horizontal proxy boxes used by the OBB path: their width and height are
    intrinsic to each (possibly oriented) rectangle and are therefore preserved,
    while the box center is rotated together with the image. Callers that track
    an orientation angle add ``k * pi / 2`` to it separately (see the YOLO9
    OBB transform); a 90-degree turn of the whole rectangle is equivalent to
    swapping its sides, so keeping width/height and turning the angle keeps the
    canonical long-side-is-width convention intact.
    """
    k = int(k) % 4
    rotated = np.ascontiguousarray(np.rot90(image, k)) if k else image
    if k == 0 or len(boxes) == 0:
        return rotated, boxes

    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    box_w = boxes[:, 2] - boxes[:, 0]
    box_h = boxes[:, 3] - boxes[:, 1]

    cur_w = image.shape[1]
    cur_h = image.shape[0]
    for _ in range(k):
        # One counter-clockwise quarter turn maps (x, y) -> (y, cur_w - x).
        cx, cy = cy, cur_w - cx
        cur_w, cur_h = cur_h, cur_w

    out = boxes.copy()
    out[:, 0] = cx - box_w * 0.5
    out[:, 1] = cy - box_h * 0.5
    out[:, 2] = cx + box_w * 0.5
    out[:, 3] = cy + box_h * 0.5
    return rotated, out


def letterbox_preproc(
    img,
    input_size,
    swap=(2, 0, 1),
    *,
    to_rgb=False,
    scale=False,
    letterbox_pad=None,
):
    """Letterbox resize + pad (114) + HWC→CHW transpose.

    The historical per-family ``preproc`` copies differed only in two finalize
    flags, exposed here as parameters:

    - ``to_rgb=False, scale=False`` — YOLOX/PicoDet/RTMDet (BGR, raw 0-255)
    - ``to_rgb=True, scale=True``   — YOLO9/RT-DETR/YOLO-NAS (RGB, /255)

    ``letterbox_pad`` is ``topleft`` (historical default) or ``center``.
    YOLOX and other non-YOLOv9 callers must leave it unset.
    """
    from libreyolo.preprocess.letterbox import apply_letterbox_hwc

    padded_img, r, _pad_left, _pad_top = apply_letterbox_hwc(
        img,
        int(input_size[0]),
        int(input_size[1]),
        pad=letterbox_pad,
        fill=114,
    )

    if to_rgb:
        padded_img = padded_img[:, :, ::-1]

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    if scale:
        padded_img = padded_img / 255.0
    return padded_img, r


def preproc(img, input_size, swap=(2, 0, 1)):
    """YOLOX-flavor letterbox (BGR, unscaled) — the historical shared default."""
    return letterbox_preproc(img, input_size, swap)
