"""Geometric augmentations and letterbox preprocessing shared by the numpy
pipelines.

Moved verbatim from ``libreyolo/training/augment.py`` (originally adapted
from the official YOLOX repository).
"""

import math
import random

import cv2
import numpy as np


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
    (rotation + scale + shear + translation), byte for byte, and draws no extra
    random numbers.

    With ``perspective != 0.0`` it returns a full 3x3 homography built from
    first principles as a standard composition of elementary transforms, read
    right to left as applied to a source pixel ``[x, y, 1]``:

        M = translate @ shear @ rotate_scale @ perspective @ center

    - ``center`` shifts the image center to the origin so every following
      transform pivots about the center;
    - ``perspective`` adds the two projective terms ``P[2, 0]`` and ``P[2, 1]``,
      each sampled uniformly in ``[-perspective, +perspective]``, which tilt the
      plane; the homogeneous divide happens in ``cv2.warpPerspective`` and in
      :func:`apply_affine_to_bboxes`;
    - ``rotate_scale`` is the same rotation+scale used by the affine path;
    - ``shear`` applies the x/y shear as a left-multiply of the rotation
      (matching the affine path's ``R[0] + shear_y * R[1]`` construction);
    - ``translate`` recenters the image and adds the sampled translation.

    The first six random draws (angle, scale, two shears, two translations)
    keep the same order as the affine path so switching perspective on does not
    reshuffle the shared draws; the two projective terms are drawn last and only
    when perspective is active.
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

    # Projective path: draw the two tilt terms last, then compose the 3x3.
    perspective_x = random.uniform(-perspective, perspective)
    perspective_y = random.uniform(-perspective, perspective)

    center = np.eye(3)
    center[0, 2] = -twidth / 2.0
    center[1, 2] = -theight / 2.0

    projective = np.eye(3)
    projective[2, 0] = perspective_x
    projective[2, 1] = perspective_y

    rotate_scale = np.eye(3)
    rotate_scale[:2] = R

    shear_m = np.eye(3)
    shear_m[0, 1] = shear_y
    shear_m[1, 0] = shear_x

    translate_m = np.eye(3)
    translate_m[0, 2] = twidth / 2.0 + translation_x
    translate_m[1, 2] = theight / 2.0 + translation_y

    M = translate_m @ shear_m @ rotate_scale @ projective @ center
    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    """Warp box corners through M, then recompute axis-aligned bounds.

    ``M`` may be a 2x3 affine or a 3x3 homography; for the projective case the
    warped corners are divided by their homogeneous coordinate.
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
        corner_points = corner_points[:, :2] / corner_points[:, 2:3]
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


def _window_origin(size, window, low, high):
    """A random origin for a ``window``-long span of a ``size``-long axis.

    The span covers ``[low, high]`` when it is long enough to; ``low`` and
    ``high`` are ``None`` when there is nothing to cover.
    """
    first, last = 0, size - window
    if low is not None:
        first = max(first, math.ceil(high) - window)
        last = min(last, math.floor(low))
    return random.randint(first, max(first, last))


def zoom_to_boxes(image, boxes, zoom_range=(1.0, 1.0), min_visible=0.6):
    """Random zoom-in crop that keeps one box whole.

    A magnification ``z`` is drawn from ``zoom_range`` and a window of ``1 / z``
    of the image, in the image's own aspect ratio, is cut out of it. One box is
    picked at random as the anchor and the window is placed uniformly among the
    positions that contain the whole of it; when the anchor is larger than the
    window, ``z`` is lowered until it fits. The crop is returned at its own
    size: the magnification happens when the caller resizes it to the network
    input, so a crop of a frame larger than that input carries real detail
    rather than interpolated pixels.

    The other boxes are shifted with the window, clipped to it, and kept only
    when at least ``min_visible`` of their area is still inside, so a sliver of
    an object is never labelled as one. Without boxes the window is placed
    anywhere.

    Args:
        image: ``(H, W[, C])`` array.
        boxes: ``[N, 4]`` xyxy in pixel coordinates of ``image``.
        zoom_range: ``(low, high)`` magnification with ``1 <= low <= high``;
            this is a zoom-in, never a zoom-out.
        min_visible: area fraction of a box the window must contain to keep it.

    Returns:
        ``(crop, kept_boxes, keep)``: ``kept_boxes`` are the surviving boxes in
        the crop's pixel coordinates and ``keep`` the boolean mask over the
        input rows that selects them, for the caller's per-box columns.

    The draws come from :mod:`random`, in a fixed order: the zoom, the anchor
    (only with boxes), then the window's x and y origin.
    """
    low, high = zoom_range
    if not 1.0 <= low <= high:
        raise ValueError(
            f"zoom_range must satisfy 1 <= low <= high (zoom-in only). Got {zoom_range}"
        )
    height, width = image.shape[:2]
    dtype = (
        boxes.dtype
        if isinstance(boxes, np.ndarray) and boxes.dtype.kind == "f"
        else np.float32
    )
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)

    zoom = random.uniform(low, high)
    anchor = None
    if len(boxes) > 0:
        anchor = boxes[random.randrange(len(boxes))]
        anchor_w = math.ceil(anchor[2]) - math.floor(anchor[0])
        anchor_h = math.ceil(anchor[3]) - math.floor(anchor[1])
        zoom = max(1.0, min(zoom, width / max(anchor_w, 1), height / max(anchor_h, 1)))

    win_w = min(width, max(1, round(width / zoom)))
    win_h = min(height, max(1, round(height / zoom)))
    if anchor is not None:
        x0 = _window_origin(width, win_w, anchor[0], anchor[2])
        y0 = _window_origin(height, win_h, anchor[1], anchor[3])
    else:
        x0 = _window_origin(width, win_w, None, None)
        y0 = _window_origin(height, win_h, None, None)

    crop = np.ascontiguousarray(image[y0 : y0 + win_h, x0 : x0 + win_w])
    if len(boxes) == 0:
        return crop, boxes.astype(dtype), np.zeros((0,), dtype=bool)

    shifted = boxes - np.array([x0, y0, x0, y0], dtype=np.float64)
    clipped = shifted.copy()
    clipped[:, 0::2] = clipped[:, 0::2].clip(0, win_w)
    clipped[:, 1::2] = clipped[:, 1::2].clip(0, win_h)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    visible = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
    keep = (area > 0) & (visible >= min_visible * area)
    return crop, clipped[keep].astype(dtype), keep


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


def letterbox_preproc(img, input_size, swap=(2, 0, 1), *, to_rgb=False, scale=False):
    """Letterbox resize + pad (114) + HWC→CHW transpose.

    The historical per-family ``preproc`` copies differed only in two finalize
    flags, exposed here as parameters:

    - ``to_rgb=False, scale=False`` — YOLOX/PicoDet/RTMDet (BGR, raw 0-255)
    - ``to_rgb=True, scale=True``   — YOLO9/RT-DETR/YOLO-NAS (RGB, /255)
    """
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
