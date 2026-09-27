"""PP-YOLOE training transform.

The inference path is a **stretch resize** plus channel normalization with
mean ``[123.675, 116.28, 103.53]`` / std ``[58.395, 57.12, 57.375]`` on the
0-255 scale, so the training transform has to produce the exact same canvas
convention. Reusing the YOLO-NAS letterbox ``/255`` transform here would put
train and val in different colour spaces (skill landmines 1 and 17).

Beyond the shared HSV / horizontal-flip knobs this adds the two PP-YOLOE
specific augmentations from the source recipe: random 90-degree rotation
(p=0.5) and a random RGB-to-BGR channel swap (p=0.25). Operation order follows
the source dataset params: rotate90, channel swap, HSV, horizontal flip,
resize, normalize. Both the swap-before-normalize and the swap-before-HSV
orderings matter, because normalization statistics are per-channel and HSV is
not channel-order independent.

Affine and mixup come from the shared ``YOLONASAffineMixupDataset`` wrapper,
which operates on the raw image before this transform runs. That places mixup
earlier in the chain than the source does (upstream mixes after the horizontal
flip); the constituent operations are the same, and the reordering is recorded
here rather than by forking the shared wrapper.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from ...data.augment.boxes import xyxy2cxcywh
from ...data.augment.color import augment_hsv
from ...data.augment.geometry import mirror
from .utils import PPYOLOE_MEAN, PPYOLOE_STD

__all__ = ["PPYOLOETrainTransform", "ppyoloe_preproc", "rot90_with_boxes"]


def ppyoloe_preproc(img: np.ndarray, input_dim):
    """Stretch-resize BGR HWC uint8 to the normalized CHW model input.

    Returns ``(chw_float32, (scale_x, scale_y))``. The two scales differ
    whenever the source image is not square, which is exactly why PP-YOLOE
    cannot reuse the single-ratio letterbox helpers.
    """
    input_h, input_w = int(input_dim[0]), int(input_dim[1])
    src_h, src_w = img.shape[:2]
    resized = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1].astype(np.float32)
    rgb -= np.array(PPYOLOE_MEAN, dtype=np.float32)
    rgb /= np.array(PPYOLOE_STD, dtype=np.float32)
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
    return chw, (input_w / src_w, input_h / src_h)


def rot90_with_boxes(img: np.ndarray, boxes: np.ndarray):
    """Rotate the image 90 degrees counter-clockwise and carry ``xyxy`` boxes.

    A point ``(x, y)`` in an ``H x W`` image maps to ``(y, W - 1 - x)`` in the
    rotated ``W x H`` image, so the box corners swap axes and the x extent is
    mirrored.
    """
    h, w = img.shape[:2]
    rotated = np.ascontiguousarray(np.rot90(img, k=1))
    if len(boxes) == 0:
        return rotated, boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    new = np.empty_like(boxes)
    new[:, 0] = y1
    new[:, 1] = w - x2
    new[:, 2] = y2
    new[:, 3] = w - x1
    return rotated, new


class PPYOLOETrainTransform:
    """Emit ``[class, cx, cy, w, h]`` pixel targets over the PP-YOLOE canvas."""

    def __init__(
        self,
        max_labels: int = 100,
        flip_prob: float = 0.5,
        hsv_prob: float = 0.5,
        rot90_prob: float = 0.5,
        rgb2bgr_prob: float = 0.25,
    ) -> None:
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.rot90_prob = rot90_prob
        self.rgb2bgr_prob = rgb2bgr_prob

    def _photometric(self, image):
        """Channel swap then HSV, in the source's order.

        ``DetectionRGB2BGR`` runs *before* ``DetectionNormalize`` upstream, and
        the normalization statistics are per-channel, so swapping after
        normalizing would pair each channel with the wrong mean and std. It
        also runs before ``DetectionHSV``, which is not order-independent
        either. Both are applied here on the raw pixels.
        """
        if self.rgb2bgr_prob > 0 and random.random() < self.rgb2bgr_prob:
            image = np.ascontiguousarray(image[:, :, ::-1])
        if self.hsv_prob > 0 and random.random() < self.hsv_prob:
            # augment_hsv mutates in place; the copy above may be a view.
            image = np.ascontiguousarray(image)
            augment_hsv(image)
        return image

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()

        if len(boxes) == 0:
            padded_labels = np.zeros((self.max_labels, 5), dtype=np.float32)
            chw, _ = ppyoloe_preproc(self._photometric(image), input_dim)
            return chw, padded_labels

        image_o = image.copy()
        boxes_o = xyxy2cxcywh(boxes.copy())
        labels_o = labels.copy()
        src_h_o, src_w_o = image_o.shape[:2]

        # Source order: affine (shared wrapper) -> rot90 -> RGB2BGR -> HSV ->
        # horizontal flip -> resize -> normalize.
        image_t = image
        if self.rot90_prob > 0 and random.random() < self.rot90_prob:
            image_t, boxes = rot90_with_boxes(image_t, boxes)
        image_t = self._photometric(image_t)
        image_t, boxes = mirror(image_t, boxes, self.flip_prob)

        src_h, src_w = image_t.shape[:2]
        chw, _ = ppyoloe_preproc(image_t, input_dim)
        input_h, input_w = int(input_dim[0]), int(input_dim[1])
        boxes = xyxy2cxcywh(boxes)
        boxes[:, 0::2] *= input_w / src_w
        boxes[:, 1::2] *= input_h / src_h

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            chw, _ = ppyoloe_preproc(image_o, input_dim)
            boxes_o[:, 0::2] *= input_w / src_w_o
            boxes_o[:, 1::2] *= input_h / src_h_o
            boxes_t = boxes_o
            labels_t = labels_o

        targets_t = np.hstack((np.expand_dims(labels_t, 1), boxes_t))
        padded_labels = np.zeros((self.max_labels, 5), dtype=np.float32)
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        return chw, np.ascontiguousarray(padded_labels, dtype=np.float32)
