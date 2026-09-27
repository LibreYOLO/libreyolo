"""YOLO9-flavor training transforms and mosaic dataset wrapper.

Moved verbatim from ``libreyolo/models/yolo9/transforms.py``. Also used by
the RT-DETR trainer (with ``max_labels=300``).

Key difference from YOLOX: outputs normalized xyxy format for loss computation.
- YOLOX: [class, cx, cy, w, h] in pixel coordinates
- YOLOv9: [class, x1, y1, x2, y2] in normalized (0-1) coordinates

The mosaic wrapper stays separate from :class:`~.yolox.MosaicMixupDataset`
on purpose: the two differ in tile scaling (yolo9 caps at 1.0), canvas
allocation, mixup style and RNG draw order, and post-affine box filtering.
Only the quadrant placement math (:func:`~.mosaic.get_mosaic_coordinate`)
is provably identical and shared.
"""

import logging
import random

import cv2
import numpy as np

from ..obb import normalize_obb_angle
from .color import augment_hsv
from .copy_paste import copy_paste as _copy_paste_instances
from .geometry import (  # noqa: F401
    letterbox_preproc,
    mirror,
    random_affine,
    rot90_image_boxes,
)
from .segments import (
    copy_segments as _copy_segments,
    filter_segments as _filter_segments,
    flip_segments_lr as _flip_segments_lr,
    flip_segments_ud as _flip_segments_ud,
    rasterize_segments as _rasterize_segments,
    transform_segments as _transform_segments,
)

logger = logging.getLogger(__name__)


def preproc(img, input_size, swap=(2, 0, 1), letterbox_pad=None):
    """Letterbox to RGB float32/0-1 (matches YOLO9ValPreprocessor and weights)."""
    return letterbox_preproc(
        img, input_size, swap, to_rgb=True, scale=True, letterbox_pad=letterbox_pad
    )


class YOLO9TrainTransform:
    """
    Transform for YOLOv9 training data.

    Outputs normalized xyxy format: [class, x1, y1, x2, y2] where coordinates
    are normalized to [0, 1] range.
    """

    def __init__(
        self,
        max_labels=300,
        flip_prob=0.5,
        vertical_flip_prob=0.0,
        hsv_prob=1.0,
        mask_downsample_ratio=4,
        output_label_dim=None,
        flipud=None,
        rot90_prob=0.0,
        letterbox_pad=None,
    ):
        """
        Args:
            max_labels: Maximum number of labels per image
            flip_prob: Probability of horizontal flip
            vertical_flip_prob: Probability of vertical flip
            hsv_prob: Probability of HSV augmentation
            flipud: Standard-name alias for ``vertical_flip_prob``. When given
                (not ``None``) it overrides ``vertical_flip_prob``; the older
                name is kept for backward compatibility.
            rot90_prob: Probability of a random k*90-degree rotation (k in
                1..3). Intended for oriented-box (OBB) training; it is only
                applied when the sample carries angle targets and draws no
                random numbers when disabled.
        """
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.vertical_flip_prob = (
            vertical_flip_prob if flipud is None else flipud
        )
        self.hsv_prob = hsv_prob
        self.mask_downsample_ratio = mask_downsample_ratio
        self.output_label_dim = output_label_dim
        self.rot90_prob = rot90_prob
        self.letterbox_pad = letterbox_pad

    def __call__(self, image, targets, input_dim, segments=None):
        """
        Apply transformations.

        Args:
            image: Input image (H, W, C) in BGR format
            targets: Annotations [N, 5] with [x1, y1, x2, y2, class] in pixel coords
            input_dim: Target size (height, width)

        Returns:
            image: Transformed image (C, H, W) as float32
            padded_labels: [max_labels, 5] with [class, x1, y1, x2, y2] normalized
        """
        return_masks = segments is not None
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        has_angles = targets.shape[1] >= 6 or self.output_label_dim == 6
        angles = (
            targets[:, 5].copy()
            if targets.shape[1] >= 6
            else np.zeros((targets.shape[0],), dtype=np.float32)
        ) if has_angles else None
        segments_t = _copy_segments(segments)
        label_dim = 6 if has_angles else 5
        mask_shape = (
            input_dim[0] // self.mask_downsample_ratio,
            input_dim[1] // self.mask_downsample_ratio,
        )

        if len(boxes) == 0:
            padded_labels = np.zeros((self.max_labels, label_dim), dtype=np.float32)
            # Fill class with -1 to indicate padding (empty slots)
            padded_labels[:, 0] = -1
            # Background images get the same photometric/flip augmentation as
            # labeled images (no label sync needed — there are no labels). The
            # old early return skipped HSV and flips entirely, so on
            # background-heavy datasets most of the training data was never
            # augmented and the model could memorize the exact negatives
            # (issue #484). Same op order and RNG-draw pattern as the labeled
            # path below.
            if random.random() < self.hsv_prob:
                augment_hsv(image)
            if random.random() < self.flip_prob:
                image = image[:, ::-1]
            if random.random() < self.vertical_flip_prob:
                image = image[::-1, :]
            image, _ = preproc(
                image, input_dim, letterbox_pad=self.letterbox_pad
            )
            if return_masks:
                # No instances -> zero mask rows (masks are variable-length
                # per image and padded to the batch max at collate, #527).
                return (
                    image,
                    padded_labels,
                    np.zeros((0, *mask_shape), dtype=np.uint8),
                )
            return image, padded_labels

        # Store original for fallback
        image_o = image.copy()
        boxes_o = boxes.copy()
        labels_o = labels.copy()
        angles_o = angles.copy() if angles is not None else None
        segments_o = _copy_segments(segments_t)

        # Apply a random k*90-degree rotation for oriented boxes. Off by
        # default and only meaningful when the sample carries angle targets, so
        # it is guarded to draw no random numbers when disabled (keeping the
        # detection path's RNG order unchanged). The image and box centers
        # rotate together; the angle turns by k*pi/2. A quarter turn of a
        # rectangle is a swap of its sides, which normalize_obb_angle folds back
        # into the canonical [-pi/2, pi/2) range, so width/height are preserved
        # and only the angle column carries the turn.
        if self.rot90_prob > 0 and angles is not None:
            if random.random() < self.rot90_prob:
                k = random.randint(1, 3)
                image, boxes = rot90_image_boxes(image, boxes, k)
                angles = np.asarray(
                    [
                        normalize_obb_angle(float(a) + k * np.pi / 2.0)
                        for a in angles
                    ],
                    dtype=np.float32,
                )

        # Apply HSV augmentation
        if random.random() < self.hsv_prob:
            augment_hsv(image)

        # Apply horizontal flip
        _, width, _ = image.shape
        if random.random() < self.flip_prob:
            image_t = image[:, ::-1]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            if angles is not None:
                angles = np.asarray(
                    [normalize_obb_angle(np.pi - float(a)) for a in angles],
                    dtype=np.float32,
                )
            segments_t = _flip_segments_lr(segments_t, width)
        else:
            image_t = image

        # Apply vertical flip. Kept disabled by default because it is not a
        # natural augmentation for all datasets, but it is useful for aerial OBB.
        height = image_t.shape[0]
        if random.random() < self.vertical_flip_prob:
            image_t = image_t[::-1, :]
            boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
            if angles is not None:
                angles = np.asarray(
                    [normalize_obb_angle(-float(a)) for a in angles],
                    dtype=np.float32,
                )
            segments_t = _flip_segments_ud(segments_t, height)

        # Resize with letterbox
        src_h, src_w = image_t.shape[:2]
        image_t, r = preproc(
            image_t, input_dim, letterbox_pad=self.letterbox_pad
        )

        # Scale boxes by resize ratio, then shift by pad (0 for topleft).
        from libreyolo.preprocess.letterbox import letterbox_geometry

        _ratio, _nh, _nw, pad_left, pad_top = letterbox_geometry(
            src_h, src_w, input_dim[0], input_dim[1], self.letterbox_pad
        )
        boxes = boxes * r
        boxes[:, [0, 2]] = boxes[:, [0, 2]] + pad_left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] + pad_top
        segments_t = _transform_segments(
            segments_t,
            scale=r,
            padw=pad_left,
            padh=pad_top,
            width=input_dim[1],
            height=input_dim[0],
        )

        # Filter out tiny boxes (after resize)
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        mask = (w > 1) & (h > 1)
        boxes_t = boxes[mask]
        labels_t = labels[mask]
        angles_t = angles[mask] if angles is not None else None
        segments_t = _filter_segments(segments_t, mask)

        # Fallback to original if all boxes filtered
        if len(boxes_t) == 0:
            src_h, src_w = image_o.shape[:2]
            image_t, r = preproc(
                image_o, input_dim, letterbox_pad=self.letterbox_pad
            )
            _ratio, _nh, _nw, pad_left, pad_top = letterbox_geometry(
                src_h, src_w, input_dim[0], input_dim[1], self.letterbox_pad
            )
            boxes_t = boxes_o * r
            boxes_t[:, [0, 2]] = boxes_t[:, [0, 2]] + pad_left
            boxes_t[:, [1, 3]] = boxes_t[:, [1, 3]] + pad_top
            labels_t = labels_o
            angles_t = angles_o
            segments_t = _transform_segments(
                segments_o,
                scale=r,
                padw=pad_left,
                padh=pad_top,
                width=input_dim[1],
                height=input_dim[0],
            )

        # Normalize coordinates to [0, 1]
        h_out, w_out = input_dim
        boxes_norm = boxes_t.copy()
        boxes_norm[:, 0] /= w_out  # x1
        boxes_norm[:, 1] /= h_out  # y1
        boxes_norm[:, 2] /= w_out  # x2
        boxes_norm[:, 3] /= h_out  # y2

        # Clip to [0, 1]
        boxes_norm = np.clip(boxes_norm, 0, 1)

        # Format: [class, x1, y1, x2, y2]
        labels_t = np.expand_dims(labels_t, 1)
        if angles_t is not None:
            targets_t = np.hstack((labels_t, boxes_norm, angles_t[:, None]))
        else:
            targets_t = np.hstack((labels_t, boxes_norm))

        # Pad to max_labels
        padded_labels = np.zeros((self.max_labels, targets_t.shape[1]), dtype=np.float32)
        # Fill class with -1 to indicate padding
        padded_labels[:, 0] = -1

        n = min(len(targets_t), self.max_labels)
        padded_labels[:n] = targets_t[:n]

        if return_masks:
            masks = _rasterize_segments(
                segments_t,
                image_shape=input_dim,
                mask_shape=mask_shape,
                max_masks=self.max_labels,
            )
            return image_t, padded_labels, masks

        return image_t, padded_labels


class YOLO9ValTransform:
    """Transform for YOLOv9 validation data."""

    def __init__(self, swap=(2, 0, 1)):
        self.swap = swap

    def __call__(self, img, res, input_size):
        """
        Apply validation transform.

        Args:
            img: Input image
            res: Annotations (ignored for validation)
            input_size: Target size

        Returns:
            img: Preprocessed image
            dummy: Dummy labels array
        """
        img, _ = preproc(
            img, input_size, self.swap, letterbox_pad=getattr(self, "letterbox_pad", None)
        )
        return img, np.zeros((1, 5))


class YOLO9MosaicMixupDataset:
    """
    Dataset wrapper that applies Mosaic and Mixup augmentation for YOLOv9.

    Similar to YOLOX MosaicMixupDataset but outputs normalized xyxy format.
    """

    def __init__(
        self,
        dataset,
        img_size,
        mosaic=True,
        preproc=None,
        degrees=0.0,
        translate=0.1,
        mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5),
        shear=0.0,
        enable_mixup=False,
        mosaic_prob=1.0,
        mixup_prob=0.0,
        copy_paste=0.0,
        copy_paste_mode="flip",
        perspective=0.0,
    ):
        """
        Initialize YOLO9MosaicMixupDataset.

        Args:
            dataset: Base dataset with pull_item method
            img_size: Target image size (height, width)
            mosaic: Enable mosaic augmentation
            preproc: Preprocessing transform (default: YOLO9TrainTransform)
            degrees: Rotation degrees (default 0 for yolo9)
            translate: Translation factor
            mosaic_scale: Scale range for mosaic
            mixup_scale: Scale range for mixup
            shear: Shear degrees (default 0 for yolo9)
            enable_mixup: Enable mixup (default False for yolo9)
            mosaic_prob: Probability of applying mosaic
            mixup_prob: Probability of applying mixup
            copy_paste: Probability of applying copy-paste instance augmentation
                (segmentation only). 0 disables it and leaves the RNG stream
                untouched.
            copy_paste_mode: "flip" pastes instances from the same sample
                mirrored, "mixup" pastes instances from a second random sample.
        """
        self.dataset = dataset
        self.img_size = img_size
        self.preproc = preproc or YOLO9TrainTransform()
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.copy_paste = copy_paste
        self.copy_paste_mode = copy_paste_mode
        self._copy_paste_warned = False

    def __len__(self):
        return len(self.dataset)

    @property
    def input_dim(self):
        return self.img_size

    def close_mosaic(self):
        """Disable mosaic and mixup augmentation (for final training epochs)."""
        self.enable_mosaic = False
        self.enable_mixup = False
        # Copy-paste is a strong augmentation too; drop it in the no-aug tail.
        self.copy_paste = 0.0

    def _pull_copy_paste_source(self):
        """Pull a second random sample as a copy-paste source.

        Returns ``(img, boxes, labels, segments)`` in the sample's own pixel
        coords, or ``(None, None, None, None)`` if the sample carries no
        segments.
        """
        idx = random.randint(0, len(self.dataset) - 1)
        item = self.dataset.pull_item(idx)
        if len(item) == 5:
            img, label, _info, _id, segments = item
            return img, label[:, :4], label[:, 4], segments
        return None, None, None, None

    def _apply_copy_paste(self, image, labels, segments):
        """Optionally paste extra instances before the final preproc.

        ``labels`` is ``[N, >=5]`` with ``[x1, y1, x2, y2, class, ...]`` in the
        pixel frame of ``image``; ``segments`` are the aligned polygon rings.
        Returns the (possibly enlarged) ``(image, labels, segments)``.
        """
        # Ordered so the default (copy_paste == 0) draws no RNG and detection
        # data (no segments) is a warn-once no-op without consuming the stream.
        if self.copy_paste <= 0.0:
            return image, labels, segments
        if not segments:
            if not self._copy_paste_warned:
                logger.warning(
                    "copy_paste is set but the dataset has no segments; "
                    "copy-paste needs instance masks and is being skipped."
                )
                self._copy_paste_warned = True
            return image, labels, segments
        if random.random() >= self.copy_paste:
            return image, labels, segments
        if labels.ndim != 2 or labels.shape[1] != 5:
            # Copy-paste only handles plain [x1, y1, x2, y2, class] seg labels.
            return image, labels, segments

        boxes = labels[:, :4]
        classes = labels[:, 4]
        if self.copy_paste_mode == "mixup":
            src_img, src_boxes, src_classes, src_segments = self._pull_copy_paste_source()
            if src_segments is None:
                return image, labels, segments
        else:
            src_img, src_boxes, src_classes, src_segments = image, boxes, classes, segments

        # In "flip" mode the source is the same sample, so force the mirror
        # (flip_prob=1.0): an unflipped self-paste would drop instances back on
        # their own location. In "mixup" mode the source is a different sample,
        # so a random flip (default 0.5) is a useful extra variation.
        cp_flip_prob = 0.5 if self.copy_paste_mode == "mixup" else 1.0
        new_image, new_boxes, new_classes, new_segments = _copy_paste_instances(
            image,
            boxes,
            classes,
            segments,
            src_img,
            src_boxes,
            src_classes,
            src_segments,
            max_instances=getattr(self.preproc, "max_labels", None),
            flip_prob=cp_flip_prob,
        )

        if len(new_boxes):
            new_labels = np.concatenate(
                [new_boxes.reshape(-1, 4), np.asarray(new_classes, dtype=np.float32).reshape(-1, 1)],
                axis=1,
            ).astype(np.float32)
        else:
            new_labels = np.zeros((0, 5), dtype=np.float32)
        return new_image, new_labels, new_segments

    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            return self._get_mosaic_item(idx)
        else:
            return self._get_normal_item(idx)

    def _get_normal_item(self, idx):
        """Get a single item without mosaic."""
        item = self.dataset.pull_item(idx)
        if len(item) == 5:
            img, label, img_info, img_id, segments = item
            img, label, segments = self._apply_copy_paste(img, label, segments)
            output = self.preproc(img, label, self.input_dim, segments)
            img, label, masks = output
            return img, label, img_info, img_id, masks

        img, label, img_info, img_id = item
        img, label = self.preproc(img, label, self.input_dim)
        return img, label, img_info, img_id

    def _rand_partner_index(self):
        """Sample a partner index, preferring images with annotations.

        Uniform draws on background-heavy datasets frequently land on
        label-free images, silently degrading mosaic tiles (and mixup
        partners) to unsupervised pixels. Retry like the YOLOX mixup does;
        if every sampled candidate is background, keep the last draw so
        the augmentation still works on datasets with no foreground
        labels. Extra RNG draws happen only after an empty candidate, so
        fully annotated datasets sample identically.
        """
        index = 0
        for _ in range(20):
            index = random.randint(0, len(self.dataset) - 1)
            if len(self.dataset.load_anno(index)) > 0:
                break
        return index

    def _get_mosaic_item(self, idx):
        """Get a mosaic-augmented item."""
        input_h, input_w = self.input_dim

        # Random center point for mosaic
        yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
        xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

        # Get 4 random indices, preferring annotated partners
        indices = [idx] + [self._rand_partner_index() for _ in range(3)]

        # Create mosaic canvas
        mosaic_img = np.full((input_h * 2, input_w * 2, 3), 114, dtype=np.uint8)
        mosaic_labels = []
        mosaic_segments = []
        has_segments = False

        for i, index in enumerate(indices):
            item = self.dataset.pull_item(index)
            if len(item) == 5:
                img, _labels, _, _, segments = item
                has_segments = True
            else:
                img, _labels, _, _ = item
                segments = None
            h0, w0 = img.shape[:2]

            # Scale for mosaic
            scale = min(1.0, min(input_h / h0, input_w / w0))
            img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))
            h, w = img.shape[:2]

            # Get placement coordinates
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, input_h * 2)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = (
                    xc,
                    yc,
                    min(xc + w, input_w * 2),
                    min(yc + h, input_h * 2),
                )
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # Adjust labels
            labels = _labels.copy()
            if len(labels) > 0:
                labels[:, :4] = labels[:, :4] * scale
                labels[:, 0] += padw  # x1
                labels[:, 1] += padh  # y1
                labels[:, 2] += padw  # x2
                labels[:, 3] += padh  # y2
                mosaic_labels.append(labels)
                if has_segments:
                    tile_segments = _transform_segments(
                        segments,
                        scale=scale,
                        padw=padw,
                        padh=padh,
                        width=input_w * 2,
                        height=input_h * 2,
                    )
                    mosaic_segments.extend(tile_segments or [[] for _ in labels])

        if len(mosaic_labels) > 0:
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            # Clip to mosaic bounds
            np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
            np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
            np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
            np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])
        else:
            label_dim = getattr(self.preproc, "output_label_dim", None) or 5
            mosaic_labels = np.zeros((0, label_dim))

        if has_segments:
            # Segment polygons are not affine-warped yet, so the mask path keeps
            # the plain 2x -> 1x resize. Affine support for segments is a
            # follow-up (see issue #432).
            mosaic_img = cv2.resize(mosaic_img, (input_w, input_h))
            mosaic_labels[:, :4] = mosaic_labels[:, :4] / 2
            mosaic_segments = _transform_segments(
                mosaic_segments, scale=0.5, width=input_w, height=input_h
            )
        else:
            # Apply random affine (rotation / translation / scale / shear) on the
            # assembled 2x mosaic, warping it down to the target size. This
            # matches the shared MosaicMixupDataset used by the other detection
            # families and restores the geometric augmentation that YOLO9 was
            # silently dropping: degrees/translate/mosaic_scale/shear were passed
            # in but never applied (issue #432). random_affine also performs the
            # 2x -> 1x downscale, so it replaces the manual resize here.
            mosaic_img, mosaic_labels = random_affine(
                mosaic_img,
                mosaic_labels,
                target_size=(input_w, input_h),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear,
                perspective=self.perspective,
            )

        # Filter small boxes
        if len(mosaic_labels) > 0:
            w = mosaic_labels[:, 2] - mosaic_labels[:, 0]
            h = mosaic_labels[:, 3] - mosaic_labels[:, 1]
            mask = (w > 2) & (h > 2)
            mosaic_labels = mosaic_labels[mask]
            if has_segments:
                mosaic_segments = _filter_segments(mosaic_segments, mask)

        # Copy-paste operates while segments are still polygons in the assembled
        # mosaic's pixel frame, so pasted masks and labels stay consistent.
        if has_segments:
            mosaic_img, mosaic_labels, mosaic_segments = self._apply_copy_paste(
                mosaic_img, mosaic_labels, mosaic_segments
            )

        # Apply preprocessing (HSV, flip, normalize)
        if has_segments:
            img, labels, masks = self.preproc(
                mosaic_img, mosaic_labels, self.input_dim, mosaic_segments
            )
        else:
            img, labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)

        # Apply mixup if enabled
        if (
            not has_segments
            and self.enable_mixup
            and random.random() < self.mixup_prob
            and len(labels) > 0
        ):
            img, labels = self._mixup(img, labels)

        if has_segments:
            return img, labels, (input_h, input_w), idx, masks
        return img, labels, (input_h, input_w), idx

    def _mixup(self, img, labels):
        """Apply mixup augmentation.

        Both ``labels`` and the second image's labels arrive already padded to
        ``max_labels`` rows with class ``-1`` marking empty slots (that is what
        ``YOLO9TrainTransform`` emits). Vstacking the second image's rows after
        this full block and truncating to ``max_labels`` would keep only the
        first image's block and silently drop every object from the second
        image — they land at rows >= max_labels — turning ~50%-opacity objects
        into unlabeled ghosts the loss punishes as background (issue #484).

        Fix: strip padding from both images, concatenate the *real* objects,
        then re-pad. Mirrors the shared YOLOX mixup, which merges labels before
        padding.
        """
        # Get another random image, preferring an annotated partner (mixup
        # only runs on the non-segment path, so _get_normal_item returns
        # (img, label, ...); ignore the tail).
        idx2 = self._rand_partner_index()
        img2, labels2, *_ = self._get_normal_item(idx2)

        # Mix images (beta(32, 32) ≈ 0.5, so both images are ~equally visible
        # and both label sets must be kept).
        r = np.random.beta(32.0, 32.0)
        img = (img * r + img2 * (1 - r)).astype(img.dtype)

        max_labels = getattr(self.preproc, "max_labels", 300)
        label_dim = labels.shape[1]

        # Drop padding (class == -1) from both, then merge the real objects.
        real1 = labels[labels[:, 0] >= 0]
        if labels2 is not None and len(labels2) > 0:
            real2 = labels2[labels2[:, 0] >= 0]
        else:
            real2 = np.zeros((0, label_dim), dtype=labels.dtype)
        merged = np.vstack([real1, real2])[:max_labels]

        # Re-pad to max_labels with class = -1 (matches YOLO9TrainTransform).
        padded = np.zeros((max_labels, label_dim), dtype=labels.dtype)
        padded[:, 0] = -1
        padded[: len(merged)] = merged

        return img, padded
