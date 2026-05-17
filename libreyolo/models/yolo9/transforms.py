"""
Data augmentation transforms for YOLOv9 training.

Key difference from YOLOX: outputs normalized xyxy format for loss computation.
- YOLOX: [class, cx, cy, w, h] in pixel coordinates
- YOLOv9: [class, x1, y1, x2, y2] in normalized (0-1) coordinates
"""

import random
import cv2
import numpy as np


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    """Apply HSV augmentation to an image."""
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]
    hsv_augs *= np.random.randint(0, 2, 3)
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)


def preproc(img, input_size, swap=(2, 0, 1)):
    """Preprocess image: resize with letterbox padding, transpose."""
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

    # Convert BGR to RGB (match YOLO9ValPreprocessor and pretrained weights)
    padded_img = padded_img[:, :, ::-1]

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) / 255.0
    return padded_img, r


def mirror(image, boxes, prob=0.5):
    """Apply horizontal flip with probability."""
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def _copy_segments(segments):
    if segments is None:
        return None
    return [[ring.copy() for ring in instance] for instance in segments]


def _transform_segments(segments, scale=1.0, padw=0.0, padh=0.0, width=None, height=None):
    if segments is None:
        return None
    transformed = []
    for instance in segments:
        rings = []
        for ring in instance:
            r = ring.astype(np.float32, copy=True)
            r[:, 0] = r[:, 0] * scale + padw
            r[:, 1] = r[:, 1] * scale + padh
            if width is not None:
                r[:, 0] = np.clip(r[:, 0], 0, width)
            if height is not None:
                r[:, 1] = np.clip(r[:, 1], 0, height)
            rings.append(r)
        transformed.append(rings)
    return transformed


def _flip_segments_lr(segments, width):
    if segments is None:
        return None
    flipped = []
    for instance in segments:
        rings = []
        for ring in instance:
            r = ring.astype(np.float32, copy=True)
            r[:, 0] = width - r[:, 0]
            rings.append(r)
        flipped.append(rings)
    return flipped


def _filter_segments(segments, keep_mask):
    if segments is None:
        return None
    keep = np.asarray(keep_mask, dtype=bool)
    return [segments[i] for i in range(min(len(segments), len(keep))) if keep[i]]


def _rasterize_segments(segments, image_shape, mask_shape, max_masks):
    masks = np.zeros((max_masks, mask_shape[0], mask_shape[1]), dtype=np.float32)
    if not segments:
        return masks

    img_h, img_w = image_shape
    mask_h, mask_w = mask_shape
    sx = mask_w / max(float(img_w), 1.0)
    sy = mask_h / max(float(img_h), 1.0)

    for idx, instance in enumerate(segments[:max_masks]):
        polygons = []
        for ring in instance:
            if ring is None or len(ring) < 3:
                continue
            poly = ring.astype(np.float32, copy=True)
            poly[:, 0] *= sx
            poly[:, 1] *= sy
            polygons.append(np.round(poly).astype(np.int32))
        if polygons:
            cv2.fillPoly(masks[idx], polygons, color=1)
    return masks


class YOLO9TrainTransform:
    """
    Transform for YOLOv9 training data.

    Outputs normalized xyxy format: [class, x1, y1, x2, y2] where coordinates
    are normalized to [0, 1] range.
    """

    def __init__(
        self,
        max_labels=100,
        flip_prob=0.5,
        hsv_prob=1.0,
        mask_downsample_ratio=4,
    ):
        """
        Args:
            max_labels: Maximum number of labels per image
            flip_prob: Probability of horizontal flip
            hsv_prob: Probability of HSV augmentation
        """
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.mask_downsample_ratio = mask_downsample_ratio

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
        segments_t = _copy_segments(segments)
        mask_shape = (
            input_dim[0] // self.mask_downsample_ratio,
            input_dim[1] // self.mask_downsample_ratio,
        )

        if len(boxes) == 0:
            padded_labels = np.zeros((self.max_labels, 5), dtype=np.float32)
            # Fill class with -1 to indicate padding (empty slots)
            padded_labels[:, 0] = -1
            image, _ = preproc(image, input_dim)
            if return_masks:
                return (
                    image,
                    padded_labels,
                    np.zeros((self.max_labels, *mask_shape), dtype=np.float32),
                )
            return image, padded_labels

        # Store original for fallback
        image_o = image.copy()
        boxes_o = boxes.copy()
        labels_o = labels.copy()
        segments_o = _copy_segments(segments_t)

        # Apply HSV augmentation
        if random.random() < self.hsv_prob:
            augment_hsv(image)

        # Apply horizontal flip
        _, width, _ = image.shape
        if random.random() < self.flip_prob:
            image_t = image[:, ::-1]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            segments_t = _flip_segments_lr(segments_t, width)
        else:
            image_t = image

        # Resize with letterbox
        image_t, r = preproc(image_t, input_dim)

        # Scale boxes by resize ratio
        boxes = boxes * r
        segments_t = _transform_segments(
            segments_t, scale=r, width=input_dim[1], height=input_dim[0]
        )

        # Filter out tiny boxes (after resize)
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        mask = (w > 1) & (h > 1)
        boxes_t = boxes[mask]
        labels_t = labels[mask]
        segments_t = _filter_segments(segments_t, mask)

        # Fallback to original if all boxes filtered
        if len(boxes_t) == 0:
            image_t, r = preproc(image_o, input_dim)
            boxes_t = boxes_o * r
            labels_t = labels_o
            segments_t = _transform_segments(
                segments_o, scale=r, width=input_dim[1], height=input_dim[0]
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
        targets_t = np.hstack((labels_t, boxes_norm))

        # Pad to max_labels
        padded_labels = np.zeros((self.max_labels, 5), dtype=np.float32)
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
        img, _ = preproc(img, input_size, self.swap)
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
        """
        self.dataset = dataset
        self.img_size = img_size
        self.preproc = preproc or YOLO9TrainTransform()
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob

    def __len__(self):
        return len(self.dataset)

    @property
    def input_dim(self):
        return self.img_size

    def close_mosaic(self):
        """Disable mosaic and mixup augmentation (for final training epochs)."""
        self.enable_mosaic = False
        self.enable_mixup = False

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
            output = self.preproc(img, label, self.input_dim, segments)
            img, label, masks = output
            return img, label, img_info, img_id, masks

        img, label, img_info, img_id = item
        img, label = self.preproc(img, label, self.input_dim)
        return img, label, img_info, img_id

    def _get_mosaic_item(self, idx):
        """Get a mosaic-augmented item."""
        input_h, input_w = self.input_dim

        # Random center point for mosaic
        yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
        xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

        # Get 4 random indices
        indices = [idx] + [random.randint(0, len(self.dataset) - 1) for _ in range(3)]

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
            mosaic_labels = np.zeros((0, 5))

        # Resize mosaic to target size
        mosaic_img = cv2.resize(mosaic_img, (input_w, input_h))
        mosaic_labels[:, :4] = mosaic_labels[:, :4] / 2
        if has_segments:
            mosaic_segments = _transform_segments(
                mosaic_segments, scale=0.5, width=input_w, height=input_h
            )

        # Filter small boxes
        if len(mosaic_labels) > 0:
            w = mosaic_labels[:, 2] - mosaic_labels[:, 0]
            h = mosaic_labels[:, 3] - mosaic_labels[:, 1]
            mask = (w > 2) & (h > 2)
            mosaic_labels = mosaic_labels[mask]
            if has_segments:
                mosaic_segments = _filter_segments(mosaic_segments, mask)

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
        """Apply mixup augmentation."""
        # Get another random image
        idx2 = random.randint(0, len(self.dataset) - 1)
        img2, labels2, _, _ = self._get_normal_item(idx2)

        # Mix images
        r = np.random.beta(32.0, 32.0)
        img = (img * r + img2 * (1 - r)).astype(img.dtype)

        # Concatenate labels (from both images)
        if labels2 is not None and len(labels2) > 0 and labels2[0, 0] >= 0:
            # Filter padding from labels2
            mask = labels2[:, 0] >= 0
            labels2 = labels2[mask]
            if len(labels2) > 0:
                # Concatenate and truncate
                all_labels = np.vstack([labels, labels2])
                max_labels = (
                    self.preproc.max_labels
                    if hasattr(self.preproc, "max_labels")
                    else 100
                )
                labels = all_labels[:max_labels]

        return img, labels
