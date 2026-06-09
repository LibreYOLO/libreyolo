"""Color-space augmentations shared by all model families."""

import random

import cv2
import numpy as np

# ImageNet statistics in CHW broadcast shape, applied after /255 scaling.
IMAGENET_MEAN_CHW = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD_CHW = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    """Random HSV jitter (in-place) on a uint8 BGR image."""
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]
    hsv_augs *= np.random.randint(0, 2, 3)  # randomly zero-out each channel
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)


def brightness_contrast(img: np.ndarray) -> None:
    """In-place brightness/contrast jitter for uint8 BGR images.

    alpha in [0.8, 1.2] scales contrast, beta in [-51, 51] shifts brightness.
    """
    alpha = random.uniform(0.8, 1.2)
    beta = random.uniform(-0.2, 0.2) * 255.0
    img[:] = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
