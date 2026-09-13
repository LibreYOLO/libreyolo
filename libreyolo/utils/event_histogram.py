"""Two-polarity numerical inputs for YOLO9 and RF-DETR detection.

Original LibreYOLO implementation. Event decoding/accumulation belongs to the
producer; this module only consumes already prepared spatial histograms.
"""

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_PROFILE_KEYS = {"format", "layout", "polarity", "encoding", "scale", "window_us"}


def validate_input_profile(profile, *, family=None, task="detect"):
    """Validate the complete v1 input contract; None retains the RGB contract."""
    if profile is None:
        return None
    if not isinstance(profile, dict) or set(profile) != _PROFILE_KEYS:
        raise ValueError(f"input_profile requires exactly {sorted(_PROFILE_KEYS)}")
    expected = {
        "format": "event_histogram",
        "layout": "HWC",
        "polarity": "positive_negative",
        "encoding": "counts",
    }
    for key, value in expected.items():
        if profile[key] != value:
            raise ValueError(f"input_profile.{key} must be {value!r}")
    scale = profile["scale"]
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("input_profile.scale must be a finite positive number")
    window = profile["window_us"]
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("input_profile.window_us must be a positive integer")
    if (family is not None and family not in {"yolo9", "rfdetr"}) or task != "detect":
        raise ValueError("Event histograms support YOLO9 and RF-DETR detection only")
    return dict(profile)


def load_histogram(source):
    """Read finite nonnegative HWC count planes without pickle or RGB conversion."""
    if isinstance(source, (str, Path)):
        if Path(source).suffix.lower() != ".npy":
            raise ValueError(
                "Event histogram input must be an HWC .npy file or NumPy array"
            )
        source = np.load(source, allow_pickle=False)
    if not isinstance(source, np.ndarray):
        raise TypeError("Event histogram input must be an HWC NumPy array or .npy file")
    if source.ndim != 3 or source.shape[2] != 2 or min(source.shape[:2]) <= 0:
        raise ValueError(
            "Event histogram shape must be (height, width, 2), positive then negative"
        )
    if (
        source.dtype.kind not in "uif"
        or not np.isfinite(source).all()
        or (source < 0).any()
    ):
        raise ValueError(
            "Event histogram counts must be finite nonnegative real numbers"
        )
    with np.errstate(over="ignore"):
        result = np.asarray(source, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Event histogram counts exceed the float32 range")
    return result


def visualize_histogram(source, *, scale):
    """Return an RGB uint8 preview: positive red, negative blue, overlap magenta.

    This is a visualization, never a detector input. The fixed scale is the
    same count saturation level declared by the dataset/model input profile.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    values = np.clip(load_histogram(source) / scale, 0, 1)
    rgb = np.zeros((*values.shape[:2], 3), dtype=np.uint8)
    rgb[..., 0] = np.rint(values[..., 0] * 255).astype(np.uint8)
    rgb[..., 2] = np.rint(values[..., 1] * 255).astype(np.uint8)
    return rgb


def histogram_geometry(height, width, input_size, family, letterbox_pad="topleft"):
    from ..utils.image_size import imgsz_to_hw

    h, w = imgsz_to_hw(input_size)
    if family == "yolo9":
        from ..preprocess.letterbox import letterbox_geometry

        ratio, rh, rw, dx, dy = letterbox_geometry(height, width, h, w, letterbox_pad)
        return h, w, rh, rw, ratio, ratio, dx, dy
    return h, w, h, w, w / width, h / height, 0, 0


def preprocess_histogram(source, profile, input_size, family, letterbox_pad="topleft"):
    """Return normalized CHW float32 and the exact box transform."""
    values = load_histogram(source)
    geometry = histogram_geometry(*values.shape[:2], input_size, family, letterbox_pad)
    h, w, rh, rw, *_ = geometry
    dx, dy = geometry[-2:]
    # Saturate before interpolation, so scale means the same thing at every size.
    values = np.clip(values / profile["scale"], 0, 1)
    resized = cv2.resize(values, (rw, rh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((h, w, 2), dtype=np.float32)
    canvas[dy : dy + rh, dx : dx + rw] = resized
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)), geometry


def predict_histogram(
    wrapper, source, input_size=None, color_format="auto", *, as_numpy=False
):
    if color_format != "auto":
        raise ValueError("color_format does not apply to event histograms; use 'auto'")
    family = getattr(wrapper, "FAMILY", None) or wrapper.model_family
    size = input_size if input_size is not None else wrapper._get_input_size()
    if family == "rfdetr" and hasattr(wrapper, "_validate_imgsz"):
        size = wrapper._validate_imgsz(size)
    values = load_histogram(source)
    chw, _ = preprocess_histogram(
        values,
        wrapper.input_profile,
        size,
        family,
        getattr(wrapper, "letterbox_pad", "topleft"),
    )
    preview = Image.fromarray(
        visualize_histogram(values, scale=wrapper.input_profile["scale"])
    )
    if as_numpy:
        tensor = chw[None]
    else:
        import torch

        tensor = torch.from_numpy(chw[None])
    return tensor, preview, (values.shape[1], values.shape[0]), 1.0


def configure_input(wrapper, profile, *, initialization=None):
    """Replace the actual input convolution, preserving every other parameter."""
    import torch
    from torch import nn

    family = wrapper.FAMILY
    profile = validate_input_profile(profile, family=family, task=wrapper.task)
    current = getattr(wrapper, "input_profile", None)
    if current is not None and profile != current:
        raise ValueError(
            "Dataset/checkpoint input_profile does not match the loaded model"
        )
    if profile is None:
        return
    if family == "yolo9":
        parent, attr = wrapper.model.backbone.conv0, "conv"
    else:
        candidates = [
            m
            for m in wrapper.model.modules()
            if type(m).__name__ == "Dinov2WithRegistersPatchEmbeddings"
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Event histograms require RF-DETR's DINOv2 patch embedding"
            )
        parent, attr = candidates[0], "projection"
    old = getattr(parent, attr)
    if old.in_channels != 2:
        if old.in_channels != 3 or old.groups != 1:
            raise ValueError(
                "Expected an RGB input convolution before histogram adaptation"
            )
        policy = initialization or (
            "random"
            if getattr(wrapper, "_training_from_scratch", False)
            or wrapper.model_path is None
            else "rgb_mean"
        )
        if policy not in {"random", "rgb_mean"}:
            raise ValueError("input_initialization must be 'random' or 'rgb_mean'")
        new = nn.Conv2d(
            2,
            old.out_channels,
            old.kernel_size,
            old.stride,
            old.padding,
            old.dilation,
            old.groups,
            old.bias is not None,
            old.padding_mode,
        )
        new = new.to(device=old.weight.device, dtype=old.weight.dtype)
        if policy == "rgb_mean":
            with torch.no_grad():
                new.weight.copy_(
                    old.weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1) * 1.5
                )
                if old.bias is not None:
                    new.bias.copy_(old.bias)
        new.train(old.training)
        setattr(parent, attr, new)
        if family == "rfdetr":
            parent.num_channels = 2
        wrapper.input_initialization = policy
    elif initialization is not None:
        if initialization not in {"random", "rgb_mean"}:
            raise ValueError("Invalid input_initialization")
        wrapper.input_initialization = initialization
    wrapper.input_profile = profile


def input_metadata(wrapper):
    profile = getattr(wrapper, "input_profile", None)
    if not isinstance(profile, dict):
        return {}
    metadata = {
        "input_profile": dict(profile),
        "input_initialization": wrapper.input_initialization,
    }
    if wrapper.FAMILY == "yolo9":
        metadata["letterbox_pad"] = wrapper.letterbox_pad
    return metadata


def restore_input(wrapper, checkpoint):
    profile = checkpoint.get("input_profile")
    if profile is not None and checkpoint.get("input_initialization") not in {
        "random",
        "rgb_mean",
    }:
        raise ValueError("Histogram checkpoints require input_initialization")
    configure_input(
        wrapper, profile, initialization=checkpoint.get("input_initialization")
    )


class HistogramTransform:
    """Shared numerical preprocessing; train labels are class/cx/cy/w/h pixels."""

    wants_unresized_image = True
    custom_normalization = True
    normalize = False

    def __init__(
        self,
        profile,
        family,
        letterbox_pad="topleft",
        *,
        training=False,
        flip_prob=0.0,
        flipud=0.0,
        max_labels=300,
    ):
        self.input_profile = profile
        self.family = family
        self.letterbox_pad = letterbox_pad
        self.training = training
        self.flip_prob = flip_prob
        self.flipud = flipud
        self.max_labels = max_labels

    @property
    def uses_letterbox(self):
        return self.family == "yolo9"

    def letterbox_scale(self, orig_h, orig_w, imgsz):
        g = histogram_geometry(orig_h, orig_w, imgsz, self.family, self.letterbox_pad)
        return g[4], g[6], g[7]

    def __call__(self, img, targets, input_size):
        chw, g = preprocess_histogram(
            img, self.input_profile, input_size, self.family, self.letterbox_pad
        )
        h, w, _, _, sx, sy, dx, dy = g
        targets = np.array(targets, dtype=np.float32, copy=True)
        if len(targets) > self.max_labels:
            raise ValueError(
                f"Histogram sample has {len(targets)} boxes; increase max_labels={self.max_labels}"
            )
        targets[:, [0, 2]] = targets[:, [0, 2]] * sx + dx
        targets[:, [1, 3]] = targets[:, [1, 3]] * sy + dy
        if self.training:
            if random.random() < self.flip_prob:
                chw = chw[:, :, ::-1]
                targets[:, [0, 2]] = w - targets[:, [2, 0]]
            if random.random() < self.flipud:
                chw = chw[:, ::-1, :]
                targets[:, [1, 3]] = h - targets[:, [3, 1]]
            boxes = targets[:, :4].copy()
            targets[:, 0] = targets[:, 4]
            if self.family == "yolo9":
                targets[:, 1:5] = boxes / np.array([w, h, w, h], dtype=np.float32)
            else:
                targets[:, 1:3] = (boxes[:, :2] + boxes[:, 2:]) / 2
                targets[:, 3:5] = boxes[:, 2:] - boxes[:, :2]
        padded = np.zeros((self.max_labels, 5), dtype=np.float32)
        if self.training and self.family == "yolo9":
            padded[:, 0] = -1
        padded[: len(targets)] = targets
        return np.ascontiguousarray(chw), padded


def val_transform(wrapper):
    family = getattr(wrapper, "FAMILY", None) or wrapper.model_family
    return HistogramTransform(
        wrapper.input_profile, family, getattr(wrapper, "letterbox_pad", "topleft")
    )


def check_dataset_profile(wrapper, config):
    profile = validate_input_profile(config.get("input_profile"))
    if profile != getattr(wrapper, "input_profile", None):
        raise ValueError(
            "Dataset input_profile must match the model's saved input_profile"
        )


def check_predict_options(source_spec, *, augment=False, kwargs=None):
    from ..utils.source import SourceKind

    if source_spec.kind not in {
        SourceKind.IMAGE,
        SourceKind.IMAGE_BATCH,
        SourceKind.DIRECTORY,
    }:
        raise ValueError(
            "Histogram prediction accepts prepared arrays/files, lists and directories"
        )
    if augment or (kwargs or {}).get("sliced"):
        raise ValueError("Histogram prediction does not support TTA or slicing")


def collect_histograms(directory):
    """Collect prepared NumPy frames without importing the training stack."""
    return sorted(
        p
        for p in Path(directory).rglob("*")
        if p.is_file() and p.suffix.lower() == ".npy"
    )
