"""Single-channel (grayscale) image input.

Original LibreYOLO implementation. The public contract follows Ultralytics:
a dataset YAML declares ``channels: 1`` and every image is converted to
grayscale for training, validation and prediction, the model's input
convolution takes one channel, and exported graphs consume ``(N, 1, H, W)``.

Inside the library images keep travelling as three identical channels, so
every family's augmentation, normalization and validation code runs
unchanged. Only the model's input convolution is replaced: it holds one
weight plane per RGB group (the pretrained RGB planes summed, which gives
the same response to a gray image) and folds a three-channel input back to
one channel by averaging. Export traces with a one-channel input, so the
exported graph has no folding step at all.

``channels: 2`` keeps its event-histogram meaning (see ``event_histogram``).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

RGB = 3
GRAY = 1
SUPPORTED_CHANNELS = (GRAY, RGB)

# Read by datasets at construction. Datasets are pickled into DataLoader
# workers with the value they captured, so a module global is safe here.
_dataset_channels = RGB


def validate_channels(value: Any) -> int:
    """Return a supported image channel count or raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(  # noqa: TRY004 - dataset YAML validation uses ValueError
            f"channels must be 1 (grayscale) or 3 (RGB), got {value!r}"
        )
    value = int(value)
    if value not in SUPPORTED_CHANNELS:
        raise ValueError(
            f"channels must be 1 (grayscale) or 3 (RGB), got {value}. "
            "channels: 2 is reserved for event histograms with an input_profile."
        )
    return value


def dataset_channels(config: dict) -> int | None:
    """Image channels a dataset config declares, or None when it declares none.

    Event-histogram datasets own their ``channels: 2`` key and return None.
    """
    if not isinstance(config, dict) or config.get("input_profile") is not None:
        return None
    if "channels" not in config or config["channels"] is None:
        return None
    return validate_channels(config["channels"])


def model_channels(model: Any) -> int:
    """Image channels a wrapper or backend consumes (3 unless configured)."""
    value = getattr(model, "input_channels", None)
    # Only a real int counts: mocks and unset attributes mean RGB.
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return RGB
    return int(value)


@contextmanager
def dataset_input_channels(channels: int) -> Iterator[None]:
    """Datasets constructed inside this block convert images to ``channels``."""
    global _dataset_channels
    previous = _dataset_channels
    _dataset_channels = validate_channels(channels)
    try:
        yield
    finally:
        _dataset_channels = previous


def active_dataset_channels() -> int:
    return _dataset_channels


def to_gray3(img: np.ndarray) -> np.ndarray:
    """Return an HWC 3-channel image whose channels all hold its luma.

    Works for BGR and RGB inputs alike only up to the luma weights, so callers
    pass decoded OpenCV images (BGR). Gray and 4-channel inputs are accepted.
    """
    import cv2

    if img.ndim == 2:
        gray = img
    elif img.shape[2] == 1:
        gray = img[..., 0]
    elif img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(np.repeat(gray[..., None], 3, axis=2))


def apply_dataset_channels(img: np.ndarray, channels: int) -> np.ndarray:
    """Dataset hook: convert a decoded BGR image for the configured channels."""
    if channels == GRAY and isinstance(img, np.ndarray) and img.ndim >= 2:
        return to_gray3(img)
    return img


def gray_pil(image: Any, color_format: str = "auto"):
    """Load any prediction input and return it as a 3-channel gray PIL image."""
    from .image_loader import ImageLoader

    return ImageLoader.load(image, color_format=color_format).convert("L").convert("RGB")


# ---------------------------------------------------------------------------
# Model input convolution
# ---------------------------------------------------------------------------


def _gray_conv_class():
    import torch
    from torch import nn

    class GrayInputConv2d(nn.Conv2d):
        """Input convolution for grayscale models.

        ``in_channels`` is the number of RGB groups the original convolution
        read (1 for a plain stem, 4 for a YOLOX space-to-depth stem). A
        three-channel input, which is how images travel inside LibreYOLO, is
        folded to one plane per group by averaging; a one-channel input, which
        is what exported graphs receive, goes straight through.
        """

        # Lets shape checks ahead of the convolution (DINOv2 patch embeddings)
        # accept the three-channel images that travel inside the library.
        folds_rgb = True

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Export traces with a one-channel input, so the fold never
            # belongs in a graph; skipping the check keeps tracing quiet.
            if not torch.jit.is_tracing() and x.shape[1] == 3 * self.in_channels:
                x = x.unflatten(1, (self.in_channels, 3)).mean(2)
            return super().forward(x)

    return GrayInputConv2d


_GRAY_CONV = None


def gray_conv_class():
    global _GRAY_CONV
    if _GRAY_CONV is None:
        _GRAY_CONV = _gray_conv_class()
    return _GRAY_CONV


class _Found(Exception):
    pass


def _sample_input(wrapper) -> Any:
    import torch

    from .image_size import imgsz_to_hw

    size = wrapper._get_input_size() if hasattr(wrapper, "_get_input_size") else 640
    h, w = imgsz_to_hw(size)
    try:
        device = next(wrapper.model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    return torch.zeros(1, 3, h, w, device=device)


def find_input_convs(wrapper):
    """Return ``[(qualified_name, conv), ...]`` for the convolutions reading the image.

    Runs the wrapper's forward on a blank image. The first ``nn.Conv2d`` that
    executes reads the image; parallel stem branches (RepVGG-style stems run
    a 3x3 and a 1x1 convolution on the same tensor) are collected with it.
    The forward stops at the first convolution fed anything else. Stems that
    are not plain convolutions over RGB groups are rejected, not guessed at.
    """
    import torch
    from torch import nn

    model = wrapper.model
    found: list = []
    first_input: list = []

    def same(a, b):
        return a is b or (
            a.shape == b.shape and a.data_ptr() == b.data_ptr() and a.dtype == b.dtype
        )

    def hook(module, inputs):
        x = inputs[0] if inputs else None
        if not isinstance(x, torch.Tensor):
            raise _Found
        if not first_input:
            first_input.append(x)
        elif not same(x, first_input[0]):
            raise _Found
        found.append(module)

    handles = [
        m.register_forward_pre_hook(hook)
        for m in model.modules()
        if isinstance(m, nn.Conv2d)
    ]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            try:
                wrapper._forward(_sample_input(wrapper))
            except _Found:
                pass
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    if not found:
        raise ValueError(
            f"{type(wrapper).__name__} has no convolution reading the image; "
            "channels: 1 is not supported for this model."
        )
    names = {m: n for n, m in model.named_modules()}
    result = []
    for conv in dict.fromkeys(found):
        if conv.in_channels % 3 or conv.groups != 1 or conv.in_channels != found[0].in_channels:
            raise ValueError(
                f"{type(wrapper).__name__}'s input convolution reads "
                f"{conv.in_channels} channels in {conv.groups} groups; "
                "channels: 1 is not supported for this model."
            )
        result.append((names[conv], conv))
    return result


def _replace_module(root, name: str, new) -> None:
    parent = root
    *path, attr = name.split(".")
    for part in path:
        parent = getattr(parent, part)
    setattr(parent, attr, new)


def configure_channels(wrapper, channels: int | None, *, pretrained: bool | None = None) -> None:
    """Make ``wrapper`` consume ``channels``-channel images (idempotent).

    ``pretrained`` chooses how a new grayscale convolution starts: summed RGB
    weights (the default when the model came from weights) or a fresh random
    initialization (scratch training). Weights loaded afterwards from a
    grayscale checkpoint overwrite either.
    """
    if channels is None:
        return
    channels = validate_channels(channels)
    current = model_channels(wrapper)
    if channels == current:
        return
    if current != RGB:
        raise ValueError(
            f"This model consumes {current}-channel images; a dataset or "
            f"checkpoint with channels: {channels} cannot be used with it."
        )
    if getattr(wrapper, "input_profile", None) is not None:
        raise ValueError("Event-histogram models cannot use channels: 1")

    if pretrained is None:
        pretrained = not (
            getattr(wrapper, "_training_from_scratch", False)
            or getattr(wrapper, "model_path", None) is None
        )
    for name, old in find_input_convs(wrapper):
        _replace_module(wrapper.model, name, _gray_conv_from(old, pretrained))
    wrapper.input_channels = channels


def _gray_conv_from(old, pretrained: bool):
    import torch

    groups = old.in_channels // 3
    new = gray_conv_class()(
        groups,
        old.out_channels,
        old.kernel_size,
        old.stride,
        old.padding,
        old.dilation,
        old.groups,
        old.bias is not None,
        old.padding_mode,
    ).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        if pretrained:
            new.weight.copy_(old.weight.unflatten(1, (groups, 3)).sum(2))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    new.weight.requires_grad_(old.weight.requires_grad)
    new.train(old.training)
    return new


def channels_metadata(wrapper) -> dict:
    """Checkpoint/export metadata; empty for RGB so RGB artifacts are unchanged."""
    channels = model_channels(wrapper)
    return {} if channels == RGB else {"channels": channels}


def restore_channels(wrapper, checkpoint: Any) -> None:
    """Apply a checkpoint's ``channels`` before its state dict is loaded."""
    if not isinstance(checkpoint, dict) or checkpoint.get("channels") is None:
        return
    channels = validate_channels(checkpoint["channels"])
    if channels != RGB:
        configure_channels(wrapper, channels, pretrained=False)


def prepare_channels_training(wrapper, args, kwargs) -> None:
    """Configure the model from the dataset YAML before a trainer is built."""
    from pathlib import Path

    data = kwargs.get("data", args[0] if args else None)
    if not isinstance(data, (str, Path)):
        return
    try:
        from ..data import load_data_config

        # Only the YAML's own keys matter here; the trainer downloads data.
        cfg = load_data_config(data, autodownload=False)
    except Exception:  # noqa: BLE001 - the trainer reports unreadable datasets
        return
    declared = dataset_channels(cfg)
    current = model_channels(wrapper)
    if declared is None:
        return
    if declared == RGB and current != RGB:
        raise ValueError(
            f"The dataset declares channels: 3 but this model consumes "
            f"{current}-channel images."
        )
    configure_channels(wrapper, declared)


def fold_input_channels(blob, expected: int | None):
    """Backends: fold a 3-channel NCHW batch for a graph expecting 1 channel."""
    if expected != GRAY or blob is None or getattr(blob, "ndim", 0) != 4:
        return blob
    if blob.shape[1] != 3:
        return blob
    if isinstance(blob, np.ndarray):
        return np.ascontiguousarray(blob.mean(axis=1, keepdims=True, dtype=np.float32).astype(blob.dtype))
    return blob.mean(dim=1, keepdim=True)
