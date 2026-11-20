"""Torch-free input preprocessing.

Mirrors ``libreyolo/postprocess/``: the family-specific numpy preprocessing
recipes live here rather than under ``libreyolo/models/``, because importing
anything under ``models/`` runs ``models/__init__.py``, which eagerly
constructs every model class to populate the ``can_load()`` registry. Those
classes are ``nn.Module`` subclasses and genuinely require torch, so a backend
that only needs a letterbox recipe would drag the whole torch wheel in with it.

Keeping these functions here is what lets ``backends/base.py`` (and therefore
``backends/onnx.py``) import without torch, so a lightweight deployment can run
ONNX inference with just numpy + onnxruntime.

``libreyolo/models/<family>/utils.py`` re-exports everything here, so existing
import paths keep working.

See https://github.com/LibreYOLO/libreyolo/discussions/711.
"""

from __future__ import annotations

import numpy as np

from ..utils.lazy import lazy_module, module_available

torch = lazy_module("torch")


def normalize_hwc_to_chw(
    img_hwc: np.ndarray,
    mean=None,
    std=None,
    *,
    scale=None,
    reverse_channels: bool = False,
) -> np.ndarray:
    """Normalize an HWC image into a contiguous float32 CHW array, one channel plane at a time.

    Per element this is the float32 arithmetic of the usual HWC recipe
    ``((img.astype(float32)[..., ::-1 if reverse_channels] / scale - mean) / std).transpose(2, 0, 1)``
    (each step only when given, scalars as float32), so the result is bitwise
    identical to it. Working plane by plane straight into the CHW output
    avoids the strided broadcast over the channel axis, the float HWC
    temporaries and the later copy of a transposed view (about 3-4x faster
    on a 640x640 image).
    """
    height, width, channels = img_hwc.shape
    mean = None if mean is None else np.asarray(mean, dtype=np.float32).reshape(-1)
    std = None if std is None else np.asarray(std, dtype=np.float32).reshape(-1)
    scale = None if scale is None else np.float32(scale)
    out = np.empty((channels, height, width), dtype=np.float32)
    for c in range(channels):
        plane = out[c]
        plane[...] = img_hwc[:, :, channels - 1 - c if reverse_channels else c]
        if scale is not None:
            np.divide(plane, scale, out=plane)
        if mean is not None:
            np.subtract(plane, mean[c], out=plane)
        if std is not None:
            np.divide(plane, std[c], out=plane)
    return out


def as_input(img_nchw: np.ndarray):
    """Wrap an already-batched array, as a ``torch.Tensor`` if torch is installed.

    Counterpart to :func:`as_batched_input` for the recipes whose
    ``preprocess_numpy`` already returns NCHW (RT-DETR), so no batch dim is
    added here.
    """
    if module_available("torch"):
        return torch.from_numpy(img_nchw)
    return img_nchw


def as_batched_input(img_chw: np.ndarray):
    """Add a batch dim, as a ``torch.Tensor`` if torch is installed.

    With torch present this is exactly ``torch.from_numpy(chw).unsqueeze(0)``,
    which is what every family preprocessor did before this module existed.
    Without torch it returns the equivalent ``(1, C, H, W)`` ndarray, which the
    ONNX backend feeds to onnxruntime directly (it converted the tensor back to
    numpy anyway).
    """
    if module_available("torch"):
        return torch.from_numpy(img_chw).unsqueeze(0)
    return img_chw[None, ...]


__all__ = ["as_batched_input", "as_input"]
