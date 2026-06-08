"""Per-model preprocessing for face-embedding ONNX heads.

Face-recognition heads differ only in a small preprocessing contract applied to
an already-aligned 112x112 crop (channel order, mean, scale, layout). Getting
this wrong silently yields garbage embeddings, so each known head pins its
convention explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreprocCfg:
    """Preprocessing contract for an aligned face crop."""

    size: int = 112
    color_order: str = "RGB"      # "RGB" or "BGR"
    mean: float = 127.5           # subtracted per channel
    scale: float = 1.0 / 127.5    # multiplied after mean subtraction
    layout: str = "NCHW"          # "NCHW" or "NHWC"

    @classmethod
    def arcface(cls) -> "PreprocCfg":
        # ArcFace / iResNet convention (e.g. the AuraFace 512-d head):
        # RGB, (x-127.5)/127.5 -> [-1, 1].
        return cls(size=112, color_order="RGB", mean=127.5, scale=1.0 / 127.5)

    @classmethod
    def raw_bgr(cls) -> "PreprocCfg":
        # Raw BGR convention (e.g. some MobileFaceNet-style 128-d heads):
        # BGR, [0, 255], no normalization.
        return cls(size=112, color_order="BGR", mean=0.0, scale=1.0)


_NAMED = {"arcface": PreprocCfg.arcface, "raw_bgr": PreprocCfg.raw_bgr}


def resolve_preproc(spec) -> PreprocCfg:
    """Coerce ``None`` / a name / a PreprocCfg into a PreprocCfg (default ArcFace)."""
    if spec is None:
        return PreprocCfg.arcface()
    if isinstance(spec, PreprocCfg):
        return spec
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in _NAMED:
            return _NAMED[key]()
        raise ValueError(f"Unknown preprocessing preset: {spec!r}. Known: {sorted(_NAMED)}.")
    raise TypeError(f"Unsupported preproc spec: {type(spec).__name__}")


def preprocess_aligned(crop_rgb_uint8: np.ndarray, cfg: PreprocCfg) -> np.ndarray:
    """Aligned HxWx3 RGB uint8 -> (1, 3, H, W) or (1, H, W, 3) float32 per cfg."""
    img = crop_rgb_uint8.astype(np.float32)
    if cfg.color_order == "BGR":
        img = img[:, :, ::-1]
    img = (img - cfg.mean) * cfg.scale
    if cfg.layout == "NCHW":
        img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img[None, ...], dtype=np.float32)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), eps, None)
