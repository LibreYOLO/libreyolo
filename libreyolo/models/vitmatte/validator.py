"""Family-local validation for trimap-guided ViTMatte.

Existing matte datasets contain image/ground-truth-alpha pairs but not always a
trimap. For a reproducible validation input, a three-level trimap is derived by
eroding and dilating the alpha-at-0.5 foreground with a fixed 15-pixel radius
(31x31 window). The band between those boundaries is marked unknown. Callers
may instead pass ``trimap_dir=`` with one guide per image stem.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...data.matte_dataset import resolve_matte_pairs
from ...validation.matte_validator import matte_mae, s_measure


logger = logging.getLogger(__name__)

DEFAULT_TRIMAP_RADIUS = 15
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
_TRIMAP_DIR: ContextVar[Path | None] = ContextVar(
    "vitmatte_validation_trimap_dir",
    default=None,
)
_TRIMAP_RADIUS: ContextVar[int] = ContextVar(
    "vitmatte_validation_trimap_radius",
    default=DEFAULT_TRIMAP_RADIUS,
)


def _load_matte(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def derive_trimap_from_matte(
    matte: np.ndarray,
    *,
    radius: int = DEFAULT_TRIMAP_RADIUS,
) -> np.ndarray:
    """Derive deterministic ``0/128/255`` guidance from a ground-truth alpha.

    The alpha is thresholded at 0.5, then the binary foreground is eroded and
    dilated by ``radius``. The region outside the dilation is known background,
    the erosion is known foreground, and their symmetric band is unknown.
    """
    alpha = np.asarray(matte, dtype=np.float32)
    if alpha.ndim != 2:
        raise ValueError(
            f"ViTMatte validation mattes must be 2D; got shape {tuple(alpha.shape)}."
        )
    if not np.isfinite(alpha).all():
        raise ValueError("ViTMatte validation matte contains NaN or infinite values.")
    radius = int(radius)
    if radius < 0:
        raise ValueError(f"trimap_radius must be >= 0, got {radius}.")

    foreground = torch.from_numpy((np.clip(alpha, 0.0, 1.0) >= 0.5).astype(np.float32))[
        None, None
    ]
    if radius:
        kernel_size = radius * 2 + 1
        dilated_foreground = F.max_pool2d(
            foreground,
            kernel_size=kernel_size,
            stride=1,
            padding=radius,
        )
        eroded_foreground = 1.0 - F.max_pool2d(
            1.0 - foreground,
            kernel_size=kernel_size,
            stride=1,
            padding=radius,
        )
    else:
        dilated_foreground = foreground
        eroded_foreground = foreground

    trimap = np.full(alpha.shape, 128, dtype=np.uint8)
    trimap[dilated_foreground[0, 0].numpy() <= 0.5] = 0
    trimap[eroded_foreground[0, 0].numpy() > 0.5] = 255
    return trimap


def _index_trimaps(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"trimap_dir is not a directory: {directory}")
    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
            indexed.setdefault(path.stem, path)
    if not indexed:
        raise ValueError(f"trimap_dir contains no supported guide images: {directory}")
    return indexed


@contextmanager
def validation_trimap_options(
    trimap_dir: str | Path | None,
    trimap_radius: int,
) -> Iterator[None]:
    """Keep family validation guide options scoped to one concurrent call."""
    radius = int(trimap_radius)
    if radius < 0:
        raise ValueError(f"trimap_radius must be >= 0, got {radius}.")
    directory = Path(trimap_dir) if trimap_dir is not None else None
    directory_token = _TRIMAP_DIR.set(directory)
    radius_token = _TRIMAP_RADIUS.set(radius)
    try:
        yield
    finally:
        _TRIMAP_RADIUS.reset(radius_token)
        _TRIMAP_DIR.reset(directory_token)


class ViTMatteValidator:
    """Reuse matte MAE/S-measure while supplying one trimap per image."""

    task = "matte"

    def __init__(self, model, config, **kwargs) -> None:
        self.model = model
        self.config = config

    def __call__(self, **kwargs) -> Dict[str, float]:
        del kwargs
        if not self.config.data:
            raise ValueError(
                "ViTMatte validation requires data= with paired images and mattes."
            )
        pairs = resolve_matte_pairs(
            self.config.data,
            split=self.config.split or "val",
        )
        trimap_dir = _TRIMAP_DIR.get()
        trimaps = _index_trimaps(trimap_dir) if trimap_dir is not None else None
        radius = _TRIMAP_RADIUS.get()

        maes: List[float] = []
        sms: List[float] = []
        for image_path, matte_path in pairs:
            gt = _load_matte(matte_path)
            if trimaps is None:
                guide = Image.fromarray(
                    derive_trimap_from_matte(gt, radius=radius),
                    mode="L",
                )
            else:
                guide = trimaps.get(image_path.stem)
                if guide is None:
                    raise ValueError(
                        f"No trimap matching image stem {image_path.stem!r} "
                        f"in {trimap_dir}."
                    )

            result = self.model.predict(
                str(image_path),
                trimap=guide,
                device=str(self.config.device),
            )
            if isinstance(result, list):
                result = result[0]
            if result.matte is None:
                raise ValueError(f"Model returned no matte for {image_path}.")
            prediction = np.asarray(result.matte.array, dtype=np.float32)
            if gt.shape != prediction.shape:
                gt = np.asarray(
                    Image.fromarray(gt, mode="F").resize(
                        (prediction.shape[1], prediction.shape[0]),
                        Image.Resampling.BILINEAR,
                    ),
                    dtype=np.float32,
                )
            maes.append(matte_mae(prediction, gt))
            sms.append(s_measure(prediction, gt))

        metrics = {
            "metrics/MAE": float(np.mean(maes)),
            "metrics/Smeasure": float(np.mean(sms)),
        }
        metrics["fitness"] = metrics["metrics/Smeasure"]
        if getattr(self.config, "verbose", True):
            self._print_results(metrics, len(pairs), trimap_dir, radius)
        return metrics

    @staticmethod
    def _print_results(
        metrics: Dict[str, float],
        count: int,
        trimap_dir: Path | None,
        radius: int,
    ) -> None:
        guide_source = (
            f"trimap_dir={trimap_dir}"
            if trimap_dir is not None
            else f"GT-derived, fixed radius={radius}px"
        )
        logger.info("ViTMatte Validation Results (%d pairs; %s)", count, guide_source)
        logger.info("  MAE:        %.4f", metrics["metrics/MAE"])
        logger.info("  S-measure:  %.4f", metrics["metrics/Smeasure"])


__all__ = [
    "DEFAULT_TRIMAP_RADIUS",
    "ViTMatteValidator",
    "derive_trimap_from_matte",
    "validation_trimap_options",
]
