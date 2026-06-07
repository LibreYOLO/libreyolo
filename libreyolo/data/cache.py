"""Optional decoded-image caching shared by dataset classes.

Every training epoch re-reads and re-decodes the same image files from disk.
Enabling caching keeps the *decoded, pre-resize, pre-augmentation* image around
so later reads skip that work. Because resize and augmentation always run
downstream of the cache, the cached pixels are augmentation-agnostic and safe to
reuse across every epoch.

The ``cache`` flag accepts:

    False / None  -> disabled (default)
    True / "ram"  -> keep decoded BGR images in RAM (per dataset instance)
    "disk"        -> store decoded images as ``.npy`` beside each source image

``"disk"`` is process- and platform-safe (each DataLoader worker just reads the
``.npy`` file), so it is the recommended mode when training with workers. RAM
caching benefits single-process loaders (``workers=0``) or loaders with
``persistent_workers=True``; with respawned workers each worker fills its own
copy, which is harmless but yields no cross-epoch reuse.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def normalize_cache(cache) -> str | None:
    """Map a user ``cache`` value to ``None`` / ``"ram"`` / ``"disk"``."""
    if cache is True:
        return "ram"
    if cache is False or cache is None:
        return None
    mode = str(cache).strip().lower()
    if mode in ("ram", "disk"):
        return mode
    if mode in ("true", "1", "yes"):
        return "ram"
    if mode in ("false", "0", "no", "none", ""):
        return None
    raise ValueError(
        f"Invalid cache mode {cache!r}; expected False, True, 'ram', or 'disk'."
    )


class ImageCacheMixin:
    """Adds optional RAM/disk image caching to a ``Dataset``.

    Subclasses must implement two hooks:

    * ``_decode_image(index) -> np.ndarray`` — the raw BGR decode from disk.
    * ``_image_path(index) -> Path`` — the source image path (used to locate the
      sibling ``.npy`` for disk caching).

    and call :meth:`enable_image_cache` once ``self.num_imgs`` is known. When
    caching is disabled (the default), :meth:`load_image` is exactly equivalent
    to ``_decode_image``.
    """

    cache: str | None = None
    _ram_cache: list | None = None

    def enable_image_cache(self, cache) -> None:
        """Configure caching from a user ``cache`` flag. Idempotent; safe with False."""
        self.cache = normalize_cache(cache)
        self._ram_cache = [None] * self.num_imgs if self.cache == "ram" else None
        if self.cache == "ram":
            self._warn_if_ram_short()
        if self.cache:
            logger.info(
                "Image cache enabled (mode=%s) for %d images", self.cache, self.num_imgs
            )

    def load_image(self, index: int) -> np.ndarray:
        cache = getattr(self, "cache", None)
        if cache == "ram":
            img = self._ram_cache[index]
            if img is None:
                img = self._decode_image(index)
                self._ram_cache[index] = img
            # Copy so downstream in-place augmentation cannot corrupt the cache.
            return img.copy()
        if cache == "disk":
            return self._load_image_from_disk(index)
        return self._decode_image(index)

    def _load_image_from_disk(self, index: int) -> np.ndarray:
        src = Path(self._image_path(index))
        # Append (not replace) the suffix so 'a.jpg' and 'a.png' never collide.
        npy = src.with_name(src.name + ".npy")
        try:
            if npy.exists() and npy.stat().st_mtime >= src.stat().st_mtime:
                return np.load(npy)
        except Exception:  # corrupt / unreadable cache -> rebuild from source
            pass
        img = self._decode_image(index)
        try:
            np.save(str(npy), img)
        except OSError:  # read-only dataset dir -> skip persistence, still train
            pass
        return img

    def _warn_if_ram_short(self) -> None:
        """Best-effort warning if a RAM cache likely exceeds available memory."""
        try:
            import psutil  # optional dependency
        except Exception:
            return
        try:
            sample = self._decode_image(0)
        except Exception:
            return
        needed = sample.nbytes * self.num_imgs
        available = psutil.virtual_memory().available
        if needed > available:
            logger.warning(
                "cache='ram' may need ~%.1f GB but only ~%.1f GB is available; "
                "consider cache='disk' instead.",
                needed / 1e9,
                available / 1e9,
            )
