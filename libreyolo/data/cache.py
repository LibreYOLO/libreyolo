"""Optional image caching shared by dataset classes.

Every training epoch re-reads, re-decodes and re-resizes the same image files
from disk. Enabling caching keeps the result of that deterministic work around
so later epochs skip it. There are two cache points, and the dataset picks the
right one per read:

* **Post-resize** (``load_resized_img``): the decoded image after the
  deterministic pre-augmentation resize to ``img_size``. This is the cache
  point for every family whose transform consumes the dataset's resized frame
  (it skips decode *and* resize), and it is the one that makes caching viable
  at all on large-image datasets: a 4000x3000 source stores ~640px pixels,
  roughly an order of magnitude smaller than the decoded original. Because
  augmentation always runs downstream of this resize, the cached pixels are
  augmentation-agnostic and safe to reuse across every epoch — later reads are
  byte-identical to a fresh decode+resize by construction.
* **Pre-resize** (``load_image``): the decoded full-resolution image. This is
  the cache point for transforms that opt into ``wants_unresized_image`` and
  own all resizing themselves (RT-DETR, DEIM, RF-DETR, PicoDet, ...). It skips
  decode only, and stores original-resolution pixels.

The ``cache`` flag accepts:

    False / None  -> disabled (default)
    True / "ram"  -> keep cached images in RAM (per dataset instance)
    "disk"        -> store cached images as ``.npy`` beside each source image

``"disk"`` is process- and platform-safe (each DataLoader worker just reads the
``.npy`` file), so it is the recommended mode when training with workers. RAM
caching benefits single-process loaders (``workers=0``) or loaders with
``persistent_workers=True``; with respawned workers each worker fills its own
copy, which is harmless but yields no cross-epoch reuse.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
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

    and expose ``self.img_size`` as an ``(h, w)`` tuple (used by
    :meth:`load_resized_img`), then call :meth:`enable_image_cache` once
    ``self.num_imgs`` is known. When caching is disabled (the default),
    :meth:`load_image` is exactly equivalent to ``_decode_image`` and
    :meth:`load_resized_img` to a fresh decode+resize.
    """

    cache: str | None = None
    _ram_cache: list | None = None
    _resized_ram_cache: list | None = None

    def enable_image_cache(self, cache) -> None:
        """Configure caching from a user ``cache`` flag. Idempotent; safe with False."""
        self.cache = normalize_cache(cache)
        ram = self.cache == "ram"
        self._ram_cache = [None] * self.num_imgs if ram else None
        self._resized_ram_cache = [None] * self.num_imgs if ram else None
        if ram:
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

    def load_resized_img(self, index: int) -> np.ndarray:
        """Load the image after the deterministic pre-augmentation resize.

        This is the post-resize cache point. It deliberately bypasses the
        decoded-image cache (``load_image``) on cache hits and fills: storing
        both the full-resolution decode and its resized product would roughly
        double the footprint for no reuse, since families that consume the
        resized frame never read the decoded one again.
        """
        cache = getattr(self, "cache", None)
        if cache == "ram":
            img = self._resized_ram_cache[index]
            if img is None:
                img = self._resize_decoded(self._decode_image(index))
                self._resized_ram_cache[index] = img
            # Copy so downstream in-place augmentation cannot corrupt the cache.
            return img.copy()
        if cache == "disk":
            return self._load_resized_from_disk(index)
        return self._resize_decoded(self._decode_image(index))

    def _resize_decoded(self, img: np.ndarray) -> np.ndarray:
        """Aspect-preserving resize so the longer side fits ``self.img_size``."""
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        return cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(img.dtype)

    def _load_image_from_disk(self, index: int) -> np.ndarray:
        src = Path(self._image_path(index))
        # Append (not replace) the suffix so 'a.jpg' and 'a.png' never collide.
        npy = src.with_name(src.name + ".npy")
        return self._disk_cached(npy, src, lambda: self._decode_image(index))

    def _load_resized_from_disk(self, index: int) -> np.ndarray:
        src = Path(self._image_path(index))
        # The resized bytes depend on the target size, so key the sidecar on it:
        # two runs at different imgsz must not read each other's pixels.
        h, w = int(self.img_size[0]), int(self.img_size[1])
        npy = src.with_name(f"{src.name}.r{h}x{w}.npy")
        return self._disk_cached(
            npy, src, lambda: self._resize_decoded(self._decode_image(index))
        )

    @staticmethod
    def _disk_cached(npy: Path, src: Path, produce) -> np.ndarray:
        try:
            if npy.exists() and npy.stat().st_mtime >= src.stat().st_mtime:
                return np.load(npy)
        except Exception:  # corrupt / unreadable cache -> rebuild from source
            pass
        img = produce()
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
            # Families that consume the dataset's resized frame cache at the
            # post-resize point, which is what actually occupies RAM; sizing
            # the warning from the full-resolution decode would overstate the
            # footprint by roughly (source pixels / resized pixels).
            wants_unresized = getattr(
                getattr(self, "preproc", None), "wants_unresized_image", False
            )
            if not wants_unresized:
                sample = self._resize_decoded(sample)
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
