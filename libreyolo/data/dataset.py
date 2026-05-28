"""
Dataset classes for YOLOX training.

Supports both COCO JSON format and YOLO txt format.
"""

import copy
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from .utils import polygon_to_cxcywh
from libreyolo.training.distributed import is_main_process

logger = logging.getLogger(__name__)

# Mirrors Ultralytics' NUM_THREADS choice for the image cache thread pool.
_CACHE_NUM_THREADS = min(8, max(1, (os.cpu_count() or 1) - 1))


def _normalize_cache(cache: Union[bool, str, None]) -> Optional[str]:
    """Normalise ``cache`` argument to ``None`` / ``"ram"`` / ``"disk"``."""
    if cache is False or cache is None:
        return None
    if cache is True:
        return "ram"
    value = str(cache).strip().lower()
    if value in ("ram", "disk"):
        return value
    if value in ("false", "none", ""):
        return None
    if value == "true":
        return "ram"
    logger.warning("Unknown image cache mode %r — disabling cache", cache)
    return None


def _available_memory_bytes() -> int:
    """Best-effort available RAM in bytes; 0 means "unknown"."""
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            return int(psutil.virtual_memory().available)
        except Exception:  # pragma: no cover - defensive
            pass
    # Fall back to POSIX sysconf where available (Linux/macOS).
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return 0


def _available_disk_bytes(path: Path) -> int:
    """Free disk space in bytes for ``path``'s filesystem; 0 means "unknown"."""
    try:
        return int(os.statvfs(str(path)).f_bavail * os.statvfs(str(path)).f_frsize)
    except (AttributeError, OSError):
        pass
    try:
        import shutil
        return int(shutil.disk_usage(str(path)).free)
    except OSError:
        return 0


def _check_cache_ram(
    image_paths: List[Path],
    img_size: Tuple[int, int],
    safety_margin: float = 0.5,
) -> Tuple[bool, float]:
    """Estimate RAM required to cache all resized images.

    Mirrors Ultralytics ``check_cache_ram``: samples up to 30 images,
    extrapolates by ``n_total / n_sample`` and applies ``1 + safety_margin``.
    Returns ``(fits, required_bytes)``.
    """
    n_total = len(image_paths)
    if n_total == 0:
        return False, 0.0
    n_sample = min(n_total, 30)
    samples = random.sample(image_paths, n_sample)
    target_long = max(img_size)
    bytes_seen = 0
    n_ok = 0
    for path in samples:
        im = cv2.imread(str(path))
        if im is None:
            continue
        h, w = im.shape[:2]
        long_side = max(h, w)
        if long_side > 0:
            r = target_long / long_side
        else:
            r = 1.0
        if r != 1.0:
            new_h = max(1, int(round(h * r)))
            new_w = max(1, int(round(w * r)))
            bytes_seen += new_h * new_w * 3
        else:
            bytes_seen += im.size
        n_ok += 1
    if n_ok == 0:
        return False, 0.0
    required = bytes_seen * (n_total / n_ok) * (1.0 + safety_margin)
    available = _available_memory_bytes()
    fits = bool(available) and required < available
    return fits, float(required)


def _check_cache_disk(
    image_paths: List[Path],
    safety_margin: float = 0.5,
) -> Tuple[bool, float]:
    """Estimate disk space required to cache all ``.npy`` decoded images.

    Mirrors Ultralytics ``check_cache_disk``: decoded numpy size scales with
    the source image's decoded pixel count, so we sample 30 images and
    extrapolate. Returns ``(fits, required_bytes)``.
    """
    n_total = len(image_paths)
    if n_total == 0:
        return False, 0.0
    n_sample = min(n_total, 30)
    samples = random.sample(image_paths, n_sample)
    bytes_seen = 0
    n_ok = 0
    for path in samples:
        im = cv2.imread(str(path))
        if im is None:
            continue
        bytes_seen += im.nbytes
        n_ok += 1
    if n_ok == 0:
        return False, 0.0
    required = bytes_seen * (n_total / n_ok) * (1.0 + safety_margin)
    # Use the first image's directory as a free-space probe.
    available = _available_disk_bytes(Path(image_paths[0]).parent)
    fits = bool(available) and required < available
    return fits, float(required)


def _cache_image_to_disk(npy_path: Path, src_path: Path) -> int:
    """Decode ``src_path`` and write it to ``npy_path``. Returns bytes written."""
    if npy_path.exists():
        try:
            return npy_path.stat().st_size
        except OSError:
            return 0
    im = cv2.imread(str(src_path))
    if im is None:
        return 0
    np.save(str(npy_path), im, allow_pickle=False)
    try:
        return npy_path.stat().st_size
    except OSError:
        return int(im.nbytes)


def _yolo_coords_to_rings(
    coords: List[float], width: int, height: int
) -> List[np.ndarray]:
    """Convert one normalized YOLO polygon row to the shared ring contract."""
    ring = np.array(coords, dtype=np.float32).reshape(-1, 2)
    ring[:, 0] *= width
    ring[:, 1] *= height
    return [ring]


def _yolo_box_to_ring(cx: float, cy: float, w: float, h: float, width: int, height: int) -> List[np.ndarray]:
    """Convert one normalized YOLO bbox row to a rectangular ring."""
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    ring = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )
    ring[:, 0] = np.clip(ring[:, 0], 0.0, float(width))
    ring[:, 1] = np.clip(ring[:, 1], 0.0, float(height))
    return [ring]


class DenseMaskRing(np.ndarray):
    """Polygon ring carrying a dense mask for mask-aware transforms.

    Used when the polygon ring is a lossy approximation of the true mask
    (e.g., a contour extracted from an RLE-decoded mask). For polygon-sourced
    annotations the ring is itself exact, so a plain ndarray is stored instead
    and consumers that need crop-fidelity materialize the mask on demand.
    """

    def __new__(cls, ring: np.ndarray, mask: np.ndarray):
        obj = np.asarray(ring, dtype=np.float32).view(cls)
        obj.dense_mask = np.ascontiguousarray(mask.astype(np.uint8))
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.dense_mask = getattr(obj, "dense_mask", None)

    def copy(self, order="C"):
        copied = super().copy(order).view(type(self))
        copied.dense_mask = None if self.dense_mask is None else self.dense_mask.copy()
        return copied


def _mask_to_rings(mask: np.ndarray) -> List[np.ndarray]:
    """Convert a binary mask to polygon rings using OpenCV contours."""
    mask_u8 = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rings = []
    for contour in contours:
        ring = contour.reshape(-1, 2)
        if ring.shape[0] >= 3:
            rings.append(ring.astype(np.float32))
    if rings or mask_u8.sum() == 0:
        return rings

    ys, xs = np.where(mask_u8 > 0)
    x1, x2 = float(xs.min()), float(xs.max() + 1)
    y1, y2 = float(ys.min()), float(ys.max() + 1)
    return [
        np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
    ]


def _coco_segmentation_to_rings(
    segmentation,
    *,
    height: int | None = None,
    width: int | None = None,
) -> List[np.ndarray]:
    """Convert COCO polygon or RLE segmentation to pixel-space rings."""
    if isinstance(segmentation, list):
        rings = []
        for polygon in segmentation:
            if polygon is None or len(polygon) < 6:
                continue
            ring = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            rings.append(ring)
        return rings

    if not isinstance(segmentation, dict):
        return []
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        return []

    rle = segmentation
    if isinstance(rle.get("counts"), list):
        if height is None or width is None:
            return []
        rle = mask_utils.frPyObjects(rle, height, width)
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    rings = _mask_to_rings(decoded)
    if rings:
        rings[0] = DenseMaskRing(rings[0], decoded)
    return rings


class YOLODataset(Dataset):
    """
    YOLO format dataset supporting both directory and file list modes.

    Mode 1 (Directory): Traditional structure
        dataset/images/{split}/*.jpg
        dataset/labels/{split}/*.txt

    Mode 2 (File List): .txt file format
        Provide img_files list directly, labels inferred via img2label_paths()

    Each label file contains one object per line:
    class_id center_x center_y width height  (all normalized 0-1)
    """

    def __init__(
        self,
        data_dir: str | None = None,
        split: str = "train",
        img_size: Tuple[int, int] = (640, 640),
        preproc=None,
        img_files: List[Path] | None = None,
        label_files: List[Path] | None = None,
        load_segments: bool = False,
        cache: Union[bool, str, None] = False,
    ):
        """
        Initialize YOLO dataset.

        Args:
            data_dir: Path to dataset root (for directory mode).
            split: "train" or "val" (for directory mode).
            img_size: Target image size (height, width).
            preproc: Preprocessing transform.
            img_files: List of image paths (for file list mode).
            label_files: List of label paths (optional, inferred if not provided).
            cache: Image cache mode. ``False`` / ``None`` disables caching
                (default), ``True`` / ``"ram"`` decodes + resizes all images
                once at construction and keeps them in RAM, ``"disk"`` saves
                decoded images as ``.npy`` files next to the originals.
        """
        self.img_size = img_size
        self.preproc = preproc
        self._input_dim = img_size
        self.load_segments = load_segments

        if img_files is not None:
            # File list mode (.txt format)
            self.img_files = [Path(f) for f in img_files]
            if label_files is not None:
                self.label_files = [Path(f) for f in label_files]
            else:
                # Infer label paths from image paths
                from libreyolo.data import img2label_paths

                self.label_files = img2label_paths(self.img_files)

            self.data_dir = None
            self.split = None
            self.img_dir = None
            self.label_dir = None
        else:
            # Directory mode (original behavior)
            if data_dir is None:
                raise ValueError("Either data_dir or img_files must be provided")

            self.data_dir = Path(data_dir)
            self.split = split
            self.img_dir = self.data_dir / "images" / split
            self.label_dir = self.data_dir / "labels" / split

            if not self.img_dir.exists():
                raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

            # Collect image files from directory
            self.img_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                self.img_files.extend(self.img_dir.glob(ext))
                self.img_files.extend(self.img_dir.glob(ext.upper()))
            self.img_files = sorted(self.img_files)

            # Generate corresponding label file paths
            self.label_files = [
                self.label_dir / (f.stem + ".txt") for f in self.img_files
            ]

        self.num_imgs = len(self.img_files)

        if self.num_imgs == 0:
            raise ValueError("No images found")

        # Pre-load annotations
        self.annotations = self._load_annotations()

        # Optional Ultralytics-style image cache. Default ``cache_mode`` is None
        # so ``load_image`` / ``load_resized_img`` behave exactly as before.
        self.cache_mode: Optional[str] = None
        self.ims: List[Optional[np.ndarray]] = []
        self.npy_files: List[Path] = []
        self._init_image_cache(cache)

    def _load_annotations(self) -> List:
        """Load all annotations."""
        total = len(self.img_files)
        source = self._annotation_source()
        main = is_main_process()
        if main:
            logger.info("Loading %d YOLO annotations from %s...", total, source)
        start = time.perf_counter()

        pairs = list(zip(self.img_files, self.label_files))
        max_workers = min(8, os.cpu_count() or 1, total)

        def load_one(pair):
            img_file, label_file = pair
            return self._load_label(label_file, img_file)

        tqdm_disable = not (main and sys.stderr.isatty())
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                annotations = list(
                    tqdm(
                        executor.map(load_one, pairs),
                        total=total,
                        desc=f"Loading YOLO annotations ({source})",
                        file=sys.stderr,
                        disable=tqdm_disable,
                    )
                )
        else:
            annotations = [
                load_one(pair)
                for pair in tqdm(
                    pairs,
                    total=total,
                    desc=f"Loading YOLO annotations ({source})",
                    file=sys.stderr,
                    disable=tqdm_disable,
                )
            ]

        if main:
            logger.info(
                "Loaded %d YOLO annotations from %s in %.2fs",
                total,
                source,
                time.perf_counter() - start,
            )
        if self.load_segments:
            self.segments = [item[1] for item in annotations]
            annotations = [item[0] for item in annotations]
        else:
            self.segments = None

        if sum(a[0].shape[0] for a in annotations) == 0:
            logger.warning("No labels found in %d files from %s.", total, source)
        return annotations

    def _annotation_source(self) -> str:
        """Return a compact source label for annotation loading progress."""
        if self.split is not None:
            return str(self.split)
        if self.label_files:
            label_dir = self.label_files[0].parent
            if label_dir.parent.name:
                return f"{label_dir.parent.name}/{label_dir.name}"
            return str(label_dir)
        return "dataset"

    def _load_label(self, label_file: Path, img_file: Path) -> Tuple:
        """Load annotation for a single image."""
        # Read image to get dimensions
        try:
            with Image.open(img_file) as im:
                width, height = im.size
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            raise FileNotFoundError(f"Cannot read image: {img_file}") from e

        # Load labels
        labels = []
        segments = []
        if label_file.exists():
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])

                        if len(parts) > 5:
                            # Segmentation format: derive bbox from polygon vertices
                            coords = [float(p) for p in parts[1:]]
                            cx, cy, w, h = polygon_to_cxcywh(coords)
                            if self.load_segments:
                                segments.append(_yolo_coords_to_rings(coords, width, height))
                        else:
                            cx, cy, w, h = map(float, parts[1:5])
                            if self.load_segments:
                                segments.append(_yolo_box_to_ring(cx, cy, w, h, width, height))

                        # Convert normalized xywh to pixel xyxy
                        x1 = (cx - w / 2) * width
                        y1 = (cy - h / 2) * height
                        x2 = (cx + w / 2) * width
                        y2 = (cy + h / 2) * height

                        labels.append([x1, y1, x2, y2, cls_id])

        # Create annotation array
        if labels:
            res = np.array(labels, dtype=np.float32)
        else:
            res = np.zeros((0, 5), dtype=np.float32)

        # Scale to target image size
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        if len(res) > 0:
            res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))
        file_name = img_file.name

        annotation = (res, img_info, resized_info, file_name)
        if self.load_segments:
            return annotation, segments
        return annotation

    def __len__(self):
        return self.num_imgs

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value):
        self._input_dim = value

    def load_anno(self, index: int) -> np.ndarray:
        """Load annotation for given index."""
        return self.annotations[index][0]

    def _source_image_path(self, index: int) -> Path:
        """Return the on-disk path of the source image for ``index``."""
        return self.img_files[index]

    def load_image(self, index: int) -> np.ndarray:
        """Load image for given index (unresized)."""
        if self.cache_mode == "disk" and self.npy_files:
            npy = self.npy_files[index]
            if npy.exists():
                try:
                    return np.load(str(npy))
                except (OSError, ValueError) as exc:  # pragma: no cover - defensive
                    logger.warning("Failed to load cached %s (%s); reading source", npy, exc)
        img_file = self._source_image_path(index)
        img = cv2.imread(str(img_file))
        assert img is not None, f"Failed to load {img_file}"
        return img

    def load_resized_img(self, index: int) -> np.ndarray:
        """Load and resize image, consulting the image cache when enabled."""
        if self.cache_mode == "ram" and self.ims:
            cached = self.ims[index]
            if cached is not None:
                return cached
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    # =========================================================================
    # Image caching (Ultralytics parity)
    # =========================================================================

    def _init_image_cache(self, cache: Union[bool, str, None]) -> None:
        """Pre-flight + populate the image cache. Falls back silently on failure."""
        mode = _normalize_cache(cache)
        if mode is None:
            return
        paths = [Path(p) for p in self.img_files]
        if mode == "ram":
            fits, required = _check_cache_ram(paths, self.img_size)
            if not fits:
                if is_main_process():
                    logger.warning(
                        "Skipping RAM image cache: estimated %.1f GB required "
                        "exceeds available memory. Falling back to no cache.",
                        required / 1e9,
                    )
                return
            self.ims = [None] * self.num_imgs
        else:  # "disk"
            fits, required = _check_cache_disk(paths)
            if not fits:
                if is_main_process():
                    logger.warning(
                        "Skipping disk image cache: estimated %.1f GB required "
                        "exceeds free disk space. Falling back to no cache.",
                        required / 1e9,
                    )
                return
            self.npy_files = [p.with_suffix(".npy") for p in paths]

        self.cache_mode = mode
        self._populate_image_cache()

    def _populate_image_cache(self) -> None:
        """Decode + resize (RAM) or write ``.npy`` (disk) for every image."""
        mode = self.cache_mode
        if mode is None:
            return
        storage = "RAM" if mode == "ram" else "Disk"
        main = is_main_process()
        disable_pbar = not (main and sys.stderr.isatty())
        gb = 0
        n = self.num_imgs

        def _job(i: int) -> int:
            if mode == "ram":
                img = self.load_image(i)
                h0, w0 = img.shape[:2]
                r = min(self.img_size[0] / h0, self.img_size[1] / w0)
                if r != 1.0:
                    img = cv2.resize(
                        img,
                        (int(w0 * r), int(h0 * r)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                img = img.astype(np.uint8, copy=False)
                self.ims[i] = img
                return int(img.nbytes)
            # disk
            return _cache_image_to_disk(self.npy_files[i], self._source_image_path(i))

        with ThreadPool(_CACHE_NUM_THREADS) as pool:
            iterator = pool.imap(_job, range(n))
            pbar = tqdm(
                iterator,
                total=n,
                desc=f"Caching images (0.0GB {storage})",
                file=sys.stderr,
                disable=disable_pbar,
            )
            for nbytes in pbar:
                gb += int(nbytes)
                pbar.set_description_str(f"Caching images ({gb / 1e9:.1f}GB {storage})")
            pbar.close()
        if main:
            logger.info(
                "Image cache ready: %d images (%.2f GB %s)",
                n,
                gb / 1e9,
                storage,
            )

    def _load_segments(self, index: int):
        if self.segments is None:
            return None
        return copy.deepcopy(self.segments[index])

    def pull_item(self, index: int):
        """Get item without preprocessing."""
        label, origin_image_size, _, _ = self.annotations[index]
        segments = self._load_segments(index)
        if getattr(self.preproc, "wants_unresized_image", False):
            img = self.load_image(index)
            label = copy.deepcopy(label)
            if label.shape[0] > 0:
                target_h, target_w = self.img_size
                r = min(target_h / origin_image_size[0], target_w / origin_image_size[1])
                if r > 0:
                    label[:, :4] = label[:, :4] / r
            if self.load_segments:
                return img, label, origin_image_size, index, segments
            return img, label, origin_image_size, index
        img = self.load_resized_img(index)
        if self.load_segments:
            return img, copy.deepcopy(label), origin_image_size, index, segments
        return img, copy.deepcopy(label), origin_image_size, index

    def __getitem__(self, index: int):
        """Get preprocessed item."""
        item = self.pull_item(index)
        if len(item) == 5:
            img, target, img_info, img_id, segments = item
        else:
            img, target, img_info, img_id = item
            segments = None

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        if self.load_segments:
            return img, target, img_info, img_id, segments
        return img, target, img_info, img_id


class COCODataset(Dataset):
    """
    COCO format dataset for YOLOX training.

    Directory structure:
    dataset/
    ├── annotations/
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    ├── train2017/
    │   ├── img1.jpg
    │   └── ...
    └── val2017/
    """

    def __init__(
        self,
        data_dir: str,
        json_file: str = "instances_train2017.json",
        name: str = "train2017",
        img_size: Tuple[int, int] = (640, 640),
        preproc=None,
        load_segments: bool = False,
        cache: Union[bool, str, None] = False,
    ):
        """
        Initialize COCO dataset.

        Args:
            data_dir: Path to dataset root
            json_file: COCO annotation JSON file name
            name: Image folder name (e.g., 'train2017')
            img_size: Target image size (height, width)
            preproc: Preprocessing transform
            cache: Image cache mode (False/None/True/"ram"/"disk"). See
                :class:`YOLODataset` for details.
        """
        try:
            from pycocotools.coco import COCO
        except ImportError:
            raise ImportError(
                "pycocotools is required for COCO format. "
                "Install with: pip install pycocotools"
            )

        self.data_dir = data_dir
        self.json_file = json_file
        self.name = name
        self.img_size = img_size
        self._input_dim = img_size
        self.preproc = preproc
        self.load_segments = load_segments

        # Load COCO annotations
        ann_file = os.path.join(data_dir, "annotations", json_file)
        self.coco = COCO(ann_file)

        # Remove useless info to save memory
        self._remove_useless_info()

        self.ids = self.coco.getImgIds()
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])

        # Pre-load annotations
        self.annotations = self._load_coco_annotations()

        # Optional Ultralytics-style image cache.
        self.cache_mode: Optional[str] = None
        self.ims: List[Optional[np.ndarray]] = []
        self.npy_files: List[Path] = []
        self._init_image_cache(cache)

    def _remove_useless_info(self):
        """Remove useless info from COCO to save memory."""
        dataset = self.coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset.get("images", []):
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if not self.load_segments:
            for anno in dataset.get("annotations", []):
                anno.pop("segmentation", None)

    def _load_coco_annotations(self) -> List:
        """Load all annotations."""
        total = len(self.ids)
        source = f"{self.name}/{self.json_file}"
        logger.info("Loading %d COCO annotations from %s...", total, source)
        start = time.perf_counter()
        annotations = [
            self._load_anno_from_id(id_)
            for id_ in tqdm(
                self.ids,
                total=total,
                desc=f"Loading COCO annotations ({self.name})",
                file=sys.stderr,
                disable=not sys.stderr.isatty(),
            )
        ]
        logger.info(
            "Loaded %d COCO annotations from %s in %.2fs",
            total,
            source,
            time.perf_counter() - start,
        )
        if self.load_segments:
            self.segments = [item[1] for item in annotations]
            annotations = [item[0] for item in annotations]
        else:
            self.segments = None

        if sum(a[0].shape[0] for a in annotations) == 0:
            logger.warning("No labels found in %d files from %s.", total, source)
        return annotations

    def _load_anno_from_id(self, id_: int) -> Tuple:
        """Load annotation for a single image ID."""
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)

        objs = []
        segments = []
        for obj in annotations:
            x1 = max(0, obj["bbox"][0])
            y1 = max(0, obj["bbox"][1])
            x2 = min(width, x1 + max(0, obj["bbox"][2]))
            y2 = min(height, y1 + max(0, obj["bbox"][3]))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)
                if self.load_segments:
                    segments.append(
                        _coco_segmentation_to_rings(
                            obj.get("segmentation", []),
                            height=height,
                            width=width,
                        )
                    )

        num_objs = len(objs)
        res = np.zeros((num_objs, 5), dtype=np.float32)
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        # Scale to target size
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))
        file_name = im_ann.get("file_name", f"{id_:012}.jpg")

        annotation = (res, img_info, resized_info, file_name)
        if self.load_segments:
            return annotation, segments
        return annotation

    def __len__(self):
        return self.num_imgs

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value):
        self._input_dim = value

    def load_anno(self, index: int) -> np.ndarray:
        """Load annotation for given index."""
        return self.annotations[index][0]

    def _source_image_path(self, index: int) -> Path:
        """Return the on-disk path of the source image for ``index``."""
        file_name = self.annotations[index][3]
        return Path(self.data_dir) / self.name / file_name

    def load_image(self, index: int) -> np.ndarray:
        """Load image for given index (unresized)."""
        if self.cache_mode == "disk" and self.npy_files:
            npy = self.npy_files[index]
            if npy.exists():
                try:
                    return np.load(str(npy))
                except (OSError, ValueError) as exc:  # pragma: no cover - defensive
                    logger.warning("Failed to load cached %s (%s); reading source", npy, exc)
        img_file = self._source_image_path(index)
        img = cv2.imread(str(img_file))
        assert img is not None, f"Failed to load {img_file}"
        return img

    def load_resized_img(self, index: int) -> np.ndarray:
        """Load and resize image, consulting the image cache when enabled."""
        if self.cache_mode == "ram" and self.ims:
            cached = self.ims[index]
            if cached is not None:
                return cached
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    # =========================================================================
    # Image caching (Ultralytics parity) — mirrors YOLODataset
    # =========================================================================

    def _init_image_cache(self, cache: Union[bool, str, None]) -> None:
        """Pre-flight + populate the image cache. Falls back silently on failure."""
        mode = _normalize_cache(cache)
        if mode is None:
            return
        paths = [self._source_image_path(i) for i in range(self.num_imgs)]
        if mode == "ram":
            fits, required = _check_cache_ram(paths, self.img_size)
            if not fits:
                logger.warning(
                    "Skipping RAM image cache: estimated %.1f GB required "
                    "exceeds available memory. Falling back to no cache.",
                    required / 1e9,
                )
                return
            self.ims = [None] * self.num_imgs
        else:  # "disk"
            fits, required = _check_cache_disk(paths)
            if not fits:
                logger.warning(
                    "Skipping disk image cache: estimated %.1f GB required "
                    "exceeds free disk space. Falling back to no cache.",
                    required / 1e9,
                )
                return
            self.npy_files = [p.with_suffix(".npy") for p in paths]

        self.cache_mode = mode
        self._populate_image_cache()

    def _populate_image_cache(self) -> None:
        """Decode + resize (RAM) or write ``.npy`` (disk) for every image."""
        mode = self.cache_mode
        if mode is None:
            return
        storage = "RAM" if mode == "ram" else "Disk"
        disable_pbar = not sys.stderr.isatty()
        gb = 0
        n = self.num_imgs

        def _job(i: int) -> int:
            if mode == "ram":
                img = self.load_image(i)
                h0, w0 = img.shape[:2]
                r = min(self.img_size[0] / h0, self.img_size[1] / w0)
                if r != 1.0:
                    img = cv2.resize(
                        img,
                        (int(w0 * r), int(h0 * r)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                img = img.astype(np.uint8, copy=False)
                self.ims[i] = img
                return int(img.nbytes)
            return _cache_image_to_disk(self.npy_files[i], self._source_image_path(i))

        with ThreadPool(_CACHE_NUM_THREADS) as pool:
            iterator = pool.imap(_job, range(n))
            pbar = tqdm(
                iterator,
                total=n,
                desc=f"Caching images (0.0GB {storage})",
                file=sys.stderr,
                disable=disable_pbar,
            )
            for nbytes in pbar:
                gb += int(nbytes)
                pbar.set_description_str(f"Caching images ({gb / 1e9:.1f}GB {storage})")
            pbar.close()
        logger.info(
            "Image cache ready: %d images (%.2f GB %s)",
            n,
            gb / 1e9,
            storage,
        )

    def _load_segments(self, index: int):
        if self.segments is None:
            return None
        return copy.deepcopy(self.segments[index])

    def pull_item(self, index: int):
        """Get item without preprocessing."""
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        segments = self._load_segments(index)
        if getattr(self.preproc, "wants_unresized_image", False):
            # Preprocessor handles all resizing in one pass (avoids the
            # letterbox-then-stretch double-resize). Targets are already
            # scaled by the dataset's letterbox ratio; we undo that here so
            # the preprocessor sees them in original-image coords matching
            # the original-image pixels we hand over.
            img = self.load_image(index)
            label = copy.deepcopy(label)
            if label.shape[0] > 0:
                target_h, target_w = self.img_size
                r = min(target_h / origin_image_size[0], target_w / origin_image_size[1])
                if r > 0:
                    label[:, :4] = label[:, :4] / r
            if self.load_segments:
                return img, label, origin_image_size, id_, segments
            return img, label, origin_image_size, id_
        img = self.load_resized_img(index)
        if self.load_segments:
            return img, copy.deepcopy(label), origin_image_size, id_, segments
        return img, copy.deepcopy(label), origin_image_size, id_

    def __getitem__(self, index: int):
        """Get preprocessed item."""
        item = self.pull_item(index)
        if len(item) == 5:
            img, target, img_info, img_id, segments = item
        else:
            img, target, img_info, img_id = item
            segments = None

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        if self.load_segments:
            return img, target, img_info, img_id, segments
        return img, target, img_info, img_id


def yolox_collate_fn(batch):
    """
    Collate function for YOLOX training.

    Returns:
        imgs: (B, C, H, W) tensor
        targets: (B, max_labels, 5) tensor
        img_infos: tuple of image info
        img_ids: tuple of image ids
    """
    has_segments = len(batch[0]) == 5
    if has_segments:
        imgs, targets, img_infos, img_ids, segments = zip(*batch)
    else:
        imgs, targets, img_infos, img_ids = zip(*batch)

    # Stack images
    imgs = torch.from_numpy(np.stack(imgs))

    # Stack targets (already padded to max_labels)
    targets = torch.from_numpy(np.stack(targets))

    if has_segments:
        if all(isinstance(s, np.ndarray) for s in segments):
            return imgs, targets, img_infos, img_ids, torch.from_numpy(np.stack(segments))
        if all(isinstance(s, torch.Tensor) for s in segments):
            return imgs, targets, img_infos, img_ids, torch.stack(segments)
        return imgs, targets, img_infos, img_ids, list(segments)
    return imgs, targets, img_infos, img_ids


def create_dataloader(
    dataset,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    sampler=None,
):
    """
    Create a DataLoader for YOLOX training.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Shuffle data (ignored when ``sampler`` is given — PyTorch
            forbids passing both)
        pin_memory: Pin memory for faster GPU transfer
        sampler: Optional sampler (e.g. ``DistributedSampler`` for DDP). When
            provided, the sampler's own shuffling takes over and ``shuffle``
            is forced to False to satisfy PyTorch's mutual-exclusion check.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=yolox_collate_fn,
        drop_last=True,
    )
