"""Unified image loader for all image input formats."""

import io
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image


# Type alias for all supported image inputs
ImageInput = Union[
    str, Path, Image.Image, np.ndarray, torch.Tensor, bytes, "io.BytesIO"
]

# Supported image file extensions for directory scanning
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}


class ImageLoader:
    """
    Unified image loader that accepts any reasonable image input
    and returns a PIL Image in RGB format.

    Example:
        >>> img = ImageLoader.load("./image.jpg")
        >>> img = ImageLoader.load("https://example.com/image.jpg")
        >>> img = ImageLoader.load(cv2.imread("image.jpg"), color_format="bgr")
        >>> img = ImageLoader.load(torch.randn(3, 224, 224))
    """

    # =========================================================================
    # Private helpers
    # =========================================================================

    @classmethod
    def _normalize_dtype(cls, arr: np.ndarray) -> np.ndarray:
        if arr.dtype in (np.float32, np.float64, np.float16):
            # [0, 1] range if max <= 1, otherwise [0, 255]
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.clip(0, 255).astype(np.uint8)
        return arr

    @classmethod
    def _from_pil(cls, img: Image.Image) -> Image.Image:
        return img.convert("RGB")

    @classmethod
    def _from_bytes(cls, data: Union[bytes, io.BytesIO]) -> Image.Image:
        if isinstance(data, bytes):
            data = io.BytesIO(data)

        try:
            return Image.open(data).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode image from bytes: {e}") from e

    @classmethod
    def _from_url(cls, url: str) -> Image.Image:
        try:
            import requests
        except ImportError:
            raise ImportError(
                "Loading images from URLs requires the 'requests' package. "
                "Install it with: pip install requests"
            )

        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to load image from URL '{url}': {e}") from e

    @classmethod
    def _from_s3(cls, uri: str) -> Image.Image:
        """Load image from S3 URI (s3://bucket/key)."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "Loading images from S3 requires the 'boto3' package. "
                "Install it with: pip install boto3"
            )

        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {uri}")

        path_parts = uri[5:].split("/", 1)
        if len(path_parts) != 2:
            raise ValueError(f"Invalid S3 URI format: {uri}. Expected s3://bucket/key")

        bucket, key = path_parts

        try:
            s3 = boto3.client("s3")
            response = s3.get_object(Bucket=bucket, Key=key)
            return Image.open(io.BytesIO(response["Body"].read())).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image from S3 '{uri}': {e}") from e

    @classmethod
    def _from_gcs(cls, uri: str) -> Image.Image:
        """Load image from Google Cloud Storage URI (gs://bucket/path)."""
        try:
            import gcsfs
        except ImportError:
            raise ImportError(
                "Loading images from GCS requires the 'gcsfs' package. "
                "Install it with: pip install gcsfs"
            )

        try:
            fs = gcsfs.GCSFileSystem()
            gcs_path = uri[5:] if uri.startswith("gs://") else uri
            with fs.open(gcs_path, "rb") as f:
                return Image.open(io.BytesIO(f.read())).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image from GCS '{uri}': {e}") from e

    @classmethod
    def _from_path_or_url(cls, path: Union[str, Path]) -> Image.Image:
        path_str = str(path)

        if path_str.startswith(("http://", "https://")):
            return cls._from_url(path_str)
        elif path_str.startswith("s3://"):
            return cls._from_s3(path_str)
        elif path_str.startswith("gs://"):
            return cls._from_gcs(path_str)
        else:
            if not Path(path_str).exists():
                raise FileNotFoundError(f"Image file not found: {path_str}")
            return Image.open(path_str).convert("RGB")

    @classmethod
    def _from_numpy(cls, arr: np.ndarray, color_format: str) -> Image.Image:
        """Convert NumPy array to PIL Image (auto-detects format, dtype, channels)."""
        if arr.ndim == 2:
            # Grayscale
            arr = cls._normalize_dtype(arr)
            return Image.fromarray(arr, mode="L").convert("RGB")

        elif arr.ndim == 3:
            # Heuristic: if first dim is 1, 3, or 4 and smaller than last dim
            if arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[2]:
                arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC

            if arr.shape[2] == 1:
                arr = np.squeeze(arr, axis=2)
                arr = cls._normalize_dtype(arr)
                return Image.fromarray(arr, mode="L").convert("RGB")

            if color_format == "bgr" and arr.shape[2] >= 3:
                arr = arr[..., ::-1].copy()  # BGR -> RGB

            if arr.shape[2] == 4:
                arr = arr[..., :3]  # RGBA -> RGB

            arr = cls._normalize_dtype(arr)
            return Image.fromarray(arr, mode="RGB")

        elif arr.ndim == 4:
            # Batch — take first image (NCHW or NHWC)
            return cls._from_numpy(arr[0], color_format)

        else:
            raise ValueError(
                f"Unsupported array dimensions: {arr.ndim}. Expected 2, 3, or 4."
            )

    @classmethod
    def _from_tensor(cls, tensor: torch.Tensor) -> Image.Image:
        """Convert PyTorch tensor (CHW or NCHW) to PIL Image."""
        tensor = tensor.detach().cpu()

        if tensor.dim() == 4:
            tensor = tensor[0]  # take first image if batched

        if tensor.dim() == 3:
            if tensor.shape[0] in (1, 3, 4) and tensor.shape[0] < tensor.shape[2]:
                tensor = tensor.permute(1, 2, 0)  # CHW -> HWC

        arr = tensor.numpy()
        # Tensors are typically RGB, not BGR
        return cls._from_numpy(arr, color_format="rgb")

    # =========================================================================
    # Public API
    # =========================================================================

    @classmethod
    def load(cls, source: ImageInput, color_format: str = "auto") -> Image.Image:
        """
        Load image from any source and return PIL Image in RGB format.

        Args:
            source: Image source. Supported types:
                - str: Local file path or URL (http/https/s3/gs)
                - pathlib.Path: Local file path
                - PIL.Image: PIL Image object
                - np.ndarray: NumPy array (HWC or CHW, RGB or BGR)
                - torch.Tensor: PyTorch tensor (CHW or NCHW)
                - bytes: Raw image bytes
                - io.BytesIO: BytesIO object containing image data
            color_format: Color format hint for NumPy arrays.
                - "auto": Auto-detect (default, uses heuristics)
                - "rgb": Input is RGB format
                - "bgr": Input is BGR format (e.g., OpenCV)

        Returns:
            PIL Image in RGB format

        Raises:
            TypeError: If source type is not supported
            ValueError: If image cannot be loaded
            ImportError: If required optional dependency is missing
        """
        color_format = color_format.lower()
        if color_format not in ("auto", "rgb", "bgr"):
            raise ValueError(
                f"color_format must be 'auto', 'rgb', or 'bgr', got '{color_format}'"
            )

        if isinstance(source, Image.Image):
            return cls._from_pil(source)

        elif isinstance(source, (str, Path)):
            return cls._from_path_or_url(source)

        elif isinstance(source, np.ndarray):
            return cls._from_numpy(source, color_format)

        elif isinstance(source, torch.Tensor):
            return cls._from_tensor(source)

        elif isinstance(source, (bytes, io.BytesIO)):
            return cls._from_bytes(source)

        else:
            raise TypeError(
                f"Unsupported image type: {type(source).__name__}. "
                f"Supported types: str, Path, PIL.Image, np.ndarray, torch.Tensor, bytes, BytesIO"
            )

    @classmethod
    def collect_images(
        cls, directory: Union[str, Path], recursive: bool = True
    ) -> List[Path]:
        """
        Recursively collect all image file paths from a directory.

        Args:
            directory: Path to the directory to scan.
            recursive: If True (default), recursively walk subdirectories.
                      If False, only scan the immediate directory.

        Returns:
            Sorted list of Path objects for all image files found.

        Raises:
            ValueError: If the path is not a directory.

        Example:
            >>> paths = ImageLoader.collect_images("./images/")
            >>> for path in paths:
            ...     img = ImageLoader.load(path)
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        pattern = "**/*" if recursive else "*"
        return sorted(
            [
                p
                for p in directory.glob(pattern)
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        )
