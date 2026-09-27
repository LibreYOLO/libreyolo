"""Compare LibreLaMa's embedded graph with direct OpenCV DNN execution.

OpenCV Zoo documents OpenCV >=5.0 for this artifact. Released OpenCV Python
4.x builds cannot execute the graph, so this external-data gate fails early on
those versions rather than reporting a misleading numerical result.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libreyolo.models.lama import LibreLaMa  # noqa: E402
from libreyolo.models.lama.nn import (  # noqa: E402
    OFFICIAL_ONNX_SHA256,
    OFFICIAL_ONNX_SIZE_BYTES,
)
from libreyolo.models.lama.utils import preprocess_image_and_mask  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_opencv_5() -> None:
    match = re.match(r"(\d+)\.(\d+)", cv2.__version__)
    if match is None or int(match.group(1)) < 5:
        raise RuntimeError(
            "Direct LaMa parity requires OpenCV >=5.0 as documented by "
            f"OpenCV Zoo; found {cv2.__version__}. LibreLaMa runtime itself "
            "uses ONNX Runtime and is unaffected."
        )


def compare(
    checkpoint_path: str,
    onnx_path: str,
    image_path: str,
    mask_path: str,
    *,
    atol: float = 1e-3,
) -> tuple[float, float]:
    """Return max/mean raw-output error after enforcing public pixel semantics."""

    _require_opencv_5()
    onnx_file = Path(onnx_path)
    if onnx_file.stat().st_size != OFFICIAL_ONNX_SIZE_BYTES:
        raise ValueError("Standalone ONNX size does not match the pinned artifact.")
    digest = _sha256(onnx_file)
    if digest != OFFICIAL_ONNX_SHA256:
        raise ValueError(
            f"Standalone ONNX SHA-256 mismatch: expected {OFFICIAL_ONNX_SHA256}, "
            f"got {digest}."
        )

    guided, _, original_size, _, context = preprocess_image_and_mask(
        image_path,
        mask_path,
        input_size=512,
    )
    arrays = guided.numpy()

    direct = cv2.dnn.readNetFromONNX(str(onnx_file))
    direct.setInput(np.ascontiguousarray(arrays[:, :3]), "image")
    direct.setInput(np.ascontiguousarray(arrays[:, 3:4]), "mask")
    direct_output = direct.forward()

    model = LibreLaMa(checkpoint_path, size="b", device="cpu")
    with torch.inference_mode():
        libre_output = model.model(guided).cpu().numpy()
    difference = np.abs(libre_output - direct_output)
    max_abs = float(difference.max())
    mean_abs = float(difference.mean())
    if max_abs > float(atol):
        raise AssertionError(
            f"LaMa raw parity failed: max_abs={max_abs:.9g} > atol={atol:.9g}; "
            f"mean_abs={mean_abs:.9g}."
        )

    public = model.predict(image_path, mask=mask_path).restored.array
    if public.shape[:2] != (original_size[1], original_size[0]):
        raise AssertionError(
            f"Public result shape {public.shape} does not match {original_size}."
        )
    if not np.array_equal(
        public[~context.fill_mask], context.original_rgb[~context.fill_mask]
    ):
        raise AssertionError("Public result changed pixels outside the fill mask.")
    return max_abs, mean_abs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify LibreLaMa tensor parity")
    parser.add_argument("checkpoint", help="Converted LibreLaMab-restore.pt")
    parser.add_argument("onnx", help="Pinned OpenCV Zoo ONNX artifact")
    parser.add_argument("image", help="Parity input image")
    parser.add_argument("mask", help="Aligned nonzero-means-fill mask")
    parser.add_argument("--atol", type=float, default=1e-3)
    arguments = parser.parse_args()
    maximum, mean = compare(
        arguments.checkpoint,
        arguments.onnx,
        arguments.image,
        arguments.mask,
        atol=arguments.atol,
    )
    print(f"max_abs_diff={maximum:.9g}")
    print(f"mean_abs_diff={mean:.9g}")
