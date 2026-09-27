"""Convert the official QuickSRNet Medium 2x checkpoint to LibreYOLO format.

The supported source artifact is the BSD-3-Clause checkpoint published by
``quic/aimet-model-zoo``. Its model tensors are copied unchanged into a lean
LibreYOLO checkpoint; training optimizer/history objects are intentionally
discarded. The output is validated before an atomic rename.

Usage::

    python weights/convert_quicksrnet_weights.py \
        quicksrnet_medium_2x_checkpoint_float32.pth.tar \
        weights/LibreQuickSRNetm2-restore.pt --verify
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch

from _conversion_utils import add_repo_root_to_path, load_checkpoint, save_checkpoint


OFFICIAL_CHECKPOINT_URL = (
    "https://github.com/quic/aimet-model-zoo/releases/download/"
    "phase_2_january_artifacts/"
    "quicksrnet_medium_2x_checkpoint_float32.pth.tar"
)
OFFICIAL_CHECKPOINT_SHA256 = (
    "a0d176b40a649e45a176c3b53f45e0237015f4f2c17b157ef5c81e38c4442a0d"
)


def file_sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of ``path``."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_quicksrnet_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract only the model tensors from the official training checkpoint."""

    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise TypeError(
            "Expected the official QuickSRNet checkpoint layout with a "
            "'state_dict' dictionary."
        )
    state_dict = checkpoint["state_dict"]
    if not state_dict or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise TypeError("QuickSRNet state_dict must contain only named tensors.")
    return dict(state_dict)


def convert_weights(
    input_path: str,
    output_path: str,
    *,
    expected_sha256: str | None = OFFICIAL_CHECKPOINT_SHA256,
) -> dict:
    """Strictly load, wrap, validate, and atomically save the Medium 2x weights."""

    source_digest = file_sha256(input_path)
    if expected_sha256 is not None and source_digest != expected_sha256.lower():
        raise ValueError(
            "QuickSRNet source SHA-256 mismatch: "
            f"expected {expected_sha256.lower()}, got {source_digest}."
        )

    raw = load_checkpoint(input_path)
    state_dict = extract_quicksrnet_state_dict(raw)

    add_repo_root_to_path()
    from libreyolo.models.quicksrnet import LibreQuickSRNet
    from libreyolo.utils.serialization import (
        validate_checkpoint_metadata,
        wrap_libreyolo_checkpoint,
    )

    model = LibreQuickSRNet(model_path=None, size="m2", device="cpu")
    model.model.load_state_dict(state_dict, strict=True)
    checkpoint = wrap_libreyolo_checkpoint(
        model.model.state_dict(),
        model_family="quicksrnet",
        size="m2",
        task="restore",
        nc=1,
        names={0: "image"},
        imgsz=64,
        scale=2,
        degradation="super-resolution",
        dataset="DIV2K",
        source_url=OFFICIAL_CHECKPOINT_URL,
        source_sha256=source_digest,
    )
    validate_checkpoint_metadata(checkpoint, strict=True)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        save_checkpoint(checkpoint, temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        validate_checkpoint_metadata(loaded, strict=True)
        if not LibreQuickSRNet.can_load(loaded["model"]):
            raise ValueError("Converted state dict failed QuickSRNet identification.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Saved {destination}")
    print(f"source sha256: {source_digest}")
    print(f"output sha256: {file_sha256(destination)}")
    return checkpoint


def verify_conversion(converted_path: str) -> bool:
    """Load the output through the public factory and run a shape smoke test."""

    add_repo_root_to_path()
    from libreyolo import LibreQuickSRNet, LibreYOLO

    model = LibreYOLO(converted_path, device="cpu")
    if not isinstance(model, LibreQuickSRNet):
        raise TypeError(
            f"Factory selected {type(model).__name__}, not LibreQuickSRNet."
        )
    model.model.eval()
    with torch.inference_mode():
        output = model.model(torch.zeros(1, 3, 17, 23))
    if tuple(output.shape) != (1, 3, 34, 46):
        raise RuntimeError(
            f"Unexpected QuickSRNet output shape: {tuple(output.shape)}."
        )
    print("Verified family=quicksrnet size=m2 task=restore scale=2")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert official QuickSRNet Medium 2x weights"
    )
    parser.add_argument("input", help="Official Medium 2x .pth.tar checkpoint")
    parser.add_argument("output", help="Output LibreYOLO .pt checkpoint")
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    convert_weights(arguments.input, arguments.output)
    if arguments.verify:
        verify_conversion(arguments.output)
