"""Convert the pinned ViTMatte-S Composition-1k safetensors checkpoint.

The source checkpoint is trained on Adobe Composition-1k and is treated as
NON-COMMERCIAL under the Adobe Deep Image Matting Dataset License Agreement.
See ``libreyolo/models/vitmatte/NOTICE`` before using or redistributing it.

Usage::

    python weights/convert_vitmatte_weights.py model.safetensors \
        weights/LibreViTMattes-matte.pt --verify
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from _conversion_utils import add_repo_root_to_path, save_checkpoint


OFFICIAL_REPO = "hustvl/vitmatte-small-composition-1k"
OFFICIAL_REVISION = "6a58ad7646403c1df626fbd746900aec7361ea1d"
OFFICIAL_FILENAME = "model.safetensors"
OFFICIAL_SIZE = 103_294_572
OFFICIAL_SHA256 = "bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb"
OFFICIAL_URL = (
    f"https://huggingface.co/{OFFICIAL_REPO}/resolve/"
    f"{OFFICIAL_REVISION}/{OFFICIAL_FILENAME}"
)
ADOBE_DIM_LICENSE_URL = "https://sites.google.com/view/deepimagematting/homepage"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_checkpoint(input_path: str | Path) -> str:
    """Fail closed unless ``input_path`` is the one audited source artifact."""
    source = Path(input_path)
    if source.suffix.lower() != ".safetensors":
        raise ValueError(
            "ViTMatte conversion accepts only the pinned safetensors file."
        )
    if not source.is_file():
        raise FileNotFoundError(f"ViTMatte source checkpoint not found: {source}")
    source_size = source.stat().st_size
    if source_size != OFFICIAL_SIZE:
        raise ValueError(
            "ViTMatte source size mismatch: "
            f"expected {OFFICIAL_SIZE} bytes, got {source_size}."
        )
    source_digest = file_sha256(source)
    if source_digest != OFFICIAL_SHA256:
        raise ValueError(
            "ViTMatte source SHA-256 mismatch: "
            f"expected {OFFICIAL_SHA256}, got {source_digest}."
        )
    return source_digest


def convert_weights(input_path: str, output_path: str) -> dict:
    """Verify, strictly load, metadata-wrap, and atomically save ViTMatte-S."""
    source_digest = verify_source_checkpoint(input_path)

    from safetensors.torch import load_file

    state_dict = dict(load_file(input_path, device="cpu"))
    if not state_dict or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise TypeError("ViTMatte safetensors must contain named tensors only.")

    add_repo_root_to_path()
    from libreyolo.models.vitmatte import LibreViTMatte
    from libreyolo.utils.serialization import (
        validate_checkpoint_metadata,
        wrap_libreyolo_checkpoint,
    )

    model = LibreViTMatte(model_path=None, size="s", device="cpu")
    model.model.load_state_dict(state_dict, strict=True)
    checkpoint = wrap_libreyolo_checkpoint(
        model.model.state_dict(),
        model_family="vitmatte",
        size="s",
        task="matte",
        nc=1,
        names={0: "matte"},
        imgsz=512,
        dataset="Adobe Composition-1k",
        weight_license="NON-COMMERCIAL: Adobe Deep Image Matting dataset terms",
        weight_license_url=ADOBE_DIM_LICENSE_URL,
        source_repo=OFFICIAL_REPO,
        source_url=OFFICIAL_URL,
        source_revision=OFFICIAL_REVISION,
        source_filename=OFFICIAL_FILENAME,
        source_size=OFFICIAL_SIZE,
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
        if not LibreViTMatte.can_load(loaded["model"]):
            raise ValueError("Converted state dict failed ViTMatte identification.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Saved {destination}")
    print(f"source sha256: {source_digest}")
    print(f"output sha256: {file_sha256(destination)}")
    return checkpoint


def verify_conversion(converted_path: str) -> bool:
    """Strictly round-trip the converted checkpoint without registry support."""
    add_repo_root_to_path()
    from libreyolo.models.vitmatte import LibreViTMatte
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    checkpoint = torch.load(converted_path, map_location="cpu", weights_only=True)
    validate_checkpoint_metadata(checkpoint, strict=True)
    model = LibreViTMatte(converted_path, device="cpu")
    if model.size != "s" or model.task != "matte":
        raise RuntimeError(
            f"Unexpected ViTMatte metadata: size={model.size}, task={model.task}."
        )
    if not LibreViTMatte.can_load(model.model.state_dict()):
        raise RuntimeError("Round-tripped model failed ViTMatte identification.")
    print("Verified family=vitmatte size=s task=matte")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the pinned non-commercial ViTMatte-S checkpoint"
    )
    parser.add_argument("input", help="Pinned upstream model.safetensors")
    parser.add_argument("output", help="Output LibreViTMattes-matte.pt")
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    convert_weights(arguments.input, arguments.output)
    if arguments.verify:
        verify_conversion(arguments.output)
