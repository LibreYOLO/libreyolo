"""Convert the official HVI-CIDNet Generalization checkpoint.

The source is the publisher's MIT-tagged safetensors artifact. Learned tensors
are unchanged; conversion adds the LibreYOLO v1 metadata wrapper only.

Usage::

    python weights/convert_hvi_cidnet_weights.py model.safetensors \
        weights/LibreHVICIDNett-restore.pt --verify
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from _conversion_utils import add_repo_root_to_path, save_checkpoint


OFFICIAL_REPO = "Fediory/HVI-CIDNet-Generalization"
OFFICIAL_REVISION = "51481ef2546f870060c43eb6d6525399f5b3d2d3"
OFFICIAL_FILENAME = "model.safetensors"
OFFICIAL_SIZE = 7_920_332
OFFICIAL_URL = (
    f"https://huggingface.co/{OFFICIAL_REPO}/resolve/"
    f"{OFFICIAL_REVISION}/{OFFICIAL_FILENAME}"
)
OFFICIAL_SHA256 = "2291407125e809cc9c0614cc2d010d21d309a66eb3da33e1ee2386a68fa05894"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_weights(
    input_path: str,
    output_path: str,
) -> dict:
    """Strictly verify, metadata-wrap, and atomically save HVI-CIDNet."""

    source_digest = file_sha256(input_path)
    source_size = Path(input_path).stat().st_size
    if source_size != OFFICIAL_SIZE:
        raise ValueError(
            "HVI-CIDNet source size mismatch: "
            f"expected {OFFICIAL_SIZE} bytes, got {source_size}."
        )
    if source_digest != OFFICIAL_SHA256:
        raise ValueError(
            "HVI-CIDNet source SHA-256 mismatch: "
            f"expected {OFFICIAL_SHA256}, got {source_digest}."
        )
    from safetensors.torch import load_file

    state_dict = dict(load_file(input_path, device="cpu"))
    if not state_dict or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise TypeError("HVI-CIDNet safetensors must contain named tensors only.")

    add_repo_root_to_path()
    from libreyolo.models.hvi_cidnet import LibreHVICIDNet
    from libreyolo.utils.serialization import (
        validate_checkpoint_metadata,
        wrap_libreyolo_checkpoint,
    )

    model = LibreHVICIDNet(model_path=None, size="t", device="cpu")
    model.model.load_state_dict(state_dict, strict=True)
    checkpoint = wrap_libreyolo_checkpoint(
        model.model.state_dict(),
        model_family="hvi_cidnet",
        size="t",
        task="restore",
        nc=1,
        names={0: "image"},
        imgsz=256,
        scale=1,
        degradation="low-light",
        dataset="LOLv2-Synthetic",
        source_repo=OFFICIAL_REPO,
        source_url=OFFICIAL_URL,
        source_revision=OFFICIAL_REVISION,
        source_filename=OFFICIAL_FILENAME,
        source_size=source_size,
        source_sha256=source_digest,
        upstream_license="MIT",
        weight_license="MIT (publisher declaration)",
        training_data_license="not stated by the canonical LOLv2 source",
    )
    validate_checkpoint_metadata(checkpoint, strict=True)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        save_checkpoint(checkpoint, temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        validate_checkpoint_metadata(loaded, strict=True)
        if not LibreHVICIDNet.can_load(loaded["model"]):
            raise ValueError("Converted state dict failed HVI-CIDNet identification.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Saved {destination}")
    print(f"source sha256: {source_digest}")
    print(f"output sha256: {file_sha256(destination)}")
    return checkpoint


def verify_conversion(converted_path: str) -> bool:
    add_repo_root_to_path()
    from libreyolo import LibreHVICIDNet, LibreYOLO

    model = LibreYOLO(converted_path, device="cpu")
    if not isinstance(model, LibreHVICIDNet):
        raise TypeError(f"Factory selected {type(model).__name__}, not LibreHVICIDNet.")
    with torch.inference_mode():
        output = model.model(torch.zeros(1, 3, 16, 24))
    if tuple(output.shape) != (1, 3, 16, 24):
        raise RuntimeError(
            f"Unexpected HVI-CIDNet output shape: {tuple(output.shape)}."
        )
    print("Verified family=hvi_cidnet size=t task=restore")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HVI-CIDNet weights")
    parser.add_argument("input", help="Official model.safetensors")
    parser.add_argument("output", help="Output LibreHVICIDNett-restore.pt")
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    convert_weights(arguments.input, arguments.output)
    if arguments.verify:
        verify_conversion(arguments.output)
