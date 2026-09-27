"""Convert exact official DDColor checkpoints to LibreYOLO format.

Only the two audited Apache-2.0 Hugging Face artifacts are accepted. Source
bytes are SHA-256 pinned, loaded with ``weights_only=True``, strictly loaded
into a fresh native model, wrapped without optimizer/history objects, validated,
and written via an atomic replace.

Usage::

    python weights/convert_ddcolor_weights.py \
        pytorch_model.bin weights/LibreDDColort-restore.pt --size t --verify
    python weights/convert_ddcolor_weights.py \
        pytorch_model.bin weights/LibreDDColorl-restore.pt --size l
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch

from _conversion_utils import add_repo_root_to_path, save_checkpoint


OFFICIAL_CHECKPOINTS: dict[str, dict[str, str | int]] = {
    "t": {
        "repo": "piddnad/ddcolor_paper_tiny",
        "revision": "cf9fd99c1d7472689ec7413441c1b799a51866a3",
        "filename": "pytorch_model.bin",
        "bytes": 220_372_845,
        "sha256": "8a1277bc90a1bfbb6d2d83933a9a6bc821931879ca93e26e4fcec12165d41fce",
    },
    "l": {
        "repo": "piddnad/ddcolor_modelscope",
        "revision": "060f67494e31883a4b13cb27f889f3154847ada4",
        "filename": "pytorch_model.bin",
        "bytes": 911_914_869,
        "sha256": "d81711971ec59200da26d5e8a1afae8dd3778d495ea8ad7a7dadc769f403f7e7",
    },
}


def official_url(size: str) -> str:
    artifact = OFFICIAL_CHECKPOINTS[size]
    return (
        f"https://huggingface.co/{artifact['repo']}/resolve/"
        f"{artifact['revision']}/{artifact['filename']}"
    )


def file_sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_ddcolor_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract the official raw or ``params``-wrapped tensor dictionary."""

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected a DDColor checkpoint dictionary, got {type(checkpoint)!r}."
        )
    candidate = checkpoint.get("params", checkpoint)
    if not isinstance(candidate, dict) or not candidate:
        raise TypeError("DDColor checkpoint does not contain a non-empty state dict.")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in candidate.items()
    ):
        raise TypeError("DDColor state dict must contain only named tensors.")
    return dict(candidate)


def _resolve_official_artifact(path: str | Path, size: str | None) -> tuple[str, str]:
    source = Path(path)
    digest = file_sha256(source)
    byte_count = source.stat().st_size
    if size is None:
        matches = [
            code
            for code, artifact in OFFICIAL_CHECKPOINTS.items()
            if digest == artifact["sha256"] and byte_count == artifact["bytes"]
        ]
        if len(matches) != 1:
            raise ValueError(
                "DDColor source is not one of the two audited official artifacts: "
                f"bytes={byte_count}, sha256={digest}."
            )
        size = matches[0]
    artifact = OFFICIAL_CHECKPOINTS[size]
    expected_digest = str(artifact["sha256"])
    expected_bytes = int(artifact["bytes"])
    if byte_count != expected_bytes or digest != expected_digest:
        raise ValueError(
            f"DDColor {size} source mismatch: expected bytes={expected_bytes}, "
            f"sha256={expected_digest}; got bytes={byte_count}, sha256={digest}."
        )
    return size, digest


def convert_weights(
    input_path: str,
    output_path: str,
    *,
    size: str | None = None,
) -> dict:
    """Strictly validate, wrap, and atomically save one official checkpoint."""

    size, source_digest = _resolve_official_artifact(input_path, size)
    raw = torch.load(input_path, map_location="cpu", weights_only=True)
    state_dict = extract_ddcolor_state_dict(raw)

    add_repo_root_to_path()
    from libreyolo.models.ddcolor import LibreDDColor
    from libreyolo.utils.serialization import (
        validate_checkpoint_metadata,
        wrap_libreyolo_checkpoint,
    )

    detected = LibreDDColor.detect_size(state_dict)
    if detected != size:
        raise ValueError(
            f"DDColor source architecture is size {detected!r}, expected {size!r}."
        )
    model = LibreDDColor(size=size, device="cpu")
    model.model.load_state_dict(state_dict, strict=True)

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="ddcolor",
        size=size,
        task="restore",
        nc=1,
        names={0: "image"},
        imgsz=512,
        degradation="colorization",
        dataset="ImageNet",
        source_repo=str(OFFICIAL_CHECKPOINTS[size]["repo"]),
        source_url=official_url(size),
        source_revision=str(OFFICIAL_CHECKPOINTS[size]["revision"]),
        source_filename=str(OFFICIAL_CHECKPOINTS[size]["filename"]),
        source_size=int(OFFICIAL_CHECKPOINTS[size]["bytes"]),
        source_sha256=source_digest,
        upstream_license="Apache-2.0",
        weight_license="Apache-2.0 (publisher declaration)",
        training_data_terms="ImageNet access agreement",
    )
    validate_checkpoint_metadata(checkpoint, strict=True)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        save_checkpoint(checkpoint, temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        validate_checkpoint_metadata(loaded, strict=True)
        if loaded.get("size") != size or not LibreDDColor.can_load(loaded["model"]):
            raise ValueError("Converted checkpoint failed DDColor identification.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Saved {destination}")
    print(f"source url: {official_url(size)}")
    print(f"source sha256: {source_digest}")
    print(f"output sha256: {file_sha256(destination)}")
    return checkpoint


def verify_conversion(converted_path: str) -> bool:
    """Strict-load the converted checkpoint and run a small tensor smoke."""

    add_repo_root_to_path()
    from libreyolo.models.ddcolor import LibreDDColor

    checkpoint = torch.load(converted_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"]
    size = str(checkpoint["size"])
    model = LibreDDColor(size=size, device="cpu")
    model.model.load_state_dict(state_dict, strict=True)
    model.model.eval()
    with torch.inference_mode():
        output = model.model(torch.zeros(1, 3, 32, 32))
    if tuple(output.shape) != (1, 2, 32, 32):
        raise RuntimeError(f"Unexpected DDColor output shape: {tuple(output.shape)}.")
    print(f"Verified family=ddcolor size={size} task=restore output=(1,2,32,32)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert an audited official DDColor checkpoint"
    )
    parser.add_argument("input", help="Official pinned pytorch_model.bin")
    parser.add_argument("output", help="Output LibreYOLO .pt checkpoint")
    parser.add_argument(
        "--size",
        choices=sorted(OFFICIAL_CHECKPOINTS),
        default=None,
        help="Expected size; omitted means infer from the pinned file digest.",
    )
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    convert_weights(arguments.input, arguments.output, size=arguments.size)
    if arguments.verify:
        verify_conversion(arguments.output)
