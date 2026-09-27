"""Embed the pinned OpenCV Zoo LaMa ONNX artifact in a LibreYOLO checkpoint.

The converter performs no architecture conversion. It verifies the exact
Apache-2.0 ONNX artifact, stores its bytes as one persistent uint8 state-dict
buffer, writes the LibreYOLO v1.0 metadata wrapper, safely reloads the result,
and atomically replaces the destination.

Usage::

    python weights/convert_lama_weights.py inpainting_lama_2025jan.onnx \
        weights/LibreLaMab-restore.pt --verify
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from _conversion_utils import add_repo_root_to_path


OFFICIAL_REPO = "opencv/inpainting_lama"
OFFICIAL_REVISION = "aee6d22f0a13e5e35af1c9a1c3afd62841fc6f3f"
OFFICIAL_FILENAME = "inpainting_lama_2025jan.onnx"
OFFICIAL_URL = (
    f"https://huggingface.co/{OFFICIAL_REPO}/resolve/"
    f"{OFFICIAL_REVISION}/{OFFICIAL_FILENAME}"
)
OFFICIAL_SIZE_BYTES = 92_591_623
OFFICIAL_SHA256 = "7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_weights(input_path: str, output_path: str) -> dict:
    """Verify, embed, safely reload, and atomically save the official graph."""

    source = Path(input_path)
    source_size = source.stat().st_size
    if source_size != OFFICIAL_SIZE_BYTES:
        raise ValueError(
            "LaMa source size mismatch: expected "
            f"{OFFICIAL_SIZE_BYTES} bytes, got {source_size}."
        )
    source_digest = file_sha256(source)
    if source_digest != OFFICIAL_SHA256:
        raise ValueError(
            "LaMa source SHA-256 mismatch: expected "
            f"{OFFICIAL_SHA256}, got {source_digest}."
        )

    payload = bytearray(source.read_bytes())
    graph = torch.from_numpy(np.frombuffer(payload, dtype=np.uint8).copy())
    del payload

    add_repo_root_to_path()
    from libreyolo.models.lama import LibreLaMa
    from libreyolo.models.lama import nn as lama_nn
    from libreyolo.utils.serialization import (
        validate_checkpoint_metadata,
        wrap_libreyolo_checkpoint,
    )

    lama_nn.validate_onnx_graph_tensor(graph)
    state_dict = {"onnx_graph": graph}
    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="lama",
        size="b",
        task="restore",
        nc=1,
        names={0: "image"},
        imgsz=512,
        scale=1,
        degradation="inpaint",
        dataset="Places365-Challenge",
        source_url=OFFICIAL_URL,
        source_revision=OFFICIAL_REVISION,
        source_sha256=source_digest,
        source_size=source_size,
        onnx_opset=21,
        runtime="onnxruntime>=1.18",
        upstream_license="Apache-2.0",
        inference_only=True,
    )
    validate_checkpoint_metadata(checkpoint, strict=True)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(checkpoint, temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        validate_checkpoint_metadata(loaded, strict=True)
        if not LibreLaMa.can_load(loaded["model"]):
            raise ValueError("Converted state dict failed LibreLaMa identification.")
        loaded_digest = lama_nn.validate_onnx_graph_tensor(
            loaded["model"]["onnx_graph"]
        )
        if loaded_digest != source_digest:
            raise ValueError(
                "Converted checkpoint changed the embedded ONNX bytes: "
                f"source={source_digest}, embedded={loaded_digest}."
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Saved {destination}")
    print(f"source sha256: {source_digest}")
    print(f"output sha256: {file_sha256(destination)}")
    return checkpoint


def verify_conversion(converted_path: str) -> bool:
    """Load the single-file checkpoint and run one public inpainting call."""

    add_repo_root_to_path()
    from libreyolo.models.lama import LibreLaMa

    image = np.zeros((17, 23, 3), dtype=np.uint8)
    image[..., 0] = 31
    mask = np.zeros((17, 23), dtype=np.uint8)
    mask[5:12, 7:16] = 255
    model = LibreLaMa(converted_path, size="b", device="cpu")
    result = model.predict(image, mask=mask)
    if result.restored is None or result.restored.array.shape != image.shape:
        raise RuntimeError("LibreLaMa converted-checkpoint prediction failed.")
    if not np.array_equal(result.restored.array[mask == 0], image[mask == 0]):
        raise RuntimeError("LibreLaMa did not preserve pixels outside the mask.")
    print("Verified family=lama size=b task=restore mask-required single-file load")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed the pinned OpenCV Zoo LaMa ONNX in LibreYOLO format"
    )
    parser.add_argument("input", help=f"Pinned {OFFICIAL_FILENAME}")
    parser.add_argument("output", help="Output LibreLaMab-restore.pt")
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    convert_weights(arguments.input, arguments.output)
    if arguments.verify:
        verify_conversion(arguments.output)
