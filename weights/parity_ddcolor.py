"""External parity proof against pinned Apache-2.0 DDColor source.

The script requires a checkout of ``piddnad/DDColor`` at exact commit
``2adb63f2656ac41cbdf7b894cddd94121a3faf13``. It strict-loads one official
checkpoint into upstream and LibreYOLO networks and requires bit-exact FP32
output on an identical tensor. With ``--image``, it additionally compares the
full official BGR/OpenCV/Lab/512 pipeline byte for byte.

Usage::

    python weights/parity_ddcolor.py CHECKPOINT --size t \
        --upstream-dir /path/to/DDColor
    python weights/parity_ddcolor.py CHECKPOINT --size t \
        --upstream-dir /path/to/DDColor --image image.jpg
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from convert_ddcolor_weights import (
    _resolve_official_artifact,
    extract_ddcolor_state_dict,
)
from _conversion_utils import add_repo_root_to_path


UPSTREAM_COMMIT = "2adb63f2656ac41cbdf7b894cddd94121a3faf13"


def _assert_upstream_pin(upstream_dir: Path) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"DDColor upstream checkout must be {UPSTREAM_COMMIT}, got {commit}."
        )


def _build_upstream(size: str, upstream_dir: Path):
    _assert_upstream_pin(upstream_dir)
    sys.path.insert(0, str(upstream_dir))
    try:
        from ddcolor import DDColor as UpstreamDDColor
    finally:
        sys.path.pop(0)

    return UpstreamDDColor(
        encoder_name="convnext-t" if size == "t" else "convnext-l",
        decoder_name="MultiScaleColorDecoder",
        input_size=(32, 32),
        num_output_channels=2,
        last_norm="Spectral",
        do_normalize=False,
        num_queries=100,
        num_scales=3,
        dec_layers=9,
    )


def run_parity(
    checkpoint_path: str,
    *,
    size: str,
    upstream_dir: str,
    image_path: str | None = None,
) -> None:
    resolved_size, _ = _resolve_official_artifact(checkpoint_path, size)
    if resolved_size != size:
        raise RuntimeError(
            f"DDColor checkpoint resolved as size {resolved_size!r}, expected {size!r}."
        )
    add_repo_root_to_path()
    from libreyolo.models.ddcolor.nn import DDColor as LibreDDColorNetwork

    state_dict = extract_ddcolor_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    upstream_root = Path(upstream_dir).resolve()
    upstream = _build_upstream(size, upstream_root)
    ours = LibreDDColorNetwork(
        encoder_name="convnext-t" if size == "t" else "convnext-l",
        decoder_name="MultiScaleColorDecoder",
        input_size=(512, 512),
        num_output_channels=2,
        last_norm="Spectral",
        do_normalize=False,
        num_queries=100,
        num_scales=3,
        dec_layers=9,
    )

    upstream.load_state_dict(state_dict, strict=True)
    ours.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()
    upstream.eval()
    ours.eval()

    generator = torch.Generator().manual_seed(0)
    tensor = torch.rand((1, 3, 32, 32), generator=generator)
    with torch.inference_mode():
        upstream_output = upstream(tensor)
        our_output = ours(tensor)
    max_abs_diff = float((upstream_output - our_output).abs().max())
    if max_abs_diff != 0.0:
        raise AssertionError(f"DDColor network parity failed: {max_abs_diff=}")
    print(f"network parity: size={size}, max_abs_diff={max_abs_diff}")

    if image_path is None:
        return

    # Import after the source checkout is pinned and upstream modules are loaded.
    from ddcolor.pipeline import ColorizationPipeline
    from libreyolo.models.ddcolor.utils import (
        DDCOLOR_ORIGINAL_L_KEY,
        preprocess_image,
    )
    from libreyolo.postprocess.ddcolor import postprocess

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not decode image: {image_path}")
    official_bgr = ColorizationPipeline(upstream, input_size=512, device="cpu").process(
        image_bgr
    )
    tensor, _, original_size, metadata = preprocess_image(
        image_bgr,
        input_size=512,
        color_format="bgr",
    )
    with torch.inference_mode():
        output_ab = ours(tensor)
    our_rgb = postprocess(
        output_ab,
        original_size,
        original_l=metadata[DDCOLOR_ORIGINAL_L_KEY],
    )
    our_bgr = np.ascontiguousarray(our_rgb[..., ::-1])
    pixel_max_abs_diff = int(
        np.abs(official_bgr.astype(np.int16) - our_bgr.astype(np.int16)).max()
    )
    if pixel_max_abs_diff != 0:
        raise AssertionError(
            f"DDColor full-pipeline parity failed: {pixel_max_abs_diff=}"
        )
    print(f"full pipeline parity: pixel_max_abs_diff={pixel_max_abs_diff}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prove LibreDDColor parity")
    parser.add_argument("checkpoint", help="Official pytorch_model.bin")
    parser.add_argument("--size", choices=("t", "l"), required=True)
    parser.add_argument("--upstream-dir", required=True)
    parser.add_argument("--image", default=None)
    arguments = parser.parse_args()
    run_parity(
        arguments.checkpoint,
        size=arguments.size,
        upstream_dir=arguments.upstream_dir,
        image_path=arguments.image,
    )
