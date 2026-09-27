"""Exact external parity for the pinned ViTMatte-S checkpoint.

This script constructs the reference from an exact pinned Transformers source
checkout, strictly loads the audited safetensors into both graphs, and requires
bit-exact CPU equality for preprocessing and raw alpha prediction.

Usage::

    python weights/parity_vitmatte.py path/to/model.safetensors \
        --upstream-dir path/to/transformers
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from _conversion_utils import add_repo_root_to_path
from convert_vitmatte_weights import verify_source_checkpoint


TRANSFORMERS_UPSTREAM_COMMIT = "7d6354e04794f3246bf9a0faf4fead080edeebb6"

CONFIG = {
    "model_type": "vitmatte",
    "hidden_size": 384,
    "batch_norm_eps": 1e-5,
    "convstream_hidden_sizes": [48, 96, 192],
    "fusion_hidden_sizes": [256, 128, 64, 32],
    "backbone_config": {
        "model_type": "vitdet",
        "hidden_size": 384,
        "image_size": 512,
        "num_attention_heads": 6,
        "num_channels": 4,
        "out_features": ["stage12"],
        "out_indices": [12],
        "residual_block_indices": [2, 5, 8, 11],
        "use_relative_position_embeddings": True,
        "window_block_indices": [0, 1, 3, 4, 6, 7, 9, 10],
        "window_size": 14,
    },
}


def _assert_upstream_pin(upstream_dir: Path) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != TRANSFORMERS_UPSTREAM_COMMIT:
        raise RuntimeError(
            "Transformers upstream checkout must be "
            f"{TRANSFORMERS_UPSTREAM_COMMIT}, got {commit}."
        )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _load_pinned_transformers(upstream_dir: str):
    upstream_root = Path(upstream_dir).resolve()
    _assert_upstream_pin(upstream_root)
    source_root = upstream_root / "src"
    package_root = source_root / "transformers"
    if not (package_root / "__init__.py").is_file():
        raise FileNotFoundError(
            f"Pinned Transformers package not found under {source_root}."
        )

    loaded = sys.modules.get("transformers")
    if loaded is None:
        sys.path.insert(0, str(source_root))
        try:
            loaded = importlib.import_module("transformers")
        finally:
            sys.path.pop(0)

    module_file = getattr(loaded, "__file__", None)
    if module_file is None or not _is_within(Path(module_file).resolve(), package_root):
        raise RuntimeError(
            "Transformers reference was not imported from the pinned upstream "
            f"checkout: {module_file!r}."
        )
    return (
        loaded.VitMatteConfig,
        loaded.VitMatteForImageMatting,
        loaded.VitMatteImageProcessor,
    )


def _require_exact(name: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    if torch.equal(expected, actual):
        print(f"{name}: exact ({tuple(actual.shape)})")
        return
    difference = (expected - actual).abs()
    raise AssertionError(
        f"{name} parity failed: max_abs={difference.max().item():.9g}, "
        f"mean_abs={difference.mean().item():.9g}."
    )


def run_parity(source_path: str, *, upstream_dir: str) -> None:
    verify_source_checkpoint(source_path)
    from safetensors.torch import load_file

    VitMatteConfig, VitMatteForImageMatting, VitMatteImageProcessor = (
        _load_pinned_transformers(upstream_dir)
    )

    add_repo_root_to_path()
    from libreyolo.models.vitmatte.nn import (
        LibreViTMatteModel,
        constrain_alpha_to_trimap,
    )
    from libreyolo.models.vitmatte.utils import preprocess_guided_image

    state_dict = load_file(source_path, device="cpu")
    config = VitMatteConfig.from_dict(CONFIG)
    reference = VitMatteForImageMatting(config).eval()
    reference.load_state_dict(state_dict, strict=True)
    candidate = LibreViTMatteModel().eval()
    candidate.load_state_dict(state_dict, strict=True)

    rng = np.random.default_rng(17)
    rgb_array = rng.integers(0, 256, (61, 93, 3), dtype=np.uint8)
    trimap_array = np.full((61, 93), 128, dtype=np.uint8)
    trimap_array[:, :19] = 0
    trimap_array[:, 71:] = 255
    rgb = Image.fromarray(rgb_array, mode="RGB")
    trimap = Image.fromarray(trimap_array, mode="L")

    processor = VitMatteImageProcessor()
    reference_pixels = processor(
        images=rgb,
        trimaps=trimap,
        return_tensors="pt",
    ).pixel_values
    candidate_pixels, _, original_size, _ = preprocess_guided_image(rgb, trimap)
    if original_size != rgb.size:
        raise AssertionError(
            f"Candidate changed original_size: {original_size} != {rgb.size}."
        )
    _require_exact("preprocess", reference_pixels, candidate_pixels)

    with torch.inference_mode():
        reference_alpha = reference(pixel_values=reference_pixels).alphas
        candidate_raw = candidate.forward_unconstrained(candidate_pixels)
        candidate_alpha = candidate(candidate_pixels)
    _require_exact("raw alpha", reference_alpha, candidate_raw)
    _require_exact(
        "known-region trimap constraint",
        constrain_alpha_to_trimap(reference_alpha, reference_pixels),
        candidate_alpha,
    )
    print("ViTMatte external exact parity passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exact ViTMatte parity")
    parser.add_argument("source", help="Pinned upstream model.safetensors")
    parser.add_argument(
        "--upstream-dir",
        required=True,
        help=(
            "Transformers checkout at "
            f"{TRANSFORMERS_UPSTREAM_COMMIT} used as the external reference"
        ),
    )
    arguments = parser.parse_args()
    run_parity(arguments.source, upstream_dir=arguments.upstream_dir)
