"""Depth Pro 2 official-checkpoint identification, checksum pinning, and conversion."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

from ...utils.download import download_url_to_path
from ...utils.hf_hub import canonical_hf_url
from ...utils.serialization import (
    load_untrusted_torch_file,
    validate_checkpoint_metadata,
    wrap_libreyolo_checkpoint,
)

INPUT_SIZES: dict[str, int] = {"s": 512, "b": 512, "l": 512, "h": 512}
EMBED_DIM_TO_SIZE: dict[int, str] = {384: "s", 768: "b", 1024: "l", 1280: "h"}

# Converted checkpoints mirrored on the LibreYOLO Hugging Face org
# (LibreYOLO/LibreDepthPro2{s,b,l,h}-depth), bundling DepthPro2-focal under
# fov.* with LibreYOLO v1.0 metadata under apple-amlr (non-commercial research).
MIRROR_SHA256: dict[str, str] = {
    "s": "79118a3c8d0a4aeccf50ef7e57e9544d028db97a788801b4337953131302f426",
    "b": "d230b836a71ca90741c2191c6463b53c84afdc77e8a75addf2f935de2fc6155d",
    "l": "7a39f1204de2d0e8461f3a44dbcd2afa4081864bd0f1e5f84b20b3c956157c27",
    "h": "c5723f7d471d713a73c30e50c9d54c49d1bf86640af75fa4dfe6ee1bb02a0dea",
}

# Official Apple Hugging Face checkpoints pinned to exact git commits and LFS SHA-256 digests.
UPSTREAM_URLS: dict[str, str] = {
    "s": (
        "https://huggingface.co/apple/DepthPro2-S/resolve/"
        "903a1c1ade918831d138adcc4aedbcc8f4da0078/model.safetensors"
    ),
    "b": (
        "https://huggingface.co/apple/DepthPro2-B/resolve/"
        "85e67fdfb7f5b36a6f6c4495e6789d331b1d8a04/model.safetensors"
    ),
    "l": (
        "https://huggingface.co/apple/DepthPro2-L/resolve/"
        "849dc5461267e0bc006ec63ab46920ea1ed98702/model.safetensors"
    ),
    "h": (
        "https://huggingface.co/apple/DepthPro2-H/resolve/"
        "68426c789f75a7c760273e857222625b4fe69576/model.safetensors"
    ),
}

UPSTREAM_SHA256: dict[str, str] = {
    "s": "583c4d630779fd231f969c738733624963ad34bd3c29f9c64088257e3436f97c",
    "b": "244ff85e0d9bde6e0f985de1c61420abc4c2110265b495c974556225ac2b60ed",
    "l": "d799d8a91f2e8dc91715f9b33a405a1ff0727212041d206ebbb04e6c1424ff9f",
    "h": "cfc532336019abd2128b04be2c2967175522e46de69222ef72b87c6675ee3b17",
}

FOV_UPSTREAM_URL = (
    "https://huggingface.co/apple/DepthPro2-focal/resolve/"
    "622659d44a2002cf9ae4333f2569afa938290d9a/model.safetensors"
)
FOV_UPSTREAM_SHA256 = (
    "8f567d1c97ff916137288668f6bd306ce2094da2e2b852b0e4539dbb139c8c86"
)
FOV_CACHE_FILENAME = "LibreDepthPro2-focal.safetensors"


def file_sha256(path: str | Path) -> str:
    """Compute hex SHA-256 of a file on disk."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_depthpro2_state_dict(state_dict: dict) -> bool:
    """Return True only for Depth Pro 2 native-keyed checkpoints (with or without ``fov.*``)."""
    required = (
        "encoder.patch_embed.proj.weight",
        "encoder.cls_token",
        "encoder.reg_token",
        "encoder.reassemble.0.0.weight",
        "decoder.fusions.0.deconv.weight",
        "head.1.weight",
    )
    return all(k in state_dict for k in required)


def detect_size(state_dict: dict) -> str | None:
    """Infer size code (``s``, ``b``, ``l``, ``h``) from ``encoder.cls_token`` width."""
    cls_token = state_dict.get("encoder.cls_token")
    if cls_token is None or getattr(cls_token, "ndim", 0) < 1:
        return None
    return EMBED_DIM_TO_SIZE.get(int(cls_token.shape[-1]))


def has_fov_weights(state_dict: dict) -> bool:
    """Return True when a state dict bundles the ``FOVEstimator`` weights under ``fov.*``."""
    return "fov.head.out.weight" in state_dict and "fov.trunk.cls_token" in state_dict


def load_raw_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load an upstream ``.safetensors`` or ``.pt`` checkpoint to CPU tensors."""
    file_path = Path(path)
    if file_path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(file_path), device="cpu")
    loaded = load_untrusted_torch_file(
        str(file_path),
        map_location="cpu",
        context="upstream Depth Pro 2 checkpoint",
    )
    if isinstance(loaded, dict) and isinstance(loaded.get("model"), dict):
        loaded = loaded["model"]
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a state dict in {file_path}, got {type(loaded).__name__}.")
    return {k: v for k, v in loaded.items() if isinstance(v, torch.Tensor)}


def merge_fov_state_dict(
    depth_state_dict: dict[str, torch.Tensor],
    fov_state_dict: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Combine a Depth Pro 2 depth state dict with its companion focal estimator weights."""
    merged = dict(depth_state_dict)
    if fov_state_dict is None:
        return merged
    for key, value in fov_state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        target_key = key if key.startswith("fov.") else f"fov.{key}"
        merged[target_key] = value
    return merged


def wrap_upstream_state_dict(state_dict: dict, size: str) -> dict:
    """Add strict LibreYOLO v1.0 depth metadata without modifying learned tensors."""
    if not is_depthpro2_state_dict(state_dict):
        raise ValueError("Checkpoint is not a supported Depth Pro 2 state dict.")
    detected = detect_size(state_dict)
    if detected != size:
        raise ValueError(
            f"Depth Pro 2 checkpoint signature is {detected!r}, expected {size!r}."
        )
    return wrap_libreyolo_checkpoint(
        dict(state_dict),
        model_family="depthpro2",
        size=size,
        task="depth",
        nc=1,
        names={0: "depth"},
        imgsz=INPUT_SIZES[size],
    )


def _verify_fov_file(local_path: str, source_url: str) -> None:
    if canonical_hf_url(source_url) != FOV_UPSTREAM_URL:
        raise ValueError(f"Unrecognized Depth Pro 2 focal URL: {source_url}")
    actual = file_sha256(local_path)
    if actual != FOV_UPSTREAM_SHA256:
        raise RuntimeError(
            "Checksum mismatch for Depth Pro 2 focal checkpoint: "
            f"expected {FOV_UPSTREAM_SHA256}, got {actual}."
        )


def _ensure_fov_state_dict(cache_dir: Path) -> dict[str, torch.Tensor]:
    """Fetch (once) and SHA-256-verify Apple's official DepthPro2-focal checkpoint."""
    focal_path = cache_dir / FOV_CACHE_FILENAME
    if focal_path.exists():
        actual = file_sha256(focal_path)
        if actual != FOV_UPSTREAM_SHA256:
            focal_path.unlink(missing_ok=True)
    if not focal_path.exists():
        download_url_to_path(
            FOV_UPSTREAM_URL,
            focal_path,
            verify=_verify_fov_file,
        )
    from safetensors.torch import load_file

    return load_file(str(focal_path), device="cpu")


def verify_and_wrap_download(local_path: str, source_url: str) -> None:
    """Verify SHA-256 of a LibreYOLO mirror or upstream Apple Depth Pro 2 download."""
    canonical = canonical_hf_url(source_url)

    if canonical.startswith("https://huggingface.co/LibreYOLO/"):
        size = next(
            (
                s
                for s in MIRROR_SHA256
                if f"/LibreDepthPro2{s}-depth/" in canonical
            ),
            None,
        )
        if size is None:
            raise RuntimeError(
                f"Refusing unpinned Depth Pro 2 mirror download {source_url!r}; "
                "no SHA-256 is known."
            )
        expected = MIRROR_SHA256[size]
        actual = file_sha256(local_path)
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch for LibreDepthPro2{size}-depth.pt: "
                f"expected {expected}, got {actual}."
            )
        return

    size = next(
        (s for s, url in UPSTREAM_URLS.items() if url == canonical),
        None,
    )
    if size is None:
        raise RuntimeError(
            f"Refusing unpinned Depth Pro 2 download {source_url!r}; no SHA-256 is known."
        )

    expected = UPSTREAM_SHA256[size]
    actual = file_sha256(local_path)
    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch for Depth Pro 2 {size} checkpoint: "
            f"expected {expected}, got {actual}."
        )

    from safetensors.torch import load_file

    depth_sd = load_file(local_path, device="cpu")
    detected = detect_size(depth_sd)
    if detected != size:
        raise RuntimeError(
            f"Depth Pro 2 checkpoint for {size!r} has unexpected architecture {detected!r}."
        )

    path = Path(local_path)
    fov_sd = _ensure_fov_state_dict(path.parent)
    merged = merge_fov_state_dict(depth_sd, fov_sd)
    wrapped = wrap_upstream_state_dict(merged, size)
    errors = validate_checkpoint_metadata(wrapped, strict=True)
    if errors:
        raise RuntimeError(
            "Invalid converted Depth Pro 2 metadata: " + "; ".join(errors)
        )

    temporary = path.with_suffix(path.suffix + ".wrapped")
    torch.save(wrapped, temporary)
    os.replace(temporary, path)


__all__ = [
    "EMBED_DIM_TO_SIZE",
    "FOV_CACHE_FILENAME",
    "FOV_UPSTREAM_SHA256",
    "FOV_UPSTREAM_URL",
    "INPUT_SIZES",
    "MIRROR_SHA256",
    "UPSTREAM_SHA256",
    "UPSTREAM_URLS",
    "detect_size",
    "file_sha256",
    "has_fov_weights",
    "is_depthpro2_state_dict",
    "load_raw_state_dict",
    "merge_fov_state_dict",
    "verify_and_wrap_download",
    "wrap_upstream_state_dict",
]
