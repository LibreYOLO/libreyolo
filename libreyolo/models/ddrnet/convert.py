"""Recognize, unwrap and identify official DDRNet checkpoints.

Shared by the runtime auto-converter, ``LibreDDRNet`` and
``weights/convert_ddrnet_weights.py``. The official segmentation checkpoints
are flat dicts of ``model.<key>`` tensors (the training wrapper) plus a
``loss.criterion.weight`` class-weight buffer; native keys are the upstream
keys with that prefix stripped, so conversion is a rename, never a reshape.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

UPSTREAM_REPO = "https://github.com/ydhongHIT/DDRNet"
UPSTREAM_COMMIT = "de0db317c5af4b0946231f475cf74ca4f51b8ac2"
UPSTREAM_LICENSE_URL = f"{UPSTREAM_REPO}/blob/{UPSTREAM_COMMIT}/LICENSE"

# Official Cityscapes checkpoints linked from the upstream README (Google
# Drive), identified by SHA-256 so their terms never depend on a filename.
CITYSCAPES_SOURCES: dict[str, dict[str, str]] = {
    "s": {
        "name": "DDRNet_23_slim Cityscapes (val mIoU 77.8)",
        "url": "https://drive.google.com/file/d/1d_K3Af5fKHYwxSo8HkxpnhiekhwovmiP",
        "sha256": "688dd42edbd623d21a6fe67c4b90ef8a5913d2824214a49bb60d94c61e6544b4",
    },
    "m": {
        "name": "DDRNet_23 Cityscapes (val mIoU 79.5)",
        "url": "https://drive.google.com/file/d/16viDZhbmuc3y7OSsUo2vhA7V6kYO0KX6",
        "sha256": "66b68a4e61b2ab41fba5e967acd708ee3d9dd4771fb1e1a245a9a6b5bbd1db4e",
    },
}

# Official ImageNet-1K classification checkpoints the segmentation recipe
# initializes from, mirrored byte-for-byte on the LibreYOLO Hugging Face org
# under their upstream filenames.
IMAGENET_BACKBONES: dict[str, dict[str, str]] = {
    "s": {
        "filename": "DDRNet23s_imagenet.pth",
        "repo": "LibreYOLO/LibreDDRNets-imagenet",
        "url": "https://drive.google.com/file/d/1mg5tMX7TJ9ZVcAiGSB4PEihPtrJyalB4",
        "sha256": "f6d03fa29b80e2e648f89a7639eb0c6d6c088e8f70f2cba1b2872b314a5c54b6",
    },
    "m": {
        "filename": "DDRNet23_imagenet.pth",
        "repo": "LibreYOLO/LibreDDRNetm-imagenet",
        "url": "https://drive.google.com/file/d/1VoUsERBeuCaiuQJufu8PqpKKtGvCTdug",
        "sha256": "571959df93dce2db4b57d788362eb049a941e21b1b076a57112f8256216d1c08",
    },
    "l": {
        "filename": "DDRNet39_imagenet.pth",
        "repo": "LibreYOLO/LibreDDRNetl-imagenet",
        "url": "https://drive.google.com/file/d/122CMx6DZBaRRf-dOHYwuDY9vG0_UQ10i",
        "sha256": "b97be13428534508791e3897f7921d58d4886142bea25a8fa2f582acb26c25fc",
    },
}

# The ImageNet classifiers carry their own head (``down5``, ``last_layer``,
# ``linear``); the segmentation network adds DAPPM and both segmentation heads,
# which upstream trains from random initialization.
IMAGENET_CLASSIFIER_ONLY_PREFIXES = ("down5.", "last_layer.", "linear.")
IMAGENET_SEGMENTATION_ONLY_PREFIXES = ("spp.", "final_layer.", "seghead_extra.")

_WRAPPER_PREFIXES = ("module.", "model.")
_DROP_PREFIXES = ("loss.",)
# PIDNet shares DDRNet's stem, residual stages, DAPPM and head names; these
# modules exist only in PIDNet and must veto a DDRNet claim.
_PIDNET_ONLY_PREFIXES = ("pag3.", "pag4.", "diff3.", "diff4.", "dfm.", "layer3_d.", "layer4_d.", "layer5_d.")
_REQUIRED_KEYS = (
    "conv1.0.weight",
    "compression4.0.weight",
    "down4.3.weight",
    "layer5_.0.conv3.weight",
    "layer5.0.conv3.weight",
    "spp.scale0.2.weight",
    "spp.process4.2.weight",
    "final_layer.conv2.weight",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_wrapper_prefixes(state_dict: dict) -> dict:
    """Drop ``module.`` / ``model.`` wrappers and the upstream loss buffers."""
    out = {}
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in _WRAPPER_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        if key.startswith(_DROP_PREFIXES):
            continue
        out[key] = value
    return out


def is_ddrnet_state_dict(state_dict: dict) -> bool:
    """True only for the DDRNet segmentation layout (any size, any head width)."""
    keys = set(strip_wrapper_prefixes(state_dict))
    if not all(key in keys for key in _REQUIRED_KEYS):
        return False
    has_stage3 = ("down3.0.weight" in keys and "layer3_.0.conv1.weight" in keys) or (
        "down3_1.0.weight" in keys and "layer3_2_.0.conv1.weight" in keys
    )
    if not has_stage3:
        return False
    return not any(key.startswith(_PIDNET_ONLY_PREFIXES) for key in keys)


def convert_upstream(state_dict: dict) -> dict[str, torch.Tensor]:
    """Native keys are upstream's minus the training wrapper: rename only."""
    return {
        key: value
        for key, value in strip_wrapper_prefixes(state_dict).items()
        if torch.is_tensor(value)
    }


def convert_upstream_ddrnet_state_dict(state_dict: dict) -> dict | None:
    if not is_ddrnet_state_dict(state_dict):
        return None
    return convert_upstream(state_dict)


def official_cityscapes_size(path: str | Path) -> str | None:
    """Return the size whose official Cityscapes artifact has this SHA-256."""
    digest = sha256_file(path)
    for size, source in CITYSCAPES_SOURCES.items():
        if source["sha256"] == digest:
            return size
    return None


def imagenet_backbone_state(state_dict: dict, target_keys: set[str]) -> dict[str, torch.Tensor]:
    """Return the ImageNet tensors the segmentation network can take.

    Raises if the file is not the matching DDRNet ImageNet classifier: the
    keys the two networks do not share must be exactly the classifier head on
    one side and DAPPM plus the segmentation heads on the other, so a wrong
    size or a foreign checkpoint fails loudly instead of half-loading.
    """
    source = {
        key: value
        for key, value in strip_wrapper_prefixes(state_dict).items()
        if torch.is_tensor(value)
    }
    unexpected = sorted(set(source) - target_keys)
    missing = sorted(target_keys - set(source))
    bad_unexpected = [k for k in unexpected if not k.startswith(IMAGENET_CLASSIFIER_ONLY_PREFIXES)]
    bad_missing = [k for k in missing if not k.startswith(IMAGENET_SEGMENTATION_ONLY_PREFIXES)]
    if bad_unexpected or bad_missing:
        raise RuntimeError(
            "Not a DDRNet ImageNet backbone for this size: "
            f"unexpected={bad_unexpected[:5]} missing={bad_missing[:5]}"
        )
    return {key: value for key, value in source.items() if key in target_keys}


__all__ = [
    "CITYSCAPES_SOURCES",
    "IMAGENET_BACKBONES",
    "UPSTREAM_COMMIT",
    "UPSTREAM_LICENSE_URL",
    "UPSTREAM_REPO",
    "convert_upstream",
    "convert_upstream_ddrnet_state_dict",
    "imagenet_backbone_state",
    "is_ddrnet_state_dict",
    "official_cityscapes_size",
    "sha256_file",
    "strip_wrapper_prefixes",
]
