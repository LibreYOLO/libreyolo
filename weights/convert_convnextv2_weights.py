"""Wrap official 224px ImageNet-1K ConvNeXt V2 classifiers without tensor changes.

Usage: python weights/convert_convnextv2_weights.py input.pt output.pt
Official weights are CC-BY-NC-4.0; the model code is MIT. FCMAE encoder-only
and other-resolution checkpoints are outside this converter's contract.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _conversion_utils import add_repo_root_to_path, extract_state_dict, imagenet1k_names, save_checkpoint

SOURCE_COMMIT = "2553895753323c6fe0b2bf390683f5ea358a42b9"
SOURCE_REPO = "https://github.com/facebookresearch/ConvNeXt-V2"
VARIANTS = {"atto": "atto", "femto": "femto", "pico": "pico", "n": "nano",
            "t": "tiny", "b": "base", "l": "large", "h": "huge"}


def convert(input_path: str | Path, output_path: str | Path) -> Path:
    add_repo_root_to_path()
    import torch
    from libreyolo.models.convnextv2.model import LibreConvNeXtV2
    from libreyolo.models.convnextv2.nn import ConvNeXtV2
    from libreyolo.utils.serialization import (
        load_untrusted_torch_file, validate_checkpoint_metadata, wrap_libreyolo_checkpoint,
    )

    input_path, output_path = Path(input_path), Path(output_path)
    state = extract_state_dict(load_untrusted_torch_file(input_path))
    if not LibreConvNeXtV2.can_load(state):
        raise ValueError("Expected a supported dense ConvNeXt V2 classifier checkpoint")
    size = LibreConvNeXtV2.detect_size(state)
    nc = LibreConvNeXtV2.detect_nb_classes(state)
    if nc != 1000:
        raise ValueError("This official ImageNet-1K converter requires a 1000-class head")
    with torch.device("meta"):
        model = ConvNeXtV2(size, nc)
    model.load_state_dict(state, strict=True, assign=True)
    digest = hashlib.sha256()
    with input_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    checkpoint = wrap_libreyolo_checkpoint(
        state, model_family="convnextv2", size=size, nc=nc,
        names=imagenet1k_names(), task="classify", imgsz=224,
        crop_pct=0.875, interpolation="bicubic",
        source=SOURCE_REPO, source_commit=SOURCE_COMMIT,
        source_sha256=digest.hexdigest(),
        weight_license="cc-by-nc-4.0",
        weight_license_url="https://creativecommons.org/licenses/by-nc/4.0/",
        weight_commercial_use=False, weight_dataset="ImageNet-1K",
    )
    validate_checkpoint_metadata(checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    save_checkpoint(checkpoint, temporary)
    validate_checkpoint_metadata(load_untrusted_torch_file(temporary))
    temporary.replace(output_path)
    print(f"Wrote {output_path} ({size}, {nc} classes; tensors unchanged)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.input, args.output)
