"""Convert an official DDRNet Cityscapes checkpoint to a LibreYOLO checkpoint.

Usage:
    python weights/convert_ddrnet_weights.py \
        /path/to/best_val_smaller.pth weights/LibreDDRNets-sem.pt

Upstream (https://github.com/ydhongHIT/DDRNet, MIT) links two Cityscapes
checkpoints from its README: DDRNet-23-slim (size ``s``) and DDRNet-23 (size
``m``). They are identified by SHA-256, not by filename, and the conversion is
a key rename: the ``model.`` training-wrapper prefix is stripped and the
``loss.criterion.weight`` buffer is dropped. Learned parameters are unchanged.

Any other DDRNet segmentation checkpoint in the upstream layout (a CamVid
model, a fine-tune) converts the same way with ``--allow-unofficial``; it then
carries no weight-license metadata, because a tensor layout does not establish
one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from _conversion_utils import (
    add_repo_root_to_path,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


def convert(input_path: str, output_path: str, *, allow_unofficial: bool = False) -> None:
    add_repo_root_to_path()
    from libreyolo.models.ddrnet.convert import (
        CITYSCAPES_SOURCES,
        UPSTREAM_COMMIT,
        UPSTREAM_REPO,
        convert_upstream_ddrnet_state_dict,
        official_cityscapes_size,
        sha256_file,
    )
    from libreyolo.models.ddrnet.model import (
        CITYSCAPES_NAMES,
        WEIGHT_LICENSE,
        WEIGHT_LICENSE_URL,
        LibreDDRNet,
    )
    from libreyolo.models.ddrnet.nn import SIZE_CONFIGS
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    official_size = official_cityscapes_size(input_path)
    if official_size is None and not allow_unofficial:
        known = ", ".join(f"{s}: {src['sha256']}" for s, src in CITYSCAPES_SOURCES.items())
        raise SystemExit(
            f"{input_path} is not an official DDRNet Cityscapes checkpoint "
            f"(known SHA-256: {known}). Pass --allow-unofficial to convert it "
            "without weight-license metadata."
        )

    # The official files are plain tensor dicts; never unpickle arbitrary code.
    raw = torch.load(input_path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and isinstance(raw.get("state_dict"), dict):
        raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise SystemExit(f"{input_path} does not hold a state dict.")
    state_dict = convert_upstream_ddrnet_state_dict(raw)
    if state_dict is None:
        raise SystemExit(f"{input_path} is not a DDRNet segmentation checkpoint.")
    print(f"Extracted {len(state_dict)} tensors from {input_path}")

    size = LibreDDRNet.detect_size(state_dict)
    nc = LibreDDRNet.detect_nb_classes(state_dict)
    if size is None or nc is None:
        raise SystemExit(f"Could not read size/classes from tensors (size={size}, nc={nc}).")
    if official_size is not None and size != official_size:
        raise SystemExit(f"Tensor evidence says size={size!r}; the digest says {official_size!r}.")

    model = LibreDDRNet(size=size, nb_classes=nc, device="cpu")
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"Strict-equivalent load failed: missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    print(f"Strict load OK: size={size} nc={nc}")

    imgsz_h, imgsz_w = SIZE_CONFIGS[size]["imgsz"]
    train_h, train_w = SIZE_CONFIGS[size]["train_crop"]
    metadata = dict(
        imgsz=max(imgsz_h, imgsz_w),
        imgsz_h=imgsz_h,
        imgsz_w=imgsz_w,
        train_imgsz_h=train_h,
        train_imgsz_w=train_w,
        normalization="imagenet",
        resize_mode="stretch",
        ignore_index=255,
        source_sha256=sha256_file(input_path),
        source_revision=f"{UPSTREAM_REPO.removeprefix('https://github.com/')}@{UPSTREAM_COMMIT}",
        architecture_licenses="MIT (ydhongHIT/DDRNet)",
    )
    if official_size is not None:
        metadata.update(
            source_url=CITYSCAPES_SOURCES[size]["url"],
            weight_license=WEIGHT_LICENSE,
            weight_license_url=WEIGHT_LICENSE_URL,
            weight_dataset="Cityscapes",
        )
    names = dict(CITYSCAPES_NAMES) if nc == len(CITYSCAPES_NAMES) else None

    wrapped = wrap_libreyolo_checkpoint(
        {key: value.float().cpu() if value.is_floating_point() else value.cpu() for key, value in state_dict.items()},
        model_family="ddrnet",
        size=size,
        nc=nc,
        names=names,
        task="semantic",
        supported_tasks=("semantic",),
        default_task="semantic",
        **metadata,
    )
    validate_checkpoint_metadata(wrapped)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.replace(out)
    print(f"Wrote {out} (size={size}, nc={nc}, imgsz={imgsz_h}x{imgsz_w})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument(
        "--allow-unofficial",
        action="store_true",
        help="convert a non-official DDRNet checkpoint without weight-license metadata",
    )
    args = parser.parse_args()
    convert(args.input, args.output, allow_unofficial=args.allow_unofficial)
