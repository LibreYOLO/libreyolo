"""Convert the official mmseg UNet-S5-D16 Cityscapes checkpoint.

Usage:
    python weights/convert_unet_weights.py \
        /tmp/unet-mmseg/fcn_unet_cityscapes.pth weights/LibreUNets-sem.pt

The released checkpoint is trained on Cityscapes and is NON-COMMERCIAL; the
conversion stamps that into the checkpoint metadata so it travels with the
file (see libreyolo/models/unet/NOTICE).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

add_repo_root_to_path()
from libreyolo.models.unet.convert import SOURCE_DIGEST
from libreyolo.models.unet.convert import checkpoint_sha256 as sha256

SOURCE_URL = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/unet/"
    "fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes/"
    "fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth"
)
SOURCE_REVISION = "open-mmlab/mmsegmentation@b040e147adfa"
CITYSCAPES_LICENSE_URL = "https://www.cityscapes-dataset.com/license/"


def convert(input_path: str, output_path: str) -> None:
    add_repo_root_to_path()
    from libreyolo.models.unet.convert import convert_upstream_unet_state_dict
    from libreyolo.models.unet.model import CITYSCAPES_NAMES, WEIGHT_LICENSE, LibreUNet
    from libreyolo.models.unet.nn import SIZE_CONFIGS

    digest = sha256(input_path)
    if digest != SOURCE_DIGEST:
        raise SystemExit(
            f"Digest mismatch for {input_path}: got {digest}, expected {SOURCE_DIGEST}. "
            "Refusing to deserialize an unverified checkpoint."
        )

    raw = load_checkpoint(input_path)
    if not isinstance(raw, dict) or "state_dict" not in raw:
        raise SystemExit(
            f"{input_path} has no top-level 'state_dict' key; not an mmseg artifact."
        )
    source_state = raw["state_dict"]
    if not isinstance(source_state, dict):
        raise SystemExit("Upstream 'state_dict' entry is not a mapping.")

    state_dict = convert_upstream_unet_state_dict(source_state)
    if state_dict is None:
        raise SystemExit(f"{input_path} is not the mmseg UNet-S5-D16 + FCN layout.")
    print(f"Extracted {len(state_dict)} tensors from {input_path}")

    nc = LibreUNet.detect_nb_classes(state_dict)
    if nc is None:
        raise SystemExit("Could not read the class count from the segmentation heads.")
    size = LibreUNet.detect_size(state_dict)
    if size != "s":
        raise SystemExit(f"Tensor evidence says size={size!r}; expected 's'.")

    model = LibreUNet(size=size, nb_classes=nc, device="cpu")
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise SystemExit(
            "Strict-equivalent load failed: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    model.model.load_state_dict(state_dict, strict=True)
    print(f"Strict load OK: size={size} nc={nc}")

    imgsz_h, imgsz_w = SIZE_CONFIGS[size]["imgsz"]
    train_h, train_w = SIZE_CONFIGS[size]["train_crop"]
    names = dict(CITYSCAPES_NAMES) if nc == len(CITYSCAPES_NAMES) else None

    wrapped = wrap_libreyolo_checkpoint(
        {key: value.cpu() for key, value in state_dict.items()},
        model_family="unet",
        size=size,
        nc=nc,
        names=names,
        task="semantic",
        supported_tasks=("semantic",),
        default_task="semantic",
        imgsz=max(imgsz_h, imgsz_w),
        imgsz_h=imgsz_h,
        imgsz_w=imgsz_w,
        train_imgsz_h=train_h,
        train_imgsz_w=train_w,
        normalization="imagenet",
        resize_mode="stretch",
        ignore_index=255,
        weight_license=WEIGHT_LICENSE,
        weight_license_url=CITYSCAPES_LICENSE_URL,
        weight_dataset="Cityscapes",
        weight_commercial_use=False,
        source_url=SOURCE_URL,
        source_sha256=digest,
        source_revision=SOURCE_REVISION,
        architecture_licenses="Apache-2.0 (mmsegmentation)",
    )

    out = Path(output_path)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.replace(out)
    print(f"Wrote {out} (size={size}, nc={nc}, imgsz={imgsz_h}x{imgsz_w})")
    print(
        "NOTE: this checkpoint derives from Cityscapes and is restricted to "
        f"NON-COMMERCIAL use ({CITYSCAPES_LICENSE_URL})."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    convert(args.input, args.output)
