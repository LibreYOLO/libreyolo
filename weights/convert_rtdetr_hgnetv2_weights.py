"""Convert lyuwenyu RT-DETRv2 HGNetv2-L/X PyTorch weights to LibreYOLO format.

Source weights (Apache-2.0, lyuwenyu/RT-DETR upstream):
  https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_l_6x_coco_from_paddle.pth
  https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_x_6x_coco_from_paddle.pth

These checkpoints are themselves Paddle→PyTorch conversions of Baidu's
official RT-DETR HGNetv2-L/X COCO training run (53.0 / 54.8 AP).

Adaptation steps:
  1. Unwrap the EMA wrapper: ckpt["ema"]["module"] -> raw state_dict.
  2. Remap encoder/decoder input_proj and decoder.enc_output keys from
     v2's named-submodule style (.conv./.norm./.proj.) to LibreYOLO's
     Sequential numeric style (.0./.1.).
  3. Drop v2-only tensors that LibreYOLO's v1-style RT-DETR module does
     not have:
       - decoder.anchors                         (precomputed eval anchors)
       - decoder.valid_mask                      (precomputed eval mask)
       - decoder.decoder.layers.{i}.cross_attn.num_points_scale
                                                 (v2 distinct sampling pts)
  4. Save as a flat fp32 state_dict: weights/LibreRTDETR{l,x}.pt

Run from repo root:
    python weights/convert_rtdetr_hgnetv2_weights.py

Verify with:
    python -c "from libreyolo import LibreYOLO; \
               m = LibreYOLO('weights/LibreRTDETRl.pt'); \
               print(m.predict('pruebitas/images.jpeg'))"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    repo_root,
    save_checkpoint,
)


SOURCES = {
    "l": {
        "url": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_l_6x_coco_from_paddle.pth",
        "src": "downloads/weights/lyuwenyu_rtdetrv2_hgnetv2_l.pth",
        "dst": "weights/LibreRTDETRl.pt",
    },
    "x": {
        "url": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_x_6x_coco_from_paddle.pth",
        "src": "downloads/weights/lyuwenyu_rtdetrv2_hgnetv2_x.pth",
        "dst": "weights/LibreRTDETRx.pt",
    },
}

# v2-only tensor key fragments to drop entirely.
DROP_FRAGMENTS = (
    "decoder.anchors",
    "decoder.valid_mask",
    "cross_attn.num_points_scale",
)


def remap_key(k: str) -> str:
    """Remap v2 named-submodule keys to LibreYOLO Sequential-numeric keys.

    Two transforms:
      encoder.input_proj.{i}.conv.*  -> encoder.input_proj.{i}.0.*
      encoder.input_proj.{i}.norm.*  -> encoder.input_proj.{i}.1.*
      decoder.enc_output.proj.*      -> decoder.enc_output.0.*
      decoder.enc_output.norm.*      -> decoder.enc_output.1.*

    decoder.input_proj keeps its named-submodule form — LibreYOLO's
    decoder uses named submodules there.
    """
    if k.startswith("encoder.input_proj."):
        parts = k.split(".")
        if len(parts) >= 4:
            sub = parts[3]
            if sub == "conv":
                parts[3] = "0"
                return ".".join(parts)
            if sub == "norm":
                parts[3] = "1"
                return ".".join(parts)

    if k.startswith("decoder.enc_output."):
        parts = k.split(".")
        if len(parts) >= 3:
            sub = parts[2]
            if sub == "proj":
                parts[2] = "0"
                return ".".join(parts)
            if sub == "norm":
                parts[2] = "1"
                return ".".join(parts)

    return k


def convert(src_path: Path, dst_path: Path) -> dict:
    """Load lyuwenyu v2 ckpt, remap keys, save as LibreYOLO state_dict."""
    print(f"Loading {src_path} ...")
    sd_v2 = extract_state_dict(load_checkpoint(src_path))

    out = {}
    dropped = []
    for k, v in sd_v2.items():
        if any(frag in k for frag in DROP_FRAGMENTS):
            dropped.append(k)
            continue
        out[remap_key(k)] = v.float().clone()  # fp32, detached copy

    print(f"  source tensors: {len(sd_v2)}")
    print(f"  dropped (v2-only): {len(dropped)}")
    print(f"  output tensors: {len(out)}")
    for k in dropped:
        print(f"    drop: {k}")

    save_checkpoint(out, dst_path)
    print(f"  saved -> {dst_path}")
    return out


def verify(state_dict: dict, size: str) -> None:
    """Round-trip: load state_dict into a LibreYOLO RTDETR model strict=True."""
    add_repo_root_to_path()
    from libreyolo.models.rtdetr.model import RTDETR_CONFIGS, LibreYOLORTDETR

    RTDETR_CONFIGS[size]["backbone_pretrained"] = False
    model = LibreYOLORTDETR(nb_classes=80, size=size, device="cpu")
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(
            f"[{size}] missing keys: {missing[:5]}{' ...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        raise RuntimeError(
            f"[{size}] unexpected keys: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}"
        )
    print(f"  [{size}] strict load OK, no missing or unexpected keys")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sizes",
        nargs="*",
        default=list(SOURCES.keys()),
        help="Sizes to convert (default: all).",
    )
    args = parser.parse_args()

    for size in args.sizes:
        if size not in SOURCES:
            raise SystemExit(f"unknown size {size!r}; valid: {list(SOURCES)}")
        cfg = SOURCES[size]
        src = repo_root() / cfg["src"]
        dst = repo_root() / cfg["dst"]
        if not src.exists():
            raise SystemExit(f"missing source {src}; download from {cfg['url']} first")
        print(f"\n=== Converting RT-DETR HGNetv2-{size.upper()} ===")
        sd = convert(src, dst)
        verify(sd, size)


if __name__ == "__main__":
    main()
