"""Convert Bo396543018/Picodet_Pytorch checkpoints to LibreYOLO format.

With ``--hf-bundle <dir>`` the script also emits the 5-file repo layout
expected by the ``libreyolo-upload-hf-model`` skill (README, LICENSE,
NOTICE, .gitattributes, and the canonical ``LibrePicoDet<size>.pt``).
After conversion::

    huggingface-cli upload LibreYOLO/LibrePicoDet<size> <hf-bundle-dir> .

Per-size repos: ``LibrePicoDets``, ``LibrePicoDetm``, ``LibrePicoDetl``.

Bo's checkpoints carry mmdet-style key naming because his ``ESNet`` /
``CSPPAN`` / ``PicoDetHead`` are wrapped in mmcv's ``ConvModule`` /
``DepthwiseSeparableConvModule`` / ``SELayer`` and registered as a
detector via ``@DETECTORS``. LibreYOLO's port keeps the same numerics
but flattens those wrappers, so the key remap is purely syntactic:

  bbox_head.*                           -> head.*
  backbone.<stage>_<i>.*                -> backbone.blocks.<flat_idx>.*
  neck.trans.trans.<i>.*                -> neck.trans.<i>.*
  *.se.conv{1,2}.conv.{w,b}             -> *.se.conv{1,2}.{w,b}

Usage::

    python weights/convert_picodet_weights.py \
        --src ~/picodet_s_320_coco-some-epoch.pth \
        --size s --nc 80 \
        --dst weights/LibrePicoDets.pt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


# ESNet stage repeats: stage_id (2,3,4) -> repeats. Used to flatten Bo's
# ``<stage>_<i>`` (1-indexed) names into ``blocks.<flat_idx>`` (0-indexed).
ESNET_STAGE_REPEATS = (3, 7, 3)


def _build_block_index_map() -> Dict[str, int]:
    """Map ``<stage>_<i>`` -> flat block index. Bo numbers stages 2,3,4."""
    out: Dict[str, int] = {}
    flat = 0
    for stage_idx, repeats in enumerate(ESNET_STAGE_REPEATS):
        stage_id = stage_idx + 2
        for i in range(repeats):
            out[f"{stage_id}_{i + 1}"] = flat
            flat += 1
    return out


_BLOCK_MAP = _build_block_index_map()
# Pattern matches Bo's per-block prefix: e.g. ``backbone.2_1.`` or ``backbone.4_3.``
_BACKBONE_BLOCK_RE = re.compile(r"^backbone\.(\d+_\d+)\.")
# SE wraps with ConvModule, adding an extra ``.conv.`` we need to drop.
_SE_CONV_RE = re.compile(r"\.se\.conv([12])\.conv\.")


def remap_key(key: str) -> str | None:
    """Translate a single Bo-style key to LibreYOLO naming.

    Returns ``None`` if the key should be dropped (e.g. a buffer LibreYOLO
    doesn't carry). Currently no keys are dropped — all numerics survive.
    """
    new = key

    # Top-level rename: bbox_head -> head
    if new.startswith("bbox_head."):
        new = "head." + new[len("bbox_head.") :]

    # Backbone block flattening: backbone.<stage>_<i>. -> backbone.blocks.<flat>.
    m = _BACKBONE_BLOCK_RE.match(new)
    if m is not None:
        token = m.group(1)
        flat = _BLOCK_MAP.get(token)
        if flat is None:
            raise ValueError(
                f"Unexpected backbone block token {token!r} in key {key!r}; "
                "expected one of " + ", ".join(sorted(_BLOCK_MAP))
            )
        new = f"backbone.blocks.{flat}." + new[m.end() :]

    # Neck transformation: neck.trans.trans.X.* -> neck.trans.X.*
    if new.startswith("neck.trans.trans."):
        new = "neck.trans." + new[len("neck.trans.trans.") :]

    # SE ConvModule unwrap: *.se.conv1.conv.X -> *.se.conv1.X (and conv2)
    new = _SE_CONV_RE.sub(lambda mm: f".se.conv{mm.group(1)}.", new)

    return new


def remap_state_dict(state_dict: dict) -> dict:
    """Remap an entire Bo-format state dict. Detects collisions early."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new = remap_key(k)
        if new is None:
            continue
        if new in out:
            raise ValueError(f"Key collision after remap: {k!r} and another -> {new!r}")
        out[new] = v
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


README_TEMPLATE = """---
license: apache-2.0
library_name: libreyolo
tags:
  - object-detection
  - picodet
---

# LibrePicoDet{size}

PicoDet-{size_upper} (ESNet backbone, CSP-PAN neck, GFL/DFL head),
repackaged for LibreYOLO.

## Source

Derived from [Bo396543018/Picodet_Pytorch](https://github.com/Bo396543018/Picodet_Pytorch),
which itself ports PaddlePaddle's PicoDet to PyTorch via mmdet/mmcv.
Copyright (c) 2018-2023 OpenMMLab. Licensed under the Apache License,
Version 2.0.

## Modifications

State-dict key remapping only: ``bbox_head.* -> head.*``,
``backbone.<stage>_<i>.* -> backbone.blocks.<flat>.*``,
``neck.trans.trans.* -> neck.trans.*``, and unwrapping mmcv's
``ConvModule`` inside SE layers. Learned parameters are unchanged.
See ``weights/convert_picodet_weights.py`` in the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo).

## License

Apache License 2.0. See the [`LICENSE`](./LICENSE) and
[`NOTICE`](./NOTICE) files in this repository.
"""


NOTICE_TEMPLATE = """LibrePicoDet weights
--------------------

This product contains weights derived from Bo396543018/Picodet_Pytorch
(https://github.com/Bo396543018/Picodet_Pytorch).
Copyright (c) 2018-2023 OpenMMLab.
Licensed under the Apache License, Version 2.0.

The PicoDet architecture originated at PaddlePaddle (PaddleDetection,
Apache-2.0); see the upstream README for full attribution chain.
"""


GITATTRIBUTES = """*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.engine filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
"""


def write_hf_bundle(bundle_dir: Path, size: str, ckpt_path: Path, license_src: Path) -> None:
    """Emit the 5-file HF upload layout next to a converted checkpoint."""
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Canonical filename
    canonical = bundle_dir / f"LibrePicoDet{size}.pt"
    if canonical.resolve() != ckpt_path.resolve():
        import shutil

        shutil.copy2(ckpt_path, canonical)

    (bundle_dir / "README.md").write_text(
        README_TEMPLATE.format(size=size, size_upper=size.upper())
    )
    (bundle_dir / "NOTICE").write_text(NOTICE_TEMPLATE)
    (bundle_dir / ".gitattributes").write_text(GITATTRIBUTES)

    if license_src is not None and license_src.exists():
        (bundle_dir / "LICENSE").write_text(license_src.read_text())
    else:
        # Fallback: a minimal Apache-2.0 marker. The user must replace
        # this with the verbatim upstream LICENSE before uploading.
        (bundle_dir / "LICENSE").write_text(
            "TODO: copy verbatim Apache-2.0 LICENSE text from\n"
            "https://github.com/Bo396543018/Picodet_Pytorch/blob/master/LICENSE\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Path to Bo's .pth checkpoint")
    parser.add_argument("--dst", required=True, help="Output LibreYOLO checkpoint path")
    parser.add_argument("--size", required=True, choices=["s", "m", "l"])
    parser.add_argument("--nc", type=int, default=80, help="Number of classes")
    parser.add_argument(
        "--hf-bundle",
        type=str,
        default=None,
        help="Optional dir to emit the 5-file HuggingFace upload layout into.",
    )
    parser.add_argument(
        "--upstream-license",
        type=str,
        default=None,
        help="Path to upstream LICENSE file (copied verbatim into the HF bundle).",
    )
    args = parser.parse_args()

    add_repo_root_to_path()
    from libreyolo.models.picodet.nn import LibrePicoDetModel

    print(f"Loading {args.src}")
    raw = load_checkpoint(args.src)
    sd = extract_state_dict(raw)
    if not isinstance(sd, dict):
        raise TypeError(f"Could not extract state dict from {args.src}")

    # Use non-EMA (regular) weights — that is what Bo's mmdet ``init_detector``
    # actually loads (the underscore-flat ``ema_*`` keys are reported as
    # "unexpected" and silently ignored by mmcv's loader). Bo's claimed
    # 26.9 mAP corresponds to this regular set, not the EMA copy.
    print(f"Filtering EMA keys; keeping regular weights from {len(sd)} keys")
    sd = {k: v for k, v in sd.items() if not k.startswith("ema_")}
    # Drop the integral.project buffer — LibreYOLO computes DFL inline in
    # PicoHead, so this constant linspace buffer is not needed.
    sd = {k: v for k, v in sd.items() if not k.endswith("integral.project")}

    print(f"Remapping {len(sd)} keys")
    sd = remap_state_dict(sd)

    # Sanity-load into a fresh LibreYOLO model and report missing/unexpected.
    target = LibrePicoDetModel(size=args.size, nb_classes=args.nc)
    missing, unexpected = target.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing keys (in target, not in source): {len(missing)}")
        for k in missing[:10]:
            print(f"  + {k}")
    if unexpected:
        print(f"Unexpected keys (in source, not in target): {len(unexpected)}")
        for k in unexpected[:10]:
            print(f"  - {k}")
    if not missing and not unexpected:
        print("All keys matched cleanly.")

    wrapped = wrap_libreyolo_checkpoint(
        sd, model_family="picodet", size=args.size, nc=args.nc,
    )
    out = save_checkpoint(wrapped, args.dst)
    print(f"Wrote {out}")

    if args.hf_bundle:
        bundle = Path(args.hf_bundle)
        license_src = Path(args.upstream_license) if args.upstream_license else None
        write_hf_bundle(bundle, args.size, out, license_src)
        print(f"HF upload bundle ready in {bundle}")
        print(f"  Next: huggingface-cli upload LibreYOLO/LibrePicoDet{args.size} {bundle} .")


if __name__ == "__main__":
    main()
