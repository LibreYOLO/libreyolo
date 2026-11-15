"""Build and upload the LibreFlowSeek{t,s,m,l}-flow Hugging Face repos.

One repo per weight file (``LibreYOLO/LibreFlowSeek<size>-flow``), each with
the 5-file contract: ``.gitattributes``, ``README.md``, ``LICENSE``,
``NOTICE`` and the converted ``.pt``.

    python weights/upload_flowseek_hf.py --weights-dir DIR --upstream FLOWSEEK_CLONE \
        --cc-by-nc LICENSE_CC_BY_NC_4.0 --gitattributes GITATTRIBUTES --staging OUT [--upload]

``--cc-by-nc`` is the verbatim CC BY-NC 4.0 legal code (the one shipped with
LibreYOLO/LibreDepthAnythingV2l-depth). m and l embed the Depth Anything V2
Base encoder, which is CC BY-NC 4.0, so their cards lead with the
non-commercial banner and their LICENSE carries both texts.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

COMMIT = "ba6407370bceeb96244cad4d60d37489f6eb471e"
INFO = {
    "t": ("ResNet-18", "ViT-S (Small)", 4, "flowseek_T_TartanCT_TSKH.pth"),
    "s": ("ResNet-18", "ViT-S (Small)", 12, "flowseek_T_TartanCT_TSKH.pth"),
    "m": ("ResNet-34", "ViT-B (Base)", 4, "flowseek_M_TartanCT_TSKH.pth"),
    "l": ("ResNet-34", "ViT-B (Base)", 12, "flowseek_M_TartanCT_TSKH.pth"),
}
NON_COMMERCIAL = {"m", "l"}


def notice(size: str) -> str:
    nc = size in NON_COMMERCIAL
    dav2 = (
        "The embedded Depth Anything V2 Base (ViT-B) encoder weights are\n"
        "Copyright (c) 2024 Depth Anything V2 authors and licensed under the\n"
        "Creative Commons Attribution-NonCommercial 4.0 International License\n"
        "(CC BY-NC 4.0). Commercial use is not permitted.\n"
        if nc
        else "The embedded Depth Anything V2 Small (ViT-S) encoder weights are\n"
        "Copyright (c) 2024 Depth Anything V2 authors and licensed under the\n"
        "Apache License, Version 2.0.\n"
    )
    return f"""LibreFlowSeek weights
---------------------

This product contains weights derived from FlowSeek
(https://github.com/mattpoggi/flowseek, commit {COMMIT}),
checkpoint {INFO[size][3]}.
Copyright (c) 2025 Matteo Poggi and Fabio Tosi.
The FlowSeek repository is licensed under the Apache License, Version 2.0.
It publishes no separate licence for its checkpoints; LibreYOLO applies the
repository's Apache-2.0 licence to them as an implied grant, which is
LibreYOLO's disclosed interpretation, not a statement by the authors.

FlowSeek includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT),
Copyright (c) 2024, Princeton Vision & Learning Lab, BSD 3-Clause License.

{dav2}The Depth Anything V2 encoder is a DINOv2 backbone
(https://github.com/facebookresearch/dinov2), Copyright (c) Meta Platforms,
Inc. and affiliates, Apache License, Version 2.0.
"""


def readme(size: str) -> str:
    ctx, enc, iters, ckpt = INFO[size]
    nc = size in NON_COMMERCIAL
    name = f"LibreFlowSeek{size}-flow"
    banner = (
        "> **Non-commercial use only.** This checkpoint embeds the Depth Anything V2 Base\n"
        "> encoder, whose weights are CC BY-NC 4.0. The t and s sizes\n"
        "> ([LibreFlowSeekt-flow](https://huggingface.co/LibreYOLO/LibreFlowSeekt-flow),\n"
        "> [LibreFlowSeeks-flow](https://huggingface.co/LibreYOLO/LibreFlowSeeks-flow))\n"
        "> embed the Apache-2.0 Small encoder and carry no such restriction.\n\n"
        if nc
        else ""
    )
    license_tag = (
        "license: cc-by-nc-4.0\n"
        if nc
        else "license: apache-2.0\n"
    )
    lic = (
        "FlowSeek parameters: Apache License 2.0 (implied by the releasing repository).\n"
        "Embedded Depth Anything V2 Base encoder: CC BY-NC 4.0. **Commercial use is not permitted.**\n"
        if nc
        else "Apache License 2.0 (FlowSeek parameters, implied by the releasing repository;\n"
        "Depth Anything V2 Small encoder, declared upstream).\n"
    )
    shares = "t" if size in ("t", "s") else "m"
    other = {"t": "s", "s": "t", "m": "l", "l": "m"}[size]
    return f"""---
{license_tag}library_name: libreyolo
pipeline_tag: image-to-image
tags:
  - optical-flow
  - flowseek
  - libreflow
---

# {name}

FlowSeek optical flow ({ctx} context and feature networks, Depth Anything V2 {enc}
encoder, {iters} refinement iterations), repackaged for LibreYOLO's `flow` task.

{banner}## Source

Derived from [mattpoggi/flowseek](https://github.com/mattpoggi/flowseek) at commit
`{COMMIT}`, checkpoint `{ckpt}` (TartanAir, then Chairs and Things, then the
Sintel + KITTI + HD1K + Things mixed fine-tune: the generalist checkpoint upstream
evaluates on Spring and LayeredFlow).
Copyright (c) 2025 Matteo Poggi and Fabio Tosi (University of Bologna).
FlowSeek: Optical Flow Made Easier with Depth Foundation Models and Motion Bases, ICCV 2025.

The checkpoint embeds the frozen Depth Anything V2 {enc} encoder
([DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)),
so this file is self-contained. FlowSeek builds on SEA-RAFT (BSD 3-Clause).

Upstream trains two networks. This size uses the same weights as
[LibreFlowSeek{other}-flow](https://huggingface.co/LibreYOLO/LibreFlowSeek{other}-flow)
with a different iteration count, as in upstream `config/eval/flowseek-{size.upper()}.json`.

## Modifications

Metadata wrap only (LibreYOLO checkpoint schema v1.0). Learned parameters are unchanged.
See `weights/convert_flowseek_weights.py` in the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo).

## Usage

```python
from libreyolo import LibreYOLO

model = LibreYOLO("{name}.pt")
results = model(["frame_000.png", "frame_001.png"])  # N frames give N - 1 flows
flow = results[0].flow.data  # (H, W, 2) float32, pixels, (dx, dy)
```

## License

{lic}
See the [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE) files in this repository.
"""


def build(size: str, a) -> Path:
    name = f"LibreFlowSeek{size}-flow"
    out = a.staging / name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.gitattributes, out / ".gitattributes")
    apache = (a.upstream / "LICENSE.md").read_text()
    if size in NON_COMMERCIAL:
        cc = Path(a.cc_by_nc).read_text()
        text = (
            "This repository contains material under two licences.\n\n"
            "1. FlowSeek (https://github.com/mattpoggi/flowseek), Apache License 2.0,\n"
            "   applied to the FlowSeek parameters:\n\n"
            + apache
            + "\n\n"
            "2. Depth Anything V2 Base encoder weights (embedded under dav2.*),\n"
            "   Creative Commons Attribution-NonCommercial 4.0 International:\n\n"
            + cc
        )
    else:
        text = apache
    (out / "LICENSE").write_text(text)
    (out / "NOTICE").write_text(notice(size))
    (out / "README.md").write_text(readme(size))
    shutil.copy(a.weights_dir / f"{name}.pt", out / f"{name}.pt")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights-dir", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--cc-by-nc", required=True)
    p.add_argument("--gitattributes", required=True)
    p.add_argument("--staging", type=Path, required=True)
    p.add_argument("--sizes", default="t,s,m,l")
    p.add_argument("--upload", action="store_true")
    a = p.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from libreyolo.utils.serialization import load_untrusted_torch_file, validate_checkpoint_metadata

    for size in a.sizes.split(","):
        out = build(size, a)
        name = out.name
        ckpt = load_untrusted_torch_file(str(out / f"{name}.pt"), map_location="cpu")
        errors = validate_checkpoint_metadata(ckpt, strict=True)
        assert not errors, errors
        assert (ckpt["model_family"], ckpt["size"], ckpt["task"]) == ("flowseek", size, "flow")
        print("built", out, sorted(p.name for p in out.iterdir()))
        if a.upload:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(f"LibreYOLO/{name}", repo_type="model", exist_ok=True)
            api.upload_folder(
                repo_id=f"LibreYOLO/{name}",
                folder_path=str(out),
                commit_message="Initial upload",
                ignore_patterns=[".cache/*"],
            )
            print("uploaded", f"https://huggingface.co/LibreYOLO/{name}")


if __name__ == "__main__":
    main()
