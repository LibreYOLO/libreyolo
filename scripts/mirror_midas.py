"""Mirror the MiDaS depth weights onto the LibreYOLO org.

    .venv/Scripts/python.exe scripts/mirror_midas.py s l

Why now
-------
ADR 0006 says hosted depth weights must be trained only on data whose licence
permits redistribution *and commercial use*. MiDaS trained on a twelve-dataset
mixture that includes non-commercial sources, so it failed that bar and the
family has been downloading from isl-org's GitHub releases ever since.

The maintainer has ruled that the isl-org/MiDaS MIT licence extends to the
released checkpoints, so LibreYOLO relies on the publisher's own grant for the
bytes it redistributes. ADR 0006's consequence needs amending to match: it
still states the stricter commercial-use bar.

The mirror ships the converted LibreYOLO checkpoint, so the upstream SHA-256
recorded in convert.py describes upstream's file and not ours.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from _mirror_common import fetch_text, gitattributes, parse_args
from huggingface_hub import HfApi

from libreyolo.models.midas.convert import UPSTREAM_SHA256, UPSTREAM_URLS
from libreyolo.utils.serialization import validate_checkpoint_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "weights"
ARCH = {"s": "EfficientNet-Lite3 (MiDaS v2.1 small, 256 px)",
        "l": "ViT-L/16 (DPT-Large, 384 px)"}

LICENSE_URL = "https://raw.githubusercontent.com/isl-org/MiDaS/master/LICENSE"

README = """---
license: mit
library_name: libreyolo
tags:
  - depth-estimation
  - monocular-depth
  - midas
---

# {repo}

MiDaS {arch} monocular depth model, repackaged in LibreYOLO checkpoint format.

## Source

Derived from [isl-org/MiDaS](https://github.com/isl-org/MiDaS), release asset
`{asset}`.
Copyright (c) Intel ISL. Licensed under the MIT License.

## Modifications

State-dict key remapping only. Learned parameters are unchanged.
See `weights/convert_midas_weights.py` in the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo).

## A note on the training data

MiDaS was trained on a mixture of twelve datasets, several of which carry
non-commercial terms of their own. LibreYOLO redistributes these weights under
the MIT licence its publisher applied to them. If your use is commercial,
satisfy yourself about the training-data terms before relying on it.

## Usage

```python
from libreyolo import LibreYOLO

model = LibreYOLO("{repo}.pt")
depth = model.predict("image.jpg")[0].depth_map
```

## License

MIT License. See the [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE) files in
this repository.
"""

NOTICE = """LibreMiDaS weights
------------------

This product contains weights derived from MiDaS
(https://github.com/isl-org/MiDaS).
Copyright (c) Intel ISL.
Licensed under the MIT License.

Source artifact:  {asset_url}
Source SHA-256:   {upstream_sha256}
Modification:     state-dict key remapping only, by
                  weights/convert_midas_weights.py in LibreYOLO. Learned
                  parameters are unchanged.

MiDaS was trained on a mixture of twelve datasets whose individual terms are
not all permissive. The redistribution here rests on the MIT licence applied
by the publisher to the released checkpoints.
"""


def stage(size: str, staging: Path) -> Path:
    repo = f"LibreMiDaS{size}-depth"
    src = WEIGHTS / f"{repo}.pt"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Load it once so the converter produces it:\n"
            f"  LibreYOLO('{repo}.pt')"
        )

    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    errors = validate_checkpoint_metadata(checkpoint, strict=False)
    if errors:
        raise SystemExit(f"{repo}: checkpoint metadata invalid: {errors}")

    out = staging / repo
    out.mkdir(parents=True, exist_ok=True)
    for name, text in (
        (".gitattributes", gitattributes()),
        ("LICENSE", fetch_text(LICENSE_URL)),
        ("NOTICE", NOTICE.format(
            asset_url=UPSTREAM_URLS[size],
            upstream_sha256=UPSTREAM_SHA256[UPSTREAM_URLS[size].rsplit("/", 1)[-1]],
        )),
        ("README.md", README.format(repo=repo, arch=ARCH[size],
                                    asset=UPSTREAM_URLS[size].rsplit("/", 1)[-1])),
    ):
        (out / name).write_text(text, encoding="utf-8")

    if not (out / f"{repo}.pt").exists():
        (out / f"{repo}.pt").write_bytes(src.read_bytes())

    files = sorted(p.name for p in out.iterdir())
    print(f"  staged {repo}: {files} "
          f"({(out / f'{repo}.pt').stat().st_size / 1e6:.0f} MB)", flush=True)
    if len(files) != 5:
        raise SystemExit(f"{repo}: expected 5 files, got {len(files)}")
    return out


def upload(size: str, folder: Path) -> None:
    repo_id = f"LibreYOLO/LibreMiDaS{size}-depth"
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    url = api.upload_folder(
        folder_path=str(folder), repo_id=repo_id, repo_type="model",
        commit_message=f"Mirror MiDaS {size} depth weights (MIT)",
    )
    print(f"  uploaded {repo_id}: {url}", flush=True)


def main() -> int:
    args = parse_args(__doc__ or "Mirror MiDaS", ["s", "l"])
    print(f"staging under {args.staging}", flush=True)
    for size in args.sizes:
        print(f"\n=== {size} ===", flush=True)
        folder = stage(size, args.staging)
        if args.no_upload:
            print("  --no-upload: staged only", flush=True)
        else:
            upload(size, folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
