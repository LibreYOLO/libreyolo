"""Mirror Meta's SAM-1 transformers snapshots onto the LibreYOLO org.

    .venv/Scripts/python.exe scripts/mirror_sam.py base [large huge]

Why
---
LibreSAM1 loads through ``transformers`` from facebook/sam-vit-{base,large,huge}.
Those weights are Apache-2.0, which permits redistribution, so the dependency
on Meta's repos is a choice rather than a constraint.

Shape
-----
This family ships snapshot directories rather than a single ``.pt``, the same
as LibreGroundingDINO and LibreOWLv2, so the 5-file contract does not apply
verbatim; the repo mirrors the upstream layout plus a card, LICENSE and NOTICE.

Only the PyTorch-relevant files are copied. Upstream also carries
``pytorch_model.bin`` (a duplicate of the safetensors weights) and
``tf_model.h5``; mirroring those would triple the repo for nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

from _mirror_common import fetch_text, gitattributes, parse_args
from huggingface_hub import HfApi, snapshot_download

# Exact upstream revisions the mirrors were taken from, recorded so a consumer
# can identify the precise source. Refresh when re-mirroring.
UPSTREAM_REVISIONS = {
    "base": "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f",
    "large": "6851e0441005b0fb96f2cc4dfac472f3d1b14af1",
    "huge": "87aecf0df4ce6b30cd7de76e87673c49644bdf67",
}

UPSTREAM = {
    "base": "facebook/sam-vit-base",
    "large": "facebook/sam-vit-large",
    "huge": "facebook/sam-vit-huge",
}

# Everything transformers needs to rebuild the model and its processor.
KEEP = ["config.json", "preprocessor_config.json", "model.safetensors"]

LICENSE_URL = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/LICENSE"

README = """---
license: apache-2.0
library_name: libreyolo
tags:
  - mask-generation
  - image-segmentation
  - sam
---

# {repo}

Segment Anything (SAM-1) ViT-{size_title}, mirrored for LibreYOLO.

## Source

Mirrored from [{upstream}](https://huggingface.co/{upstream}).
Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the Apache License, Version 2.0.

## Modifications

None. The weights and configuration are Meta's, byte for byte. This repository
exists so LibreYOLO can serve them from its own organisation rather than
depending on a third-party repository remaining available.

Upstream also publishes `pytorch_model.bin` (a duplicate of the safetensors
weights) and `tf_model.h5`. Neither is mirrored, because LibreYOLO loads the
safetensors file.

## Usage

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreSAM{size}.pt")
```

## License

Apache License 2.0. See the [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE)
files in this repository.
"""

NOTICE = """LibreSAM weights
----------------

This product contains weights from Segment Anything (SAM)
(https://github.com/facebookresearch/segment-anything).
Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the Apache License, Version 2.0.

Source artifact:  https://huggingface.co/{upstream}
Source revision:  {revision}
Source files:     config.json, preprocessor_config.json, model.safetensors
Modification:     none. The weights and configuration are redistributed
                  unmodified. Upstream's pytorch_model.bin (a duplicate of the
                  safetensors weights) and tf_model.h5 are not mirrored.
"""


def stage(size: str, staging: Path) -> Path:
    upstream = UPSTREAM[size]
    repo_name = f"LibreSAM{size}"
    out = staging / repo_name
    out.mkdir(parents=True, exist_ok=True)

    print(f"  downloading {upstream} ({', '.join(KEEP)}) ...", flush=True)
    snapshot_download(
        repo_id=upstream,
        local_dir=str(out),
        allow_patterns=KEEP,
    )

    (out / "LICENSE").write_text(
        fetch_text(LICENSE_URL), encoding="utf-8"
    )
    (out / "NOTICE").write_text(
        NOTICE.format(upstream=upstream, revision=UPSTREAM_REVISIONS[size]),
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        README.format(repo=repo_name, size=size, size_title=size.title(), upstream=upstream),
        encoding="utf-8",
    )

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"  staged {repo_name}: {sorted(p.name for p in out.iterdir())} "
          f"({total / 1e6:.0f} MB)", flush=True)
    return out


def upload(size: str, folder: Path) -> None:
    repo_id = f"LibreYOLO/LibreSAM{size}"
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    url = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Mirror SAM-1 ViT-{size} from {UPSTREAM[size]} (Apache-2.0)",
    )
    print(f"  uploaded {repo_id}: {url}", flush=True)


def main() -> int:
    args = parse_args(__doc__ or "Mirror SAM-1", list(UPSTREAM))
    print(f"staging under {args.staging}", flush=True)
    for size in args.sizes:
        if size not in UPSTREAM:
            raise SystemExit(f"unknown size {size!r}; expected one of {list(UPSTREAM)}")
        print(f"\n=== {size} ===", flush=True)
        folder = stage(size, args.staging)
        if args.no_upload:
            print("  --no-upload: staged only", flush=True)
        else:
            upload(size, folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
