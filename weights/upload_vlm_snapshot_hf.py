"""Stage and upload LibreVLM snapshot mirrors to the LibreYOLO HF org.

Mirrors the Hugging Face snapshot layout used by Grounding DINO / OWLv2, not
a single .pt. Learned parameters are unchanged.

  python weights/upload_vlm_snapshot_hf.py --dry-run
  python weights/upload_vlm_snapshot_hf.py --upload
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "weights"
STAGE = REPO_ROOT / "tmp" / "hf_vlm_snapshots"
TEMPLATE_REPO = "LibreYOLO/LibreGroundingDINOt"

SKIP_NAMES = {
    ".cache",
    ".gitattributes",
    ".libreyolo_snapshot_complete",
    "README.md",
    "LICENSE.md",
}
SKIP_SUFFIXES = {".gguf", ".bin"}

MIRRORS = (
    {
        "local": "LibreGemma4e2b",
        "repo": "LibreYOLO/LibreGemma4e2b",
        "upstream": "google/gemma-4-E2B-it",
        "upstream_sha": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "title": "LibreGemma4e2b",
        "blurb": "Gemma 4 E2B instruct weights mirrored for LibreYOLO's LibreVLM tier.",
        "authors": "Google DeepMind",
        "alias": "gemma-4-e2b",
    },
    {
        "local": "LibreGemma4e4b",
        "repo": "LibreYOLO/LibreGemma4e4b",
        "upstream": "google/gemma-4-E4B-it",
        "upstream_sha": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "title": "LibreGemma4e4b",
        "blurb": "Gemma 4 E4B instruct weights mirrored for LibreYOLO's LibreVLM tier.",
        "authors": "Google DeepMind",
        "alias": "gemma-4",
    },
    {
        "local": "LibreMoondream2",
        "repo": "LibreYOLO/LibreMoondream2",
        "upstream": "vikhyatk/moondream2",
        "upstream_sha": "9a7d4024050840e001defacec2b00727e89149e6",
        "title": "LibreMoondream2",
        "blurb": "Moondream 2 (2025-06-21) weights mirrored for LibreYOLO's LibreVLM tier.",
        "authors": "vikhyatk / M87 Labs",
        "alias": "moondream",
    },
    {
        "local": "LibreMoondream3",
        "repo": "LibreYOLO/LibreMoondream3",
        "upstream": "moondream/moondream3-preview",
        "upstream_sha": "5112966d1a723413b1c9a1e8bea272b72e647b35",
        "title": "LibreMoondream3",
        "blurb": "Moondream 3 Preview weights mirrored for LibreYOLO's LibreVLM tier.",
        "authors": "M87 Labs, Inc.",
        "alias": "moondream-3",
        "license": "bsl-1.1",
        "upstream_license_file": "LICENSE.md",
    },
)


def _readme(spec: dict) -> str:
    if spec.get("license") == "bsl-1.1":
        return f"""---
license: other
license_name: business-source-license-1.1
license_link: https://huggingface.co/{spec["upstream"]}/blob/main/LICENSE.md
library_name: libreyolo
base_model: {spec["upstream"]}
tags:
  - object-detection
  - image-text-to-text
  - libreyolo
  - vlm
---

# {spec["title"]}

{spec["blurb"]}

> ## Custom-license weights (BSL 1.1)
>
> These weights are **not** covered by LibreYOLO's MIT license. They are
> Moondream 3 (Preview) from M87 Labs, licensed under the
> [Business Source License 1.1](https://huggingface.co/{spec["upstream"]}/blob/main/LICENSE.md)
> with an Additional Use Grant. Redistribution is allowed. Production use
> is allowed except offering the model to third parties as a paid hosted
> or embedded service that competes with M87 Labs's paid versions.
> That restriction binds you, the downloader. The LibreYOLO **code** stays
> MIT. See [`LICENSE`](./LICENSE).

## Source

Mirrored from [{spec["upstream"]}](https://huggingface.co/{spec["upstream"]})
at commit `{spec["upstream_sha"]}`.
Copyright (c) {spec["authors"]}. Licensed under BSL 1.1.

## Modifications

No learned parameters were changed. This repository preserves the Hugging Face
snapshot files needed by LibreYOLO's VLM wrapper, ships the upstream license
verbatim, and replaces the model card with LibreYOLO-specific loading notes.

## Usage

```python
from libreyolo import LibreVLM

model = LibreVLM("{spec["alias"]}")
model.set_classes(["person", "helmet"])
results = model.predict("image.jpg")
```

Official runtime notes ask for about 24 GB of GPU memory.

## License

Business Source License 1.1 with Additional Use Grant. See [`LICENSE`](./LICENSE)
and [`NOTICE`](./NOTICE). Change License is Apache-2.0 two years after first
public release of this version.
"""
    return f"""---
license: apache-2.0
library_name: libreyolo
base_model: {spec["upstream"]}
tags:
  - object-detection
  - image-text-to-text
  - libreyolo
  - vlm
---

# {spec["title"]}

{spec["blurb"]}

## Source

Mirrored from [{spec["upstream"]}](https://huggingface.co/{spec["upstream"]})
at commit `{spec["upstream_sha"]}`.
Copyright (c) {spec["authors"]}. Licensed under the Apache License 2.0.

## Modifications

No learned parameters were changed. This repository preserves the Hugging Face
snapshot files needed by LibreYOLO's VLM wrapper and replaces the model card
with LibreYOLO-specific loading notes.

## Usage

```python
from libreyolo import LibreVLM

model = LibreVLM("{spec["alias"]}")
model.set_classes(["person", "helmet"])
results = model.predict("image.jpg")
```

## License

Apache License 2.0. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
"""


def _notice(spec: dict) -> str:
    if spec.get("license") == "bsl-1.1":
        return (
            f"{spec['title']} weights\n"
            f"{'-' * (len(spec['title']) + 8)}\n\n"
            f"This product contains weights derived from {spec['upstream']}\n"
            f"(https://huggingface.co/{spec['upstream']}) at commit\n"
            f"{spec['upstream_sha']}.\n"
            f"Copyright (c) {spec['authors']}.\n"
            "Licensed under the Business Source License 1.1 with an\n"
            "Additional Use Grant (no third-party competing hosted/embedded\n"
            "paid service). The upstream license is shipped verbatim as LICENSE.\n"
            "BSL grants the right to copy, modify, create derivative works,\n"
            "and redistribute. Change License: Apache-2.0.\n\n"
            "No learned parameters were changed by LibreYOLO.\n"
        )
    return (
        f"{spec['title']} weights\n"
        f"{'-' * (len(spec['title']) + 8)}\n\n"
        f"This product contains weights derived from {spec['upstream']}\n"
        f"(https://huggingface.co/{spec['upstream']}) at commit\n"
        f"{spec['upstream_sha']}.\n"
        f"Copyright (c) {spec['authors']}.\n"
        "Licensed under the Apache License, Version 2.0.\n\n"
        "No learned parameters were changed by LibreYOLO.\n"
    )


def stage_one(spec: dict) -> Path:
    src = WEIGHTS / spec["local"]
    if not any(src.glob("*.safetensors")):
        raise FileNotFoundError(f"Missing snapshot at {src}")
    dest = STAGE / spec["local"]
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for path in src.iterdir():
        if path.name in SKIP_NAMES or path.suffix in SKIP_SUFFIXES:
            continue
        target = dest / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
    from huggingface_hub import hf_hub_download

    shutil.copy2(hf_hub_download(TEMPLATE_REPO, ".gitattributes"), dest / ".gitattributes")
    if spec.get("license") == "bsl-1.1":
        license_name = spec.get("upstream_license_file", "LICENSE.md")
        src_license = src / license_name
        if not src_license.exists():
            src_license = Path(
                hf_hub_download(
                    spec["upstream"],
                    license_name,
                    revision=spec["upstream_sha"],
                )
            )
        shutil.copy2(src_license, dest / "LICENSE")
    else:
        shutil.copy2(hf_hub_download(TEMPLATE_REPO, "LICENSE"), dest / "LICENSE")
    (dest / "README.md").write_text(_readme(spec), encoding="utf-8")
    (dest / "NOTICE").write_text(_notice(spec), encoding="utf-8")
    return dest


def upload_one(spec: dict, dest: Path) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(spec["repo"], repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=spec["repo"],
        folder_path=str(dest),
        repo_type="model",
        commit_message=f"Mirror {spec['upstream']} @{spec['upstream_sha'][:12]}",
    )
    info = api.repo_info(spec["repo"], repo_type="model")
    return info.sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    selected = MIRRORS
    if args.only:
        wanted = set(args.only)
        selected = tuple(s for s in MIRRORS if s["local"] in wanted or s["repo"] in wanted)
    for spec in selected:
        dest = stage_one(spec)
        files = sorted(p.name for p in dest.iterdir())
        print(f"staged {spec['repo']} -> {dest} ({len(files)} files): {files}")
        if args.upload:
            sha = upload_one(spec, dest)
            print(f"uploaded {spec['repo']} sha={sha}")


if __name__ == "__main__":
    main()
