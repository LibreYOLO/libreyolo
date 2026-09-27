"""Stage and upload LibreGround snapshot mirrors to the LibreYOLO HF org.

Follows the VLM / open-vocab snapshot-directory exception in
``skills/libreyolo-upload-hf-model`` (not the 5-file ``.pt`` contract).

  python weights/upload_ground_snapshot_hf.py --dry-run
  python weights/upload_ground_snapshot_hf.py --upload
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "weights"
STAGE = REPO_ROOT / "tmp" / "hf_ground_snapshots"
TEMPLATE_REPO = "LibreYOLO/LibreGroundingDINOt"

SKIP_NAMES = {
    ".cache",
    ".gitattributes",
    ".libreyolo_snapshot_complete",
    "README.md",
    "LICENSE.md",
    "LICENSE",
    "LICENSE-APACHE-2.0",
    "NOTICE",
}

MIRRORS = (
    {
        "local": "LibreGroundFlorence2base",
        "repo": "LibreYOLO/LibreGroundFlorence2base",
        "upstream": "florence-community/Florence-2-base",
        "upstream_sha": "00921df66db728a9ceb750f5eca43e5c203a2051",
        "title": "LibreGroundFlorence2base",
        "blurb": "Florence-2-base snapshot mirrored for LibreYOLO's LibreGround tier.",
        "authors": "Microsoft / florence-community",
        "alias": "florence-2-base",
        "license": "mit",
        "license_source": {
            "kind": "hf",
            "repo": "microsoft/Florence-2-base",
            "revision": "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac",
            "filename": "LICENSE",
            "sha256": "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383",
        },
        "allow_bin": False,
    },
    {
        "local": "LibreShowUI2b",
        "repo": "LibreYOLO/LibreShowUI2b",
        "upstream": "showlab/ShowUI-2B",
        "upstream_sha": "cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60",
        "title": "LibreShowUI2b",
        "blurb": "ShowUI-2B snapshot mirrored for LibreYOLO's LibreGround tier.",
        "authors": "Show Lab",
        "alias": "showui-2b",
        "license": "mit",
        "license_source": {
            "kind": "url",
            "url": "https://raw.githubusercontent.com/showlab/ShowUI/59e059c4df62db0857cc7a7ef0b15d067a30c274/LICENSE",
            "sha256": "dc3ad771428560c99f7605c777c9c37f5b8ef3cd37158f668985910d6dbf3f47",
        },
        "additional_license_sources": (
            {
                "kind": "hf",
                "repo": "Qwen/Qwen2-VL-2B-Instruct",
                "revision": "895c3a49bc3fa70a340399125c650a463535e71c",
                "filename": "LICENSE",
                "destination": "LICENSE-APACHE-2.0",
                "sha256": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
            },
        ),
        "license_note": (
            "Pinned showlab/ShowUI-2B card declares MIT for the weights. "
            "The MIT text is preserved from ShowUI's repository history. "
            "The Qwen2-VL-2B base is Apache-2.0 and its license is shipped "
            "separately as LICENSE-APACHE-2.0."
        ),
        "allow_bin": True,
    },
    {
        "local": "LibreGroundQwen3VL2b",
        "repo": "LibreYOLO/LibreGroundQwen3VL2b",
        "upstream": "Qwen/Qwen3-VL-2B-Instruct",
        "upstream_sha": "89644892e4d85e24eaac8bacfd4f463576704203",
        "title": "LibreGroundQwen3VL2b",
        "blurb": "Qwen3-VL-2B-Instruct snapshot mirrored for LibreYOLO's LibreGround tier.",
        "authors": "Qwen / Alibaba",
        "alias": "qwen3-vl-2b",
        "license": "apache-2.0",
        "license_source": {
            "kind": "url",
            "url": "https://raw.githubusercontent.com/QwenLM/Qwen3-VL/96588727e44c78b25ba03ea03b8e12f7e64fd0da/LICENSE",
            "sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        },
        "allow_bin": False,
    },
)


def _readme(spec: dict) -> str:
    license_yaml = spec["license"]
    license_label = "MIT License" if license_yaml == "mit" else "Apache License 2.0"
    extra_license = spec.get("license_note", "")
    license_body = (
        f"{license_label}. {extra_license}".strip() if extra_license else license_label
    )
    license_files = ["LICENSE"] + [
        source["destination"] for source in spec.get("additional_license_sources", ())
    ]
    license_links = ", ".join(
        f"[`{filename}`](./{filename})" for filename in license_files
    )
    return f"""---
license: {license_yaml}
library_name: libreyolo
base_model: {spec["upstream"]}
tags:
  - image-text-to-text
  - libreyolo
  - grounding
---

# {spec["title"]}

{spec["blurb"]}

## Source

Mirrored from [{spec["upstream"]}](https://huggingface.co/{spec["upstream"]})
at commit `{spec["upstream_sha"]}`.
Copyright (c) {spec["authors"]}.

## Modifications

No learned parameters were changed. This repository preserves the Hugging Face
snapshot files needed by LibreYOLO's LibreGround wrapper and replaces the
model card with LibreYOLO-specific loading notes.

## Usage

```python
from libreyolo import LibreGround

r = LibreGround("{spec["alias"]}")("screen.png", prompt="Submit")
r.points.xy
```

## License

{license_body}

See {license_links} and [`NOTICE`](./NOTICE).
"""


def _notice(spec: dict) -> str:
    extra = spec.get("license_note")
    license_label = (
        "the MIT License"
        if spec["license"] == "mit"
        else "the Apache License, Version 2.0"
    )
    extra_block = f"\n{extra}\n" if extra else ""
    return (
        f"{spec['title']} weights\n"
        f"{'-' * (len(spec['title']) + 8)}\n\n"
        f"This product contains weights derived from {spec['upstream']}\n"
        f"(https://huggingface.co/{spec['upstream']}) at commit\n"
        f"{spec['upstream_sha']}.\n"
        f"Copyright (c) {spec['authors']}.\n"
        f"Weight grant: {license_label}.\n"
        f"{extra_block}\n"
        "No learned parameters were changed by LibreYOLO.\n"
    )


def _has_weights(directory: Path, allow_bin: bool) -> bool:
    if any(directory.glob("*.safetensors")):
        return True
    return allow_bin and any(directory.glob("*.bin"))


def _copy_license_source(source: dict, destination: Path) -> None:
    """Copy a pinned upstream license verbatim and verify its digest."""
    if source["kind"] == "hf":
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            source["repo"],
            source["filename"],
            revision=source["revision"],
        )
        payload = Path(downloaded).read_bytes()
    elif source["kind"] == "url":
        with urlopen(source["url"]) as response:
            payload = response.read()
    else:
        raise ValueError(f"Unknown license source kind: {source['kind']!r}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != source["sha256"]:
        raise RuntimeError(
            f"License digest mismatch for {destination.name}: "
            f"expected {source['sha256']}, got {digest}."
        )
    destination.write_bytes(payload)


def stage_one(spec: dict) -> Path:
    """Write card files into the existing local snapshot (no second copy)."""
    from huggingface_hub import hf_hub_download

    dest = WEIGHTS / spec["local"]
    if not _has_weights(dest, spec["allow_bin"]):
        raise FileNotFoundError(f"Missing snapshot weights at {dest}")
    shutil.copy2(
        hf_hub_download(TEMPLATE_REPO, ".gitattributes"), dest / ".gitattributes"
    )
    _copy_license_source(spec["license_source"], dest / "LICENSE")
    for source in spec.get("additional_license_sources", ()):
        _copy_license_source(source, dest / source["destination"])
    (dest / "README.md").write_text(_readme(spec), encoding="utf-8")
    (dest / "NOTICE").write_text(_notice(spec), encoding="utf-8")
    return dest


def upload_one(spec: dict, dest: Path) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(spec["repo"], repo_type="model", exist_ok=True, private=False)
    ignore = [".cache", ".libreyolo_snapshot_complete", "examples"]
    if not spec["allow_bin"]:
        ignore.append("*.bin")
    api.upload_folder(
        repo_id=spec["repo"],
        folder_path=str(dest),
        repo_type="model",
        ignore_patterns=ignore,
        commit_message=f"Mirror {spec['upstream']} @{spec['upstream_sha'][:12]}",
    )
    info = api.repo_info(spec["repo"], repo_type="model")
    return info.sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = MIRRORS
    if args.only:
        wanted = set(args.only)
        selected = tuple(
            spec
            for spec in MIRRORS
            if spec["local"] in wanted
            or spec["repo"] in wanted
            or spec["alias"] in wanted
        )
    for spec in selected:
        dest = stage_one(spec)
        files = sorted(path.name for path in dest.iterdir())
        print(f"staged {spec['repo']} -> {dest} ({len(files)} files): {files}")
        if args.upload and not args.dry_run:
            sha = upload_one(spec, dest)
            print(f"uploaded {spec['repo']} sha={sha}")


if __name__ == "__main__":
    main()
