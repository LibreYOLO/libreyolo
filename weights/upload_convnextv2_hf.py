"""Stage and optionally publish the eight ConvNeXt V2 weight repositories.

Run conversion first. Default is local staging only; --publish uploads to
LibreYOLO and adds each model to the classification collection. Existing
nonempty repositories are never overwritten. Uses the active HF login.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from _conversion_utils import add_repo_root_to_path
from convert_convnextv2_weights import SOURCE_COMMIT, SOURCE_REPO, VARIANTS

COLLECTION = "LibreYOLO/libreyolo-classification-6a4164414d64a10aa8576885"


def stage(size: str, weights: Path, output: Path) -> Path:
    add_repo_root_to_path()
    from huggingface_hub import HfApi, hf_hub_download

    from libreyolo import LibreConvNeXtV2
    from libreyolo.utils.serialization import (
        load_untrusted_torch_file,
        validate_checkpoint_metadata,
    )

    name = f"LibreConvNeXtV2{size}-cls"
    filename = name + ".pt"
    source = weights / filename
    checkpoint = load_untrusted_torch_file(source)
    validate_checkpoint_metadata(checkpoint)
    assert checkpoint["model_family"] == "convnextv2" and checkpoint["size"] == size
    assert checkpoint["weight_license"] == "cc-by-nc-4.0"
    assert checkpoint["weight_commercial_use"] is False
    assert checkpoint["nc"] == 1000 and checkpoint["imgsz"] == 224
    assert (
        LibreConvNeXtV2.get_download_url(filename)
        == f"https://huggingface.co/LibreYOLO/{name}/resolve/main/{filename}"
    )
    destination = output / name
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination / filename)
    template_repo = "LibreYOLO/LibreConvNeXtt-cls"
    revision = HfApi().model_info(template_repo).sha
    attributes = hf_hub_download(template_repo, ".gitattributes", revision=revision)
    shutil.copy2(attributes, destination / ".gitattributes")
    license_url = f"https://raw.githubusercontent.com/facebookresearch/ConvNeXt-V2/{SOURCE_COMMIT}/LICENSE"
    with urllib.request.urlopen(license_url) as response:
        (destination / "LICENSE").write_bytes(response.read())
    upstream_url = f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_{VARIANTS[size]}_1k_224_ema.pt"
    (destination / "NOTICE").write_text(
        f"ConvNeXt V2 {VARIANTS[size]} weights\n\nCopyright (c) Meta Platforms, Inc. and affiliates.\n"
        f"Source: {SOURCE_REPO}\nCommit: {SOURCE_COMMIT}\nCheckpoint: {upstream_url}\n"
        f"Source SHA-256: {checkpoint['source_sha256']}\n"
        "Weights: Creative Commons Attribution-NonCommercial 4.0 International.\n"
        "https://creativecommons.org/licenses/by-nc/4.0/\n"
        "Architecture code: MIT, separate from the weight license.\n"
        "Modifications: LibreYOLO metadata and class names added; learned tensors unchanged.\n",
        encoding="utf-8",
    )
    (destination / "README.md").write_text(
        f"""---
license: cc-by-nc-4.0
library_name: libreyolo
pipeline_tag: image-classification
datasets:
  - imagenet-1k
tags:
  - convnextv2
  - image-classification
---

# {name}

**NON-COMMERCIAL WEIGHTS: CC-BY-NC-4.0.** The MIT architecture code has a separate license.

ConvNeXt V2 {VARIANTS[size].title()}, 224px, 1000 ImageNet classes, converted for LibreYOLO.

## Usage

Requires LibreYOLO with ConvNeXt V2 support (the development branch until released).

```python
from libreyolo import LibreYOLO
model = LibreYOLO("{filename}")
result = model.predict("image.jpg")[0]
print(result.names[result.probs.top1])
```

## Source

[{SOURCE_REPO}]({SOURCE_REPO}) at `{SOURCE_COMMIT}`.
Copyright (c) Meta Platforms, Inc. and affiliates.
[Official ImageNet-1K fine-tuned EMA checkpoint]({upstream_url}).
Source SHA-256: `{checkpoint["source_sha256"]}`.
[Paper: ConvNeXt V2](https://arxiv.org/abs/2301.00808).

## Modifications

LibreYOLO schema-v1 metadata and ImageNet class names added. Learned parameters
and state-dict keys are unchanged. Conversion: `weights/convert_convnextv2_weights.py`
in the [LibreYOLO source](https://github.com/LibreYOLO/libreyolo).
Evaluation uses bicubic resize of the shorter side to 256, a 224 center crop,
and ImageNet mean/std normalization. This is the supervised classifier, not an
FCMAE encoder-only checkpoint. Fine-tunes derived from these weights retain
their non-commercial terms. No independent full-ImageNet accuracy claim is made.

## License

Weights are CC-BY-NC-4.0. See [LICENSE](./LICENSE) and [NOTICE](./NOTICE).
The LICENSE reproduces the upstream combined MIT-code and CC-BY-NC-weight text.
""",
        encoding="utf-8",
    )
    assert {p.name for p in destination.iterdir()} == {
        filename,
        "README.md",
        "LICENSE",
        "NOTICE",
        ".gitattributes",
    }
    return destination


def publish(directory: Path) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    api = HfApi()
    repo = f"LibreYOLO/{directory.name}"
    try:
        existing = api.list_repo_files(repo)
    except RepositoryNotFoundError:
        existing = []
    if set(existing) - {".gitattributes"}:
        raise RuntimeError(f"Refusing to overwrite existing files in {repo}")
    api.create_repo(repo, exist_ok=True, private=False)
    api.upload_folder(
        repo_id=repo,
        folder_path=directory,
        commit_message="Publish ConvNeXt V2 ImageNet-1K weights",
    )
    assert set(api.list_repo_files(repo)) == {p.name for p in directory.iterdir()}
    api.add_collection_item(COLLECTION, item_id=repo, item_type="model", exists_ok=True)
    print(f"Published https://huggingface.co/{repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", choices=list(VARIANTS))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    for size in [args.size] if args.size else VARIANTS:
        directory = stage(size, args.weights, args.output)
        print(f"Staged {directory}")
        if args.publish:
            publish(directory)
