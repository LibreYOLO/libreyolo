"""Stage (and optionally upload) the V-JEPA 2 Hugging Face weight repos.

Builds the exact 5-file contract from ``skills/libreyolo-upload-hf-model``:

    .gitattributes  README.md  LICENSE  NOTICE  Libre<...>.pt

Weight licences are per artifact. The two ViT-g encoders are Apache-2.0 while
the other six artifacts are MIT, so each repo ships the licence text of its
own source rather than a family-wide approximation.

Usage::

    python weights/stage_vjepa2_hf_repos.py --stage-dir build/hf            # stage only
    python weights/stage_vjepa2_hf_repos.py --stage-dir build/hf --upload   # stage + push
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from convert_vjepa2_weights import ENCODER_SOURCES, PROBE_SOURCES

ORG = "LibreYOLO"

# name -> (source repo, revision, licence id, description)
ARTIFACTS: dict[str, tuple[str, str, str, str]] = {}
for _size, (_repo, _rev, _lic) in ENCODER_SOURCES.items():
    ARTIFACTS[f"LibreVJEPA2{_size}-embed"] = (
        _repo,
        _rev,
        _lic,
        f"V-JEPA 2.0 {_size} video encoder (clip embedding)",
    )
for (_size, _variant), (_repo, _rev, _lic) in PROBE_SOURCES.items():
    ARTIFACTS[f"LibreVJEPA2{_size}-cls-{_variant}"] = (
        _repo,
        _rev,
        _lic,
        f"V-JEPA 2.0 {_size} attentive probe, {_variant} video classification",
    )

LICENSE_URLS = {
    "MIT": "https://raw.githubusercontent.com/facebookresearch/vjepa2/204698b45b3712590f06245fbfba32d3be539812/LICENSE",
    "Apache-2.0": "https://raw.githubusercontent.com/huggingface/transformers/v5.1.0/LICENSE",
}

LICENSE_TAG = {"MIT": "mit", "Apache-2.0": "apache-2.0"}


def _license_text(license_id: str) -> str:
    import urllib.request

    with urllib.request.urlopen(LICENSE_URLS[license_id]) as response:
        return response.read().decode("utf-8")


def _readme(name: str, repo: str, revision: str, license_id: str, description: str) -> str:
    is_probe = "-cls-" in name
    task = "video-classification" if is_probe else "feature-extraction"
    return f"""---
license: {LICENSE_TAG[license_id]}
library_name: libreyolo
pipeline_tag: {task}
tags:
  - libreyolo
  - vjepa2
  - video
base_model: {repo}
---

# {name}

{description}, converted for [LibreYOLO](https://github.com/LibreYOLO/libreyolo).

```python
from libreyolo import LibreYOLO

model = LibreYOLO("{name}.pt")
result = model.predict("clip.mp4")
```

## Source

Converted from [`{repo}`](https://huggingface.co/{repo}) at revision
`{revision}`.

## Modifications

The upstream checkpoint is remapped into LibreYOLO's native V-JEPA 2 module
and wrapped with LibreYOLO v1.0 checkpoint metadata (family, size, task,
dataset variant, clip geometry, preprocessing and the pooling rule). Tensor
values are unchanged: the conversion is a key remap, loaded strictly, and the
self-supervised `predictor` tower is dropped as a named, asserted set rather
than by substring match.

Parity against unmodified `transformers==5.1.0` on float32 CPU is exact
(`max_abs_diff == 0.0`){" for single-view logits" if is_probe else
 " for both the full final token tensor and the mean-pooled, L2-normalized vector"}.

## Embedding contract

{"This probe applies the exact upstream three-layer attentive pooler and linear classifier. One temporal view is used by default; the published multi-view accuracy is NOT claimed from single-view inference." if is_probe else
 "The public embedding is a LibreYOLO pooling contract: the arithmetic mean of the final encoder tokens, L2-normalized. Upstream designates no global retrieval vector, and no retrieval benchmark is claimed for it. The native spatiotemporal token grid is available separately via `model.embed_tokens(...)`."}

V-JEPA 2 is trained on video. An image is accepted as a single-frame input,
which is a static appearance representation, not a motion one.

## License

These weights are **{license_id}**, inherited from the source checkpoint above.
The full text is in `LICENSE`, with attribution in `NOTICE`. Licences in this
family differ per artifact, so do not assume a family-wide licence.

{"The probe was trained on a third-party video dataset, named for provenance only. Its dataset terms are not the terms of these weights, and LibreYOLO does not mirror or auto-download it." if is_probe else ""}
"""


def _notice(name: str, repo: str, revision: str, license_id: str) -> str:
    return f"""{name}
{"=" * len(name)}

Converted by the LibreYOLO project from:

    {repo}
    revision {revision}
    weight license: {license_id}

Original work: V-JEPA 2, Copyright (c) Meta Platforms, Inc. and affiliates.
Upstream project: https://github.com/facebookresearch/vjepa2 (MIT)

The LibreYOLO port of the V-JEPA 2 architecture is adapted from Hugging Face
Transformers v5.1.0 (Apache License 2.0), Copyright 2025 The HuggingFace Inc.
team. That Apache-2.0 code is not relicensed; see the NOTICE file in
libreyolo/models/vjepa2/ in the LibreYOLO source tree.

This conversion changes checkpoint packaging only. Tensor values are
unchanged.
"""


def stage(stage_dir: Path, gitattributes: Path, weights_dir: Path) -> list[Path]:
    staged = []
    for name, (repo, revision, license_id, description) in sorted(ARTIFACTS.items()):
        weight = weights_dir / f"{name}.pt"
        if not weight.exists():
            print(f"[skip] {name}: {weight} not converted yet")
            continue
        out = stage_dir / name
        out.mkdir(parents=True, exist_ok=True)

        shutil.copy(gitattributes, out / ".gitattributes")
        (out / "README.md").write_text(
            _readme(name, repo, revision, license_id, description), encoding="utf-8"
        )
        (out / "LICENSE").write_text(_license_text(license_id), encoding="utf-8")
        (out / "NOTICE").write_text(
            _notice(name, repo, revision, license_id), encoding="utf-8"
        )
        target = out / f"{name}.pt"
        if not target.exists():
            # Hardlink rather than copy: these artifacts are 1-6 GB each and a
            # staging copy of all eight would need tens of GB for no benefit.
            try:
                os.link(weight, target)
            except OSError:
                shutil.copy(weight, target)

        files = sorted(p.name for p in out.iterdir())
        expected = sorted([".gitattributes", "README.md", "LICENSE", "NOTICE", f"{name}.pt"])
        if files != expected:
            raise SystemExit(f"[{name}] 5-file contract violated: {files}")
        print(f"[ok] staged {name} ({license_id}): {files}")
        staged.append(out)
    return staged


def upload(staged: list[Path]) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    for path in staged:
        repo_id = f"{ORG}/{path.name}"
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(path), repo_id=repo_id, repo_type="model")
        print(f"[ok] uploaded https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--gitattributes",
        type=Path,
        required=True,
        help="canonical .gitattributes copied from an existing LibreYOLO weight repo",
    )
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    staged = stage(args.stage_dir, args.gitattributes, args.weights_dir)
    if args.upload:
        upload(staged)
    else:
        print(f"\nStaged {len(staged)} repo(s). Re-run with --upload to push.")


if __name__ == "__main__":
    main()
