"""Build or publish Dome-DETR's five-file Hugging Face weight repos.

The upstream weight card is internally inconsistent: it has no license
metadata or LICENSE file, calls the project Apache-2.0, and separately says
the weights are for academic research purposes only. The LibreYOLO maintainer
approved mirroring on the disclosed interpretation documented here: the
Apache statement is the redistribution basis and the academic-only sentence
is preserved as the controlling use restriction.

A real upload requires ``--confirm-academic-license``. ``--dry-run`` only
builds and validates a local five-file payload.

Example::

    python weights/upload_domedetr_hf.py --size s --variant aitod \
        --pt LibreDOMEDETRs-aitod.pt --out ./LibreDOMEDETRs-aitod --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_UPSTREAM_MODEL_REPO = "RicePasteM/Dome-DETR"
_UPSTREAM_MODEL_REVISION = "530230620d1f3261a267d462989cddf204cc6e10"
_UPSTREAM_CODE_COMMIT = "2dde3bc1946a3e9fad9abd0612b59fc39bd6b861"
_COLLECTION = "LibreYOLO/libreyolo-models-698875bf2b5f695708415169"
_LICENSE_NAME = "dome-detr-academic-research-only"
_LICENSE_LINK = (
    f"https://huggingface.co/{_UPSTREAM_MODEL_REPO}/blob/"
    f"{_UPSTREAM_MODEL_REVISION}/README.md"
)
_WEIGHT_TERMS = (
    "Academic research purposes only (upstream model card); the same card also "
    "claims Apache-2.0. See weights/LICENSE_NOTICE.txt."
)

_SPECS = {
    ("s", "aitod"): {
        "upstream_file": "aitod-s-best.pth",
        "source_sha256": "1b10ac2c78a83363b9577ddd74d1001ac8414a479476fea22138fc769a016ba8",
        "dataset_name": "AI-TOD-V2",
        "nc": 9,
    },
    ("m", "aitod"): {
        "upstream_file": "aitod-m-best.pth",
        "source_sha256": "d0242732410ac329a441e2a39b484bbfbb5f7196170cc0379a761ac0cf835d6b",
        "dataset_name": "AI-TOD-V2",
        "nc": 9,
    },
    ("l", "aitod"): {
        "upstream_file": "aitod-l-best.pth",
        "source_sha256": "08525b4bd445e1f193bbc2bb252ddae0f144f542912a95a38111c9a4b9a53cbe",
        "dataset_name": "AI-TOD-V2",
        "nc": 9,
    },
    ("s", "visdrone"): {
        "upstream_file": "dome-s-visdrone_converted.pth",
        "source_sha256": "42874302fd97e17c2a23a46a8ec35d2ea4d2ffe479f0d632c1327fafbf910eb7",
        "dataset_name": "VisDrone",
        "nc": 12,
    },
    ("m", "visdrone"): {
        "upstream_file": "dome-m-visdrone_converted.pth",
        "source_sha256": "3a0d1db3c68fac5239e1b43b4c6f765b1330e6961070113f4ea39dbdbf8846a6",
        "dataset_name": "VisDrone",
        "nc": 12,
    },
    ("l", "visdrone"): {
        "upstream_file": "dome-l-visdrone_converted.pth",
        "source_sha256": "e230db14242f95097efa8f19e9420e17f9fe7a61c4e6d4f7f1881d48f6789039",
        "dataset_name": "VisDrone",
        "nc": 12,
    },
}

_GITATTRIBUTES = """*.7z filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.bz2 filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.ftz filter=lfs diff=lfs merge=lfs -text
*.gz filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.lfs.* filter=lfs diff=lfs merge=lfs -text
*.mlmodel filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.rar filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
saved_model/**/* filter=lfs diff=lfs merge=lfs -text
*.tar.* filter=lfs diff=lfs merge=lfs -text
*.tar filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.tgz filter=lfs diff=lfs merge=lfs -text
*.wasm filter=lfs diff=lfs merge=lfs -text
*.xz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.zst filter=lfs diff=lfs merge=lfs -text
*tfevents* filter=lfs diff=lfs merge=lfs -text
"""


def _canonical_name(size: str, variant: str) -> str:
    return f"LibreDOMEDETR{size}-{variant}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(spec: dict) -> str:
    return f"best_ckpts_dome_2026/{spec['upstream_file']}"


def _license_text() -> str:
    apache = (_REPO_ROOT / "licenses" / "Apache-2.0.txt").read_text(
        encoding="utf-8"
    ).rstrip()
    return f"""Dome-DETR pretrained checkpoint terms and provenance
====================================================

Upstream weight repository:
https://huggingface.co/{_UPSTREAM_MODEL_REPO}
Revision: {_UPSTREAM_MODEL_REVISION}

The upstream weight repository has no standalone LICENSE file and its Hugging
Face card metadata has no `license` field. Its README contains both of the
following statements (reproduced verbatim):

  - The weight files in this repository are for academic research purposes only
  - This project is licensed under the Apache 2.0 license.

At the pinned revision, the second statement links to a `LICENSE` file that is
not present in the weight repository. The Dome-DETR source-code repository does
contain the standard Apache License 2.0 text reproduced below.

LibreYOLO's maintainer approved mirroring these checkpoints by treating the
upstream Apache statement as the redistribution basis while preserving
"academic research purposes only" as the controlling use restriction. This is
a disclosed LibreYOLO interpretation, not an upstream clarification and not a
claim that the two upstream statements are legally consistent.

THESE PRETRAINED WEIGHTS ARE FOR ACADEMIC RESEARCH PURPOSES ONLY. THEY ARE NOT
COVERED BY LIBREYOLO'S MIT LICENSE AND MUST NOT BE TREATED AS COMMERCIALLY
CLEARED. DOWNSTREAM USERS ARE RESPONSIBLE FOR REVIEWING AND COMPLYING WITH THE
UPSTREAM TERMS.

The complete Apache License 2.0 text from the pinned Dome-DETR source-code
repository follows for the full context of the upstream Apache statement.

{apache}
"""


def _readme(size: str, variant: str, pt_path: Path) -> str:
    spec = _SPECS[(size, variant)]
    name = _canonical_name(size, variant)
    source_path = _source_path(spec)
    source_url = (
        f"https://huggingface.co/{_UPSTREAM_MODEL_REPO}/blob/"
        f"{_UPSTREAM_MODEL_REVISION}/{source_path}"
    )
    return f"""---
license: other
license_name: {_LICENSE_NAME}
license_link: {_LICENSE_LINK}
library_name: libreyolo
pipeline_tag: object-detection
tags:
  - object-detection
  - dome-detr
  - tiny-object-detection
  - {variant}
  - academic-research-only
  - non-commercial
---

# {name}

Dome-DETR {size.upper()} trained on {spec['dataset_name']} ({spec['nc']} output
classes), converted for LibreYOLO at 800x800.

> ## ACADEMIC-RESEARCH-ONLY WEIGHTS
>
> The upstream model card says these weight files are **for academic research
> purposes only**. They are **not covered by LibreYOLO's MIT license** and must
> not be treated as commercially cleared.
>
> The same upstream card also says the project is Apache-2.0, but it has no
> license metadata and links to a LICENSE file that does not exist in the
> weight repository. LibreYOLO's maintainer approved this mirror by treating
> the Apache statement as the redistribution basis and preserving the stricter
> academic-only sentence as the use restriction. This is LibreYOLO's disclosed
> interpretation, not clarification from the Dome-DETR authors. Review the
> pinned upstream card and the [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE)
> in this repository before use.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("{name}.pt")
results = model.predict("image.jpg")
```

## Source

Official upstream weight file:
[`{source_path}`]({source_url})

- Upstream model revision: `{_UPSTREAM_MODEL_REVISION}`
- Source SHA-256: `{spec['source_sha256']}`
- Converted SHA-256: `{_sha256(pt_path)}`

The Dome-DETR architecture source is Apache-2.0 at commit
[`{_UPSTREAM_CODE_COMMIT}`](https://github.com/RicePasteM/Dome-DETR/commit/{_UPSTREAM_CODE_COMMIT}).
Copyright (c) 2025 The Dome-DETR Authors, as stated in its source headers.
That source-code license and LibreYOLO's MIT code license do not remove the
academic-only restriction stated for these pretrained weights.

## Modifications

LibreYOLO adds checkpoint-schema metadata, including the dataset variant,
class names, pinned source revision, and source SHA-256. State-dict keys and
learned tensors are unchanged. The converted checkpoint loads strictly into
LibreYOLO's native implementation. See
[`weights/convert_domedetr_weights.py`](https://github.com/LibreYOLO/libreyolo/blob/dev/weights/convert_domedetr_weights.py).

## License and terms

**Academic research purposes only**, following the stricter statement on the
upstream weight card. The upstream Apache-2.0 statement and its missing weight
repository LICENSE file are documented without omission. See
[`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
"""


def _notice(size: str, variant: str, pt_path: Path) -> str:
    spec = _SPECS[(size, variant)]
    name = _canonical_name(size, variant)
    return f"""{name} weights
{'-' * (len(name) + 8)}

This product contains the official Dome-DETR {size.upper()} checkpoint trained
on {spec['dataset_name']} and published by the Dome-DETR authors at:
https://huggingface.co/{_UPSTREAM_MODEL_REPO}

Pinned model revision: {_UPSTREAM_MODEL_REVISION}
Upstream file: {_source_path(spec)}
Source SHA-256: {spec['source_sha256']}
Converted SHA-256: {_sha256(pt_path)}

ACADEMIC-RESEARCH-ONLY WEIGHTS

The upstream weight repository has no license metadata and no LICENSE file.
Its README says both that the weight files are "for academic research purposes
only" and that the project is licensed under Apache 2.0; the latter links to a
missing file. LibreYOLO's maintainer approved mirroring by treating the Apache
statement as the redistribution basis while preserving the academic-only
sentence as the controlling use restriction. This is a LibreYOLO
interpretation, not an upstream clarification or a claim that those statements
are legally consistent. The complete context and Apache text are included in
this repository's LICENSE file.

These pretrained weights are NOT covered by LibreYOLO's MIT license and must
not be treated as commercially cleared. Users are responsible for reviewing
and complying with the upstream terms.

Architecture source: https://github.com/RicePasteM/Dome-DETR
Pinned code commit: {_UPSTREAM_CODE_COMMIT}
Source headers: Copyright (c) 2025 The Dome-DETR Authors.
Source-code license: Apache License 2.0.

Conversion adds LibreYOLO checkpoint metadata only. State-dict keys and learned
tensors are unchanged.
"""


def _validate_checkpoint(size: str, variant: str, path: Path) -> None:
    from libreyolo import LibreYOLO
    from libreyolo.models.domedetr.model import LibreDOMEDETR
    from libreyolo.utils.serialization import (
        load_untrusted_torch_file,
        validate_checkpoint_metadata,
    )

    spec = _SPECS[(size, variant)]
    name = _canonical_name(size, variant)
    filename = f"{name}.pt"
    if path.name != filename:
        raise ValueError(f"Expected canonical filename {filename}, got {path.name}")

    whitelist = (
        _REPO_ROOT / "skills" / "libreyolo-upload-hf-model" / "SKILL.md"
    ).read_text(encoding="utf-8")
    if filename not in whitelist:
        raise ValueError(f"Canonical filename is absent from upload whitelist: {filename}")

    expected_url = (
        f"https://huggingface.co/LibreYOLO/{name}/resolve/main/{filename}"
    )
    actual_url = LibreDOMEDETR.get_download_url(filename)
    if actual_url != expected_url:
        raise ValueError(
            f"Loader URL mismatch for {filename}: {actual_url!r} != {expected_url!r}"
        )

    checkpoint = load_untrusted_torch_file(path, context="converted checkpoint")
    errors = validate_checkpoint_metadata(checkpoint, strict=False)
    if errors:
        raise ValueError(f"Invalid checkpoint metadata: {'; '.join(errors)}")
    expected = {
        "model_family": "domedetr",
        "size": size,
        "task": "detect",
        "nc": spec["nc"],
        "imgsz": 800,
        "weight_variant": variant,
        "source_sha256": spec["source_sha256"],
        "license": _WEIGHT_TERMS,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    expected_source = (
        f"{_UPSTREAM_MODEL_REPO}@{_UPSTREAM_MODEL_REVISION}/{_source_path(spec)}"
    )
    if checkpoint.get("source") != expected_source:
        mismatches["source"] = (checkpoint.get("source"), expected_source)
    if mismatches:
        raise ValueError(f"Checkpoint metadata mismatch: {mismatches}")

    model = LibreYOLO(str(path), device="cpu")
    loaded = (
        model.family,
        model.size,
        model.nb_classes,
        model.task,
        model.weight_variant,
    )
    expected_loaded = ("domedetr", size, spec["nc"], "detect", variant)
    if loaded != expected_loaded:
        raise ValueError(
            f"Factory load mismatch: expected {expected_loaded}, got {loaded}"
        )


def build_repo_dir(size: str, variant: str, pt_path: Path, out_dir: Path) -> Path:
    """Build and validate exactly five files for one weight repository."""
    if not pt_path.is_file():
        raise FileNotFoundError(f"Weight file not found: {pt_path}")
    _validate_checkpoint(size, variant, pt_path)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / ".gitattributes").write_text(
        _GITATTRIBUTES, encoding="utf-8", newline="\n"
    )
    (out_dir / "README.md").write_text(
        _readme(size, variant, pt_path), encoding="utf-8", newline="\n"
    )
    (out_dir / "LICENSE").write_text(
        _license_text(), encoding="utf-8", newline="\n"
    )
    (out_dir / "NOTICE").write_text(
        _notice(size, variant, pt_path), encoding="utf-8", newline="\n"
    )
    name = _canonical_name(size, variant)
    shutil.copy2(pt_path, out_dir / f"{name}.pt")

    expected = {".gitattributes", "README.md", "LICENSE", "NOTICE", f"{name}.pt"}
    actual = {path.name for path in out_dir.iterdir()}
    if actual != expected:
        raise RuntimeError(
            f"Five-file contract mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )
    return out_dir


def _upload(size: str, variant: str, repo_dir: Path) -> str:
    from huggingface_hub import HfApi

    name = _canonical_name(size, variant)
    repo_id = f"LibreYOLO/{name}"
    api = HfApi()
    if api.repo_exists(repo_id=repo_id, repo_type="model"):
        raise FileExistsError(
            f"Refusing to overwrite existing Hugging Face repository: {repo_id}"
        )
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=False)
    api.upload_folder(
        folder_path=str(repo_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=(
            f"Initial upload: {name} (Dome-DETR, academic-research-only)"
        ),
    )
    api.add_collection_item(
        collection_slug=_COLLECTION,
        item_id=repo_id,
        item_type="model",
        exists_ok=True,
    )
    return f"https://huggingface.co/{repo_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", required=True, choices=["s", "m", "l"])
    parser.add_argument("--variant", required=True, choices=["aitod", "visdrone"])
    parser.add_argument("--pt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate the five files without changing Hugging Face.",
    )
    parser.add_argument(
        "--confirm-academic-license",
        action="store_true",
        help="Confirm the maintainer-approved restricted-use mirroring decision.",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.confirm_academic_license:
        parser.error("a real upload requires --confirm-academic-license")

    if (args.size, args.variant) not in _SPECS:
        parser.error(f"unsupported Dome-DETR variant: {args.size}/{args.variant}")
    repo_dir = build_repo_dir(args.size, args.variant, args.pt, args.out)
    print(f"Built five-file repository: {repo_dir}")
    if args.dry_run:
        print("Dry run complete; no external state changed.")
        return 0

    print(f"Uploaded: {_upload(args.size, args.variant, repo_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
