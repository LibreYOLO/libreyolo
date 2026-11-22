"""Build or publish Apple Depth Pro 2 five-file Hugging Face weight repos.

Official Depth Pro 2 checkpoints (``apple/DepthPro2-{S,B,L,H}`` and
``apple/DepthPro2-focal``) are licensed under the Apple Machine Learning
Research Model License Agreement (``apple-amlr``), which explicitly permits
redistribution exclusively for non-commercial Research Purposes when the
license text, required attribution sentence, and modification disclosure
travel with the redistributed weights.

Each published repo contains exactly the 5-file LibreYOLO contract:

- ``.gitattributes``
- ``README.md`` (``license: other`` + ``apple-amlr`` + non-commercial banner)
- ``LICENSE`` (verbatim ``libreyolo/models/depthpro2/LICENSE_MODEL``)
- ``NOTICE`` (required Apple attribution + pinned upstream provenance)
- ``LibreDepthPro2{s,b,l,h}-depth.pt`` (bundled with ``fov.*`` + v1.0 metadata)

Example::

    python weights/upload_depthpro2_hf.py --sizes s --out ./stage --dry-run
    python weights/upload_depthpro2_hf.py --sizes s b l h --in-memory \\
        --confirm-amlr-license --token-file /path/to/hf_token
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from libreyolo.models.depthpro2.convert import (
    FOV_UPSTREAM_SHA256,
    FOV_UPSTREAM_URL,
    UPSTREAM_SHA256,
    UPSTREAM_URLS,
    merge_fov_state_dict,
    wrap_upstream_state_dict,
)
from libreyolo.utils.serialization import validate_checkpoint_metadata

_UPSTREAM_CODE_REPO = "apple-aiml-research/ml-depth-pro-2"
_UPSTREAM_CODE_COMMIT = "aa389cbb0b74341664ed663a0966c859905cd6c0"
_COLLECTION = "LibreYOLO/libreyolo-models-698875bf2b5f695708415169"
_LICENSE_NAME = "apple-amlr"
_LICENSE_LINK = (
    f"https://github.com/{_UPSTREAM_CODE_REPO}/blob/"
    f"{_UPSTREAM_CODE_COMMIT}/LICENSE_MODEL"
)
_FOV_REPO = "apple/DepthPro2-focal"
_FOV_REVISION = "622659d44a2002cf9ae4333f2569afa938290d9a"

_SPECS: dict[str, dict[str, str]] = {
    "s": {
        "upstream_repo": "apple/DepthPro2-S",
        "upstream_revision": "903a1c1ade918831d138adcc4aedbcc8f4da0078",
        "backbone": "DINOv3 ViT-S/16",
    },
    "b": {
        "upstream_repo": "apple/DepthPro2-B",
        "upstream_revision": "85e67fdfb7f5b36a6f6c4495e6789d331b1d8a04",
        "backbone": "DINOv3 ViT-B/16",
    },
    "l": {
        "upstream_repo": "apple/DepthPro2-L",
        "upstream_revision": "849dc5461267e0bc006ec63ab46920ea1ed98702",
        "backbone": "DINOv3 ViT-L/16",
    },
    "h": {
        "upstream_repo": "apple/DepthPro2-H",
        "upstream_revision": "68426c789f75a7c760273e857222625b4fe69576",
        "backbone": "DINOv3 ViT-H+/16",
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


def _canonical_name(size: str) -> str:
    return f"LibreDepthPro2{size}-depth"


def _license_text() -> str:
    path = _REPO_ROOT / "libreyolo" / "models" / "depthpro2" / "LICENSE_MODEL"
    return path.read_text(encoding="utf-8")


def _notice_text(size: str) -> str:
    spec = _SPECS[size]
    name = _canonical_name(size)
    return f"""{name} weights
{"-" * (len(name) + 8)}

Apple Machine Learning Research Model is licensed under the Apple Machine
Learning Research Model License Agreement.

Copyright (C) 2026 Apple Inc. All Rights Reserved.

Source checkpoints
------------------
- Depth model: https://huggingface.co/{spec["upstream_repo"]}
  Revision: {spec["upstream_revision"]}
  File: model.safetensors
  SHA-256: {UPSTREAM_SHA256[size]}
- Focal length estimator: https://huggingface.co/{_FOV_REPO}
  Revision: {_FOV_REVISION}
  File: model.safetensors
  SHA-256: {FOV_UPSTREAM_SHA256}
- Upstream reference code: https://github.com/{_UPSTREAM_CODE_REPO}
  Commit: {_UPSTREAM_CODE_COMMIT}

Modifications
-------------
This file is a repackaged checkpoint for LibreYOLO. The learned tensors of
{spec["upstream_repo"]} and {_FOV_REPO} are completely unchanged (bitwise
identical to upstream). The only modifications are:

1. Bundling the {_FOV_REPO} state dict under the ``fov.*`` key prefix in the
   same checkpoint file as the depth model state dict.
2. Wrapping the combined state dict with LibreYOLO checkpoint v1.0 metadata
   (``model_family="depthpro2"``, ``size="{size}"``, ``task="depth"``).

See ``weights/convert_depthpro2_weights.py`` and
``weights/upload_depthpro2_hf.py`` in https://github.com/LibreYOLO/libreyolo.

License & non-endorsement
-------------------------
These weights are governed exclusively by the Apple Machine Learning Research
Model License Agreement (see the verbatim ``LICENSE`` file in this repository),
which restricts use and redistribution to non-commercial Research Purposes
only. Downstream users are solely responsible for complying with those terms.
Apple's name, trademarks, service marks, and logos are not used to endorse or
promote LibreYOLO or any derivative works.
"""


def _readme_text(size: str) -> str:
    spec = _SPECS[size]
    name = _canonical_name(size)
    return f"""---
license: other
license_name: {_LICENSE_NAME}
license_link: {_LICENSE_LINK}
library_name: libreyolo
pipeline_tag: depth-estimation
base_model:
  - {spec["upstream_repo"]}
  - {_FOV_REPO}
tags:
  - depth-estimation
  - video-depth
  - metric-depth
  - depth-pro-2
  - non-commercial
  - research-only
---

# {name}

Apple Depth Pro 2 ({spec["backbone"]}) bundled with Apple's companion focal-length
estimator (`DepthPro2-focal`) for metric single-image and streaming video depth
estimation, repackaged in [LibreYOLO](https://github.com/LibreYOLO/libreyolo)
checkpoint format.

> ## NON-COMMERCIAL RESEARCH-ONLY WEIGHTS
>
> These weights are **not** covered by LibreYOLO's MIT license.
>
> **Apple Machine Learning Research Model is licensed under the Apple Machine
> Learning Research Model License Agreement.**
>
> Under the [Apple Machine Learning Research Model License Agreement](./LICENSE)
> (`apple-amlr`, Copyright (C) 2026 Apple Inc. All Rights Reserved), use,
> modification, and redistribution of these pretrained weights are limited
> exclusively to **non-commercial Research Purposes** (scientific research and
> academic development; no commercial exploitation, product development, or use
> in any commercial product or service). Downstream users are solely responsible
> for complying with the license terms.

## Source

- **Depth checkpoint**: [{spec["upstream_repo"]}](https://huggingface.co/{spec["upstream_repo"]})
  `model.safetensors` at pinned revision `{spec["upstream_revision"]}`
  (SHA-256: `{UPSTREAM_SHA256[size]}`).
- **Focal-length checkpoint**: [{_FOV_REPO}](https://huggingface.co/{_FOV_REPO})
  `model.safetensors` at pinned revision `{_FOV_REVISION}`
  (SHA-256: `{FOV_UPSTREAM_SHA256}`).
- **Upstream reference code**: [{_UPSTREAM_CODE_REPO}](https://github.com/{_UPSTREAM_CODE_REPO}/tree/{_UPSTREAM_CODE_COMMIT})
  at commit `{_UPSTREAM_CODE_COMMIT}` (Apple Sample Code License).

## Modifications

None to the learned parameters: every tensor from `{spec["upstream_repo"]}` and
`{_FOV_REPO}` is preserved unchanged (bit-identical inference parity verified
against upstream). The only modifications are bundling the `{_FOV_REPO}` state
dict under the `fov.*` key prefix alongside the depth state dict and adding
LibreYOLO checkpoint v1.0 metadata (`task="depth"`).

See `weights/convert_depthpro2_weights.py` and `weights/upload_depthpro2_hf.py`
in the [LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo).

## Usage

```python
from libreyolo import LibreYOLO

model = LibreYOLO("{name}.pt")
result = model.predict("image.jpg")
depth = result.depth_map            # metric depth in metres (inf on sky pixels)
K = depth.intrinsics                # (3, 3) OpenCV pixel intrinsics
```

## License

Apple Machine Learning Research Model License Agreement (`apple-amlr`).
Apple Machine Learning Research Model is licensed under the Apple Machine
Learning Research Model License Agreement. See the verbatim
[`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE) files in this repository.
"""


def _download_bytes(url: str, expected_sha256: str, label: str) -> bytes:
    print(f"Downloading {label} from {url} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "libreyolo-uploader/1.0"})
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    with urllib.request.urlopen(req) as resp:
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
            total += len(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {label}: expected {expected_sha256}, got {actual}."
        )
    print(f"Verified {label}: {total / 1e6:.1f} MB (sha256={actual[:16]}...)", flush=True)
    return b"".join(chunks)


def _load_safetensors_bytes(raw: bytes) -> dict:
    from safetensors.torch import load as load_safetensors_bytes

    return load_safetensors_bytes(raw)


def _build_pt_bytes(
    size: str,
    fov_sd: dict,
    local_depth_path: Path | None = None,
) -> tuple[bytes, str]:
    import torch

    if local_depth_path is not None:
        raw = local_depth_path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != UPSTREAM_SHA256[size]:
            raise ValueError(
                f"Local depth file SHA-256 mismatch: expected {UPSTREAM_SHA256[size]}, got {actual}."
            )
    else:
        raw = _download_bytes(UPSTREAM_URLS[size], UPSTREAM_SHA256[size], f"DepthPro2-{size.upper()}")

    depth_sd = _load_safetensors_bytes(raw)
    del raw
    merged = merge_fov_state_dict(depth_sd, fov_sd)
    del depth_sd
    wrapped = wrap_upstream_state_dict(merged, size)
    del merged

    validate_checkpoint_metadata(wrapped)
    assert wrapped["model_family"] == "depthpro2"
    assert wrapped["size"] == size
    assert wrapped["task"] == "depth"

    buf = io.BytesIO()
    torch.save(wrapped, buf)
    del wrapped
    pt_bytes = buf.getvalue()
    buf.close()

    pt_sha256 = hashlib.sha256(pt_bytes).hexdigest()
    print(
        f"Built {_canonical_name(size)}.pt: {len(pt_bytes) / 1e6:.1f} MB "
        f"(sha256={pt_sha256})",
        flush=True,
    )
    return pt_bytes, pt_sha256


def _upload_one(
    size: str,
    pt_bytes: bytes,
    token: str,
    *,
    add_to_collection: bool = True,
) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    name = _canonical_name(size)
    repo_id = f"LibreYOLO/{name}"

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    operations = [
        CommitOperationAdd(
            path_in_repo=".gitattributes",
            path_or_fileobj=_GITATTRIBUTES.encode("utf-8"),
        ),
        CommitOperationAdd(
            path_in_repo="LICENSE",
            path_or_fileobj=_license_text().encode("utf-8"),
        ),
        CommitOperationAdd(
            path_in_repo="NOTICE",
            path_or_fileobj=_notice_text(size).encode("utf-8"),
        ),
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=_readme_text(size).encode("utf-8"),
        ),
        CommitOperationAdd(
            path_in_repo=f"{name}.pt",
            path_or_fileobj=io.BytesIO(pt_bytes),
        ),
    ]
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=f"Initial upload of {name} (Apple Depth Pro 2 + focal)",
    )
    if add_to_collection:
        from huggingface_hub.errors import HfHubHTTPError

        try:
            api.add_collection_item(
                collection_slug=_COLLECTION,
                item_id=repo_id,
                item_type="model",
                exists_ok=True,
            )
        except HfHubHTTPError as exc:  # pragma: no cover - collection scope may differ
            print(f"Warning: could not add {repo_id} to collection: {exc}", flush=True)

    print(f"Uploaded {repo_id} @ {commit.oid}", flush=True)
    return str(commit.oid)


def _write_local_stage(out_dir: Path, size: str, pt_bytes: bytes) -> None:
    name = _canonical_name(size)
    repo_dir = out_dir / name
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / ".gitattributes").write_text(_GITATTRIBUTES, encoding="utf-8")
    (repo_dir / "LICENSE").write_text(_license_text(), encoding="utf-8")
    (repo_dir / "NOTICE").write_text(_notice_text(size), encoding="utf-8")
    (repo_dir / "README.md").write_text(_readme_text(size), encoding="utf-8")
    (repo_dir / f"{name}.pt").write_bytes(pt_bytes)
    print(f"Staged 5-file payload at {repo_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=("s", "b", "l", "h"),
        default=["s", "b", "l", "h"],
        help="Model sizes to convert and upload.",
    )
    parser.add_argument(
        "--fov-path",
        type=Path,
        default=None,
        help="Optional local copy of apple/DepthPro2-focal model.safetensors.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory to stage local 5-file payloads (used with --dry-run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate locally without uploading to Hugging Face.",
    )
    parser.add_argument(
        "--confirm-amlr-license",
        action="store_true",
        help="Confirm compliance with Apple Machine Learning Research Model License (apple-amlr).",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=None,
        help="Optional path to a file containing the LibreYOLO HF token.",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.confirm_amlr_license:
        raise SystemExit(
            "Refusing to upload without --confirm-amlr-license "
            "(verifies non-commercial apple-amlr terms, verbatim LICENSE_MODEL, "
            "and required Apple attribution notice)."
        )

    token = None
    if not args.dry_run:
        if args.token_file is not None:
            token = args.token_file.read_text(encoding="utf-8").strip()
        else:
            from huggingface_hub import get_token

            token = get_token()
        if not token:
            raise SystemExit("No Hugging Face token found; pass --token-file or login first.")

    if args.fov_path is not None:
        fov_raw = args.fov_path.read_bytes()
        fov_actual = hashlib.sha256(fov_raw).hexdigest()
        if fov_actual != FOV_UPSTREAM_SHA256:
            raise ValueError(
                f"Local focal file SHA-256 mismatch: expected {FOV_UPSTREAM_SHA256}, got {fov_actual}."
            )
    else:
        fov_raw = _download_bytes(FOV_UPSTREAM_URL, FOV_UPSTREAM_SHA256, "DepthPro2-focal")
    fov_sd = _load_safetensors_bytes(fov_raw)
    del fov_raw

    summary: dict[str, dict[str, str]] = {}
    for size in args.sizes:
        pt_bytes, pt_sha256 = _build_pt_bytes(size, fov_sd)
        entry: dict[str, str] = {"sha256": pt_sha256}
        if args.dry_run:
            if args.out is not None:
                _write_local_stage(args.out, size, pt_bytes)
        else:
            assert token is not None
            oid = _upload_one(size, pt_bytes, token)
            entry["commit"] = oid
        del pt_bytes
        summary[size] = entry

    print("\n=== Depth Pro 2 HF Mirror Summary ===", flush=True)
    for size, info in summary.items():
        print(f"  {size}: sha256={info['sha256']} commit={info.get('commit', 'dry-run')}", flush=True)


if __name__ == "__main__":
    main()
