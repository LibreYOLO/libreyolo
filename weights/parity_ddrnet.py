"""Developer-only DDRNet parity harness: LibreYOLO port vs the official code.

Builds the official ``DualResNet`` from a checkout of
https://github.com/ydhongHIT/DDRNet at the pinned commit, loads the official
Cityscapes checkpoints into it and into ``LibreDDRNetNet``, and compares the
main logits in eval mode and the (main, auxiliary) pair in train mode. Also
checks that the official ImageNet backbones load into every size with exactly
the expected key difference, and runs every size (including DDRNet-39, which
has no released Cityscapes checkpoint) on shared random weights. Exits
non-zero on any mismatch.

The upstream code is not a LibreYOLO dependency:

    DDRNET_OFFICIAL_REPO=/path/to/DDRNet \
    DDRNET_OFFICIAL_CKPT_DIR=/path/to/checkpoints \
    python weights/parity_ddrnet.py [--image street.png]

``DDRNET_OFFICIAL_CKPT_DIR`` holds any of the official files under any name;
they are recognized by SHA-256. Gate: ``max_abs_diff == 0.0`` on every output.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from _conversion_utils import add_repo_root_to_path

add_repo_root_to_path()

from libreyolo.models.ddrnet.convert import (  # noqa: E402
    CITYSCAPES_SOURCES,
    IMAGENET_BACKBONES,
    UPSTREAM_COMMIT,
    convert_upstream,
    imagenet_backbone_state,
    sha256_file,
)
from libreyolo.models.ddrnet.nn import ARCH_CONFIGS, LibreDDRNetNet  # noqa: E402

_UPSTREAM_FILES = {"s": "DDRNet_23_slim.py", "m": "DDRNet_23.py", "l": "DDRNet_39.py"}


def _upstream_module(repo: Path, size: str):
    path = repo / "segmentation" / _UPSTREAM_FILES[size]
    spec = importlib.util.spec_from_file_location(f"_ddrnet_upstream_{size}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Upstream declares SyncBatchNorm for multi-GPU training; eval semantics
    # are those of BatchNorm2d, which is what a single-process check can run.
    module.BatchNorm2d = nn.BatchNorm2d
    return module


def _build_upstream(repo: Path, size: str, augment: bool) -> nn.Module:
    cfg = ARCH_CONFIGS[size]
    module = _upstream_module(repo, size)
    return module.DualResNet(
        module.BasicBlock,
        list(cfg.layers),
        num_classes=19,
        planes=cfg.planes,
        spp_planes=cfg.spp_planes,
        head_planes=cfg.head_planes,
        augment=augment,
    )


def _check(label: str, a: torch.Tensor, b: torch.Tensor, failures: list) -> None:
    if a.shape != b.shape:
        failures.append(f"{label}: shape {tuple(a.shape)} != {tuple(b.shape)}")
        print(f"  {label}: SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
        return
    diff = (a.double() - b.double()).abs().max().item()
    status = "OK" if diff == 0.0 else "FAIL"
    print(f"  {label}: max_abs_diff={diff:.3e}  {status}")
    if diff != 0.0:
        failures.append(f"{label}: max_abs_diff={diff}")


def _fixtures(image: Path | None) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    fixtures = {"rand_1024x2048": torch.rand(1, 3, 1024, 2048)}
    if image is not None:
        from PIL import Image

        from libreyolo.models.ddrnet.model import preprocess_numpy

        rgb = np.asarray(Image.open(image).convert("RGB"))
        chw, _ = preprocess_numpy(rgb, (1024, 2048))
        fixtures[f"image:{image.name}"] = torch.from_numpy(chw)[None]
    return fixtures


def _standardize(ours: LibreDDRNetNet, x01: torch.Tensor) -> torch.Tensor:
    # Upstream standardizes in its data pipeline; ours inside forward. Feed
    # upstream the identical expression so the comparison is exact.
    return (x01 - ours.pixel_mean) / ours.pixel_std


def run_cityscapes(repo: Path, path: Path, size: str, fixtures: dict, failures: list) -> None:
    print(f"\n=== {size}: {CITYSCAPES_SOURCES[size]['name']} ({path.name}) ===")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    upstream_state = {k[len("model.") :]: v for k, v in raw.items() if k.startswith("model.")}

    upstream = _build_upstream(repo, size, augment=True)
    upstream.load_state_dict(upstream_state, strict=True)
    ours = LibreDDRNetNet(size=size, num_classes=19)
    ours.load_state_dict(convert_upstream(raw), strict=True)

    with torch.no_grad():
        for name, x01 in fixtures.items():
            upstream.eval()
            ours.eval()
            up_main, _ = upstream(_standardize(ours, x01))
            _check(f"eval/{name}/main", up_main, ours(x01), failures)
            if size == "s":
                pred_up = up_main.argmax(1)
                pred_ours = ours(x01).argmax(1)
                _check(f"eval/{name}/argmax", pred_up.float(), pred_ours.float(), failures)

        # Train-mode BN uses batch statistics; both nets see the same batch.
        x01 = torch.rand(2, 3, 512, 1024)
        upstream.train()
        ours.train()
        up_main, up_aux = upstream(_standardize(ours, x01))
        our_main, our_aux = ours(x01)
        _check("train/main", up_main, our_main, failures)
        _check("train/aux", up_aux, our_aux, failures)


def run_imagenet(path: Path, size: str, failures: list) -> None:
    print(f"\n=== {size}: ImageNet backbone {IMAGENET_BACKBONES[size]['filename']} ({path.name}) ===")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    ours = LibreDDRNetNet(size=size, num_classes=19)
    try:
        state = imagenet_backbone_state(raw, set(ours.state_dict()))
    except RuntimeError as exc:
        failures.append(f"imagenet/{size}: {exc}")
        print(f"  FAIL {exc}")
        return
    ours.load_state_dict(state, strict=False)
    loaded = ours.state_dict()
    same = all(torch.equal(loaded[k], v) for k, v in state.items())
    print(f"  loaded {len(state)}/{len(loaded)} tensors; values identical: {same}")
    if not same:
        failures.append(f"imagenet/{size}: tensor values changed on load")


def run_graph(repo: Path, size: str, failures: list) -> None:
    """Same tensors in both graphs; covers sizes without a released checkpoint."""
    print(f"\n=== {size}: graph parity on shared random weights ===")
    torch.manual_seed(1)
    ours = LibreDDRNetNet(size=size, num_classes=19)
    with torch.no_grad():
        for name, buffer in ours.named_buffers():
            if name.endswith("running_mean"):
                buffer.normal_(0.0, 0.1)
            elif name.endswith("running_var"):
                buffer.uniform_(0.5, 1.5)
    upstream = _build_upstream(repo, size, augment=True)
    upstream.load_state_dict(ours.state_dict(), strict=True)
    with torch.no_grad():
        x01 = torch.rand(1, 3, 1024, 2048)
        upstream.eval()
        ours.eval()
        up_main, _ = upstream(_standardize(ours, x01))
        _check("eval/main", up_main, ours(x01), failures)
        x01 = torch.rand(2, 3, 512, 1024)
        upstream.train()
        ours.train()
        up_main, up_aux = upstream(_standardize(ours, x01))
        our_main, our_aux = ours(x01)
        _check("train/main", up_main, our_main, failures)
        _check("train/aux", up_aux, our_aux, failures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None, help="optional street image fixture")
    args = parser.parse_args()

    repo = os.environ.get("DDRNET_OFFICIAL_REPO")
    ckpt_dir = os.environ.get("DDRNET_OFFICIAL_CKPT_DIR")
    if not repo or not ckpt_dir:
        print("Set DDRNET_OFFICIAL_REPO and DDRNET_OFFICIAL_CKPT_DIR.", file=sys.stderr)
        return 2
    repo_path, ckpt_path = Path(repo), Path(ckpt_dir)
    print(f"upstream: {repo_path} (expected commit {UPSTREAM_COMMIT})")

    digests = {}
    for path in sorted(ckpt_path.glob("*.pth")):
        digests[sha256_file(path)] = path
    fixtures = _fixtures(args.image)
    failures: list[str] = []
    ran = 0
    for size, source in CITYSCAPES_SOURCES.items():
        path = digests.get(source["sha256"])
        if path is None:
            print(f"\n=== {size}: official Cityscapes checkpoint not found, skipped ===")
            continue
        run_cityscapes(repo_path, path, size, fixtures, failures)
        ran += 1
    for size, spec in IMAGENET_BACKBONES.items():
        path = digests.get(spec["sha256"])
        if path is None:
            print(f"\n=== {size}: official ImageNet backbone not found, skipped ===")
            continue
        run_imagenet(path, size, failures)
        ran += 1
    for size in ARCH_CONFIGS:
        run_graph(repo_path, size, failures)

    if not ran:
        print("No official checkpoint found.", file=sys.stderr)
        return 2
    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nAll parity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
