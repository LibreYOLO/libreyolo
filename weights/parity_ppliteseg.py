"""Developer-only PP-LiteSeg parity harness: LibreYOLO port vs pinned upstream.

Validates the upstream digest, builds the pinned SuperGradients model with
``use_aux_heads=True``, converts the same tensors through the production
mapper, and probes each semantic stage. Exits non-zero on any mismatch.

The upstream oracle is not a LibreYOLO dependency. Point ``PPLITESEG_ORACLE``
at a module exposing ``build_upstream(size, num_classes, use_aux_heads)`` --
or install ``super-gradients`` and leave it unset -- plus
``PPLITESEG_OFFICIAL_CKPT_DIR`` at the directory holding the four
``pp_lite_{t,b}_seg{50,75}_cityscapes.pth`` artifacts.

    PPLITESEG_OFFICIAL_CKPT_DIR=/path/to/upstream \
    PPLITESEG_ORACLE=/path/to/oracle.py \
    python weights/parity_ppliteseg.py

Gate: main-logit ``max_abs_diff == 0.0`` for all four checkpoints, and the same
for all three training auxiliary logits on t50 and b75.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

from _conversion_utils import add_repo_root_to_path
from convert_ppliteseg_weights import SOURCE_DIGESTS, sha256

add_repo_root_to_path()

from libreyolo.models.ppliteseg.nn import SIZE_CONFIGS, LibrePPLiteSegNet  # noqa: E402

SIZES = ("t50", "b50", "t75", "b75")
# Sizes that additionally gate on exact auxiliary-logit parity: one per backbone.
AUX_PARITY_SIZES = ("t50", "b75")

CKPT_DIR = os.environ.get("PPLITESEG_OFFICIAL_CKPT_DIR")
ORACLE_PATH = os.environ.get("PPLITESEG_ORACLE")


def _load_oracle():
    if ORACLE_PATH:
        spec = importlib.util.spec_from_file_location("_ppliteseg_oracle", ORACLE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build_upstream

    from super_gradients.training.models.segmentation_models.ppliteseg import (
        PPLiteSegB,
        PPLiteSegT,
    )
    from super_gradients.training.utils import HpmStruct

    def build_upstream(size: str, num_classes: int = 19, use_aux_heads: bool = True):
        cls = PPLiteSegT if size.startswith("t") else PPLiteSegB
        return cls(HpmStruct(num_classes=num_classes, use_aux_heads=use_aux_heads, dropout=0.0))

    return build_upstream


def _source_state(path: Path, size: str) -> dict:
    digest = sha256(path)
    if digest != SOURCE_DIGESTS[size]:
        raise SystemExit(
            f"Digest mismatch for {path}: {digest} != {SOURCE_DIGESTS[size]}. "
            "Refusing to deserialize an unverified checkpoint."
        )
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw["net"]
    return {(k[len("module.") :] if k.startswith("module.") else k): v for k, v in state.items()}


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.double() - b.double()).abs().max().item()


def _check(label: str, a: torch.Tensor, b: torch.Tensor, failures: list) -> None:
    if a.shape != b.shape:
        failures.append(f"{label}: shape {tuple(a.shape)} != {tuple(b.shape)}")
        print(f"  {label}: SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
        return
    diff = _max_abs_diff(a, b)
    status = "OK" if diff == 0.0 else "FAIL"
    print(f"  {label}: max_abs_diff={diff:.3e}  {status}")
    if diff != 0.0:
        failures.append(f"{label}: max_abs_diff={diff}")


def run_size(size: str, ckpt_dir: Path, build_upstream, failures: list) -> None:
    path = ckpt_dir / f"pp_lite_{size[0]}_seg{size[1:]}_cityscapes.pth"
    state = _source_state(path, size)
    height, width = SIZE_CONFIGS[size]["imgsz"]
    print(f"\n=== {size} ({height}x{width}) {path.name} ===")

    upstream = build_upstream(size, num_classes=19, use_aux_heads=True)
    upstream.load_state_dict(state, strict=True)
    upstream.eval()

    ours = LibrePPLiteSegNet(size=size, num_classes=19, use_aux_heads=True)
    ours.load_state_dict(state, strict=True)
    ours.eval()

    torch.manual_seed(0)
    fixtures = {
        "zeros": torch.zeros(1, 3, height, width),
        "randn": torch.rand(1, 3, height, width),
        # A real preprocessed RGB frame at the native rectangle.
        "image": _image_fixture(height, width),
    }

    for name, x01 in fixtures.items():
        # Our forward standardizes internally; feed upstream the identical
        # standardized tensor produced by the same buffers, so any difference
        # is architecture, never a re-derived normalization constant.
        x_norm = ours.normalize(x01)
        with torch.no_grad():
            up_out = upstream(x_norm)
            our_main = ours(x01)
        up_main = up_out[0] if isinstance(up_out, (tuple, list)) else up_out
        _check(f"{name}/main_logits", our_main, up_main, failures)

    # Stage probes on the seeded tensor: backbone features, projections, SPPM,
    # each UAFM output. A stage-level mismatch localizes a logit mismatch.
    x01 = fixtures["randn"]
    x_norm = ours.normalize(x01)
    with torch.no_grad():
        up_backbone = upstream.encoder.backbone(x_norm)
        our_backbone = ours.encoder.backbone(x_norm)
        for index, (a, b) in enumerate(zip(our_backbone, up_backbone)):
            _check(f"backbone/stride{8 << index}", a, b, failures)

        up_feats = upstream.encoder(x_norm)
        our_feats = ours.encoder(x_norm)
        for index in range(len(our_feats) - 1):
            _check(f"encoder/proj{index}", our_feats[index], up_feats[index], failures)
        _check("encoder/sppm", our_feats[-1], up_feats[-1], failures)

        our_x = our_feats[::-1][0]
        up_x = list(up_feats)[::-1][0]
        for index, (our_stage, up_stage) in enumerate(
            zip(ours.decoder.up_stages, upstream.decoder.up_stages)
        ):
            our_x = our_stage(our_x, our_feats[::-1][index + 1])
            up_x = up_stage(up_x, list(up_feats)[::-1][index + 1])
            _check(f"decoder/uafm{index}", our_x, up_x, failures)

    if size in AUX_PARITY_SIZES:
        # Auxiliary logits only exist on the training forward. Batch 2 keeps
        # SPPM's 1x1 pooling branch out of BatchNorm's single-sample error.
        torch.manual_seed(1)
        x01 = torch.rand(2, 3, height, width)
        x_norm = ours.normalize(x01)
        upstream.train()
        ours.train()
        with torch.no_grad():
            up_out = upstream(x_norm)
            our_out = ours(x01)
        for index, label in enumerate(["main", "aux_s8", "aux_s16", "aux_s32"]):
            _check(f"train/{label}", our_out[index], up_out[index], failures)
        upstream.eval()
        ours.eval()


def _image_fixture(height: int, width: int) -> torch.Tensor:
    import numpy as np

    from libreyolo.models.ppliteseg.model import preprocess_numpy

    root = Path(add_repo_root_to_path())
    for candidate in ("tests/assets/bus.jpg", "libreyolo/assets/bus.jpg", "assets/bus.jpg"):
        image_path = root / candidate
        if image_path.exists():
            from PIL import Image

            with Image.open(image_path) as img:
                array = np.asarray(img.convert("RGB"))
            break
    else:
        rng = np.random.default_rng(7)
        array = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    chw, _ = preprocess_numpy(array, (height, width))
    return torch.from_numpy(chw).unsqueeze(0)


def main() -> int:
    if not CKPT_DIR:
        raise SystemExit("Set PPLITESEG_OFFICIAL_CKPT_DIR to the upstream checkpoint directory.")
    ckpt_dir = Path(CKPT_DIR)
    build_upstream = _load_oracle()
    failures: list[str] = []
    for size in SIZES:
        run_size(size, ckpt_dir, build_upstream, failures)

    print("\n" + "=" * 60)
    if failures:
        print(f"PARITY FAILED ({len(failures)} mismatches):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PARITY OK: max_abs_diff == 0.0 for every probe on all four checkpoints.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
