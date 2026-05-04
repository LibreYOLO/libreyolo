"""PICODET conversion-script parity test.

Bo's checkpoints aren't available in CI without installing mmcv-full +
mmdet (a non-trivial chain on modern Python). To still validate the
conversion logic, this test:

1. Builds a fresh ``LibrePICODETModel``.
2. Reverses the conversion key remap on its ``state_dict`` to produce
   a synthetic "Bo-format" state dict.
3. Runs ``convert_picodet_weights.remap_state_dict`` on it.
4. Loads the result back into a *fresh* ``LibrePICODETModel`` and asserts
   bit-equivalent forward outputs.

This proves the remap is a clean bijection. When real Bo checkpoints
appear, only the **upstream** numerics need to be verified; the libreyolo
side is already covered by this test.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.picodet]

# weights/ isn't importable as a package; expose the conversion module path.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "weights"))

from convert_picodet_weights import (  # type: ignore[import-not-found]
    ESNET_STAGE_REPEATS,
    remap_state_dict,
)
from libreyolo.models.picodet.nn import LibrePICODETModel


def _flat_to_token() -> dict[int, str]:
    """Inverse of the conversion script's ``_BLOCK_MAP``."""
    out: dict[int, str] = {}
    flat = 0
    for stage_idx, repeats in enumerate(ESNET_STAGE_REPEATS):
        stage_id = stage_idx + 2
        for i in range(repeats):
            out[flat] = f"{stage_id}_{i + 1}"
            flat += 1
    return out


_FLAT_TO_TOKEN = _flat_to_token()
_BLOCKS_RE = re.compile(r"^backbone\.blocks\.(\d+)\.")
_SE_RE = re.compile(r"\.se\.conv([12])\.")


def _libreyolo_to_bo_key(key: str) -> str:
    """Reverse the conversion remap: turn LibreYOLO names into Bo's form."""
    new = key

    if new.startswith("head."):
        new = "bbox_head." + new[len("head.") :]

    m = _BLOCKS_RE.match(new)
    if m is not None:
        flat = int(m.group(1))
        token = _FLAT_TO_TOKEN[flat]
        new = f"backbone.{token}." + new[m.end() :]

    if new.startswith("neck.trans."):
        new = "neck.trans.trans." + new[len("neck.trans.") :]

    # *.se.convN.X -> *.se.convN.conv.X (only the leaf weight/bias gets
    # wrapped, not all chains, because mmcv's SELayer wraps each 1x1 in a
    # ConvModule).
    new = _SE_RE.sub(
        lambda mm: f".se.conv{mm.group(1)}.conv.",
        new,
    )
    return new


def _make_bo_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        _libreyolo_to_bo_key(k): v for k, v in model.state_dict().items()
    }


@pytest.mark.parametrize("size", ["s", "m", "l"])
def test_conversion_roundtrip_bit_equivalent(size: str) -> None:
    torch.manual_seed(0)
    src = LibrePICODETModel(size=size, nb_classes=80).eval()
    sd_orig = src.state_dict()

    # Produce a synthetic Bo-format state dict, then run our converter.
    bo_sd = _make_bo_state_dict(src)
    converted = remap_state_dict(bo_sd)

    # Same keyset, same shapes, same values — bijection check.
    assert set(converted) == set(sd_orig), (
        "Round-trip changed keyset; "
        f"missing={set(sd_orig) - set(converted)}, "
        f"unexpected={set(converted) - set(sd_orig)}"
    )
    for k in sd_orig:
        assert torch.equal(converted[k], sd_orig[k]), f"Tensor mismatch at {k!r}"

    # Forward equivalence.
    dst = LibrePICODETModel(size=size, nb_classes=80).eval()
    dst.load_state_dict(converted, strict=True)

    isz = {"s": 320, "m": 416, "l": 640}[size]
    x = torch.randn(1, 3, isz, isz)
    with torch.no_grad():
        out_src = src(x)
        out_dst = dst(x)

    for cs_src, cs_dst in zip(out_src[0], out_dst[0]):
        assert torch.equal(cs_src, cs_dst), "cls_scores diverged"
    for bp_src, bp_dst in zip(out_src[1], out_dst[1]):
        assert torch.equal(bp_src, bp_dst), "bbox_preds diverged"


def test_remap_handles_se_modules() -> None:
    """Spot-check the SE-specific remap rule (mmcv ConvModule wrapper unwrap)."""
    # A representative Bo-style SE key — an ESBlock's SE inside a stage-2 block.
    bo_key = "backbone.2_2.se.conv1.conv.weight"
    expected = "backbone.blocks.1.se.conv1.weight"
    from convert_picodet_weights import remap_key  # type: ignore[import-not-found]

    assert remap_key(bo_key) == expected
    assert remap_key("backbone.4_3.se.conv2.conv.bias") == "backbone.blocks.12.se.conv2.bias"
    # Final stage-end indices: stage 2 ends at flat idx 12 (stages 0..2 with 3+7+3=13).
    assert remap_key("bbox_head.gfl_cls.0.weight") == "head.gfl_cls.0.weight"
    assert remap_key("neck.trans.trans.0.conv.weight") == "neck.trans.0.conv.weight"
