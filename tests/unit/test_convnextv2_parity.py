"""Exact logits against the pinned official MIT implementation and weights.

Set CONVNEXTV2_UPSTREAM to a checkout of Meta's commit
2553895753323c6fe0b2bf390683f5ea358a42b9 and CONVNEXTV2_CHECKPOINTS to a
directory containing <size>.pt official 224px ImageNet-1K EMA checkpoints.
Only the dense LayerNorm and GRN classes are loaded from upstream utils.py;
the unrelated sparse pretraining classes require MinkowskiEngine.
"""

import ast
import hashlib
import os
import types
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from libreyolo.models.convnextv2.nn import ARCH_DEFS, ConvNeXtV2

pytestmark = [pytest.mark.unit, pytest.mark.external_data]


def upstream_model(root, size):
    utils_path = root / "models" / "utils.py"
    source_hashes = {
        "utils.py": "f7eca4be0696ccdae8b7a05387870cacc66d6408a7acc44b0b7f1ae56b2df631",
        "convnextv2.py": "c553af39d6ed5e89d1ceec4598fb48001c3757f7a547c535812291ecb6fc2f1a",
    }
    for filename, digest in source_hashes.items():
        assert (
            hashlib.sha256((root / "models" / filename).read_bytes()).hexdigest()
            == digest
        )
    tree = ast.parse(utils_path.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in {"LayerNorm", "GRN"}
    ]
    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(compile(tree, str(utils_path), "exec"), namespace)  # noqa: S102 - pinned external parity oracle
    path = root / "models" / "convnextv2.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module == "utils"
            and node.level == 1
        )
    ]
    module = types.ModuleType("official_convnextv2")
    module.__dict__.update(namespace)
    exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102 - pinned external parity oracle
    depths, dims = ARCH_DEFS[size]
    return module.ConvNeXtV2(depths=depths, dims=dims)


@pytest.mark.parametrize("size", list(ARCH_DEFS))
def test_official_checkpoint_parity(size):
    root = os.environ.get("CONVNEXTV2_UPSTREAM")
    checkpoints = os.environ.get("CONVNEXTV2_CHECKPOINTS")
    if not root or not checkpoints:
        pytest.skip("Set CONVNEXTV2_UPSTREAM and CONVNEXTV2_CHECKPOINTS")
    path = Path(checkpoints) / f"{size}.pt"
    if not path.exists():
        pytest.skip(f"Missing {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    reference = upstream_model(Path(root), size).eval()
    reference.load_state_dict(state, strict=True)
    native = ConvNeXtV2(size)
    native.load_state_dict(state, strict=True)
    native.eval()
    generator = torch.Generator().manual_seed(17)
    image = torch.randn(1, 3, 224, 224, generator=generator)
    with torch.inference_mode():
        expected = reference(image)
        actual = native(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
