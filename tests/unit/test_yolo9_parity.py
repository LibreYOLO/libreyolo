"""Exact main-path logits against the pinned MIT YOLOv9 implementation.

Set YOLO9_UPSTREAM to a checkout of MultimediaTechLab/YOLO at commit
c4cb5f6f56102eceeaa7d75e23e1125cd0373eaf. The upstream model files import
``omegaconf`` and ``einops``; the test skips when they are not installed. No
weights are needed: the upstream model is built with seeded random
parameters, its state dict goes through LibreYOLO's converter, and both
networks run on the same input.
"""

import hashlib
import logging
import os
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.external_data]

SOURCE_HASHES = {
    "yolo/model/module.py": "1f51fcda3e63ef85eaa157efed258f745f32c84894c49e94edbed04819cde17b",
    "yolo/model/yolo.py": "46f63506a077052c8f0e197438b72494f351bf4c1d9f628bd0f8fc4f4294e097",
    "yolo/utils/module_utils.py": "8fd9a690e987d207a1c9de1a2069e96916ad7bad337b7dcb90f80f046fa54399",
}


@pytest.fixture(scope="module")
def upstream_yolo():
    root = os.environ.get("YOLO9_UPSTREAM")
    if not root:
        pytest.skip("set YOLO9_UPSTREAM to the pinned MultimediaTechLab/YOLO checkout")
    pytest.importorskip("omegaconf")
    pytest.importorskip("einops")
    root = Path(root)
    for relative, digest in SOURCE_HASHES.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative

    # Import the architecture files unchanged, without the upstream CLI and
    # its Lightning logger.
    saved = {name: sys.modules.get(name) for name in ("yolo", "yolo.utils.logger")}
    package = types.ModuleType("yolo")
    package.__path__ = [str(root / "yolo")]
    logger_module = types.ModuleType("yolo.utils.logger")
    logger_module.logger = logging.getLogger("yolo9-upstream")
    sys.modules["yolo"] = package
    sys.modules["yolo.utils.logger"] = logger_module
    try:
        from omegaconf import OmegaConf
        from yolo.model.yolo import YOLO

        def build(size):
            config = OmegaConf.load(root / f"yolo/config/model/v9-{size}.yaml")
            return YOLO(config, class_num=80).eval()

        yield build
    finally:
        for name in [n for n in sys.modules if n == "yolo" or n.startswith("yolo.")]:
            del sys.modules[name]
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module


@pytest.mark.parametrize("size", ["t", "s", "m", "c"])
def test_main_path_matches_pinned_upstream(upstream_yolo, size):
    from libreyolo.models.yolo9.convert import convert_state_dict
    from libreyolo.models.yolo9.model import LibreYOLO9
    from libreyolo.models.yolo9.nn import Anchor2Vec

    torch.manual_seed(37)
    reference = upstream_yolo(size)
    state, _stats = convert_state_dict(reference.model.state_dict(), size)

    ours = LibreYOLO9(None, size=size, nb_classes=80, device="cpu")
    ours._align_class_towers_for_transfer(state)
    main = {k: v for k, v in state.items() if not k.startswith(("aux.", "aux_head."))}
    ours.model.load_state_dict(main, strict=True)
    ours.model.eval()

    image = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        upstream_levels = reference(image, shortcut="Main")["Main"]
        our_levels = ours.model(image)["raw_outputs"]

    assert len(upstream_levels) == len(our_levels) == 3
    decode = Anchor2Vec()
    for (class_x, anchor_x, vector_x), raw in zip(upstream_levels, our_levels):
        box_channels = anchor_x.shape[1] * anchor_x.shape[2]
        box_logits = anchor_x.permute(0, 2, 1, 3, 4).reshape_as(raw[:, :box_channels])
        assert torch.equal(box_logits, raw[:, :box_channels])
        assert torch.equal(class_x, raw[:, box_channels:])
        # Same expectation, different reduction layout: float-level only.
        torch.testing.assert_close(
            decode(raw[:, :box_channels]), vector_x, rtol=0, atol=1e-5
        )
