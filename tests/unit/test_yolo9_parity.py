"""Exact main and auxiliary logits against the pinned MIT YOLOv9 implementation.

Set YOLO9_UPSTREAM to a checkout of MultimediaTechLab/YOLO at commit
c4cb5f6f56102eceeaa7d75e23e1125cd0373eaf. The upstream model files import
``omegaconf`` and ``einops``; the test skips when they are not installed. No
weights are needed: the upstream model is built with seeded random
parameters, its state dict goes through LibreYOLO's converter, and both
networks run on the same input. The auxiliary (PGI) branch is compared too:
``AuxNeck`` for t/s and ``AuxBackbone`` for m/c against the upstream ``AUX``
output.
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
    "yolo/config/model/v9-t.yaml": "85a0c8788639f125d0ac2243191f85840b82f7b2ec63006abcaed3c70aa1177d",
    "yolo/config/model/v9-s.yaml": "901ad583a26fb99294aa3f085b43990158e8bd29e3a897d16f6c6cc66c60dde1",
    "yolo/config/model/v9-m.yaml": "5b08b6aa827a51ac237ce05e7868df1ff4c24154236812f0c4947f561626b2eb",
    "yolo/config/model/v9-c.yaml": "add1ca3cb1029a94469e6413a41192c5a09f0447504b41bca9e7a6ee4dc12c2f",
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

    _assert_levels_identical(upstream_levels, our_levels, Anchor2Vec())


def _assert_levels_identical(upstream_levels, our_levels, decode=None):
    """Upstream ``(class_x, anchor_x, vector_x)`` per level against our raw maps."""
    assert len(upstream_levels) == len(our_levels) == 3
    for (class_x, anchor_x, vector_x), raw in zip(upstream_levels, our_levels):
        box_channels = anchor_x.shape[1] * anchor_x.shape[2]
        box_logits = anchor_x.permute(0, 2, 1, 3, 4).reshape_as(raw[:, :box_channels])
        assert torch.equal(box_logits, raw[:, :box_channels])
        assert torch.equal(class_x, raw[:, box_channels:])
        if decode is not None:
            # Same expectation, different reduction layout: float-level only.
            torch.testing.assert_close(
                decode(raw[:, :box_channels]), vector_x, rtol=0, atol=1e-5
            )


def _is_aux(key):
    return key.startswith(("aux.", "aux_head."))


@pytest.mark.parametrize("mode", ["eval", "train"])
@pytest.mark.parametrize("size", ["t", "s", "m", "c"])
def test_auxiliary_path_matches_pinned_upstream(upstream_yolo, size, mode):
    """Main and AUX outputs are bit-identical with the PGI branch attached.

    ``eval`` compares with frozen batch-norm statistics; ``train`` runs the
    trainer's own capture boundary (``_PGITrainForward``) with batch
    statistics, as a training step does.
    """
    from libreyolo.models.yolo9.convert import convert_key, convert_state_dict
    from libreyolo.models.yolo9.model import LibreYOLO9
    from libreyolo.models.yolo9.nn import AuxBackbone, AuxNeck
    from libreyolo.models.yolo9.trainer import _PGITrainForward

    torch.manual_seed(37)
    reference = upstream_yolo(size)
    upstream_state = reference.model.state_dict()
    state, stats = convert_state_dict(upstream_state, size)

    # Only the frozen anc2vec bin weights (3 per head) stay unconverted.
    leftovers = [k for k in upstream_state if not convert_key(k, size)[1]]
    assert len(leftovers) == 6 and all(".anc2vec." in k for k in leftovers)
    assert stats == {"converted": len(upstream_state) - 6, "skipped": 3, "failed": 3}

    ours = LibreYOLO9(None, size=size, nb_classes=80, device="cpu")
    ours._align_class_towers_for_transfer(state)
    ours.model.enable_aux(0.25)
    assert type(ours.model.aux) is (AuxBackbone if size in ("m", "c") else AuxNeck)
    aux_keys = [k for k in state if _is_aux(k)]
    assert ours._load_aux_tensors(state) == len(aux_keys) > 0
    # Every auxiliary tensor of our model came from the upstream file.
    assert sorted(aux_keys) == sorted(k for k in ours.model.state_dict() if _is_aux(k))
    ours.model.load_state_dict(state, strict=True)

    # The auxiliary branch has exactly the upstream trainable parameter count.
    first_aux_layer = 23
    upstream_aux_params = sum(
        p.numel()
        for name, p in reference.model.named_parameters()
        if int(name.split(".", 1)[0]) >= first_aux_layer and p.requires_grad
    )
    our_aux_params = sum(
        p.numel() for name, p in ours.model.named_parameters() if _is_aux(name)
    )
    assert our_aux_params == upstream_aux_params

    training = mode == "train"
    reference.train(training)
    ours.model.train(training)
    image = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        upstream_out = reference(image)
        if training:
            our_out = _PGITrainForward(ours.model)(image)
            our_main, our_aux = our_out["main"], our_out["aux"]
        else:
            our_main = ours.model(image)["raw_outputs"]
            b3, b4, _p5, b5 = ours.model.backbone(image, return_b5=True)
            features = ours.model.aux_features(image, b3, b4, b5)
            our_aux = ours.model.aux_head(list(features))[1]

    assert set(upstream_out) == {"Main", "AUX"}
    _assert_levels_identical(upstream_out["Main"], our_main)
    _assert_levels_identical(upstream_out["AUX"], our_aux)
