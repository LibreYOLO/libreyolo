"""Hermetic tests for the LibreRAFT optical flow family (no downloads)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from libreyolo.models.raft.model import LibreRAFT
from libreyolo.models.raft.nn import DEFAULT_ITERS, LibreRAFTNet
from libreyolo.models.raft.utils import pad_amounts, postprocess_flow
from libreyolo.utils.serialization import validate_checkpoint_metadata

pytestmark = [pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[2]


def _load_converter():
    weights_dir = str(_REPO / "weights")
    if weights_dir not in sys.path:
        sys.path.insert(0, weights_dir)
    spec = importlib.util.spec_from_file_location(
        "convert_raft_weights", _REPO / "weights" / "convert_raft_weights.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=["s", "l"])
def upstream_like(request):
    """A randomly initialised net saved the way upstream does (DataParallel keys)."""
    torch.manual_seed(0)
    net = LibreRAFTNet(small=request.param == "s")
    sd = {f"module.{k}": v for k, v in net.state_dict().items()}
    return request.param, net, sd


def test_can_load_and_detect_size(upstream_like):
    size, net, _ = upstream_like
    sd = net.state_dict()
    assert LibreRAFT.can_load(sd)
    assert LibreRAFT.detect_size(sd) == size
    assert LibreRAFT.detect_nb_classes(sd) == 1


def test_can_load_rejects_foreign_layouts(upstream_like):
    _, net, _ = upstream_like
    sd = dict(net.state_dict())
    # A RAFT descendant with an extra top-level module (SEA-RAFT style) is not RAFT.
    assert not LibreRAFT.can_load({**sd, "init_conv.weight": torch.zeros(1)})
    # Partial layouts are rejected.
    assert not LibreRAFT.can_load({k: v for k, v in sd.items() if not k.startswith("cnet.")})
    assert not LibreRAFT.can_load({"backbone.weight": torch.zeros(1)})
    assert not LibreRAFT.can_load({})


def test_can_load_rejects_other_dense_families():
    from libreyolo.models.nafnet.model import LibreNAFNet

    nafnet = LibreNAFNet(model_path=None, size="s", device="cpu")
    assert not LibreRAFT.can_load(nafnet.model.state_dict())


def test_convert_roundtrip(tmp_path, upstream_like):
    size, net, sd = upstream_like
    conv = _load_converter()
    src = tmp_path / "raft-up.pth"
    torch.save(sd, src)
    variant = "sintel" if size == "l" else None
    name = f"LibreRAFT{size}-flow" + (f"-{variant}" if variant else "") + ".pt"
    out = conv.convert(str(src), str(tmp_path / name), variant=variant)
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(ckpt)
    assert ckpt["model_family"] == "raft"
    assert ckpt["task"] == "flow"
    assert ckpt["size"] == size
    assert ckpt["nc"] == 1 and ckpt["names"] == {0: "flow"}
    assert ckpt.get("variant") == variant
    assert not any(k.startswith("module.") for k in ckpt["model"])
    for k, v in net.state_dict().items():
        assert torch.equal(ckpt["model"][k], v)


def test_convert_rejects_wrong_size(tmp_path, upstream_like):
    size, _, sd = upstream_like
    conv = _load_converter()
    src = tmp_path / "raft.pth"
    torch.save(sd, src)
    with pytest.raises(ValueError):
        conv.convert(str(src), str(tmp_path / "x.pt"), size="l" if size == "s" else "s")


def test_filename_grammar():
    assert LibreRAFT.detect_size_from_filename("LibreRAFTs-flow.pt") == "s"
    assert LibreRAFT.detect_size_from_filename("LibreRAFTl-flow.pt") == "l"
    assert LibreRAFT.detect_task_from_filename("LibreRAFTl-flow.pt") == "flow"
    assert LibreRAFT.detect_variant_from_filename("LibreRAFTl-flow-sintel.pt") == "sintel"
    assert LibreRAFT.detect_variant_from_filename("LibreRAFTl-flow-kitti.pt") == "kitti"
    assert LibreRAFT.get_download_url("LibreRAFTl-flow-kitti.pt") == (
        "https://huggingface.co/LibreYOLO/LibreRAFTl-flow-kitti/resolve/main/LibreRAFTl-flow-kitti.pt"
    )
    assert LibreRAFT.get_download_url("LibreRAFTs-flow.pt").endswith("LibreRAFTs-flow/resolve/main/LibreRAFTs-flow.pt")


def test_class_contract():
    assert LibreRAFT.SUPPORTED_TASKS == ("flow",)
    assert LibreRAFT.DEFAULT_TASK == "flow"
    assert LibreRAFT.FLOW_CONTEXT == (0, 0)
    assert DEFAULT_ITERS == 20


@pytest.mark.parametrize("h,w", [(436, 1024), (375, 1242), (360, 640), (64, 64)])
def test_padding_matches_inputpadder(h, w):
    left, right, top, bottom = pad_amounts(h, w)
    assert (h + top + bottom) % 8 == 0 and (w + left + right) % 8 == 0
    assert left == (w + left + right - w) // 2 and top == (h + top + bottom - h) // 2


def test_postprocess_unpads_and_rescales():
    h, w = 37, 50
    left, right, top, bottom = pad_amounts(h, w)
    padded = torch.zeros(1, 2, h + top + bottom, w + left + right)
    padded[:, 0, top : top + h, left : left + w] = 3.0
    out = postprocess_flow(padded, (w, h))
    assert out.shape == (h, w, 2)
    assert torch.all(out[..., 0] == 3.0) and torch.all(out[..., 1] == 0.0)

    # A half-resolution field is upsampled and its vectors doubled.
    half = torch.ones(1, 2, 24, 40)
    up = postprocess_flow(half, (80, 48))
    assert up.shape == (48, 80, 2)
    assert torch.allclose(up, torch.full_like(up, 2.0))


def test_forward_pair_shapes():
    model = LibreRAFT(model_path=None, size="s", device="cpu")
    model.model.iters = 2
    a = torch.rand(1, 3, 128, 192) * 255
    b = torch.rand(1, 3, 128, 192) * 255
    with torch.no_grad():
        f1 = model._forward((a, b))
        f2 = model._forward(torch.stack([a, b], dim=1))
        f3 = model._forward(torch.cat([a, b], dim=1))
    assert f1.shape == (1, 2, 128, 192)
    assert torch.isfinite(f1).all()
    assert torch.equal(f1, f2) and torch.equal(f1, f3)
    with pytest.raises(ValueError):
        model._forward(a)


def test_train_is_not_supported():
    model = LibreRAFT(model_path=None, size="s", device="cpu")
    with pytest.raises(NotImplementedError):
        model.train(data="x.yaml")
