"""Unit tests for the NeuFlow v2 optical flow family (CPU, random weights)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.models.base.model import BaseModel
from libreyolo.models.neuflowv2.model import LibreNeuFlowv2
from libreyolo.models.neuflowv2.nn import LibreNeuFlowv2Net
from libreyolo.utils.serialization import validate_checkpoint_metadata

pytestmark = [pytest.mark.unit, pytest.mark.neuflowv2]

REPO = Path(__file__).resolve().parents[2]


def _converter():
    weights_dir = REPO / "weights"
    import sys

    if str(weights_dir) not in sys.path:
        sys.path.insert(0, str(weights_dir))
    spec = importlib.util.spec_from_file_location(
        "convert_neuflowv2_weights", weights_dir / "convert_neuflowv2_weights.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def upstream_state_dict():
    torch.manual_seed(0)
    return LibreNeuFlowv2Net().state_dict()


def test_class_contract():
    assert LibreNeuFlowv2.FAMILY == "neuflowv2"
    assert LibreNeuFlowv2.SUPPORTED_TASKS == ("flow",)
    assert LibreNeuFlowv2.DEFAULT_TASK == "flow"
    assert LibreNeuFlowv2.FLOW_CONTEXT == (0, 0)
    assert LibreNeuFlowv2.INPUT_SIZES == {"s": (432, 768)}
    assert LibreNeuFlowv2 in BaseModel._registry
    import libreyolo

    assert libreyolo.LibreNeuFlowv2 is LibreNeuFlowv2


def test_parameter_count_matches_release(upstream_state_dict):
    # NeuFlow v2 reports about 9M parameters; the released checkpoints have 155 tensors.
    net = LibreNeuFlowv2Net()
    assert sum(p.numel() for p in net.parameters()) == 9_029_536
    assert len(upstream_state_dict) == 155


def test_can_load_and_detect_size(upstream_state_dict):
    assert LibreNeuFlowv2.can_load(upstream_state_dict)
    assert LibreNeuFlowv2.detect_size(upstream_state_dict) == "s"
    partial = {
        k: v for k, v in upstream_state_dict.items() if not k.startswith("upsample_s8.")
    }
    assert not LibreNeuFlowv2.can_load(partial)


def test_no_other_family_claims_neuflowv2(upstream_state_dict):
    claimants = [
        cls.__name__
        for cls in BaseModel._registry
        if cls is not LibreNeuFlowv2 and cls.can_load(upstream_state_dict)
    ]
    assert claimants == []


def test_neuflowv2_rejects_depth_anything_layout():
    from libreyolo.models.depth_anything.nn import LibreDepthAnythingV2Net

    assert not LibreNeuFlowv2.can_load(LibreDepthAnythingV2Net("vits").state_dict())


@pytest.mark.parametrize(
    ("filename", "variant"),
    [
        ("LibreNeuFlowv2s-flow.pt", None),
        ("LibreNeuFlowv2s-flow-sintel.pt", "sintel"),
        ("LibreNeuFlowv2s-flow-things.pt", "things"),
    ],
)
def test_filenames_and_download_urls(filename, variant):
    assert LibreNeuFlowv2.detect_size_from_filename(filename) == "s"
    assert LibreNeuFlowv2.detect_task_from_filename(filename) == "flow"
    assert LibreNeuFlowv2.detect_variant_from_filename(filename) == variant
    stem = filename[:-3]
    assert LibreNeuFlowv2.get_download_url(filename) == (
        f"https://huggingface.co/LibreYOLO/{stem}/resolve/main/{filename}"
    )


@pytest.mark.parametrize(
    ("upstream_name", "variant", "expected"),
    [
        ("neuflow_mixed.pth", "mixed", "LibreNeuFlowv2s-flow.pt"),
        ("neuflow_sintel.pth", "sintel", "LibreNeuFlowv2s-flow-sintel.pt"),
        ("neuflow_things.pth", "things", "LibreNeuFlowv2s-flow-things.pt"),
    ],
)
def test_conversion_round_trip(
    tmp_path, upstream_state_dict, upstream_name, variant, expected
):
    conv = _converter()
    assert conv.detect_variant(upstream_name) == variant
    assert conv.output_filename(variant) == expected

    src = tmp_path / upstream_name
    torch.save({"model": upstream_state_dict, "epoch": 0}, src)
    out = tmp_path / expected
    conv.convert_weights(str(src), str(out))

    ckpt = torch.load(out, map_location="cpu", weights_only=True)
    assert validate_checkpoint_metadata(ckpt, strict=True) == []
    assert ckpt["model_family"] == "neuflowv2"
    assert ckpt["size"] == "s"
    assert ckpt["task"] == "flow"
    assert ckpt["nc"] == 1
    assert ckpt["names"] == {0: "flow"}
    assert ckpt["variant"] == variant
    assert (ckpt["imgsz_h"], ckpt["imgsz_w"]) == (432, 768)
    assert ckpt["libreyolo_version"]
    for key, value in upstream_state_dict.items():
        assert torch.equal(ckpt["model"][key], value)

    from libreyolo import LibreYOLO

    model = LibreYOLO(str(out), device="cpu")
    assert isinstance(model, LibreNeuFlowv2)
    assert model.task == "flow"
    assert model.names == {0: "flow"}


def test_converter_rejects_foreign_checkpoint(tmp_path):
    conv = _converter()
    src = tmp_path / "neuflow_mixed.pth"
    torch.save({"model": {"conv.weight": torch.zeros(1)}}, src)
    with pytest.raises(ValueError, match="does not look like a NeuFlow v2"):
        conv.convert_weights(str(src), str(tmp_path / "out.pt"))


def test_forward_shapes_and_odd_canvas():
    net = LibreNeuFlowv2Net().eval()
    a = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        flow = net(a, a.clone())
    assert flow.shape == (1, 2, 64, 96)
    assert torch.isfinite(flow).all()
    with pytest.raises(ValueError, match="multiples of 16"):
        net(torch.rand(1, 3, 60, 96), torch.rand(1, 3, 60, 96))


def test_forward_does_not_mutate_inputs():
    net = LibreNeuFlowv2Net().eval()
    a, b = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    a0, b0 = a.clone(), b.clone()
    with torch.no_grad():
        net(a, b)
    assert torch.equal(a, a0) and torch.equal(b, b0)


def test_preprocess_and_postprocess_geometry():
    model = LibreNeuFlowv2(None, size="s", device="cpu")
    img = Image.fromarray(np.full((90, 160, 3), 128, dtype=np.uint8))
    tensor, _pil, orig_size, ratio = model._preprocess(img, input_size=(48, 64))
    assert tensor.shape == (1, 3, 48, 64)
    assert orig_size == (160, 90)
    assert ratio == pytest.approx((160 / 64, 90 / 48))

    # A constant (1, 2) canvas field becomes (2.5, 3.75) px on the original canvas.
    out = torch.ones(1, 2, 48, 64)
    out[:, 1] = 2.0
    flow = model._postprocess(out, 0.0, 0.0, orig_size)["flow"]
    assert tuple(flow.shape) == (90, 160, 2)
    assert torch.allclose(flow[..., 0], torch.full((90, 160), 2.5))
    assert torch.allclose(flow[..., 1], torch.full((90, 160), 2.0 * 90 / 48))

    with pytest.raises(ValueError, match="divisible by 16"):
        model._preprocess(img, input_size=(50, 64))


def test_forward_flow_window_and_single_image_error():
    model = LibreNeuFlowv2(None, size="s", device="cpu")
    window = torch.rand(2, 3, 32, 48)
    with torch.no_grad():
        out = model._forward_flow(window)
    assert out.shape == (1, 2, 32, 48)
    with pytest.raises(ValueError, match="pair of frames"):
        model._forward(torch.rand(1, 3, 32, 48))


def test_predict_sequence_returns_n_minus_one_flows(tmp_path):
    model = LibreNeuFlowv2(None, size="s", device="cpu")
    rng = np.random.default_rng(0)
    paths = []
    for i in range(3):
        p = tmp_path / f"{i:03d}.png"
        Image.fromarray(rng.integers(0, 255, (40, 56, 3), dtype=np.uint8)).save(p)
        paths.append(str(p))
    results = model.predict(paths, imgsz=(32, 48))
    assert len(results) == 2
    for r in results:
        assert r.boxes is None
        assert tuple(r.flow.data.shape) == (40, 56, 2)
    with pytest.raises(ValueError):
        model.predict(paths[0])


def test_train_is_not_supported():
    model = LibreNeuFlowv2(None, size="s", device="cpu")
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="x.yaml")
