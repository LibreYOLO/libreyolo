"""Hermetic unit coverage for the DPFlow optical flow family."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.unit, pytest.mark.dpflow]

ROOT = Path(__file__).resolve().parents[2]


def _converter():
    weights_dir = ROOT / "weights"
    sys.path.insert(0, str(weights_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "convert_dpflow_weights", weights_dir / "convert_dpflow_weights.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


@pytest.fixture(scope="module")
def native_state():
    from libreyolo.models.dpflow._vendor.dpflow import DPFlow

    torch.manual_seed(0)
    return DPFlow().state_dict()


@pytest.fixture(scope="module")
def model():
    from libreyolo import LibreDPFlow

    return LibreDPFlow(None, size="m", device="cpu")


def test_registration_and_filenames():
    from libreyolo import LibreDPFlow
    from libreyolo.models.base import BaseModel

    assert LibreDPFlow in BaseModel._registry
    assert LibreDPFlow.FAMILY == "dpflow"
    assert LibreDPFlow.SUPPORTED_TASKS == ("flow",)
    assert LibreDPFlow.DEFAULT_TASK == "flow"
    assert LibreDPFlow.FLOW_CONTEXT == (0, 0)
    assert LibreDPFlow.detect_size_from_filename("LibreDPFlowm-flow.pt") == "m"
    assert LibreDPFlow.detect_task_from_filename("LibreDPFlowm-flow.pt") == "flow"
    assert LibreDPFlow.detect_variant_from_filename("LibreDPFlowm-flow-kitti.pt") == "kitti"
    assert LibreDPFlow.detect_size_from_filename("LibreDPFlowm.pt") is None
    assert LibreDPFlow.get_download_url("LibreDPFlowm-flow.pt") == (
        "https://huggingface.co/LibreYOLO/LibreDPFlowm-flow/resolve/main/LibreDPFlowm-flow.pt"
    )
    assert LibreDPFlow.get_download_url("LibreDPFlowm-flow-sintel.pt") == (
        "https://huggingface.co/LibreYOLO/LibreDPFlowm-flow-sintel/resolve/main/"
        "LibreDPFlowm-flow-sintel.pt"
    )
    notice = LibreDPFlow.get_download_notice("LibreDPFlowm-flow.pt", "")
    assert "CC BY-NC-SA 4.0" in notice


def test_can_load_is_specific(native_state):
    from libreyolo import LibreDPFlow

    assert len(native_state) == 498
    assert LibreDPFlow.can_load(native_state)
    assert LibreDPFlow.detect_size(native_state) == "m"
    assert LibreDPFlow.detect_nb_classes(native_state) == 1

    # A RAFT-style layout shares the fnet/update_block vocabulary but not the
    # dual-pyramid encoder: it must not be claimed.
    raft_like = {
        "fnet.conv1.weight": torch.zeros(1),
        "cnet.conv1.weight": torch.zeros(1),
        "update_block.gru.convz1.weight": torch.zeros(1),
        "update_block.mask.0.weight": torch.zeros(1),
    }
    assert not LibreDPFlow.can_load(raft_like)
    partial = {k: v for k, v in native_state.items() if not k.startswith("fnet.backward_gru.")}
    assert not LibreDPFlow.can_load(partial)


def test_no_other_family_claims_dpflow(native_state):
    from libreyolo import LibreDPFlow
    from libreyolo.models.base import BaseModel

    claimants = [
        cls for cls in BaseModel._registry if cls is not LibreDPFlow and cls.can_load(native_state)
    ]
    assert claimants == []


def test_converter_wraps_lightning_checkpoint(tmp_path, native_state):
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    conv = _converter()
    src = tmp_path / "dpflow-sintel-test.ckpt"
    torch.save({"state_dict": dict(native_state), "epoch": 3}, src)
    out = tmp_path / conv.output_name("sintel")
    assert out.name == "LibreDPFlowm-flow-sintel.pt"
    assert conv.output_name("things") == "LibreDPFlowm-flow.pt"

    conv.convert(str(src), str(out), "sintel")
    ckpt = torch.load(out, map_location="cpu", weights_only=True)
    assert validate_checkpoint_metadata(ckpt, strict=True) == []
    assert ckpt["model_family"] == "dpflow"
    assert ckpt["size"] == "m"
    assert ckpt["task"] == "flow"
    assert ckpt["nc"] == 1
    assert ckpt["names"] == {0: "flow"}
    assert ckpt["variant"] == "sintel"
    assert ckpt["weight_license"] == "CC-BY-NC-SA-4.0"
    assert ckpt["weight_commercial_use"] is False
    assert len(ckpt["model"]) == 498
    for key, value in native_state.items():
        assert torch.equal(ckpt["model"][key], value)


def test_converter_rejects_foreign_state(tmp_path):
    conv = _converter()
    src = tmp_path / "other.ckpt"
    torch.save({"state_dict": {"backbone.weight": torch.zeros(1)}}, src)
    with pytest.raises(ValueError, match="not a ptlflow DPFlow"):
        conv.convert(str(src), str(tmp_path / "x.pt"), "things")
    with pytest.raises(ValueError, match="variant"):
        conv.convert(str(src), str(tmp_path / "x.pt"), "chairs")


def test_converted_checkpoint_loads_through_factory(tmp_path, native_state):
    from libreyolo import LibreYOLO, LibreDPFlow

    conv = _converter()
    src = tmp_path / "dpflow-things.ckpt"
    torch.save({"state_dict": dict(native_state)}, src)
    out = tmp_path / "LibreDPFlowm-flow.pt"
    conv.convert(str(src), str(out), "things", verify=False)
    model = LibreYOLO(str(out), device="cpu")
    assert isinstance(model, LibreDPFlow)
    assert model.task == "flow"
    assert model.names == {0: "flow"}
    for key, value in model.model.state_dict().items():
        assert torch.equal(value, native_state[key])


def test_preprocess_caps_the_long_side_only_above_imgsz(model):
    img = Image.fromarray(np.full((37, 53, 3), 255, dtype=np.uint8))
    # A cap above the frame leaves it at native resolution.
    tensor, pil, size, ratio = model._preprocess(img, input_size=960)
    assert tuple(tensor.shape) == (1, 3, 37, 53)
    assert size == (53, 37)
    assert ratio == (1.0, 1.0)
    assert float(tensor.max()) == 1.0
    # The family default is a cap too (8192, in practice no cap).
    native, _, _, ratio_native = model._preprocess(img)
    assert tuple(native.shape) == (1, 3, 37, 53) and ratio_native == (1.0, 1.0)
    # An explicit smaller imgsz downscales, aspect kept, and reports the factors back.
    small, _, size, ratio = model._preprocess(img, input_size=40)
    assert tuple(small.shape) == (1, 3, 28, 40) and size == (53, 37)
    assert ratio == (53 / 40, 37 / 28)
    with pytest.raises(ValueError, match="single int"):
        model._preprocess(img, input_size=(40, 40))


def test_imgsz_runs_the_network_on_the_capped_canvas(model):
    rng = np.random.default_rng(0)
    frames = [Image.fromarray(rng.integers(0, 255, (90, 140, 3), dtype=np.uint8)) for _ in range(2)]
    seen = []
    forward_flow = model._forward_flow

    def record(window):
        seen.append(tuple(window.shape))
        return forward_flow(window)

    model._forward_flow = record
    try:
        results = list(model.predict(frames, imgsz=70))
    finally:
        del model._forward_flow
    assert seen == [(2, 3, 45, 70)]
    assert tuple(results[0].flow.data.shape) == (90, 140, 2)
    assert torch.isfinite(torch.as_tensor(results[0].flow.data)).all()


def test_postprocess_rescales_vectors_to_original_canvas(model):
    flow = torch.zeros(1, 2, 10, 20)
    flow[:, 0] = 1.0  # dx = 1 px on a 20 px wide canvas
    flow[:, 1] = 2.0  # dy = 2 px on a 10 px tall canvas
    out = model._postprocess(flow, 0.0, 0.0, (40, 30))["flow"]
    assert tuple(out.shape) == (30, 40, 2)
    assert torch.allclose(out[..., 0], torch.full((30, 40), 2.0))
    assert torch.allclose(out[..., 1], torch.full((30, 40), 6.0))

    same = model._postprocess(flow, 0.0, 0.0, (20, 10))["flow"]
    assert torch.equal(same, flow[0].permute(1, 2, 0))


def test_pair_predict_returns_flow_on_original_canvas(model):
    rng = np.random.default_rng(0)
    frames = [Image.fromarray(rng.integers(0, 255, (45, 70, 3), dtype=np.uint8)) for _ in range(3)]
    results = list(model.predict(frames))
    assert len(results) == 2
    for r in results:
        assert r.boxes is None
        assert tuple(r.flow.data.shape) == (45, 70, 2)
        assert torch.isfinite(torch.as_tensor(r.flow.data)).all()


def test_single_image_and_train_are_rejected(model):
    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        model.predict(img)
    with pytest.raises(ValueError, match="two frames"):
        model._forward(torch.zeros(1, 3, 32, 32))
    with pytest.raises(NotImplementedError):
        model.train(data="x.yaml")


def test_torch_corr_mode_matches_allpairs(native_state):
    from libreyolo.models.dpflow._vendor.dpflow import DPFlow

    torch.manual_seed(0)
    a, b = torch.rand(1, 3, 48, 64), torch.rand(1, 3, 48, 64)
    allpairs = DPFlow().eval()
    allpairs.load_state_dict(native_state)
    local = DPFlow(corr_mode="torch").eval()
    local.load_state_dict(native_state)
    with torch.no_grad():
        ref = allpairs(a, b)
        alt = local(a, b)
    assert ref.shape == (1, 2, 48, 64)
    assert torch.allclose(ref, alt, atol=1e-3)
