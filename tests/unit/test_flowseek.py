"""Unit tests for the FlowSeek optical flow family (CPU, random weights)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreYOLO
from libreyolo.models.depth_anything.model import LibreDepthAnythingV2
from libreyolo.models.depth_anything.nn import LibreDepthAnythingV2Net
from libreyolo.models.flowseek.model import LibreFlowSeek, build_flowseek
from libreyolo.utils.serialization import load_untrusted_torch_file

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]


def _load_converter():
    sys.path.insert(0, str(REPO / "weights"))
    spec = importlib.util.spec_from_file_location(
        "convert_flowseek_weights", REPO / "weights" / "convert_flowseek_weights.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tiny_state_dict():
    torch.manual_seed(0)
    return build_flowseek("t").state_dict()


def _frames(h=128, w=160, shift=2):
    rng = np.random.default_rng(0)
    base = (rng.random((h, w + shift, 3)) * 255).astype(np.uint8)
    return [Image.fromarray(base[:, shift:]), Image.fromarray(base[:, :w])]


class TestCanLoad:
    def test_accepts_flowseek_layout(self, tiny_state_dict):
        assert LibreFlowSeek.can_load(tiny_state_dict)

    def test_rejects_plain_depth_anything(self):
        dav2 = LibreDepthAnythingV2Net("vits").state_dict()
        assert not LibreFlowSeek.can_load(dav2)

    def test_depth_anything_rejects_flowseek(self, tiny_state_dict):
        assert not LibreDepthAnythingV2.can_load(tiny_state_dict)

    def test_rejects_missing_encoder(self, tiny_state_dict):
        sd = {k: v for k, v in tiny_state_dict.items() if not k.startswith("dav2.")}
        assert not LibreFlowSeek.can_load(sd)


class TestSizes:
    def test_detect_size_from_encoder(self, tiny_state_dict):
        assert LibreFlowSeek.detect_size(tiny_state_dict) == "t"

    @pytest.mark.parametrize("size", ["t", "s", "m", "l"])
    def test_filename_size_and_download_url(self, size):
        name = f"LibreFlowSeek{size}-flow.pt"
        assert LibreFlowSeek.detect_size_from_filename(name) == size
        assert LibreFlowSeek.get_download_url(name) == (
            f"https://huggingface.co/LibreYOLO/LibreFlowSeek{size}-flow/resolve/main/{name}"
        )

    @pytest.mark.parametrize("size,non_commercial", [("t", False), ("s", False), ("m", True), ("l", True)])
    def test_non_commercial_notice(self, size, non_commercial):
        notice = LibreFlowSeek.get_download_notice(f"LibreFlowSeek{size}-flow.pt", "")
        assert (notice is not None) == non_commercial

    def test_iterations_follow_upstream_configs(self):
        assert LibreFlowSeek.ITERATIONS == {"t": 4, "s": 12, "m": 4, "l": 12}


class TestConversion:
    def test_convert_roundtrip(self, tiny_state_dict, tmp_path):
        conv = _load_converter()
        src = tmp_path / "flowseek_T_TartanCT_TSKH.pth"
        torch.save(tiny_state_dict, src)
        out = tmp_path / "LibreFlowSeeks-flow.pt"
        conv.convert_weights(str(src), str(out), size="s")
        ckpt = load_untrusted_torch_file(str(out), map_location="cpu")
        assert ckpt["model_family"] == "flowseek"
        assert ckpt["size"] == "s"
        assert ckpt["task"] == "flow"
        assert ckpt["nc"] == 1
        assert ckpt["names"] == {0: "flow"}
        assert ckpt["libreyolo_version"]
        assert len(ckpt["model"]) == len(tiny_state_dict)

        model = LibreYOLO(str(out), device="cpu")
        assert isinstance(model, LibreFlowSeek)
        assert (model.size, model.task, model.iterations) == ("s", "flow", 12)
        for key, value in tiny_state_dict.items():
            assert torch.equal(model.model.state_dict()[key], value), key

    def test_embeds_separate_dav2(self, tiny_state_dict, tmp_path):
        conv = _load_converter()
        flow_only = {k: v for k, v in tiny_state_dict.items() if not k.startswith("dav2.")}
        dav2 = {k[5:]: v for k, v in tiny_state_dict.items() if k.startswith("dav2.")}
        merged = conv.prepare_state_dict(flow_only, "t", dav2)
        assert set(merged) == set(tiny_state_dict)

    def test_rejects_size_encoder_mismatch(self, tiny_state_dict):
        conv = _load_converter()
        with pytest.raises(ValueError, match="ViT width"):
            conv.prepare_state_dict(dict(tiny_state_dict), "m")


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return LibreFlowSeek(None, size="t", device="cpu")


class TestPredict:
    def test_pair_gives_one_flow_on_original_canvas(self, model):
        model.model.eval()
        results = model(_frames(124, 165))
        assert len(results) == 1
        flow = results[0].flow
        assert tuple(flow.data.shape) == (124, 165, 2)
        assert results[0].boxes is None

    def test_single_image_is_rejected(self, model):
        with pytest.raises(ValueError):
            model(_frames()[0])

    def test_postprocess_rescales_vectors(self, model):
        out = torch.ones(1, 2, 64, 80)
        flow = model._postprocess(out, 0.0, 0.0, (160, 128))["flow"]
        assert tuple(flow.shape) == (128, 160, 2)
        assert torch.allclose(flow[..., 0], torch.full((128, 160), 2.0))
        assert torch.allclose(flow[..., 1], torch.full((128, 160), 2.0))

    def test_imgsz_caps_the_long_side(self, model):
        model.model.eval()
        frames = _frames(248, 330)
        # The cap must leave a padded canvas of at least 128 px per side:
        # 330 -> 200 gives 150x200, padded to 152x200.
        tensor, _, size, ratio = model._preprocess(frames[0], input_size=200)
        assert tuple(tensor.shape) == (1, 3, 150, 200) and size == (330, 248)
        assert ratio == (330 / 200, 248 / 150)
        native, _, _, ratio_native = model._preprocess(frames[0])
        assert tuple(native.shape) == (1, 3, 248, 330) and ratio_native == (1.0, 1.0)
        seen = []
        forward_flow = model._forward_flow

        def record(window):
            seen.append(tuple(window.shape))
            return forward_flow(window)

        model._forward_flow = record
        # Cap a 248x330 pair onto the 124x165 canvas the pair test above runs at.
        big = _frames(248, 330)
        try:
            results = model(big, imgsz=165)
        finally:
            del model._forward_flow
        assert seen == [(2, 3, 124, 165)]
        assert tuple(results[0].flow.data.shape) == (248, 330, 2)
        with pytest.raises(ValueError, match="single int"):
            model(big, imgsz=(165, 165))

    def test_flow_context_and_train(self, model):
        assert LibreFlowSeek.FLOW_CONTEXT == (0, 0)
        with pytest.raises(NotImplementedError):
            model.train()


def test_small_frames_are_rejected_with_a_clear_error(model):
    from PIL import Image as _Image

    tiny = [_Image.new("RGB", (80, 60)), _Image.new("RGB", (80, 60))]
    with pytest.raises(ValueError, match="at least 128x128"):
        model(tiny)


def test_imgsz_cap_below_the_minimum_names_the_cap(model):
    from PIL import Image as _Image

    frames = [_Image.new("RGB", (400, 300)), _Image.new("RGB", (400, 300))]
    with pytest.raises(ValueError, match="imgsz=100"):
        model(frames, imgsz=100)
