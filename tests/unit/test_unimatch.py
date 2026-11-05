"""Unit tests for the UniMatch / GMFlow optical flow family (CPU, no weights)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]


def _converter():
    weights_dir = _ROOT / "weights"
    sys.path.insert(0, str(weights_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "convert_unimatch_weights", weights_dir / "convert_unimatch_weights.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(weights_dir))


@pytest.fixture(scope="module")
def upstream_state_dicts():
    from libreyolo.models.unimatch.model import LibreUniMatch

    torch.manual_seed(0)
    return {
        size: LibreUniMatch(None, size=size, device="cpu").model.state_dict()
        for size in ("s", "l")
    }


def test_class_contract():
    from libreyolo import LibreUniMatch
    from libreyolo.models.base import BaseModel
    from libreyolo.models.registry import MODEL_GROUPS

    assert LibreUniMatch in BaseModel._registry
    assert LibreUniMatch.FAMILY == "unimatch"
    assert LibreUniMatch.SUPPORTED_TASKS == ("flow",)
    assert LibreUniMatch.DEFAULT_TASK == "flow"
    assert LibreUniMatch.FLOW_CONTEXT == (0, 0)
    assert set(LibreUniMatch.INPUT_SIZES) == {"s", "l"}
    assert MODEL_GROUPS["unimatch"] == "g3"


def test_filenames_and_download_urls():
    from libreyolo.models.unimatch.model import LibreUniMatch

    assert LibreUniMatch.detect_size_from_filename("LibreUniMatchs-flow.pt") == "s"
    assert LibreUniMatch.detect_size_from_filename("LibreUniMatchl-flow.pt") == "l"
    assert LibreUniMatch.detect_task_from_filename("LibreUniMatchl-flow.pt") == "flow"
    assert LibreUniMatch.detect_variant_from_filename("LibreUniMatchl-flow-kitti.pt") == "kitti"
    # The flow suffix is required.
    assert LibreUniMatch.detect_size_from_filename("LibreUniMatchl.pt") is None
    for name in ("LibreUniMatchs-flow", "LibreUniMatchl-flow", "LibreUniMatchl-flow-sintel"):
        assert LibreUniMatch.get_download_url(f"{name}.pt") == (
            f"https://huggingface.co/LibreYOLO/{name}/resolve/main/{name}.pt"
        )
    with pytest.raises(FileNotFoundError):
        LibreUniMatch.get_download_url("LibreUniMatchs-flow-kitti.pt")


def test_can_load_and_detect_size(upstream_state_dicts):
    from libreyolo.models.unimatch.model import LibreUniMatch

    sd_s, sd_l = upstream_state_dicts["s"], upstream_state_dicts["l"]
    assert len(sd_s) == 123 and len(sd_l) == 152  # upstream key counts
    assert LibreUniMatch.can_load(sd_s) and LibreUniMatch.can_load(sd_l)
    assert LibreUniMatch.detect_size(sd_s) == "s"
    assert LibreUniMatch.detect_size(sd_l) == "l"
    assert LibreUniMatch.detect_nb_classes(sd_l) == 1
    # Without the GMFlow propagation block it is not this family.
    stripped = {k: v for k, v in sd_s.items() if not k.startswith("feature_flow_attn.")}
    assert not LibreUniMatch.can_load(stripped)
    # scale2 without refinement is not a shipped size.
    scale2 = {k: v for k, v in sd_l.items() if not k.startswith("refine")}
    assert LibreUniMatch.detect_size(scale2) is None


def test_unimatch_wins_factory_arbitration(upstream_state_dicts):
    """No family registered ahead of UniMatch claims its checkpoints.

    RF-DETR's lazily registered recognizer is intentionally broad and also
    accepts GMFlow keys; it registers after the eager core families, so
    registry order (the factory's arbitration rule) still routes to UniMatch.
    """
    from libreyolo.models.base import BaseModel
    from libreyolo.models.unimatch.model import LibreUniMatch

    for size, sd in upstream_state_dicts.items():
        claims = [
            c
            for c in BaseModel._registry
            # Test doubles registered by other test modules are not families.
            if c.__module__.startswith("libreyolo.models.") and c.can_load(sd)
        ]
        assert claims and claims[0] is LibreUniMatch, (size, [c.__name__ for c in claims])
        assert {c.__name__ for c in claims} <= {"LibreUniMatch", "LibreRFDETR"}


@pytest.mark.parametrize("size", ["s", "l"])
def test_conversion_round_trip(tmp_path, upstream_state_dicts, size):
    from libreyolo import LibreYOLO
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    conv = _converter()
    upstream_name = (
        "gmflow-scale1-things-e9887eda.pth"
        if size == "s"
        else "gmflow-scale2-regrefine6-sintelft-6e39e2b9.pth"
    )
    src = tmp_path / upstream_name
    torch.save({"model": upstream_state_dicts[size]}, src)
    out = conv.convert_weights(str(src), str(tmp_path / "out.pt"))
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert validate_checkpoint_metadata(ckpt, strict=True) == []
    assert ckpt["task"] == "flow" and ckpt["nc"] == 1 and ckpt["names"] == {0: "flow"}
    assert ckpt["model_family"] == "unimatch" and ckpt["size"] == size
    assert ckpt.get("variant") == (None if size == "s" else "sintel")
    assert ckpt["upstream_checkpoint"] == upstream_name
    assert conv.canonical_name(size, ckpt.get("variant")) == (
        "LibreUniMatchs-flow.pt" if size == "s" else "LibreUniMatchl-flow-sintel.pt"
    )

    model = LibreYOLO(str(out), device="cpu")
    assert type(model).__name__ == "LibreUniMatch"
    assert model.size == size and model.task == "flow" and model.names == {0: "flow"}
    loaded = model.model.state_dict()
    for key, value in upstream_state_dicts[size].items():
        assert torch.equal(loaded[key], value), key


def test_conversion_rejects_foreign_checkpoint(tmp_path):
    conv = _converter()
    src = tmp_path / "foreign.pth"
    torch.save({"model": {"backbone.conv1.weight": torch.zeros(1)}}, src)
    with pytest.raises(ValueError, match="UniMatch"):
        conv.convert_weights(str(src), str(tmp_path / "x.pt"))


def test_preprocess_geometry():
    from libreyolo.models.unimatch.model import LibreUniMatch

    s = LibreUniMatch(None, size="s", device="cpu")
    l = LibreUniMatch(None, size="l", device="cpu")
    frame = np.zeros((90, 130, 3), dtype=np.uint8)
    assert tuple(s.preprocess_array(frame).shape) == (1, 3, 96, 144)  # multiples of 16
    assert tuple(l.preprocess_array(frame).shape) == (1, 3, 96, 160)  # multiples of 32
    # Portrait frames are transposed to landscape, as upstream.
    assert tuple(s.preprocess_array(frame.transpose(1, 0, 2)).shape) == (1, 3, 96, 144)
    tensor, img, size, ratio = s._preprocess(Image.fromarray(frame))
    assert size == (130, 90) and ratio == 1.0 and tensor.max() <= 255


def test_postprocess_rescales_vectors_to_original_canvas():
    from libreyolo.models.unimatch.model import LibreUniMatch

    m = LibreUniMatch(None, size="s", device="cpu")
    # Canvas 96x144 for an original 90x130 frame: a constant (dx, dy) = (1, 1)
    # on the canvas is (130 / 144, 90 / 96) original pixels.
    canvas = torch.ones(1, 2, 96, 144)
    flow = m._postprocess(canvas, 0.0, 0.0, (130, 90))["flow"]
    assert tuple(flow.shape) == (90, 130, 2)
    assert torch.allclose(flow[..., 0], torch.full((90, 130), 130 / 144))
    assert torch.allclose(flow[..., 1], torch.full((90, 130), 90 / 96))


def test_postprocess_portrait_swaps_axes():
    from libreyolo.models.unimatch.model import LibreUniMatch

    m = LibreUniMatch(None, size="s", device="cpu")
    # Portrait original 128x96 (w=96, h=128) runs transposed on a 96x128
    # canvas: canvas dx (along the original y axis) becomes dy.
    canvas = torch.zeros(1, 2, 96, 128)
    canvas[:, 0] = 3.0
    canvas[:, 1] = -2.0
    flow = m._postprocess(canvas, 0.0, 0.0, (96, 128))["flow"]
    assert tuple(flow.shape) == (128, 96, 2)
    assert torch.allclose(flow[..., 0], torch.full((128, 96), -2.0))
    assert torch.allclose(flow[..., 1], torch.full((128, 96), 3.0))


def test_forward_flow_window_and_single_image_error():
    from libreyolo.models.unimatch.model import LibreUniMatch

    m = LibreUniMatch(None, size="s", device="cpu")
    window = torch.rand(2, 3, 64, 96) * 255
    with torch.no_grad():
        flow = m._forward_flow(window)
    assert tuple(flow.shape) == (1, 2, 64, 96)
    with pytest.raises(ValueError, match="pair"):
        m._forward(torch.zeros(1, 3, 64, 96))


def test_predict_sequence_returns_flow_results():
    from libreyolo.models.unimatch.model import LibreUniMatch

    m = LibreUniMatch(None, size="s", device="cpu")
    frames = [Image.fromarray((np.random.rand(40, 56, 3) * 255).astype(np.uint8)) for _ in range(3)]
    results = m(frames)
    assert len(results) == 2
    assert tuple(results[0].flow.data.shape) == (40, 56, 2)
    assert results[0].boxes is None
    with pytest.raises(ValueError):
        m(frames[0])


def test_train_is_not_supported():
    from libreyolo.models.unimatch.model import LibreUniMatch

    with pytest.raises(NotImplementedError):
        LibreUniMatch(None, size="s", device="cpu").train(data="x.yaml")
