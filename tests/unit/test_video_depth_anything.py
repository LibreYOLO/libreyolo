"""LibreVideoDepthAnything: recognition, conversion, window stitching and the
per-frame sequence runner (random weights, no downloads)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreVideoDepthAnything
from libreyolo.models.depth_anything.nn import LibreDepthAnythingV2Net
from libreyolo.models.video_depth_anything.nn import (
    INFER_LEN,
    LibreVideoDepthAnythingNet,
)
from libreyolo.models.video_depth_anything.utils import (
    effective_input_size,
    stitch_video_depth,
)
from libreyolo.utils.serialization import validate_checkpoint_metadata

pytestmark = [pytest.mark.unit, pytest.mark.video_depth_anything]

REPO_ROOT = Path(__file__).resolve().parents[2]
STEP = 22  # INFER_LEN - OVERLAP


@pytest.fixture(scope="module")
def upstream_state_dict():
    # Same keys and shapes as video_depth_anything_vits.pth (strict load).
    return LibreVideoDepthAnythingNet("vits").state_dict()


@pytest.fixture(scope="module")
def model():
    return LibreVideoDepthAnything(None, size="s", device="cpu")


@pytest.fixture(scope="module")
def converter():
    sys.path.insert(0, str(REPO_ROOT / "weights"))
    try:
        import convert_video_depth_anything_weights
    finally:
        sys.path.remove(str(REPO_ROOT / "weights"))
    return convert_video_depth_anything_weights


# ---------------------------------------------------------------- recognition


def test_can_load_and_detect_size(upstream_state_dict):
    assert LibreVideoDepthAnything.can_load(upstream_state_dict)
    assert LibreVideoDepthAnything.detect_size(upstream_state_dict) == "s"


def test_depth_anything_v2_and_vda_do_not_claim_each_other(upstream_state_dict):
    from libreyolo.models.depth_anything.model import LibreDepthAnythingV2

    dav2 = LibreDepthAnythingV2Net("vits").state_dict()
    assert not LibreVideoDepthAnything.can_load(dav2)
    assert not LibreDepthAnythingV2.can_load(upstream_state_dict)


def test_filenames_and_licence_notice():
    name = "LibreVideoDepthAnythings-depth.pt"
    assert LibreVideoDepthAnything.detect_size_from_filename(name) == "s"
    assert LibreVideoDepthAnything.detect_size_from_filename("LibreVideoDepthAnythings.pt") is None
    url = LibreVideoDepthAnything.get_download_url(name)
    assert url.endswith("LibreYOLO/LibreVideoDepthAnythings-depth/resolve/main/LibreVideoDepthAnythings-depth.pt")
    assert LibreVideoDepthAnything.get_download_notice(name, url) is None
    for size in ("b", "l"):
        notice = LibreVideoDepthAnything.get_download_notice(
            f"LibreVideoDepthAnything{size}-depth.pt", ""
        )
        assert "CC-BY-NC-4.0" in notice


# ---------------------------------------------------------------- conversion


def test_converter_wraps_metadata(tmp_path, converter, upstream_state_dict):
    src = tmp_path / "video_depth_anything_vits.pth"
    torch.save(upstream_state_dict, src)
    out = tmp_path / "LibreVideoDepthAnythings-depth.pt"
    wrapped = converter.convert_weights(str(src), str(out))
    assert validate_checkpoint_metadata(wrapped) == []
    assert wrapped["model_family"] == "video_depth_anything"
    assert wrapped["task"] == "depth" and wrapped["size"] == "s"
    loaded = LibreVideoDepthAnything(str(out), device="cpu")
    assert loaded.size == "s" and loaded.names == {0: "depth"}


def test_metric_checkpoints_are_rejected(tmp_path, converter, upstream_state_dict):
    src = tmp_path / "metric_video_depth_anything_vits.pth"
    torch.save(upstream_state_dict, src)
    with pytest.raises(ValueError, match="metric"):
        converter.convert_weights(str(src), str(tmp_path / "out.pt"))
    with pytest.raises(ValueError, match="metric"):
        LibreVideoDepthAnything.upstream_checkpoint_metadata({}, source=src)


# ---------------------------------------------------------------- geometry


def test_effective_input_size_shrinks_wider_than_16_9():
    assert effective_input_size(540, 960, 518) == 518
    assert effective_input_size(400, 1600, 518) == 224  # int(518*1.777/4)=230 -> 16*14
    assert effective_input_size(400, 1600, 518) % 14 == 0


# ---------------------------------------------------------------- stitching


def _frames(n, h=2, w=3):
    """Frame ``i`` is a tensor filled with ``i``, so windows reveal their slots."""
    return [(torch.full((1, 1, h, w), float(i)), (w, h)) for i in range(n)]


def _map(frame_id, h=2, w=3):
    # A per-frame map with spatial structure so the scale/shift fit is well posed.
    return (frame_id + 1.0) * (1.0 + np.arange(h * w, dtype=np.float32).reshape(h, w))


@pytest.mark.parametrize("n", [1, 5, 22, 32, 33, 54, 77])
def test_stitch_yields_one_map_per_frame_in_order(n):
    windows = []

    def run_window(window, canvas):
        windows.append(window[0, :, 0, 0, 0].tolist())
        return [_map(int(v)) for v in window[0, :, 0, 0, 0]]

    out = list(stitch_video_depth(_frames(n), run_window))
    assert len(out) == n
    # Frame-only network: alignment must reduce to identity.
    for i, depth in enumerate(out):
        np.testing.assert_allclose(depth, _map(i), rtol=1e-4)
    # Upstream window schedule and keyframe slots.
    assert len(windows) == -(-n // STEP)
    for k, slots in enumerate(windows):
        expected = [min(STEP * k + i, n - 1) for i in range(INFER_LEN)]
        if k:
            expected[0] = 0  # keyframe 0 of the previous window is always frame 0
            expected[1] = STEP * k - 10  # its slot 12
        assert slots == expected


def test_stitch_aligns_each_window_to_the_first():
    # Each window sees the same scene under its own affine distortion, as a
    # relative-depth network does; the stitched output must undo it.
    calls = []

    def run_window(window, canvas):
        k = len(calls)
        calls.append(k)
        scale, shift = 1.0 + 0.5 * k, 0.25 * k
        return [_map(int(v)) * scale + shift for v in window[0, :, 0, 0, 0]]

    out = list(stitch_video_depth(_frames(70), run_window))
    assert len(calls) == 4
    for i, depth in enumerate(out):
        np.testing.assert_allclose(depth, _map(i), rtol=1e-3, atol=1e-3)


def test_stitch_streams_with_bounded_lag():
    pulled = []

    def frames():
        for i, item in enumerate(_frames(200)):
            pulled.append(i)
            yield item

    stream = stitch_video_depth(frames(), lambda w, c: [_map(0)] * INFER_LEN)
    next(stream)
    assert len(pulled) == INFER_LEN


def test_stitch_rejects_empty_and_mixed_sizes():
    with pytest.raises(ValueError, match="at least one frame"):
        list(stitch_video_depth(iter([]), lambda w, c: []))
    mixed = _frames(2) + [(torch.zeros(1, 1, 4, 4), (4, 4))]
    with pytest.raises(ValueError, match="share a size"):
        list(stitch_video_depth(mixed, lambda w, c: [_map(0)] * INFER_LEN))


# ---------------------------------------------------------------- runner


def _images(n, size=(84, 56)):
    rng = np.random.default_rng(0)
    return [
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
        for _ in range(n)
    ]


def test_network_forward_shape(model):
    out = model.model(torch.rand(1, 3, 3, 70, 98))
    assert tuple(out.shape) == (1, 3, 70, 98)


def test_predict_sequence_returns_one_result_per_frame(model):
    images = _images(3)
    results = model.predict(images, imgsz=70)
    assert isinstance(results, list) and len(results) == 3
    for result in results:
        assert result.boxes is None
        assert tuple(result.depth_map.data.shape) == (56, 84)


def test_single_image_is_a_one_frame_sequence(model):
    image = _images(1)[0]
    result = model.predict(image, imgsz=70)
    assert not isinstance(result, list)
    # Same as the independent-image path validation uses.
    tensor, _, size, _ = model._preprocess(image, input_size=70)
    with torch.no_grad():
        direct = model._postprocess(model._forward(tensor), 0.25, 0.45, size)["depth"]
    torch.testing.assert_close(result.depth_map.data, direct, rtol=1e-4, atol=1e-5)


def test_sequence_options_are_rejected(model):
    images = _images(2)
    with pytest.raises(ValueError, match="batch > 1"):
        model.predict(images, imgsz=70, batch=2)
    with pytest.raises(ValueError, match="optical flow"):
        model.predict(images, imgsz=70, save_flow=True)
    with pytest.raises(ValueError, match="divisible by 14"):
        model.predict(images, imgsz=64)


def test_runner_checks_output_count(model, monkeypatch):
    def short(frames):
        for _ in frames:
            pass
        yield {"depth": torch.zeros(56, 84)}

    monkeypatch.setattr(model, "_forward_sequence", short)
    with pytest.raises(RuntimeError, match="fewer outputs"):
        model.predict(_images(2), imgsz=70)


def test_train_and_export_are_not_supported(model):
    with pytest.raises(NotImplementedError):
        model.train(data="x.yaml")
    with pytest.raises(NotImplementedError):
        model.export(format="onnx")
