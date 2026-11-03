"""Runner tests for the optical flow task: sequences, windows, video, saving."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.models.base.inference import InferenceRunner
from libreyolo.utils.image_loader import ImageLoader
from libreyolo.utils.results import FlowField, Results

pytestmark = pytest.mark.unit

H, W = 24, 32


class _StubFlowModel:
    """A flow family double: the flow it returns encodes which frames it saw.

    ``_forward_flow`` records every window and returns the mean-brightness
    difference between frame t + 1 and frame t; ``_postprocess`` spreads it
    over a constant ``(H, W, 2)`` field. Frames are built with distinct
    shades, so the field value identifies the pair.
    """

    task = "flow"
    family = "stubflow"
    TTA_ENABLED = False
    FLOW_CONTEXT = (0, 0)
    size = "s"
    names = {0: "flow"}
    device = torch.device("cpu")

    def __init__(self, context=(0, 0), with_confidence=False):
        self.FLOW_CONTEXT = context
        self.with_confidence = with_confidence
        self.windows = []
        self.window_means = []

    def _get_input_size(self):
        return 32

    def _get_model_name(self):
        return "stub"

    def _preprocess(self, image, color_format="auto", input_size=None):
        pil = ImageLoader.load(image, color_format=color_format)
        tensor = torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1).float().unsqueeze(0)
        return tensor, pil, pil.size, 1.0

    def _forward_flow(self, window):
        self.windows.append(tuple(window.shape))
        self.window_means.append([round(float(f.mean())) for f in window])
        before = self.FLOW_CONTEXT[0]
        delta = float(window[before + 1].mean() - window[before].mean())
        return {"delta": delta}

    def _postprocess(self, output, conf, iou, original_size, max_det=300, ratio=1.0, classes=None, **kwargs):
        w, h = original_size
        detections = {"flow": torch.full((h, w, 2), output["delta"])}
        if self.with_confidence:
            detections["flow_confidence"] = np.full((h, w), 0.5, dtype=np.float32)
        return detections


def _frame(shade):
    return np.full((H, W, 3), shade, dtype=np.uint8)


def _frames(n):
    return [_frame(10 * i) for i in range(n)]


def test_single_image_is_an_error():
    runner = InferenceRunner(_StubFlowModel())
    with pytest.raises(ValueError, match="at least two frames"):
        runner(_frame(0))


def test_list_of_three_frames_yields_two_results():
    model = _StubFlowModel()
    results = InferenceRunner(model)(_frames(3))
    assert isinstance(results, list) and len(results) == 2
    for t, result in enumerate(results):
        assert isinstance(result, Results)
        assert result.boxes is None
        assert isinstance(result.flow, FlowField)
        assert tuple(result.flow.data.shape) == (H, W, 2)
        assert result.flow.orig_shape == (H, W)
        assert float(result.flow.data[0, 0, 0]) == pytest.approx(10.0)
        assert result.path is None
        # In-memory frames keep their pixels: result t shows frame t.
        assert int(np.asarray(result.orig_img).mean()) == 10 * t
    assert model.windows == [(2, 3, H, W), (2, 3, H, W)]
    assert model.window_means == [[0, 10], [10, 20]]


def test_tuple_source_is_a_sequence_too():
    results = InferenceRunner(_StubFlowModel())(tuple(_frames(2)))
    assert len(results) == 1


def test_context_window_is_edge_padded():
    model = _StubFlowModel(context=(1, 1))
    results = InferenceRunner(model)(_frames(3))
    assert len(results) == 2
    assert model.windows == [(4, 3, H, W), (4, 3, H, W)]
    assert model.window_means == [[0, 0, 10, 20], [0, 10, 20, 20]]
    assert all(float(r.flow.data[0, 0, 0]) == pytest.approx(10.0) for r in results)


def test_stream_returns_a_generator_and_drops_the_last_frame():
    gen = InferenceRunner(_StubFlowModel())(_frames(4), stream=True)
    assert hasattr(gen, "__next__")
    results = list(gen)
    assert len(results) == 3


def test_frame_iterator_source_is_lazy():
    def frames():
        for shade in (0, 10, 20):
            yield Image.fromarray(_frame(shade))

    results = InferenceRunner(_StubFlowModel())(frames())
    assert len(results) == 2


def test_directory_pairs_frames_in_name_order(tmp_path):
    for index, shade in enumerate((0, 10, 20, 30)):
        Image.fromarray(_frame(shade)).save(tmp_path / f"frame_{index:03d}.png")
    results = InferenceRunner(_StubFlowModel())(str(tmp_path))
    assert len(results) == 3
    assert [r.path for r in results] == [
        str(tmp_path / f"frame_{index:03d}.png") for index in range(3)
    ]
    assert all(float(r.flow.data[0, 0, 0]) == pytest.approx(10.0) for r in results)


def test_confidence_passes_through():
    results = InferenceRunner(_StubFlowModel(with_confidence=True))(_frames(2))
    assert results[0].flow.confidence is not None
    assert tuple(results[0].flow.confidence.shape) == (H, W)
    assert float(results[0].flow.confidence[0, 0]) == pytest.approx(0.5)


def test_save_writes_one_image_per_pair(tmp_path):
    out = tmp_path / "out"
    results = InferenceRunner(_StubFlowModel())(_frames(3), save=True, output_path=str(out))
    assert len(results) == 2
    written = sorted(p.name for p in out.iterdir())
    assert written == ["image0.jpg", "image1.jpg"]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"tiling": True}, "Tiled inference"),
        ({"augment": True}, "augmentation"),
        ({"batch": 2}, "batch > 1"),
    ],
)
def test_unsupported_modes_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        InferenceRunner(_StubFlowModel())(_frames(3), **kwargs)


def test_missing_flow_hook_is_a_clear_error():
    class _NoHook(_StubFlowModel):
        def _forward_flow(self, window):
            raise NotImplementedError("stubflow does not implement _forward_flow")

    with pytest.raises(NotImplementedError, match="_forward_flow"):
        InferenceRunner(_NoHook())(_frames(2))


def test_postprocess_without_flow_is_an_error():
    class _NoFlow(_StubFlowModel):
        def _postprocess(self, output, *args, **kwargs):
            return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    with pytest.raises(ValueError, match="no 'flow' entry"):
        InferenceRunner(_NoFlow())(_frames(2))


def test_mixed_frame_sizes_are_an_error():
    frames = [_frame(0), np.full((H + 8, W, 3), 10, dtype=np.uint8)]
    with pytest.raises(ValueError, match="share a size"):
        InferenceRunner(_StubFlowModel())(frames)


@pytest.fixture
def shaded_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = str(tmp_path / "shades.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    for i in range(10):
        writer.write(np.full((64, 64, 3), i * 25, dtype=np.uint8))
    writer.release()
    return path


def test_video_pairs_consecutive_frames(shaded_video):
    model = _StubFlowModel()
    results = InferenceRunner(model)(shaded_video)
    assert len(results) == 9
    assert [r.frame_idx for r in results] == list(range(9))
    assert all(r.path == shaded_video for r in results)
    # Lossy encoding keeps constant frames within a few levels of the shade.
    assert all(abs(float(r.flow.data[0, 0, 0]) - 25.0) < 4 for r in results)
    # Collected lists do not hold every decoded frame.
    assert all(getattr(r, "_orig_img", None) is None for r in results)


def test_video_vid_stride_pairs_the_kept_frames(shaded_video):
    results = list(InferenceRunner(_StubFlowModel())(shaded_video, vid_stride=3, stream=True))
    assert [r.frame_idx for r in results] == [0, 3, 6]
    assert all(abs(float(r.flow.data[0, 0, 0]) - 75.0) < 6 for r in results)
    # Streamed results keep frame t for plotting.
    assert all(r.orig_img is not None for r in results)


def test_video_save_writes_a_video(shaded_video, tmp_path):
    out = tmp_path / "flow.mp4"
    results = InferenceRunner(_StubFlowModel())(shaded_video, save=True, output_path=str(out))
    assert len(results) == 9
    assert out.exists() and out.stat().st_size > 0


def test_screen_without_stream_is_an_error():
    with pytest.raises(ValueError, match="stream=True"):
        InferenceRunner(_StubFlowModel())("screen")


def test_live_source_without_stream_is_an_error():
    with pytest.raises(ValueError, match="stream=True"):
        InferenceRunner(_StubFlowModel())(0)


def test_multiple_streams_are_rejected():
    with pytest.raises(NotImplementedError, match="one live source"):
        InferenceRunner(_StubFlowModel())([0, 1], stream=True)
