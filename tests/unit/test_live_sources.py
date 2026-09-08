"""Local-only coverage for live prediction-source dispatch and capture."""

from __future__ import annotations

import io
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.utils import source as source_module
from libreyolo.utils.results import Probs, Results
from libreyolo.utils.source import (
    ImageSequenceSource,
    MultiStreamSource,
    SourceKind,
    StreamFrame,
    StreamSource,
    classify_source,
    redact_source,
    resolve_youtube_stream,
)
from libreyolo.utils.video import run_video_inference

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", [0, 2, "0", "3"])
def test_webcam_indices_dispatch_as_streams(value):
    spec = classify_source(value)
    assert spec.kind == SourceKind.STREAM
    assert spec.items == (int(value),)


@pytest.mark.parametrize(
    "value",
    [
        "rtsp://127.0.0.1:8554/camera",
        "rtmp://127.0.0.1/live/camera",
        "tcp://127.0.0.1:9000",
        "udp://127.0.0.1:9000",
        "https://example.test/live/camera.m3u8",
    ],
)
def test_network_urls_dispatch_before_image_paths(value):
    spec = classify_source(value)
    assert spec.kind == SourceKind.STREAM
    assert spec.items == (value,)


@pytest.mark.parametrize(
    "value",
    [
        "https://www.youtube.com/watch?v=abc123",
        "https://youtu.be/abc123",
    ],
)
def test_youtube_pages_dispatch_as_streams(value):
    assert classify_source(value).kind == SourceKind.STREAM


def test_image_url_remains_an_image():
    assert classify_source("https://example.test/camera.jpg").kind == SourceKind.IMAGE


def test_stream_list_file_supports_comments_webcams_and_rtsp(tmp_path):
    path = tmp_path / "cameras.streams"
    path.write_text(
        "# loading dock\n0\n\nrtsp://127.0.0.1:8554/warehouse\n",
        encoding="utf-8",
    )
    spec = classify_source(path)
    assert spec.kind == SourceKind.STREAMS
    assert spec.items == (0, "rtsp://127.0.0.1:8554/warehouse")


def test_multi_source_list_cannot_mix_images_and_streams():
    with pytest.raises(TypeError, match="cannot mix"):
        classify_source([0, np.zeros((4, 4, 3), dtype=np.uint8)])


def test_rtsp_credentials_are_redacted_from_result_label():
    label = redact_source("rtsp://alice:secret@camera.test:8554/live")
    assert "secret" not in label
    assert label == "rtsp://alice:***@camera.test:8554/live"


def test_youtube_resolution_uses_yt_dlp_without_downloading(monkeypatch):
    calls = {}

    class FakeYDL:
        def __init__(self, options):
            calls["options"] = options

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def extract_info(self, url, download):
            calls["url"] = url
            calls["download"] = download
            return {"url": "https://media.test/direct.mp4"}

    monkeypatch.setattr(
        source_module,
        "_import_yt_dlp",
        lambda: SimpleNamespace(YoutubeDL=FakeYDL),
    )
    resolved = resolve_youtube_stream("https://youtu.be/example")
    assert resolved == "https://media.test/direct.mp4"
    assert calls["download"] is False
    assert calls["options"]["noplaylist"] is True


class _FakeCapture:
    def __init__(self, source, frames, *, first_read_barrier=None):
        self.source = source
        self.frames = list(frames)
        self.first_read_barrier = first_read_barrier
        self.read_count = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2

        values = {
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_WIDTH: 8,
            cv2.CAP_PROP_FRAME_HEIGHT: 6,
        }
        return values.get(prop, 0)

    def read(self):
        if self.read_count == 0 and self.first_read_barrier is not None:
            self.first_read_barrier.wait(timeout=2)
        self.read_count += 1
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


def _frames(source_index, count=3):
    return [
        np.full((6, 8, 3), source_index * 10 + i, dtype=np.uint8) for i in range(count)
    ]


def test_single_webcam_capture_runs_on_reader_thread(monkeypatch):
    import cv2

    captures = []

    def fake_video_capture(source, *args):
        capture = _FakeCapture(source, _frames(0))
        captures.append(capture)
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", fake_video_capture)
    with StreamSource(0, buffer=True) as stream:
        packets = list(stream)

    assert [packet.frame_idx for packet in packets] == [0, 1, 2]
    assert [int(packet.frame_bgr[0, 0, 0]) for packet in packets] == [0, 1, 2]
    assert captures[0].released is True
    assert stream._thread is not None
    assert stream._thread.name == "libreyolo-stream-0"


def test_multiple_cameras_open_and_read_independently(monkeypatch):
    import cv2

    barrier = threading.Barrier(2)
    captures = {}

    def fake_video_capture(source, *args):
        capture = _FakeCapture(
            source,
            _frames(int(source), count=2),
            first_read_barrier=barrier,
        )
        captures[source] = capture
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", fake_video_capture)
    with MultiStreamSource([0, 1], buffer=True) as streams:
        packets = list(streams)

    assert {(packet.source_index, packet.frame_idx) for packet in packets} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    assert {packet.source_label for packet in packets} == {"0", "1"}
    assert all(capture.released for capture in captures.values())


def test_existing_numeric_filename_is_not_claimed_as_webcam(tmp_path, monkeypatch):
    numeric_file = tmp_path / "0"
    numeric_file.write_bytes(b"not an image")
    monkeypatch.chdir(tmp_path)
    assert classify_source("0").kind == SourceKind.IMAGE


def test_missing_stream_list_fails_at_dispatch(tmp_path):
    path = Path(tmp_path) / "missing.streams"
    with pytest.raises(FileNotFoundError, match="Stream list not found"):
        classify_source(path)


def test_shared_video_loop_preserves_per_camera_path_and_frame_index():
    class FakeMultiSource:
        total_frames = 0
        fps = 25.0
        width = 8
        height = 6
        save_name = "streams"
        num_streams = 2

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def __iter__(self):
            for source_index, label in enumerate(("camera-a", "camera-b")):
                yield StreamFrame(
                    frame_bgr=np.zeros((6, 8, 3), dtype=np.uint8),
                    frame_idx=4 + source_index,
                    source_index=source_index,
                    source_label=label,
                    fps=25.0,
                )

    def predict_frame(_image):
        return Results(
            boxes=None,
            orig_shape=(6, 8),
            probs=Probs(torch.tensor([1.0])),
            names={0: "frame"},
        )

    results = list(
        run_video_inference(
            FakeMultiSource(),
            predict_frame,
            progress=False,
        )
    )

    assert [(result.path, result.frame_idx) for result in results] == [
        ("camera-a", 4),
        ("camera-b", 5),
    ]


def test_list_of_images_dispatches_as_image_batch():
    images = [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
    spec = classify_source(images)
    assert spec.kind == SourceKind.IMAGE_BATCH
    assert spec.items == tuple(images)


def test_bare_generator_dispatches_as_image_sequence():
    def frames():
        yield Image.new("RGB", (4, 4))

    gen = frames()
    spec = classify_source(gen)
    assert spec.kind == SourceKind.IMAGE_SEQUENCE
    assert spec.source is gen


def test_ndarray_is_not_mistaken_for_an_image_sequence():
    # np.ndarray has __iter__ but not __next__; must stay a single IMAGE.
    assert classify_source(np.zeros((4, 4, 3), dtype=np.uint8)).kind == SourceKind.IMAGE


def test_bytesio_is_not_mistaken_for_an_image_sequence():
    assert classify_source(io.BytesIO(b"image bytes")).kind == SourceKind.IMAGE


class TestImageSequenceSource:
    def test_iterates_a_finite_list_reporting_its_length(self):
        images = [Image.new("RGB", (4, 4)) for _ in range(3)]
        src = ImageSequenceSource(images, fps=15.0, save_name="clip")

        assert src.total_frames == 3
        with src as opened:
            packets = list(opened)

        assert [p.frame_idx for p in packets] == [0, 1, 2]
        assert all(p.fps == 15.0 for p in packets)
        assert all(p.frame_bgr.shape == (4, 4, 3) for p in packets)
        assert src.save_name == "clip"

    def test_iterates_a_lazy_iterator_reporting_unknown_length(self):
        def frames():
            for _ in range(5):
                yield Image.new("RGB", (4, 4))

        src = ImageSequenceSource(frames())
        assert src.total_frames == 0

        packets = list(src)
        assert [p.frame_idx for p in packets] == [0, 1, 2, 3, 4]

    def test_applies_vid_stride(self):
        images = [Image.new("RGB", (4, 4)) for _ in range(5)]
        src = ImageSequenceSource(images, vid_stride=2)

        packets = list(src)
        assert [p.frame_idx for p in packets] == [0, 2, 4]
        assert src.total_frames == 3

    def test_vid_stride_scales_down_reported_fps(self):
        # Retaining half the frames must halve the reported fps, or a saved
        # output video (written one retained frame per tick) plays back
        # vid_stride times too fast.
        images = [Image.new("RGB", (4, 4)) for _ in range(4)]
        src = ImageSequenceSource(images, vid_stride=2, fps=30.0)

        packets = list(src)
        assert all(p.fps == 15.0 for p in packets)

    def test_bgr_numpy_frames_preserve_their_color_format(self):
        frame_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        frame_bgr[:] = [0, 0, 255]
        src = ImageSequenceSource([frame_bgr], color_format="bgr")

        (packet,) = list(src)
        assert packet.frame_bgr[0, 0].tolist() == [0, 0, 255]

    @pytest.mark.parametrize("fps", [0, -1, np.nan, np.inf])
    def test_rejects_invalid_fps(self, fps):
        with pytest.raises(ValueError, match="fps must be a finite value > 0"):
            ImageSequenceSource([Image.new("RGB", (4, 4))], fps=fps)

    def test_path_items_are_reported_as_the_source_label(self, tmp_path):
        path = tmp_path / "frame.png"
        Image.new("RGB", (4, 4)).save(path)
        src = ImageSequenceSource([path])

        (packet,) = list(src)
        assert packet.source_label == str(path)

    def test_in_memory_items_have_no_source_label(self):
        src = ImageSequenceSource([Image.new("RGB", (4, 4))])

        (packet,) = list(src)
        assert packet.source_label is None

    def test_reiteration_raises(self):
        src = ImageSequenceSource([Image.new("RGB", (4, 4))])
        list(src)
        with pytest.raises(RuntimeError, match="already been consumed"):
            list(src)
