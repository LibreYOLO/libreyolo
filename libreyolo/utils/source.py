"""Prediction-source dispatch and threaded live-stream capture.

This module owns the distinction between images, finite video, screen capture,
and live inputs. Keeping that decision in one place prevents stream locators
from falling through to :class:`ImageLoader` as if they were file paths.
"""

from __future__ import annotations

import io
import re
import sys
import threading
from collections import deque
from collections.abc import Sized
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Iterator, Sequence
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from .screen import is_screen_source
from .video import is_video_file


NETWORK_STREAM_SCHEMES = frozenset({"rtsp", "rtmp", "tcp", "udp"})
YOUTUBE_HOSTS = frozenset(
    {
        "youtu.be",
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
    }
)
STREAM_LIST_EXTENSION = ".streams"
_HLS_EXTENSIONS = frozenset({".m3u8"})
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


class SourceKind(str, Enum):
    """Kinds understood by the public prediction entry points."""

    IMAGE = "image"
    IMAGE_BATCH = "image_batch"
    IMAGE_SEQUENCE = "image_sequence"
    DIRECTORY = "directory"
    VIDEO = "video"
    SCREEN = "screen"
    STREAM = "stream"
    STREAMS = "streams"


@dataclass(frozen=True)
class SourceSpec:
    """A classified prediction source."""

    kind: SourceKind
    source: Any
    items: tuple[Any, ...] = ()

    @property
    def live(self) -> bool:
        return self.kind in {SourceKind.STREAM, SourceKind.STREAMS}


@dataclass(frozen=True)
class StreamFrame:
    """One frame from a live source, including per-camera identity."""

    frame_bgr: np.ndarray
    frame_idx: int
    source_index: int
    source_label: str | None
    fps: float


def _url_parts(value: Any):
    if not isinstance(value, (str, Path)):
        return None
    try:
        return urlsplit(str(value))
    except ValueError:
        return None


def is_youtube_url(source: Any) -> bool:
    """Return whether *source* is a YouTube page URL."""
    parts = _url_parts(source)
    if parts is None or parts.scheme.lower() not in {"http", "https"}:
        return False
    host = (parts.hostname or "").lower().rstrip(".")
    return host in YOUTUBE_HOSTS or host.endswith(".youtube.com")


def is_network_stream(source: Any) -> bool:
    """Return whether *source* is a direct network-video locator."""
    parts = _url_parts(source)
    if parts is None:
        return False
    if parts.scheme.lower() in NETWORK_STREAM_SCHEMES:
        return True
    return (
        parts.scheme.lower() in {"http", "https"}
        and Path(parts.path).suffix.lower() in _HLS_EXTENSIONS
    )


def _existing_path(source: Any) -> Path | None:
    if not isinstance(source, (str, Path)):
        return None
    try:
        path = Path(source)
        return path if path.exists() else None
    except (OSError, ValueError):
        return None


def _webcam_index(source: Any) -> int | None:
    if isinstance(source, bool):
        return None
    if isinstance(source, int):
        if source < 0:
            raise ValueError(f"Webcam index must be non-negative, got {source}")
        return source
    if isinstance(source, str) and source.strip().isdigit():
        # An existing file named "0" remains an ordinary file source.
        if _existing_path(source) is None:
            return int(source.strip())
    return None


def normalize_stream_locator(source: Any) -> int | str:
    """Normalize one webcam/video-stream locator for OpenCV."""
    webcam = _webcam_index(source)
    if webcam is not None:
        return webcam
    if isinstance(source, (str, Path)):
        return str(source)
    raise TypeError(
        "Live stream sources must be webcam indices or video stream URLs, "
        f"got {type(source).__name__}"
    )


def _is_stream_item(source: Any) -> bool:
    return (
        _webcam_index(source) is not None
        or is_network_stream(source)
        or is_youtube_url(source)
        or is_video_file(source)
    )


def _read_stream_list(path: Path) -> tuple[int | str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Stream list not found: {path}")
    sources = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        if not _is_stream_item(value):
            raise ValueError(
                f"Invalid stream source on line {line_number} of {path}: {value!r}"
            )
        sources.append(normalize_stream_locator(value))
    if not sources:
        raise ValueError(f"Stream list is empty: {path}")
    return tuple(sources)


def classify_source(source: Any) -> SourceSpec:
    """Classify one public prediction source without opening it."""
    if is_screen_source(source):
        return SourceSpec(SourceKind.SCREEN, source)

    webcam = _webcam_index(source)
    if webcam is not None:
        return SourceSpec(SourceKind.STREAM, webcam, (webcam,))

    if is_network_stream(source) or is_youtube_url(source):
        normalized = normalize_stream_locator(source)
        return SourceSpec(SourceKind.STREAM, normalized, (normalized,))

    if isinstance(source, (list, tuple)):
        items = tuple(source)
        stream_flags = tuple(_is_stream_item(item) for item in items)
        if items and all(stream_flags):
            normalized = tuple(normalize_stream_locator(item) for item in items)
            return SourceSpec(SourceKind.STREAMS, source, normalized)
        if any(stream_flags):
            raise TypeError(
                "A source list cannot mix live/video streams with image inputs"
            )
        return SourceSpec(SourceKind.IMAGE_BATCH, source, items)

    # BytesIO is both a supported atomic ImageInput and an iterator over bytes.
    # Keep it on the single-image path before recognizing lazy frame iterators.
    if isinstance(source, io.BytesIO):
        return SourceSpec(SourceKind.IMAGE, source)

    if hasattr(source, "__next__"):
        # A bare iterator/generator of already-loaded images (e.g. PIL frames
        # produced incrementally). Unlike IMAGE_BATCH its length is unknown
        # ahead of time, so it must stay lazy for the whole call.
        return SourceSpec(SourceKind.IMAGE_SEQUENCE, source)

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == STREAM_LIST_EXTENSION:
            items = _read_stream_list(path)
            return SourceSpec(SourceKind.STREAMS, source, items)
        if is_video_file(source):
            return SourceSpec(SourceKind.VIDEO, source)
        existing = _existing_path(source)
        if existing is not None and existing.is_dir():
            return SourceSpec(SourceKind.DIRECTORY, source)

    return SourceSpec(SourceKind.IMAGE, source)


def redact_source(source: int | str) -> str:
    """Create a user-facing label without exposing URL credentials."""
    if isinstance(source, int):
        return str(source)
    parts = _url_parts(source)
    if parts is None or not parts.scheme or not parts.netloc:
        return str(source)
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    if parts.username is not None:
        host = f"{parts.username}:***@{host}"
    # Direct stream query strings commonly contain bearer tokens or signed
    # URLs. A YouTube page's public video id is useful and not a credential;
    # retain that one query while dropping direct-stream queries from labels.
    query = parts.query if is_youtube_url(source) else ""
    return urlunsplit((parts.scheme, host, parts.path, query, ""))


def source_save_stem(source: int | str, source_index: int = 0) -> str:
    """Return a filesystem-safe name for a live source."""
    if isinstance(source, int):
        return f"webcam{source}"
    if is_youtube_url(source):
        return f"youtube{source_index}"
    parts = _url_parts(source)
    candidate = ""
    if parts is not None and parts.scheme:
        candidate = Path(parts.path).stem or parts.hostname or "stream"
    else:
        candidate = Path(str(source)).stem
    safe = _SAFE_STEM_RE.sub("_", candidate).strip("._")
    return safe or f"stream{source_index}"


def _import_yt_dlp():
    try:
        import yt_dlp
    except ImportError as exc:
        raise ImportError(
            "YouTube sources require yt-dlp. Install it with "
            '`pip install "libreyolo[stream]"`.'
        ) from exc
    return yt_dlp


def resolve_youtube_stream(source: str) -> str:
    """Resolve a YouTube page to a direct media URL without downloading it."""
    yt_dlp = _import_yt_dlp()
    options = {
        "format": "best[ext=mp4]/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(source, download=False)
    except Exception as exc:
        raise ConnectionError(f"Could not resolve YouTube source: {source}") from exc

    if isinstance(info, dict) and info.get("entries"):
        info = next((entry for entry in info["entries"] if entry), None)
    direct_url = info.get("url") if isinstance(info, dict) else None
    if not direct_url:
        raise ConnectionError(f"YouTube source has no playable media URL: {source}")
    return str(direct_url)


class StreamSource:
    """Read the latest frames from one webcam or network stream on a thread."""

    total_frames = 0

    def __init__(
        self,
        source: int | str | Path,
        *,
        vid_stride: int = 1,
        buffer: bool = False,
        source_index: int = 0,
        ready_event: threading.Event | None = None,
    ):
        self.source = normalize_stream_locator(source)
        self.source_index = int(source_index)
        self.source_label = redact_source(self.source)
        self.save_name = source_save_stem(self.source, self.source_index)
        self._vid_stride = max(1, int(vid_stride))
        self._buffer = bool(buffer)
        self._queue: Deque[StreamFrame] = deque(maxlen=None if buffer else 1)
        self._condition = threading.Condition()
        self._ready_event = ready_event
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap = None
        self._terminal = False
        self._error: BaseException | None = None
        self._iterated = False
        self._next_frame_idx = 0
        self.fps = 30.0
        self.width = 0
        self.height = 0

    @property
    def num_streams(self) -> int:
        return 1

    def _open_capture(self):
        import cv2

        capture_source: int | str = self.source
        if isinstance(capture_source, str) and is_youtube_url(capture_source):
            capture_source = resolve_youtube_stream(capture_source)

        if isinstance(capture_source, int) and sys.platform.startswith("win"):
            backend = getattr(cv2, "CAP_DSHOW", None)
            cap = (
                cv2.VideoCapture(capture_source, backend)
                if backend is not None
                else cv2.VideoCapture(capture_source)
            )
            if not cap.isOpened() and backend is not None:
                cap.release()
                cap = cv2.VideoCapture(capture_source)
        else:
            cap = cv2.VideoCapture(capture_source)
        return cap

    def _open(self) -> None:
        if self._cap is not None:
            return
        import cv2

        cap = self._open_capture()
        if not cap.isOpened():
            cap.release()
            raise ConnectionError(f"Cannot open live stream: {self.source_label}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if fps > 0 else 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise ConnectionError(
                f"Live stream opened but returned no frames: {self.source_label}"
            )

        self._cap = cap
        self._next_frame_idx = 1
        self._enqueue(frame, 0)

    def _enqueue(self, frame: np.ndarray, frame_idx: int) -> None:
        if frame_idx % self._vid_stride:
            return
        packet = StreamFrame(
            frame_bgr=frame,
            frame_idx=frame_idx,
            source_index=self.source_index,
            source_label=self.source_label,
            fps=self.fps,
        )
        with self._condition:
            self._queue.append(packet)
            self._condition.notify_all()
        if self._ready_event is not None:
            self._ready_event.set()

    def _mark_terminal(self, error: BaseException | None = None) -> None:
        with self._condition:
            self._terminal = True
            self._error = error
            self._condition.notify_all()
        if self._ready_event is not None:
            self._ready_event.set()

    def _reader(self) -> None:
        cap = self._cap
        assert cap is not None
        error = None
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_idx = self._next_frame_idx
                self._next_frame_idx += 1
                self._enqueue(frame, frame_idx)
        except BaseException as exc:
            error = RuntimeError(f"Live stream failed: {self.source_label}")
            error.__cause__ = exc
        finally:
            # The reader thread owns release after it starts. In particular,
            # this prevents VideoCapture.release() from racing FFmpeg read().
            cap.release()
            if self._cap is cap:
                self._cap = None
            self._mark_terminal(error)

    def _start_reader(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._reader,
            name=f"libreyolo-stream-{self.source_index}",
            daemon=True,
        )
        self._thread.start()

    def __enter__(self) -> "StreamSource":
        self._open()
        self._start_reader()
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def _take_nowait(self) -> tuple[StreamFrame | None, bool, BaseException | None]:
        with self._condition:
            packet = self._queue.popleft() if self._queue else None
            return packet, self._terminal and not self._queue, self._error

    def __iter__(self) -> Iterator[StreamFrame]:
        if self._iterated:
            raise RuntimeError("StreamSource has already been consumed")
        self._iterated = True
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._queue or self._terminal)
                if self._queue:
                    packet = self._queue.popleft()
                elif self._error is not None:
                    raise self._error
                else:
                    break
            yield packet

    def release(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._ready_event is not None:
            self._ready_event.set()
        thread = self._thread
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            # OpenCV/FFmpeg can crash if VideoCapture.release() races a read()
            # on another thread. A live reader normally returns once per frame,
            # observes _stop, and joins before the capture is closed.
            thread.join(timeout=2.0)
        if thread is not None and thread.is_alive():
            # A backend read can remain blocked while a network endpoint is
            # disappearing. The daemon reader retains ownership of its capture
            # and releases it when read() returns; closing it concurrently is
            # unsafe in FFmpeg.
            return
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()


class MultiStreamSource:
    """Multiplex multiple independently threaded video captures."""

    total_frames = 0
    fps = 30.0
    width = 0
    height = 0
    save_name = "streams"

    def __init__(
        self,
        sources: Sequence[int | str | Path],
        *,
        vid_stride: int = 1,
        buffer: bool = False,
    ):
        if not sources:
            raise ValueError("At least one live stream source is required")
        self._ready = threading.Event()
        self.streams = [
            StreamSource(
                source,
                vid_stride=vid_stride,
                buffer=buffer,
                source_index=index,
                ready_event=self._ready,
            )
            for index, source in enumerate(sources)
        ]
        self._iterated = False

    @property
    def num_streams(self) -> int:
        return len(self.streams)

    def __enter__(self) -> "MultiStreamSource":
        try:
            with ThreadPoolExecutor(max_workers=len(self.streams)) as executor:
                futures = [executor.submit(stream._open) for stream in self.streams]
                for future in futures:
                    future.result()
            for stream in self.streams:
                stream._start_reader()
        except BaseException:
            self.release()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def __iter__(self) -> Iterator[StreamFrame]:
        if self._iterated:
            raise RuntimeError("MultiStreamSource has already been consumed")
        self._iterated = True
        active = set(range(len(self.streams)))
        while active:
            self._ready.clear()
            emitted = False
            for index in tuple(active):
                packet, terminal, error = self.streams[index]._take_nowait()
                if packet is not None:
                    emitted = True
                    yield packet
                if error is not None:
                    raise error
                if terminal:
                    active.remove(index)
            if not emitted and active:
                self._ready.wait(timeout=0.1)

    def release(self) -> None:
        for stream in self.streams:
            stream.release()


class ImageSequenceSource:
    """Iterate an in-memory or lazily-produced sequence of images as frames.

    Wraps a finite list/tuple or a lazy iterator of already-loaded images
    (paths, PIL Images, ndarrays, tensors) so it can drive the same
    frame-by-frame loop (:func:`~libreyolo.utils.video.run_video_inference`)
    used for video files and live streams. This is what lets
    :meth:`~libreyolo.models.base.model.BaseModel.track` track across a
    batch or stream of images instead of only video files.

    Args:
        images: A finite sequence or an iterator/generator of image inputs.
            A finite sequence reports its length up front; an iterator's
            length is unknown and iteration stays lazy for the whole call.
        vid_stride: Process every N-th image (default ``1`` = every image).
        fps: Frame rate to report for the sequence. There is no real
            capture rate for a bag of images, so this only affects derived
            timing (a tracker's lost-track buffer, or the fps written into
            a saved output video).
        save_name: Base name used to derive an output path when saving.
        color_format: Color format hint passed to ``ImageLoader`` for NumPy
            frames: ``"auto"``, ``"rgb"``, or ``"bgr"``.

    Note:
        Can only be iterated once, like :class:`~libreyolo.utils.video.VideoSource`.
    """

    num_streams = 1

    def __init__(
        self,
        images: Sequence[Any] | Iterator[Any],
        *,
        vid_stride: int = 1,
        fps: float = 30.0,
        save_name: str = "sequence",
        color_format: str = "auto",
    ):
        self._images = images
        self._vid_stride = max(1, int(vid_stride))
        self.fps = float(fps)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"fps must be a finite value > 0, got {fps}")
        self.save_name = save_name
        raw_total = len(images) if isinstance(images, Sized) else 0
        self.total_frames = (
            (raw_total + self._vid_stride - 1) // self._vid_stride
            if raw_total
            else 0
        )
        self._color_format = color_format
        self.width = 0
        self.height = 0
        self._iterated = False

    def __enter__(self) -> "ImageSequenceSource":
        return self

    def __exit__(self, *exc) -> None:
        pass

    def __iter__(self) -> Iterator[StreamFrame]:
        if self._iterated:
            raise RuntimeError(
                "ImageSequenceSource has already been consumed. "
                "Create a new instance to iterate again."
            )
        self._iterated = True

        import cv2

        from .image_loader import ImageLoader

        # Fewer frames are retained at higher strides, so the fps reported
        # alongside them must drop proportionally -- otherwise a saved
        # output video plays back vid_stride times too fast (mirrors how
        # VideoSource's stride is compensated for in run_video_inference).
        effective_fps = self.fps / self._vid_stride

        for frame_idx, item in enumerate(self._images):
            if frame_idx % self._vid_stride:
                continue
            pil_img = ImageLoader.load(item, color_format=self._color_format)
            frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            label = str(item) if isinstance(item, (str, Path)) else None
            yield StreamFrame(
                frame_bgr=frame_bgr,
                frame_idx=frame_idx,
                source_index=0,
                source_label=label,
                fps=effective_fps,
            )


def build_stream_source(
    spec: SourceSpec,
    *,
    vid_stride: int = 1,
    stream_buffer: bool = False,
) -> StreamSource | MultiStreamSource:
    """Build a lazy threaded capture from a classified live source."""
    if spec.kind == SourceKind.STREAM:
        return StreamSource(spec.items[0], vid_stride=vid_stride, buffer=stream_buffer)
    if spec.kind == SourceKind.STREAMS:
        return MultiStreamSource(
            spec.items, vid_stride=vid_stride, buffer=stream_buffer
        )
    raise TypeError(f"Source kind {spec.kind.value!r} is not a live stream")
