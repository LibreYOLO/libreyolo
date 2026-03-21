"""Video source utilities for LibreYOLO."""

from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

# Video extensions supported via OpenCV's VideoCapture
VIDEO_EXTENSIONS = {
    ".asf",
    ".avi",
    ".gif",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ts",
    ".wmv",
    ".webm",
}


def is_video_file(source) -> bool:
    """Check whether *source* looks like a path to a video file."""
    if not isinstance(source, (str, Path)):
        return False
    return Path(source).suffix.lower() in VIDEO_EXTENSIONS


class VideoSource:
    """Iterate over video frames using OpenCV.

    Args:
        path: Path to a video file.
        vid_stride: Process every N-th frame (default ``1`` = every frame).

    Yields:
        ``(frame_bgr, frame_idx)`` tuples where *frame_bgr* is a
        ``np.ndarray`` in BGR/uint8 format and *frame_idx* is the
        zero-based index of the frame in the original video.
    """

    def __init__(self, path: str | Path, vid_stride: int = 1):
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "Video support requires 'opencv-python'. "
                "Install it with: pip install opencv-python"
            )

        self._path = str(path)
        self._vid_stride = max(1, int(vid_stride))

        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video file: {self._path}")

        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Tuple[np.ndarray, int]]:
        frame_idx = 0
        while self._cap.isOpened():
            grabbed = self._cap.grab()
            if not grabbed:
                break

            # Only decode on the stride boundary
            if frame_idx % self._vid_stride == 0:
                ok, frame = self._cap.retrieve()
                if ok:
                    yield frame, frame_idx

            frame_idx += 1

        self.release()

    def release(self):
        """Release the underlying VideoCapture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        self.release()

    def __repr__(self) -> str:
        return (
            f"VideoSource(path='{self._path}', "
            f"fps={self.fps:.1f}, "
            f"frames={self.total_frames}, "
            f"size={self.width}x{self.height}, "
            f"vid_stride={self._vid_stride})"
        )


class VideoWriter:
    """Write annotated frames to a video file using OpenCV.

    Args:
        path: Output video file path (should end in ``.mp4``).
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "Video writing requires 'opencv-python'. "
                "Install it with: pip install opencv-python"
            )

        self._path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self._path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise ValueError(f"Cannot open video writer for: {self._path}")

    def write_frame(self, frame_bgr: np.ndarray):
        """Write a single BGR frame."""
        self._writer.write(frame_bgr)

    def release(self):
        """Flush and close the writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __del__(self):
        self.release()

    def __repr__(self) -> str:
        return f"VideoWriter(path='{self._path}')"
