"""Video source utilities for LibreYOLO."""

import logging
import warnings
from pathlib import Path
from typing import Callable, Generator, Iterator, Tuple, Union

import numpy as np

from .general import increment_path

logger = logging.getLogger(__name__)

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


def resolve_video_save_path(
    source: Union[str, Path], output_path: Union[str, None]
) -> str:
    """Determine the output path for a saved video.

    If *output_path* is provided, uses it directly. Otherwise creates an
    auto-incrementing directory under ``runs/detect/predict*/``.
    """
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        return str(out)

    save_dir = Path("runs/detect") / "predict"
    save_dir = increment_path(save_dir, exist_ok=False, mkdir=True)
    stem = Path(source).stem
    return str(save_dir / f"{stem}.mp4")


class VideoSource:
    """Iterate over video frames using OpenCV.

    Supports use as a context manager::

        with VideoSource("clip.mp4", vid_stride=2) as src:
            for frame_bgr, frame_idx in src:
                ...

    Args:
        path: Path to a video file.
        vid_stride: Process every N-th frame (default ``1`` = every frame).

    Note:
        A ``VideoSource`` instance can only be iterated **once**. After
        iteration completes (or the source is released), create a new
        instance to iterate again.
    """

    def __init__(self, path: Union[str, Path], vid_stride: int = 1):
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
            self._cap.release()
            raise ValueError(f"Cannot open video file: {self._path}")

        self._iterated = False

        detected_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not detected_fps:
            logger.warning("Could not detect video FPS, defaulting to 30.0")
            detected_fps = 30.0
        self.fps: float = detected_fps
        self.total_frames: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Tuple[np.ndarray, int]]:
        if self._cap is None or self._iterated:
            raise RuntimeError(
                "VideoSource has been consumed or released. "
                "Create a new instance to iterate again."
            )
        self._iterated = True

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
                else:
                    logger.warning(
                        "Failed to decode frame %d in %s, skipping",
                        frame_idx,
                        self._path,
                    )

            frame_idx += 1

    def release(self):
        """Release the underlying VideoCapture. Safe to call multiple times."""
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

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

    Supports use as a context manager::

        with VideoWriter("out.mp4", fps=25, width=1920, height=1080) as w:
            w.write_frame(frame_bgr)

    Args:
        path: Output video file path (should end in ``.mp4``).
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    def __init__(self, path: Union[str, Path], fps: float, width: int, height: int):
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

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # ------------------------------------------------------------------

    def write_frame(self, frame_bgr: np.ndarray):
        """Write a single BGR frame."""
        self._writer.write(frame_bgr)

    def release(self):
        """Flush and close the writer. Safe to call multiple times."""
        if self._writer is not None:
            try:
                self._writer.release()
            finally:
                self._writer = None

    def __repr__(self) -> str:
        return f"VideoWriter(path='{self._path}')"


# ---------------------------------------------------------------------------
# Shared video inference helpers
# ---------------------------------------------------------------------------

_LARGE_VIDEO_THRESHOLD = 500


def collect_video_results(
    gen: Generator,
    source: Union[str, Path],
    vid_stride: int = 1,
) -> list:
    """Collect all video results into a list, warning for large videos."""
    vs = VideoSource(source, vid_stride=vid_stride)
    est_frames = vs.total_frames // max(1, vid_stride)
    vs.release()

    if est_frames > _LARGE_VIDEO_THRESHOLD:
        warnings.warn(
            f"Video has ~{est_frames} frames to process. "
            f"Consider using stream=True to avoid high memory usage.",
            stacklevel=3,
        )
    return list(gen)


def run_video_inference(
    source: Union[str, Path],
    predict_frame_fn: Callable,
    *,
    vid_stride: int = 1,
    save: bool = False,
    show: bool = False,
    output_path: Union[str, None] = None,
    annotate_fn: Union[Callable, None] = None,
) -> Generator:
    """Generic video inference loop shared by all backends.

    Args:
        source: Path to video file.
        predict_frame_fn: Callable that takes a PIL RGB image and returns
            a ``Results`` object.
        vid_stride: Process every N-th frame.
        save: Write annotated output video.
        show: Display frames in a cv2 window.
        output_path: Output path for saved video.
        annotate_fn: Optional callable ``(pil_img, result) -> pil_img`` for
            custom annotation (e.g. tracking labels). When *None*, the default
            ``draw_boxes()`` annotation is used.

    Yields:
        ``Results`` for each processed frame.
    """
    import cv2
    import torch
    from PIL import Image

    from .drawing import draw_boxes, draw_masks

    with VideoSource(source, vid_stride=vid_stride) as video_src:
        writer = None
        out_path = None
        if save:
            out_path = resolve_video_save_path(source, output_path)
            effective_fps = video_src.fps / max(1, vid_stride)
            writer = VideoWriter(
                out_path, effective_fps, video_src.width, video_src.height
            )

        try:
            for frame_bgr, frame_idx in video_src:
                # Convert BGR frame to PIL RGB for the model pipeline
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                # Run model-specific inference
                result = predict_frame_fn(pil_img)
                result.frame_idx = frame_idx

                # Annotate frame for save/show
                if save or show:
                    if annotate_fn is not None:
                        annotated_pil = annotate_fn(pil_img, result)
                    elif len(result) > 0:
                        annotated_pil = pil_img
                        if result.masks is not None:
                            masks_np = result.masks.data
                            if isinstance(masks_np, torch.Tensor):
                                masks_np = masks_np.cpu().numpy()
                            annotated_pil = draw_masks(
                                annotated_pil,
                                masks_np,
                                result.boxes.cls.tolist(),
                            )
                        annotated_pil = draw_boxes(
                            annotated_pil,
                            result.boxes.xyxy.tolist(),
                            result.boxes.conf.tolist(),
                            result.boxes.cls.tolist(),
                            class_names=result.names,
                        )
                    else:
                        annotated_pil = pil_img

                    annotated_bgr = cv2.cvtColor(
                        np.array(annotated_pil), cv2.COLOR_RGB2BGR
                    )

                    if save and writer is not None:
                        writer.write_frame(annotated_bgr)

                    if show:
                        cv2.imshow("LibreYOLO", annotated_bgr)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                yield result

        finally:
            if writer is not None:
                writer.release()
                logger.info("Video saved to %s", out_path)
            if show:
                cv2.destroyAllWindows()
