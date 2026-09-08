"""Video source utilities for LibreYOLO."""

import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, List, Protocol, Tuple, Union

import numpy as np

from .general import increment_path

logger = logging.getLogger(__name__)

MP4_CODEC_CANDIDATES = ("avc1", "mp4v")

# ``(codec, width, height)`` combinations that failed to open in this process.
# OpenCV's FFmpeg backend prints its own errors to stderr while probing (for
# example ``h264_v4l2m2m`` or ``Failed to initialize VideoWriter``), so probe
# once per process rather than once per video. The frame size is part of the
# key because some H.264 encoders reject odd dimensions that ``mp4v`` accepts,
# and one such video must not disable H.264 for every later one.
_UNAVAILABLE_CODECS: set = set()

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


class FrameSource(Protocol):
    """What ``run_video_inference`` needs from a frame source.

    :class:`VideoSource` and :class:`~libreyolo.utils.screen.ScreenSource` both
    satisfy it: a context manager that iterates ``(BGR frame, frame index)``
    and reports the geometry needed to write an output video.
    """

    fps: float
    total_frames: int
    width: int
    height: int
    save_name: str

    def __enter__(self) -> "FrameSource": ...

    def __exit__(self, *exc) -> None: ...

    def __iter__(self) -> Iterator[Any]: ...


def is_video_file(source) -> bool:
    """Check whether *source* looks like a path to a video file."""
    if not isinstance(source, (str, Path)):
        return False
    return Path(source).suffix.lower() in VIDEO_EXTENSIONS


def _codec_candidates(path: Union[str, Path]) -> Tuple[str, ...]:
    if Path(path).suffix.lower() == ".mp4":
        return MP4_CODEC_CANDIDATES
    return ("mp4v",)


def resolve_video_save_path(
    source: Union[str, Path], output_path: Union[str, None]
) -> str:
    """Determine the output path for a saved video.

    If *output_path* is provided, uses it directly. Otherwise creates an
    auto-incrementing directory under ``runs/detect/predict*/``.
    """
    if output_path is not None:
        out = Path(output_path)
        if out.suffix:
            out.parent.mkdir(parents=True, exist_ok=True)
            return str(out)
        out.mkdir(parents=True, exist_ok=True)
        return str(out / f"{Path(source).stem}.mp4")

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
        self.save_name = self._path
        self._vid_stride = max(1, int(vid_stride))

        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            self._cap.release()
            raise ValueError(f"Cannot open video file: {self._path}")

        self._iterated = False

        detected_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not detected_fps:
            detected_fps = 30.0
            logger.warning(f"Could not detect video FPS, defaulting to {detected_fps}")
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
        self._frame_size = (int(width), int(height))
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        self.codec = None
        self._writer = None
        candidates = _codec_candidates(self._path)
        failed = []
        last_resort = candidates[-1]
        for codec in candidates:
            key = (codec, width, height)
            if key in _UNAVAILABLE_CODECS and codec != last_resort:
                continue
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(self._path, fourcc, fps, (width, height))
            if writer.isOpened():
                self.codec = codec
                self._writer = writer
                break
            writer.release()
            failed.append(key)

        if self._writer is None:
            raise ValueError(f"Cannot open video writer for: {self._path}")
        # Only remember failures once a later codec proved the path itself is
        # writable, so a bad output path does not poison the cache.
        _UNAVAILABLE_CODECS.update(failed)

        if self.codec != candidates[0]:
            # Expected on most Linux OpenCV wheels, which ship without an H.264
            # encoder. The output is still a valid MP4, so this is not a warning.
            logger.info(
                "H.264 encoder not available in this OpenCV build; saving %s with %s. "
                "Any FFmpeg errors printed above come from that probe and can be ignored.",
                self._path,
                self.codec,
            )

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
        if self._writer is None:
            raise RuntimeError("VideoWriter has already been released")
        height, width = frame_bgr.shape[:2]
        if (width, height) != self._frame_size:
            expected_width, expected_height = self._frame_size
            raise ValueError(
                "Video frame size changed from "
                f"{expected_width}x{expected_height} to {width}x{height}. "
                "All frames in a saved video must have the same dimensions."
            )
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


def uniform_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    """Deterministic uniform frame indices over a finite video.

    The sampling rule is fixed so validation and inference agree:

    * the temporal endpoints are always included when at least two frames are
      requested and available;
    * the last available frame is repeated **only** when the video yields fewer
      frames than requested.

    Args:
        total_frames: Number of frames the video actually decodes.
        num_frames: Number of frames to sample (must be positive).

    Returns:
        A list of ``num_frames`` frame indices in non-decreasing order.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive; got {num_frames}.")
    if total_frames < 1:
        raise ValueError("Video decoded zero frames.")
    if num_frames == 1 or total_frames == 1:
        return [0] * num_frames if total_frames == 1 else [0]
    if total_frames >= num_frames:
        step = (total_frames - 1) / (num_frames - 1)
        return [int(round(i * step)) for i in range(num_frames)]
    # Short video: take every frame, then hold the last one.
    return list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)


def sample_clip_frames(path: Union[str, Path], num_frames: int) -> List:
    """Uniformly sample ``num_frames`` RGB ``PIL.Image`` frames from a finite video.

    Family-independent: decoding and sampling live here, while tensor layout,
    preprocessing and temporal pooling stay family-local.
    """
    import cv2
    from PIL import Image

    with VideoSource(path) as source:
        total = int(source.total_frames)
        cap = source._cap
        if total < 1:
            # Some containers do not report a frame count; fall back to a scan.
            total = 0
            while cap.grab():
                total += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if total < 1:
            raise ValueError(f"Video decoded zero frames: {path}")

        wanted = uniform_frame_indices(total, num_frames)
        frames = []
        for index in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame_bgr = cap.read()
            if not ok:
                if not frames:
                    raise ValueError(f"Could not decode frame {index} of {path}.")
                frames.append(frames[-1])
                continue
            frames.append(Image.fromarray(frame_bgr[:, :, ::-1].copy()))
    return frames


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
    source: Union[str, Path, "FrameSource"],
    predict_frame_fn: Callable,
    *,
    vid_stride: int = 1,
    save: bool = False,
    show: bool = False,
    output_path: Union[str, None] = None,
    annotate_fn: Union[Callable, None] = None,
    progress: bool = True,
) -> Generator:
    """Generic frame-by-frame inference loop shared by all backends.

    Args:
        source: Path to a video file, or an already-built frame source such as
            :class:`VideoSource` or
            :class:`~libreyolo.utils.screen.ScreenSource`. A frame source
            yields ``(BGR frame, frame index)`` and has already applied its own
            ``vid_stride``, so *vid_stride* is ignored for that form.
        predict_frame_fn: Callable that takes a PIL RGB image and returns
            a ``Results`` object.
        vid_stride: Process every N-th frame (file sources only).
        save: Write annotated output video.
        show: Display frames in a cv2 window.
        output_path: Output path for saved video.
        annotate_fn: Optional callable ``(pil_img, result) -> pil_img`` for
            custom annotation (e.g. tracking labels). When *None*, the default
            ``draw_boxes()`` annotation is used.
        progress: Show a tqdm progress bar (frames processed, fps).

    Yields:
        ``Results`` for each processed frame.
    """
    import cv2
    import torch
    from PIL import Image
    from tqdm import tqdm

    from .drawing import (
        draw_boxes,
        draw_depth_map,
        draw_edge_map,
        draw_normal_map,
        draw_keypoints,
        draw_masks,
        draw_matte,
        draw_obb,
        draw_points,
    )

    if isinstance(source, (str, Path)):
        frame_source = VideoSource(source, vid_stride=vid_stride)
        # VideoSource decodes every frame and this loop sees only the kept ones,
        # so the frame count and output fps both scale down by the stride.
        stride_divisor = max(1, vid_stride)
        save_name = source
        desc = Path(source).name
    else:
        # A pre-built frame source has already applied its own stride.
        frame_source = source
        stride_divisor = 1
        save_name = getattr(source, "save_name", "stream")
        desc = str(save_name)

    with frame_source as video_src:
        writers = {}
        out_paths = {}
        out_path = None
        effective_fps = None
        is_multi = int(getattr(video_src, "num_streams", 1)) > 1
        multi_output_dir = None
        multi_output_template = None
        if save:
            if is_multi:
                if output_path is None:
                    multi_output_dir = increment_path(
                        Path("runs/detect") / "predict",
                        exist_ok=False,
                        mkdir=True,
                    )
                else:
                    requested = Path(output_path)
                    if requested.suffix:
                        requested.parent.mkdir(parents=True, exist_ok=True)
                        multi_output_dir = requested.parent
                        multi_output_template = requested
                    else:
                        requested.mkdir(parents=True, exist_ok=True)
                        multi_output_dir = requested
            else:
                out_path = resolve_video_save_path(save_name, output_path)
            effective_fps = video_src.fps / stride_divisor
            # The writer is created lazily from the first output frame instead
            # of the source dimensions: restore/super-resolution results render
            # on a canvas ``restore_scale`` times the source frame.

        total = video_src.total_frames // stride_divisor or None
        pbar = (
            tqdm(total=total, desc=desc, unit="frame", dynamic_ncols=True)
            if progress
            else None
        )

        try:
            for frame_item in video_src:
                if hasattr(frame_item, "frame_bgr"):
                    frame_bgr = frame_item.frame_bgr
                    frame_idx = int(frame_item.frame_idx)
                    source_index = int(frame_item.source_index)
                    source_label = (
                        str(frame_item.source_label)
                        if frame_item.source_label is not None
                        else None
                    )
                    frame_fps = float(frame_item.fps or effective_fps or 30.0)
                else:
                    frame_bgr, frame_idx = frame_item
                    source_index = 0
                    source_label = None
                    frame_fps = float(effective_fps or 30.0)

                # Convert BGR frame to PIL RGB for the model pipeline
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                # Run model-specific inference
                result = predict_frame_fn(pil_img)
                result.frame_idx = frame_idx
                if source_label is not None:
                    result.path = source_label

                # Annotate frame for save/show
                if save or show:
                    if annotate_fn is not None:
                        annotated_pil = annotate_fn(pil_img, result)
                    elif (
                        result.boxes is None
                        and getattr(result, "probs", None) is not None
                    ):
                        annotated_pil = pil_img
                    elif (
                        result.boxes is None
                        and getattr(result, "points", None) is not None
                    ):
                        if len(result.points) > 0:
                            annotated_pil = draw_points(
                                pil_img,
                                result.points.xy.tolist(),
                                result.points.conf.tolist(),
                                result.points.cls.tolist(),
                                class_names=result.names,
                            )
                        else:
                            annotated_pil = pil_img
                    elif (
                        result.boxes is None
                        and getattr(result, "restored", None) is not None
                    ):
                        annotated_pil = Image.fromarray(
                            result.restored.array, mode="RGB"
                        )
                    elif (
                        result.boxes is None
                        and getattr(result, "matte", None) is not None
                    ):
                        # Checkerboard-composited cutout preview (video frames
                        # cannot carry an alpha channel, so the transparency is
                        # visualized instead).
                        annotated_pil = draw_matte(pil_img, result.matte.array)
                    elif (
                        result.boxes is None
                        and getattr(result, "depth_map", None) is not None
                    ):
                        depth_np = result.depth_map.data
                        if isinstance(depth_np, torch.Tensor):
                            depth_np = depth_np.cpu().numpy()
                        annotated_pil = draw_depth_map(pil_img, depth_np)
                    elif (
                        result.boxes is None
                        and getattr(result, "normal_map", None) is not None
                    ):
                        normal_np = result.normal_map.data
                        if isinstance(normal_np, torch.Tensor):
                            normal_np = normal_np.cpu().numpy()
                        annotated_pil = draw_normal_map(pil_img, normal_np)
                    elif (
                        result.boxes is None
                        and getattr(result, "edges", None) is not None
                    ):
                        edge_np = result.edges.data
                        if isinstance(edge_np, torch.Tensor):
                            edge_np = edge_np.cpu().numpy()
                        annotated_pil = draw_edge_map(pil_img, edge_np)
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
                        if result.obb is not None:
                            annotated_pil = draw_obb(
                                annotated_pil,
                                result.obb.xywhr.tolist(),
                                result.obb.conf.tolist(),
                                result.obb.cls.tolist(),
                                class_names=result.names,
                                track_ids=(
                                    result.obb.id.tolist()
                                    if result.obb.id is not None
                                    else None
                                ),
                            )
                        else:
                            annotated_pil = draw_boxes(
                                annotated_pil,
                                result.boxes.xyxy.tolist(),
                                result.boxes.conf.tolist(),
                                result.boxes.cls.tolist(),
                                class_names=result.names,
                            )
                        if result.keypoints is not None:
                            kpts_np = result.keypoints.data
                            if isinstance(kpts_np, torch.Tensor):
                                kpts_np = kpts_np.cpu().numpy()
                            annotated_pil = draw_keypoints(annotated_pil, kpts_np)
                    else:
                        annotated_pil = pil_img

                    annotated_bgr = cv2.cvtColor(
                        np.array(annotated_pil), cv2.COLOR_RGB2BGR
                    )

                    if save:
                        writer = writers.get(source_index)
                        if writer is None:
                            frame_h, frame_w = annotated_bgr.shape[:2]
                            if is_multi:
                                from .source import source_save_stem

                                assert source_label is not None
                                assert multi_output_dir is not None
                                if multi_output_template is not None:
                                    suffix = multi_output_template.suffix or ".mp4"
                                    filename = (
                                        f"{multi_output_template.stem}_{source_index}"
                                        f"{suffix}"
                                    )
                                else:
                                    stem = source_save_stem(source_label, source_index)
                                    filename = f"{stem}_{source_index}.mp4"
                                current_out_path = str(multi_output_dir / filename)
                            else:
                                current_out_path = out_path
                            assert current_out_path is not None
                            writer = VideoWriter(
                                current_out_path, frame_fps, frame_w, frame_h
                            )
                            writers[source_index] = writer
                            out_paths[source_index] = current_out_path
                        writer.write_frame(annotated_bgr)

                    if show:
                        window_name = (
                            f"LibreYOLO: {source_label}"
                            if is_multi and source_label is not None
                            else "LibreYOLO"
                        )
                        cv2.imshow(window_name, annotated_bgr)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                if pbar is not None:
                    n_dets = len(result) if result is not None else 0
                    pbar.set_postfix(dets=n_dets, refresh=False)
                    pbar.update(1)

                yield result

        finally:
            if pbar is not None:
                pbar.close()
            for source_index, writer in writers.items():
                writer.release()
                logger.info("Video saved to %s", out_paths[source_index])
            if show:
                cv2.destroyAllWindows()
