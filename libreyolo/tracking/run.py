"""Shared ``track()`` driver for native models and exported backends.

``BaseModel.track`` and ``BaseBackend.track`` both delegate here: they differ
only in how one frame is predicted, so everything else (task guards, tracker
construction, ``persist``, custom-tracker checks, drawing and the
``runs/track`` save path) lives in one place.
"""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..utils.results import Results
from .protocol import Tracker

BUILTIN_TRACKERS = frozenset({"bytetrack", "botsort", "ocsort", "deepocsort"})

# Tasks whose results carry no axis-aligned objects to associate over time.
_UNSUPPORTED_TASKS: dict[str, str] = {
    "classify": "Tracking does not support classification models. Use predict().",
    "obb": (
        "Tracking does not support oriented boxes yet. Use predict() for OBB models."
    ),
    "point": (
        "Tracking does not support point results yet. Use predict() for point models."
    ),
    "depth": (
        "Tracking does not support depth maps yet. Use predict() for depth models."
    ),
    "flow": (
        "Tracking does not apply to optical flow: a flow result is a "
        "dense displacement field with no objects to associate. "
        "Use predict() for flow models."
    ),
    "normal": (
        "Tracking does not support surface-normal maps. "
        "Use predict() for normal models."
    ),
    "albedo": "Tracking does not support albedo maps. Use predict().",
    "edge": "Tracking does not support edge maps. Use predict() for edge models.",
    "semantic": (
        "Tracking does not support semantic segmentation yet. "
        "Use predict() for semantic models."
    ),
    "panoptic": (
        "Tracking does not support panoptic segmentation yet. "
        "Use predict() for panoptic models."
    ),
    "restore": "Tracking does not support restoration models. Use predict().",
    "mesh": (
        "Tracking does not support body-mesh models yet. Use predict(). "
        "Associating meshes over time also needs a temporal contract "
        "(track IDs on the mesh rows, and a world frame) that the mesh "
        "task does not define yet."
    ),
    "ocr": "Tracking does not support OCR models yet. Use predict().",
}

# Tasks that exported backends can predict but whose results the trackers
# cannot consume either; native models reject them in their own predict path.
_BACKEND_UNSUPPORTED_TASKS: dict[str, str] = {
    "embed": "Tracking does not support embedding models. Use predict().",
    "matte": "Tracking does not support matting models. Use predict().",
    "gaze": "Tracking does not support gaze models. Use predict().",
}


def check_track_task(task: str | None, *, backend: bool = False) -> None:
    """Raise NotImplementedError when *task* has nothing to track."""
    message = _UNSUPPORTED_TASKS.get(task or "detect")
    if message is None and backend:
        message = _BACKEND_UNSUPPORTED_TASKS.get(task or "detect")
    if message is not None:
        raise NotImplementedError(message)


def run_track(
    owner: Any,
    source,
    *,
    predict_frame: Callable[..., Results],
    task: str | None,
    reid_device: Callable[[], str | None],
    track_conf: float,
    conf: float | None,
    iou: float,
    imgsz,
    classes,
    max_det: int,
    save: bool,
    show: bool,
    vid_stride: int,
    fps: float,
    color_format: str,
    output_path: str | None,
    tracker: str | Tracker,
    tracker_config,
    persist: bool,
    tracker_kwargs: dict[str, Any],
    backend: bool = False,
) -> Generator[Results, None, None]:
    """Track objects frame by frame; see ``BaseModel.track`` for the contract.

    Args:
        owner: The model or backend. Its ``names`` label drawn boxes, and
            ``persist`` keeps its tracker on ``owner._track_state``.
        predict_frame: ``predict_frame(rgb_pil_image, conf=..., iou=...,
            imgsz=..., classes=..., max_det=...)`` returning one Results.
        task: The owner's task, checked against the trackable tasks.
        reid_device: Returns the torch device for Deep OC-SORT's ReID
            embedder, or None to let the embedder pick one. Only called when
            Deep OC-SORT is selected.
        backend: True for exported backends (rejects a few extra tasks).
    """
    check_track_task(task, backend=backend)

    from ..utils.drawing import draw_boxes, draw_masks
    from ..utils.image_loader import ImageLoader
    from ..utils.source import ImageSequenceSource, SourceKind, classify_source
    from ..utils.video import run_video_inference

    # Read from the package at call time so patched package attributes
    # (tests spy on the tracker classes this way) take effect.
    from . import (
        BoTSortConfig,
        BoTSortTracker,
        ByteTracker,
        DeepOCSortConfig,
        DeepOCSortTracker,
        OCSortConfig,
        OCSortTracker,
        TrackConfig,
    )

    # Classified up front (rather than after the tracker is built) so a
    # non-video source can seed the tracker's frame_rate default below.
    source_spec = classify_source(source)
    image_sequence_kinds = {
        SourceKind.IMAGE,
        SourceKind.IMAGE_BATCH,
        SourceKind.IMAGE_SEQUENCE,
        SourceKind.DIRECTORY,
    }
    if source_spec.kind in image_sequence_kinds:
        try:
            fps = float(fps)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"fps must be a finite value > 0, got {fps!r}") from exc
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"fps must be a finite value > 0, got {fps!r}")

    # persist=True continues the tracker kept by the previous persist=True
    # call (per-frame loops); a different tracker or config, or a different
    # video file/directory, starts a fresh one. In-memory frames and image
    # paths carry no stream identity, so they continue the stream.
    # persist=False always starts fresh and drops the kept tracker.
    source_key = (
        str(Path(source_spec.source).resolve())
        if source_spec.kind in (SourceKind.VIDEO, SourceKind.DIRECTORY)
        else None
    )
    kept = getattr(owner, "_track_state", None) if persist else None
    if kept is not None and kept["source"] != source_key:
        kept = None

    custom_tracker = tracker is not None and not isinstance(tracker, str)
    if custom_tracker:
        if isinstance(tracker, type) or not all(
            callable(getattr(tracker, method, None)) for method in ("update", "reset")
        ):
            raise TypeError(
                "tracker must be a built-in name or an instance with "
                "update(results, image=None) and reset() methods."
            )
        if tracker_config is not None or tracker_kwargs:
            raise ValueError(
                "Configure a custom tracker instance directly; tracker_config "
                "and tracker kwargs cannot be used with it."
            )
        if not math.isfinite(track_conf) or not 0 <= track_conf <= 1:
            raise ValueError("track_conf must be finite and between 0 and 1.")
        tracker_obj = tracker
        effective_conf = track_conf
        tracker_key = tracker_obj
        reused = kept is not None and kept["key"] is tracker_obj
    else:
        # A provided config picks the tracker; otherwise honour the selector.
        if isinstance(tracker_config, BoTSortConfig):
            # BoTSortConfig subclasses TrackConfig, so it must be checked first.
            tracker = "botsort"
        elif isinstance(tracker_config, DeepOCSortConfig):
            tracker = "deepocsort"
        elif isinstance(tracker_config, OCSortConfig):
            tracker = "ocsort"
        elif isinstance(tracker_config, TrackConfig):
            tracker = "bytetrack"
        tracker = (tracker or "bytetrack").lower()
        tracker_file = Path(tracker)
        if (
            tracker_file.suffix in (".yaml", ".yml")
            and tracker_file.parent == Path(".")
            and tracker_file.stem in BUILTIN_TRACKERS
        ):
            # The ecosystem names its built-in trackers by config file.
            if tracker_file.is_file():
                raise ValueError(
                    f"tracker={tracker!r} names a local file; LibreYOLO does not "
                    f"read tracker yaml files. Use tracker={tracker_file.stem!r} "
                    "and pass its settings as keyword arguments or tracker_config."
                )
            tracker = tracker_file.stem

        if (
            tracker in ("bytetrack", "botsort")
            and source_spec.kind in image_sequence_kinds
        ):
            # update() runs once per retained frame, so frame_rate must be
            # fps / vid_stride.
            retained_fps = fps / max(1, vid_stride)
            if tracker_config is None:
                # Pass the exact fraction: tracker.py already truncates via
                # int(), so pre-rounding here would double-round and can
                # shift max_time_lost by a frame at the expiry boundary.
                tracker_kwargs.setdefault("frame_rate", retained_fps)
            else:
                # frame_rate is a typed int field, so compare against the
                # nearest achievable value rather than the exact fraction.
                nearest_retained_fps = max(1, round(retained_fps))
                current = getattr(tracker_config, "frame_rate", nearest_retained_fps)
                if current != nearest_retained_fps:
                    # tracker_config always wins and is never overwritten --
                    # just warn so a silent mismatch doesn't linger.
                    warnings.warn(
                        f"tracker_config.frame_rate={tracker_config.frame_rate} does "
                        f"not match this image sequence's retained-frame rate "
                        f"(fps / vid_stride ≈ {nearest_retained_fps}). "
                        "tracker_config is used as-is for tracker timing, so lost "
                        "tracks will be kept recoverable for the wrong duration "
                        "unless "
                        f"you set frame_rate={nearest_retained_fps} on it yourself.",
                        stacklevel=3,
                    )

        if tracker == "deepocsort":
            if tracker_config is None:
                tracker_kwargs.setdefault("det_thresh", track_conf)
                tracker_config = DeepOCSortConfig.from_kwargs(**tracker_kwargs)
            # Deep OC-SORT has no low-score recovery band; the detector only
            # needs to produce boxes down to det_thresh.
            effective_conf = tracker_config.det_thresh
            tracker_cls = functools.partial(DeepOCSortTracker, device=reid_device())
        elif tracker == "ocsort":
            if tracker_config is None:
                tracker_kwargs.setdefault("det_thresh", track_conf)
                tracker_config = OCSortConfig.from_kwargs(**tracker_kwargs)
            # OC-SORT consumes low-score detections (>0.1) for recovery.
            effective_conf = min(0.1, tracker_config.det_thresh)
            tracker_cls = OCSortTracker
        elif tracker == "botsort":
            if tracker_config is None:
                tracker_kwargs.setdefault("track_high_thresh", track_conf)
                tracker_config = BoTSortConfig.from_kwargs(**tracker_kwargs)
            # BoT-SORT keeps ByteTrack's low-confidence recovery stage.
            effective_conf = tracker_config.track_low_thresh
            tracker_cls = BoTSortTracker
        elif tracker == "bytetrack":
            if tracker_config is None:
                tracker_kwargs.setdefault("track_high_thresh", track_conf)
                tracker_config = TrackConfig.from_kwargs(**tracker_kwargs)
            # ByteTrack needs to see low-confidence detections.
            effective_conf = tracker_config.track_low_thresh
            tracker_cls = ByteTracker
        else:
            raise ValueError(
                f"Unknown tracker {tracker!r}; "
                "choose 'bytetrack', 'botsort', 'ocsort' or 'deepocsort' "
                "(a .yaml suffix is accepted)."
            )
        tracker_key = (
            tracker,
            tracker_config,
            tracker_cls.keywords["device"] if tracker == "deepocsort" else None,
        )
        reused = kept is not None and kept["key"] == tracker_key
        tracker_obj = kept["tracker"] if reused else tracker_cls(config=tracker_config)

    default_stem = "sequence"
    if source_spec.kind == SourceKind.VIDEO:
        frame_source: Path | ImageSequenceSource = Path(source_spec.source)
        if not frame_source.exists():
            raise FileNotFoundError(f"Video file not found: {frame_source}")
        default_stem = frame_source.stem
    elif source_spec.kind == SourceKind.DIRECTORY:
        images = ImageLoader.collect_images(source_spec.source)
        if not images:
            return
        default_stem = Path(source_spec.source).name
        frame_source = ImageSequenceSource(
            images,
            vid_stride=vid_stride,
            fps=fps,
            save_name=default_stem,
            color_format=color_format,
        )
    elif source_spec.kind == SourceKind.IMAGE_BATCH:
        frame_source = ImageSequenceSource(
            list(source_spec.items),
            vid_stride=vid_stride,
            fps=fps,
            save_name=default_stem,
            color_format=color_format,
        )
    elif source_spec.kind == SourceKind.IMAGE_SEQUENCE:
        frame_source = ImageSequenceSource(
            source_spec.source,
            vid_stride=vid_stride,
            fps=fps,
            save_name=default_stem,
            color_format=color_format,
        )
    elif source_spec.kind == SourceKind.IMAGE:
        # A single already-loaded image or path, treated as a one-frame
        # sequence so predict()-style single-image inputs also work.
        frame_source = ImageSequenceSource(
            [source_spec.source],
            vid_stride=vid_stride,
            fps=fps,
            save_name=default_stem,
            color_format=color_format,
        )
    else:
        raise NotImplementedError(
            f"track() does not yet support {source_spec.kind.value!r} sources. "
            "Pass a video file, a directory or list of images, or an "
            "image iterator."
        )

    if custom_tracker and not reused:
        tracker_obj.reset()
    owner._track_state = (
        {"key": tracker_key, "source": source_key, "tracker": tracker_obj}
        if persist
        else None
    )

    if conf is not None:
        if not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError("conf must be finite and between 0 and 1.")
        effective_conf = conf

    model_names = owner.names

    def predict_and_track(pil_img):
        result = predict_frame(
            pil_img,
            conf=effective_conf,
            iou=iou,
            imgsz=imgsz,
            classes=classes,
            max_det=max_det,
        )
        if custom_tracker:
            return _check_custom_output(tracker_obj.update(result, image=pil_img))
        if isinstance(tracker_obj, (BoTSortTracker, DeepOCSortTracker)):
            # BoT-SORT needs pixels for camera motion; Deep OC-SORT needs
            # them for ReID crops.
            return tracker_obj.update(result, pil_img)
        return tracker_obj.update(result)

    def annotate_tracked(pil_img, result):
        if len(result) == 0:
            return pil_img
        img = pil_img
        if result.masks is not None:
            masks_np = result.masks.data
            if isinstance(masks_np, torch.Tensor):
                masks_np = masks_np.cpu().numpy()
            img = draw_masks(img, masks_np, result.boxes.cls.tolist())
        tid_list = result.track_id.tolist() if result.track_id is not None else None
        return draw_boxes(
            img,
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
            class_names=model_names,
            track_ids=tid_list,
        )

    # Use runs/track/ prefix instead of runs/detect/
    track_output = output_path
    if save and output_path is None:
        from ..utils.general import increment_path

        track_output = str(
            increment_path(
                Path("runs") / "track" / f"{default_stem}.mp4", exist_ok=False
            )
        )

    yield from run_video_inference(
        frame_source,
        predict_and_track,
        vid_stride=vid_stride,
        save=save,
        show=show,
        output_path=track_output,
        annotate_fn=annotate_tracked,
    )


def _check_custom_output(tracked) -> Results:
    """Validate a custom tracker's update() output and attach its IDs."""
    if not isinstance(tracked, Results):
        raise TypeError("Custom tracker update() must return Results.")
    ids = tracked.track_id
    if (
        not isinstance(ids, (torch.Tensor, np.ndarray))
        or ids.ndim != 1
        or len(ids) != len(tracked)
    ):
        raise ValueError(
            "Custom tracker Results.track_id must be a one-dimensional "
            "integer tensor/array aligned with the result rows."
        )
    if isinstance(ids, torch.Tensor):
        integer_ids = ids.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        )
    else:
        integer_ids = ids.dtype.kind in "iu"
    if not integer_ids:
        raise ValueError("Custom tracker track_id must contain integers.")
    boxes = tracked.boxes.xyxy
    if isinstance(ids, torch.Tensor) != isinstance(boxes, torch.Tensor):
        raise ValueError("Custom tracker IDs and boxes must use the same backend.")
    if isinstance(ids, torch.Tensor) and ids.device != boxes.device:
        raise ValueError("Custom tracker IDs and boxes must use the same device.")
    tracked.boxes._id = ids
    return tracked
