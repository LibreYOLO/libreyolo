"""Track command: follow objects across video frames or image sequences."""

from pathlib import Path

import typer

from ..command_utils import (
    exit_if_out_of_range,
    exit_with_error,
    get_loaded_model_family,
    get_loaded_model_input_size,
    help_json_callback,
    load_model_or_exit,
    parse_imgsz_str,
    resolve_model_or_exit,
)
from ..output import OutputHandler
from .predict import _exit_inference_error, _split_source_list


def _parse_classes(out: OutputHandler, classes: str | None) -> list[int] | None:
    """Parse ``0``, ``[0,2]`` or ``0,2`` into a list of class IDs."""
    if classes is None:
        return None
    import ast

    try:
        evaluated = ast.literal_eval(classes)
    except (ValueError, SyntaxError):
        exit_with_error(out, "config_type_error", f"Invalid classes value: {classes}")
    if isinstance(evaluated, int):
        return [evaluated]
    try:
        return list(evaluated)
    except TypeError:
        exit_with_error(out, "config_type_error", f"Invalid classes value: {classes}")


def _frame_detections(result) -> list[dict]:
    """One dict per tracked box, with its track ID."""
    boxes = result.boxes
    if boxes is None:
        return []
    ids = result.track_id
    detections = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        detections.append(
            {
                "track_id": int(ids[i]) if ids is not None else None,
                "class": result.names.get(cls_id, str(cls_id)),
                "class_id": cls_id,
                "confidence": round(float(boxes.conf[i]), 4),
                "bbox_xyxy": [round(float(c), 1) for c in boxes.xyxy[i]],
            }
        )
    return detections


def track_cmd(
    source: str = typer.Option(
        ..., help="Video file, image directory, image path, or a,b,c image list"
    ),
    model: str = typer.Option("yolox-s", help="Model name or path"),
    tracker: str = typer.Option(
        "bytetrack", help="Tracker: bytetrack, botsort, ocsort, deepocsort"
    ),
    track_conf: float = typer.Option(
        0.25, help="Confidence threshold of the tracker's first association stage"
    ),
    conf: float | None = typer.Option(
        None,
        help="Detection confidence threshold (default: the tracker's own low "
        "threshold, so low-confidence boxes stay available for recovery)",
    ),
    iou: float = typer.Option(0.45, help="NMS IoU threshold"),
    imgsz: str | None = typer.Option(
        None, help="Input image size: 640 (square) or 480x640 (HxW)"
    ),
    classes: str | None = typer.Option(None, help="Filter by class IDs, e.g. [0,2,5]"),
    max_det: int = typer.Option(300, help="Max detections per frame"),
    save: bool = typer.Option(False, help="Save the annotated video"),
    show: bool = typer.Option(False, help="Display tracked frames; q stops"),
    vid_stride: int = typer.Option(1, help="Process every N-th frame"),
    fps: float = typer.Option(
        30.0, help="Frame rate assumed for image sequences (ignored for videos)"
    ),
    output_path: str | None = typer.Option(
        None, help="Explicit output video path (default runs/track/<name>.mp4)"
    ),
    color_format: str = typer.Option("auto", help="Input color: auto, rgb, bgr"),
    device: str = typer.Option("auto", help="Device: 0, cpu, mps, auto"),
    # Agent flags
    json_output: bool = typer.Option(
        False, "--json", help="JSON output to stdout, one line per frame"
    ),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose stderr output"),
    help_json: bool = typer.Option(
        False,
        "--help-json",
        is_eager=True,
        callback=help_json_callback,
        help="Dump command schema as JSON",
    ),
) -> None:
    """Track objects across video frames with native or exported models."""
    out = OutputHandler(json_mode=json_output, quiet=quiet)
    try:
        imgsz = parse_imgsz_str(imgsz) if imgsz is not None else None
    except ValueError as exc:
        exit_with_error(out, "invalid_imgsz", str(exc))
    exit_if_out_of_range(
        out,
        conf=conf,
        track_conf=track_conf,
        iou=iou,
        max_det=max_det,
        vid_stride=vid_stride,
    )

    from libreyolo.tracking.run import BUILTIN_TRACKERS

    tracker_name = (
        Path(tracker.lower()).stem
        if tracker.lower().endswith((".yaml", ".yml"))
        else tracker.lower()
    )
    if tracker_name not in BUILTIN_TRACKERS:
        exit_with_error(
            out,
            "config_type_error",
            f"Unknown tracker {tracker!r}.",
            suggestion="Use one of: " + ", ".join(sorted(BUILTIN_TRACKERS)),
        )

    from libreyolo.utils.source import SourceKind, classify_source

    source = _split_source_list(source)
    try:
        source_spec = classify_source(source)
    except FileNotFoundError as exc:
        exit_with_error(out, "source_not_found", str(exc))
    except (TypeError, ValueError) as exc:
        exit_with_error(out, "config_type_error", str(exc))

    if source_spec.live or source_spec.kind == SourceKind.STREAM:
        exit_with_error(
            out,
            "config_unsupported",
            "track does not support live sources yet.",
            suggestion="Pass a video file, an image directory, or an image list.",
        )
    remote_prefixes = ("http://", "https://", "s3://", "gs://")
    if (
        isinstance(source, str)
        and not Path(source).exists()
        and not source.startswith(remote_prefixes)
    ):
        exit_with_error(out, "source_not_found", f"Source not found: {source}")

    model_path = resolve_model_or_exit(out, model)
    loaded_model = load_model_or_exit(
        out, model=model, model_path=model_path, device=device
    )
    if not callable(getattr(loaded_model, "track", None)):
        exit_with_error(
            out,
            "config_unsupported",
            f"This model ({type(loaded_model).__name__}) does not support track().",
        )

    if save and output_path is None:
        from libreyolo.utils.general import increment_path

        if source_spec.kind in (SourceKind.VIDEO, SourceKind.DIRECTORY):
            stem = (
                Path(source).stem
                if source_spec.kind == SourceKind.VIDEO
                else Path(source).name
            )
        else:
            stem = "sequence"
        output_path = str(
            increment_path(Path("runs") / "track" / f"{stem}.mp4", exist_ok=False)
        )

    image_size = get_loaded_model_input_size(loaded_model, imgsz=imgsz)
    image_shape = (
        [image_size[0], image_size[1]]
        if isinstance(image_size, tuple)
        else [image_size, image_size]
    )
    family = get_loaded_model_family(loaded_model) or "unknown"

    parsed_classes = _parse_classes(out, classes)
    out.progress(f"Tracking {source} with {tracker}...")
    frames = 0
    track_ids: set[int] = set()
    results = None
    try:
        results = loaded_model.track(
            source,
            track_conf=track_conf,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=parsed_classes,
            max_det=max_det,
            save=save,
            show=show,
            vid_stride=vid_stride,
            fps=fps,
            color_format=color_format,
            output_path=output_path,
            tracker=tracker,
        )
        for result in results:
            frames += 1
            detections = _frame_detections(result)
            frame_ids = [d["track_id"] for d in detections if d["track_id"] is not None]
            track_ids.update(frame_ids)
            path = result.path or str(source)
            human = (
                f"frame {frames} {Path(path).name}: {result.orig_shape[1]}x"
                f"{result.orig_shape[0]} {len(detections)} track"
                f"{'s' if len(detections) != 1 else ''}"
                + (
                    f" (ids {', '.join(str(i) for i in frame_ids)})"
                    if frame_ids
                    else ""
                )
            )
            data = {
                "source": str(source),
                "model": model,
                "model_family": family,
                "image_size": image_shape,
                "device": str(loaded_model.device),
                "tracker": tracker,
                "frame_index": getattr(result, "frame_idx", None),
                "results": [
                    {
                        "path": path,
                        "original_shape": list(result.orig_shape),
                        "detections": detections,
                    }
                ],
                "_human_text": human,
            }
            if save and output_path:
                data["output_path"] = output_path
            out.result(data)
    except KeyboardInterrupt:
        out.progress("Tracking stopped.")
    except NotImplementedError as exc:
        exit_with_error(out, "config_unsupported", str(exc))
    except Exception as exc:
        _exit_inference_error(out, exc, imgsz_provided=imgsz is not None)
    finally:
        close = getattr(results, "close", None)
        if close is not None:
            close()

    out.progress(f"{frames} frames, {len(track_ids)} unique track IDs.")
    if save and output_path:
        out.progress(f"Results saved to {output_path}")
