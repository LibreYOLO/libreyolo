"""Predict command: run inference on images."""

import time
from pathlib import Path
from typing import Optional

import typer

from ..command_utils import exit_with_error, help_json_callback, load_model_or_exit
from ..output import OutputHandler


def predict_cmd(
    source: str = typer.Option(..., help="Image path, directory, or URL"),
    model: str = typer.Option("yolox-s", help="Model name or path"),
    conf: float = typer.Option(0.25, help="Confidence threshold"),
    iou: float = typer.Option(0.45, help="NMS IoU threshold"),
    imgsz: Optional[int] = typer.Option(None, help="Input image size"),
    classes: Optional[str] = typer.Option(
        None, help="Filter by class IDs, e.g. [0,2,5]"
    ),
    max_det: int = typer.Option(300, help="Max detections per image"),
    half: bool = typer.Option(False, help="FP16 inference (CUDA only, requires model support)"),
    save: bool = typer.Option(False, help="Save annotated images"),
    batch: int = typer.Option(1, help="Batch size for directories"),
    tiling: bool = typer.Option(False, help="Tiled inference for large images"),
    overlap_ratio: float = typer.Option(0.2, help="Tile overlap ratio"),
    output_path: Optional[str] = typer.Option(None, help="Explicit output path"),
    color_format: str = typer.Option("auto", help="Input color: auto, rgb, bgr"),
    output_file_format: Optional[str] = typer.Option(
        None, help="Output format: jpg, png, webp"
    ),
    device: str = typer.Option("auto", help="Device: 0, cpu, mps, auto"),
    project: str = typer.Option("runs/detect", help="Output directory root"),
    name: str = typer.Option("predict", help="Experiment name"),
    exist_ok: bool = typer.Option(False, help="Reuse existing output directory"),
    # Agent flags
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
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
    """Run inference on images."""
    from libreyolo.utils.general import increment_path

    from ..config import resolve_model_name

    out = OutputHandler(json_mode=json_output, quiet=quiet)

    # Validate source exists
    source_path = Path(source)
    if not source_path.exists() and not source.startswith(("http://", "https://")):
        exit_with_error(out, "source_not_found", f"Source not found: {source}")

    # Resolve CLI model name (yolox-s → LibreYOLOXs.pt)
    model_path = resolve_model_name(model)

    # Load model
    loaded_model = load_model_or_exit(
        out, model=model, model_path=model_path, device=device
    )

    # NOTE: half for PyTorch inference is not yet supported in the inference
    # pipeline (model converts to FP16 but input stays FP32 → dtype mismatch).
    # FP16 works correctly through exported models (ONNX, TensorRT).
    # For now, warn and skip.
    if half:
        out.progress(
            "Warning: half (FP16) is not yet supported for PyTorch inference. "
            "Use exported models (ONNX/TensorRT) for FP16. Ignoring."
        )

    # Resolve output path
    if output_path is None and save:
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok, mkdir=True)
        output_path = str(save_dir)

    # Parse classes list
    parsed_classes = None
    if classes is not None:
        import ast

        try:
            parsed_classes = list(ast.literal_eval(classes))
        except (ValueError, SyntaxError):
            exit_with_error(
                out, "config_type_error", f"Invalid classes value: {classes}"
            )

    # Run inference
    out.progress(f"Running inference on {source}...")
    t0 = time.time()
    results = loaded_model(
        source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=parsed_classes,
        max_det=max_det,
        save=save,
        batch=batch,
        output_path=output_path,
        color_format=color_format,
        tiling=tiling,
        overlap_ratio=overlap_ratio,
        output_file_format=output_file_format,
    )
    elapsed = time.time() - t0

    # Normalize to list
    if not isinstance(results, list):
        results = [results]
    total_images = len(results)

    # Format output
    result_list = []
    human_lines = []
    for r in results:
        boxes = r.boxes
        detections = []
        # Count detections per class
        class_counts: dict[str, int] = {}
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            cls_name = r.names.get(cls_id, str(cls_id))
            det = {
                "class": cls_name,
                "class_id": cls_id,
                "confidence": round(float(boxes.conf[i]), 4),
                "bbox_xyxy": [round(float(c), 1) for c in boxes.xyxy[i]],
            }
            detections.append(det)
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        result_list.append(
            {
                "path": r.path or str(source),
                "original_shape": list(r.orig_shape),
                "detections": detections,
            }
        )
        # Human summary line (ultralytics format):
        # image 1/3 parkour.jpg: 640x480 3 persons, 2 skateboards, 45.2ms
        h, w = r.orig_shape
        counts_str = ", ".join(
            f"{v} {k}{'s' if v > 1 else ''}" for k, v in class_counts.items()
        )
        img_name = Path(r.path or source).name
        idx = len(human_lines) + 1
        elapsed_ms = elapsed * 1000 / max(len(results), 1)
        human_lines.append(
            f"image {idx}/{total_images} {img_name}: "
            f"{w}x{h} "
            f"{counts_str or '(no detections)'}, "
            f"{elapsed_ms:.1f}ms"
        )

    data = {
        "source": str(source),
        "model": model,
        "model_family": loaded_model.FAMILY,
        "image_size": [loaded_model.INPUT_SIZES.get(loaded_model.size, 640)] * 2,
        "device": str(loaded_model.device),
        "results": result_list,
    }
    if save and output_path:
        data["output_path"] = output_path

    if not json_output:
        if save and output_path:
            human_lines.append(f"Results saved to {output_path}")
        data["_human_text"] = "\n".join(human_lines)

    out.result(data)
