"""Export command: export a model to a deployment format."""

from pathlib import Path
from typing import Optional

import typer

from ..command_utils import (
    exit_stage_error,
    exit_with_error,
    help_json_callback,
    load_model_or_exit,
    parse_imgsz_str,
    resolve_model_or_exit,
)
from ..output import OutputHandler


def export_cmd(
    model: str = typer.Option(..., help="Model weights (.pt)"),
    format: str = typer.Option(
        "onnx",
        help=(
            "Export format: onnx, torchscript, executorch, tensorrt, openvino, "
            "paddle, mnn, rknn, ncnn, tflite (alias: litert), coreml, coreai "
            "(Apple, macOS only)"
        ),
    ),
    name: Optional[str] = typer.Option(
        None,
        help="RKNN target platform (currently rk3588 only)",
    ),
    imgsz: Optional[str] = typer.Option(
        None,
        help="Input image size: 640 or 480x640 (HxW); 480,640 remains supported",
    ),
    batch: int = typer.Option(1, help="Export batch size"),
    half: bool = typer.Option(False, help="FP16 precision"),
    int8: bool = typer.Option(False, help="INT8 quantization"),
    dynamic: bool = typer.Option(False, help="Dynamic input shapes (ONNX)"),
    simplify: bool = typer.Option(True, help="ONNX graph simplification"),
    nms: bool = typer.Option(
        False,
        help="Embed NMS in the model (ONNX YOLO9 detection or CoreML)",
    ),
    conf: float = typer.Option(0.25, help="Confidence threshold for embedded NMS"),
    iou: float = typer.Option(0.45, help="IoU threshold for embedded NMS"),
    max_det: int = typer.Option(300, help="Maximum detections for ONNX embedded NMS"),
    opset: Optional[int] = typer.Option(
        None, help="ONNX opset version (auto if omitted)"
    ),
    data: Optional[str] = typer.Option(None, help="Calibration data for INT8"),
    fraction: float = typer.Option(1.0, help="Fraction of calibration data"),
    device: str = typer.Option("auto", help="Device for tracing"),
    allow_download_scripts: bool = typer.Option(
        False,
        "--allow-download-scripts",
        help="Allow embedded Python in dataset YAML download blocks",
    ),
    # Agent flags
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
    verbose: bool = typer.Option(False, help="Verbose export logging"),
    verify: bool = typer.Option(
        False,
        help="Run RKNN Toolkit2's PC simulator and compare with ONNX Runtime",
    ),
    help_json: bool = typer.Option(
        False,
        "--help-json",
        is_eager=True,
        callback=help_json_callback,
        help="Dump command schema as JSON",
    ),
) -> None:
    """Export a model to a deployment format."""
    out = OutputHandler(json_mode=json_output, quiet=quiet)

    # Resolve format aliases (engine -> tensorrt, litert -> tflite) so JSON
    # output and messages always report the canonical format name.
    from libreyolo.export.exporter import BaseExporter

    fmt = format.lower()
    fmt = BaseExporter._aliases.get(fmt, fmt)

    if half and int8:
        out.warning("Both half and int8 were requested. Using INT8 precision.")
        half = False

    if nms and fmt not in {"onnx", "coreml"}:
        exit_with_error(
            out,
            "nms_unsupported_format",
            "Embedded NMS (--nms) is only supported for ONNX and CoreML, "
            f"not {fmt!r}.",
        )
    if nms and fmt == "onnx" and dynamic:
        out.warning("Embedded ONNX NMS uses a fixed batch-1 graph. Using dynamic=False.")
        dynamic = False
    if nms and fmt == "coreml" and max_det != 300:
        exit_with_error(
            out,
            "config_unsupported",
            "max_det is only supported for ONNX embedded NMS; CoreML embedded "
            "NMS does not expose max_det.",
        )
    if name is not None and fmt != "rknn":
        exit_with_error(
            out,
            "config_unsupported",
            "--name is currently an RKNN target option; use it with --format rknn.",
        )
    if verify and fmt != "rknn":
        exit_with_error(
            out,
            "config_unsupported",
            "--verify is currently supported only with --format rknn.",
        )

    model_path = resolve_model_or_exit(out, model)

    if allow_download_scripts and data is not None:
        out.warning(
            "Dataset download scripts are enabled. Embedded Python from the dataset YAML may execute locally."
        )

    # Load model
    loaded_model = load_model_or_exit(
        out, model=model, model_path=model_path, device=device
    )

    # Build export kwargs
    export_kwargs: dict = {
        "half": half,
        "int8": int8,
        "dynamic": dynamic,
        "simplify": simplify,
        "opset": opset,
        "batch": batch,
        "device": device,
        "verbose": verbose,
    }
    if nms:
        export_kwargs["nms"] = True
        export_kwargs["conf"] = conf
        export_kwargs["iou"] = iou
        if fmt == "onnx":
            export_kwargs["max_det"] = max_det
    if fmt == "rknn":
        export_kwargs["name"] = name or "rk3588"
        export_kwargs["verify"] = verify
    parsed_imgsz = None
    if imgsz is not None:
        try:
            parsed_imgsz = parse_imgsz_str(imgsz)
        except ValueError as exc:
            exit_with_error(out, "invalid_imgsz", str(exc))
        export_kwargs["imgsz"] = parsed_imgsz
    if data is not None:
        export_kwargs["data"] = data
    if data is not None or int8:
        export_kwargs["fraction"] = fraction
        export_kwargs["allow_download_scripts"] = allow_download_scripts

    # Run export
    out.progress(f"Exporting {model} to {fmt}...")
    try:
        output_path = loaded_model.export(format=fmt, **export_kwargs)
    except ValueError as e:
        if "Unsupported export format" in str(e):
            exit_with_error(
                out,
                "export_format_unknown",
                str(e),
                suggestion="Run: libreyolo formats",
            )
        else:
            exit_stage_error(out, stage="Export", detail=e)
    except ImportError as e:
        exit_with_error(out, "export_dep_missing", str(e))
    except NotImplementedError as e:
        exit_with_error(out, "format_precision_unsupported", str(e))
    except Exception as e:
        exit_stage_error(out, stage="Export", detail=e)

    # File size
    export_path = Path(output_path)
    if export_path.is_file():
        size_mb = export_path.stat().st_size / (1024 * 1024)
    elif export_path.is_dir():
        size_mb = sum(
            f.stat().st_size for f in export_path.rglob("*") if f.is_file()
        ) / (1024 * 1024)
    else:
        size_mb = 0.0

    if parsed_imgsz is not None:
        if isinstance(parsed_imgsz, int):
            input_h = input_w = parsed_imgsz
        else:
            input_h, input_w = parsed_imgsz
    else:
        native = (
            loaded_model._get_input_size()
            if hasattr(loaded_model, "_get_input_size")
            else loaded_model.INPUT_SIZES.get(loaded_model.size, 640)
        )
        input_h = input_w = native

    channels = 2 if isinstance(getattr(loaded_model, "input_profile", None), dict) else 3
    data_out = {
        "source_model": model,
        "model_family": loaded_model.FAMILY,
        "format": fmt,
        "output_path": str(output_path),
        "file_size_mb": round(size_mb, 1),
        "input_shape": [batch, channels, input_h, input_w],
        "dynamic": dynamic,
        "half": half,
        "int8": int8,
    }
    if fmt == "rknn":
        data_out["target"] = name or "rk3588"
        data_out["verified"] = verify

    if not json_output:
        data_out["_human_text"] = (
            f"Exported {loaded_model.FAMILY}-{loaded_model.size} to {fmt.upper()}: "
            f"{output_path} ({size_mb:.1f} MB)\n"
            f"  Input: [{batch}, {channels}, {input_h}, {input_w}], "
            f"dynamic={dynamic}, half={half}, int8={int8}"
        )

    out.result(data_out)
