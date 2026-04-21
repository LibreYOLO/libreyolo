"""Export command: export a model to a deployment format."""

from pathlib import Path
from typing import Optional

import typer

from ..command_utils import (
    exit_stage_error,
    exit_with_error,
    help_json_callback,
    load_model_or_exit,
    resolve_model_or_exit,
)
from ..output import OutputHandler


def export_cmd(
    model: str = typer.Option(..., help="Model weights (.pt)"),
    format: str = typer.Option(
        "onnx", help="Export format: onnx, torchscript, tensorrt, openvino, ncnn"
    ),
    imgsz: Optional[int] = typer.Option(None, help="Input image size"),
    batch: int = typer.Option(1, help="Export batch size"),
    half: bool = typer.Option(False, help="FP16 precision"),
    int8: bool = typer.Option(False, help="INT8 quantization"),
    dynamic: bool = typer.Option(False, help="Dynamic input shapes (ONNX)"),
    simplify: bool = typer.Option(True, help="ONNX graph simplification"),
    opset: int = typer.Option(13, help="ONNX opset version"),
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

    # Resolve format alias
    fmt = format.lower()
    if fmt == "engine":
        fmt = "tensorrt"

    # Validate precision conflict
    if half and int8:
        exit_with_error(
            out,
            "config_conflict",
            "Cannot use both half (FP16) and int8 simultaneously.",
            suggestion="Choose one: half or int8",
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
    if imgsz is not None:
        export_kwargs["imgsz"] = imgsz
    if data is not None:
        export_kwargs["data"] = data
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
                "export_format_unknown", str(e), suggestion="Run: libreyolo formats"
            )
        else:
            exit_stage_error(out, stage="Export", detail=e)
    except ImportError as e:
        exit_with_error(out, "export_dep_missing", str(e))
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

    input_size = loaded_model.INPUT_SIZES.get(loaded_model.size, 640)
    if imgsz is not None:
        input_size = imgsz

    data_out = {
        "source_model": model,
        "model_family": loaded_model.FAMILY,
        "format": fmt,
        "output_path": str(output_path),
        "file_size_mb": round(size_mb, 1),
        "input_shape": [batch, 3, input_size, input_size],
        "dynamic": dynamic,
        "half": half,
    }

    if not json_output:
        data_out["_human_text"] = (
            f"Exported {loaded_model.FAMILY}-{loaded_model.size} to {fmt.upper()}: "
            f"{output_path} ({size_mb:.1f} MB)\n"
            f"  Input: [{batch}, 3, {input_size}, {input_size}], "
            f"dynamic={dynamic}, half={half}"
        )

    out.result(data_out)
