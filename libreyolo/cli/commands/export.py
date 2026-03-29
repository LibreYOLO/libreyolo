"""Export command: export a model to a deployment format."""

from pathlib import Path
from typing import Optional

import typer

from ..errors import CLIError
from ..output import OutputHandler
from ..parsing import KeyValueCommand


def export_cmd(
    model: str = typer.Option(..., help="Model weights (.pt)"),
    format: str = typer.Option("onnx", help="Export format: onnx, torchscript, tensorrt, openvino, ncnn"),
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
    # Agent flags
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
    verbose: bool = typer.Option(False, help="Verbose export logging"),
    help_json: bool = typer.Option(
        False, "--help-json", is_eager=True,
        callback=lambda ctx, param, v: _help_json(ctx, v),
        help="Dump command schema as JSON",
    ),
) -> None:
    """Export a model to a deployment format."""
    from libreyolo import LibreYOLO

    out = OutputHandler(json_mode=json_output, quiet=quiet)

    # Resolve format alias
    fmt = format.lower()
    if fmt == "engine":
        fmt = "tensorrt"

    # Validate precision conflict
    if half and int8:
        err = CLIError(
            "config_conflict",
            "Cannot use both half (FP16) and int8 simultaneously.",
            suggestion="Choose one: half or int8",
        )
        out.error(err)
        raise SystemExit(err.exit_code)

    # Resolve CLI model name
    from ..config import resolve_model_name

    model_path = resolve_model_name(model)

    # Load model
    out.progress(f"Loading {model}...")
    try:
        loaded_model = LibreYOLO(model_path, device=device)
    except Exception as e:
        err = CLIError("model_load_failed", str(e))
        out.error(err)
        raise SystemExit(err.exit_code)

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

    # Run export
    out.progress(f"Exporting {model} to {fmt}...")
    try:
        output_path = loaded_model.export(format=fmt, **export_kwargs)
    except ValueError as e:
        if "Unsupported export format" in str(e):
            err = CLIError("export_format_unknown", str(e), suggestion="Run: libreyolo formats")
        else:
            err = CLIError("io_error", str(e))
        out.error(err)
        raise SystemExit(err.exit_code)
    except ImportError as e:
        err = CLIError("export_dep_missing", str(e))
        out.error(err)
        raise SystemExit(err.exit_code)
    except Exception as e:
        err = CLIError("io_error", str(e))
        out.error(err)
        raise SystemExit(err.exit_code)

    # File size
    export_path = Path(output_path)
    if export_path.is_file():
        size_mb = export_path.stat().st_size / (1024 * 1024)
    elif export_path.is_dir():
        size_mb = sum(f.stat().st_size for f in export_path.rglob("*") if f.is_file()) / (1024 * 1024)
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


def _help_json(ctx: typer.Context, value: bool) -> None:
    if not value:
        return
    from ..commands.special import _help_json_callback

    _help_json_callback(ctx, value)
