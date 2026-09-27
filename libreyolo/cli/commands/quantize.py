"""Quantize command: PyTorch-native model quantization."""

from pathlib import Path
from typing import Optional

import typer

from ..command_utils import (
    exit_with_error,
    help_json_callback,
    load_model_or_exit,
    resolve_model_or_exit,
)
from ..output import OutputHandler


def quantize_cmd(
    model: str = typer.Option(..., help="Model weights (.pt)"),
    recipe: str = typer.Option(
        "int8",
        help="Quantization recipe: fp16, bf16, fp8, int8, w4a16, w4a8, "
        "nvfp4, mxfp4, int2 (research; w4/nvfp4/mxfp4/int2 target "
        "transformer families)",
    ),
    calib: Optional[str] = typer.Option(
        "coco128.yaml",
        help="Calibration images (data.yaml or built-in dataset name); "
        "unlabeled, forward-only. Use 'none' to skip calibration.",
    ),
    samples: int = typer.Option(128, help="Maximum calibration images"),
    batch: int = typer.Option(8, help="Calibration batch size"),
    algorithm: str = typer.Option(
        "auto",
        help="Activation range estimation: auto (minmax), minmax, "
        "percentile (can reduce transformer accuracy), mse, entropy "
        "(histogram sweeps; help on outlier-heavy activations)",
    ),
    out: Optional[str] = typer.Option(
        None,
        help="Output checkpoint path (default: <model>-<recipe>.pt)",
    ),
    device: str = typer.Option("auto", help="Device"),
    allow_download_scripts: bool = typer.Option(
        False,
        "--allow-download-scripts",
        help="Allow embedded Python in dataset YAML download blocks",
    ),
    # Agent flags
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
    help_json: bool = typer.Option(
        False,
        "--help-json",
        is_eager=True,
        callback=help_json_callback,
        help="Dump command schema as JSON",
    ),
) -> None:
    """Quantize a model in PyTorch and save a quantized checkpoint.

    Calibration data is separate from training data: a small unlabeled image
    set used forward-only to derive scales. To recover accuracy afterwards,
    run `libreyolo train` on the quantized checkpoint (QAT), optionally with
    --distill-model for QAD.
    """
    out_handler = OutputHandler(json_mode=json_output, quiet=quiet)

    model_path = resolve_model_or_exit(out_handler, model)
    loaded_model = load_model_or_exit(
        out_handler, model=model, model_path=model_path, device=device
    )

    calib_arg = None if calib is None or calib.lower() == "none" else calib

    if allow_download_scripts and calib_arg is not None:
        out_handler.warning(
            "Dataset download scripts are enabled. Embedded Python from the "
            "dataset YAML may execute locally."
        )

    try:
        loaded_model.quantize(
            recipe=recipe,
            calib=calib_arg,
            samples=samples,
            batch=batch,
            algorithm=algorithm,
            allow_download_scripts=allow_download_scripts,
        )
    except Exception as exc:
        exit_with_error(out_handler, "quantize_failed", f"Quantization failed: {exc}")

    if out is None:
        src = Path(model_path)
        out = str(src.with_name(f"{src.stem}-{recipe.lower()}{src.suffix or '.pt'}"))

    try:
        saved = loaded_model.save(out)
    except Exception as exc:
        exit_with_error(out_handler, "save_failed", f"Saving checkpoint failed: {exc}")

    info = loaded_model.quant_info() or {}
    out_handler.result(
        {
            "path": saved,
            "recipe": recipe.lower(),
            "execution": info.get("execution"),
            "calibrated": info.get("calibrated"),
            "module_counts": info.get("module_counts"),
        }
    )
