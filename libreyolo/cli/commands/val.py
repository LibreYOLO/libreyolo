"""Val command: evaluate a model on a dataset."""

from pathlib import Path
from typing import Optional

import typer

from ..aliases import resolve_aliases
from ..errors import CLIError
from ..output import OutputHandler


def val_cmd(
    model: str = typer.Option(..., help="Model weights path"),
    data: str = typer.Option(..., help="Dataset YAML path"),
    data_dir: Optional[str] = typer.Option(None, help="Direct dataset directory"),
    split: str = typer.Option("val", help="Dataset split: val, test, train"),
    batch: int = typer.Option(16, help="Batch size"),
    imgsz: Optional[int] = typer.Option(None, help="Image size"),
    conf: float = typer.Option(0.001, help="Confidence threshold"),
    iou: float = typer.Option(0.6, help="NMS IoU threshold"),
    max_det: int = typer.Option(300, help="Max detections per image"),
    half: bool = typer.Option(False, help="FP16 inference"),
    save_json: bool = typer.Option(False, help="Save COCO-format JSON results"),
    workers: int = typer.Option(4, help="Dataloader workers"),
    device: str = typer.Option("auto", help="Device"),
    project: str = typer.Option("runs/val", help="Output directory root"),
    name: str = typer.Option("exp", help="Experiment name"),
    exist_ok: bool = typer.Option(False, help="Reuse output directory"),
    use_coco_eval: bool = typer.Option(True, help="Use pycocotools evaluator"),
    # Agent flags
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
    verbose: bool = typer.Option(True, help="Verbose output"),
    help_json: bool = typer.Option(
        False,
        "--help-json",
        is_eager=True,
        callback=lambda ctx, param, v: _help_json(ctx, v),
        help="Dump command schema as JSON",
    ),
) -> None:
    """Evaluate a model on a dataset."""
    from libreyolo import LibreYOLO
    from libreyolo.utils.general import increment_path

    out = OutputHandler(json_mode=json_output, quiet=quiet)

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

    # Resolve save directory
    save_dir = str(increment_path(Path(project) / name, exist_ok=exist_ok, mkdir=True))

    # Build val kwargs — use internal field names via alias resolution
    val_overrides = resolve_aliases(
        {"batch": batch, "conf": conf, "iou": iou, "workers": workers},
        "val",
    )

    # Run validation
    out.progress(f"Validating {model} on {data} ({split} split)...")
    try:
        metrics = loaded_model.val(
            data=data,
            imgsz=imgsz,
            device=device,
            split=split,
            save_json=save_json,
            verbose=verbose and not quiet,
            save_dir=save_dir,
            data_dir=data_dir,
            use_coco_eval=use_coco_eval,
            half=half,
            max_det=max_det,
            **val_overrides,
        )
    except FileNotFoundError as e:
        err = CLIError("data_not_found", str(e))
        out.error(err)
        raise SystemExit(err.exit_code)
    except Exception as e:
        err = CLIError("io_error", str(e))
        out.error(err)
        raise SystemExit(err.exit_code)

    # Extract metrics (keys like "metrics/mAP50", "metrics/mAP50-95")
    mAP50 = metrics.get("metrics/mAP50", 0.0)
    mAP50_95 = metrics.get("metrics/mAP50-95", metrics.get("metrics/mAP50_95", 0.0))
    precision = metrics.get("metrics/precision", 0.0)
    recall = metrics.get("metrics/recall", 0.0)

    data_out = {
        "model": model,
        "model_family": loaded_model.FAMILY,
        "data": data,
        "split": split,
        "device": str(loaded_model.device),
        "metrics": {
            "mAP50": round(mAP50, 4),
            "mAP50_95": round(mAP50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        },
    }

    if not json_output:
        data_out["_human_text"] = (
            f"Validating {loaded_model.FAMILY}-{loaded_model.size} on {data} ({split}):\n"
            f"  mAP50: {mAP50:.4f}  mAP50-95: {mAP50_95:.4f}  "
            f"P: {precision:.4f}  R: {recall:.4f}"
        )

    out.result(data_out)


def _help_json(ctx: typer.Context, value: bool) -> None:
    if not value:
        return
    from ..commands.special import _help_json_callback

    _help_json_callback(ctx, value)
