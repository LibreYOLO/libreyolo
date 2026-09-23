"""Val command: evaluate a model on a dataset."""

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


def _rounded_metric(metrics: dict, key: str) -> float | None:
    if key not in metrics:
        return None
    return round(float(metrics[key]), 4)


def _metric_group(metrics: dict, suffix: str) -> dict[str, float] | None:
    group = {
        "mAP50": _rounded_metric(metrics, f"metrics/mAP50{suffix}"),
        "mAP50_95": _rounded_metric(metrics, f"metrics/mAP50-95{suffix}"),
        "precision": _rounded_metric(metrics, f"metrics/precision{suffix}"),
        "recall": _rounded_metric(metrics, f"metrics/recall{suffix}"),
    }
    out = {k: v for k, v in group.items() if v is not None}
    return out or None


def val_cmd(
    model: str = typer.Option(..., help="Model weights path"),
    data: str = typer.Option(
        ..., help="Path to dataset YAML (YOLO format, e.g. coco8.yaml)"
    ),
    data_dir: Optional[str] = typer.Option(None, help="Direct dataset directory"),
    split: str = typer.Option("val", help="Dataset split: val, test, train"),
    classes: Optional[str] = typer.Option(
        None,
        help="Evaluate on only these original dataset class ids, "
        "comma-separated (e.g. '0,3,5'); every other class's boxes are "
        "dropped from ground truth and predictions. Defaults to the "
        "classes= the checkpoint was trained with, if any",
    ),
    batch: int = typer.Option(16, help="Batch size"),
    imgsz: Optional[str] = typer.Option(
        None, help="Image size: 640 (square) or 480x640 (HxW)"
    ),
    conf: float = typer.Option(0.001, help="Confidence threshold"),
    iou: float = typer.Option(0.6, help="NMS IoU threshold"),
    max_det: int = typer.Option(300, help="Max predictions per image after NMS"),
    eval_max_det: Optional[int] = typer.Option(
        None,
        help="COCO evaluator cap (default: pycocotools AP@100)",
    ),
    faster_coco_eval: bool = typer.Option(
        True,
        "--faster-coco-eval/--no-faster-coco-eval",
        help="Use the faster-coco-eval C++ backend for COCO metrics "
        "(default: on when installed; falls back to pycocotools)",
    ),
    half: bool = typer.Option(False, help="FP16 inference"),
    crop_pct: Optional[float] = typer.Option(
        None,
        help="Classification eval resize ratio before the center crop "
        "(default: the model family's native value)",
    ),
    amp_dtype: str = typer.Option(
        "float16", help="CUDA autocast dtype when half=true: float16 or bfloat16"
    ),
    save_json: bool = typer.Option(False, help="Save COCO-format JSON results"),
    save_plots: bool = typer.Option(
        False,
        help="Save validation plots (metrics, per-class AP, confusion matrix, samples)",
    ),
    plot_samples: int = typer.Option(
        8,
        help="Sample images in the validation sample plot: 0 for none, "
        "-1 for every validated image (does not change the metrics)",
    ),
    plot_errors: int = typer.Option(
        0,
        help="Incorrect images drawn for error analysis in plots/errors/ "
        "(detect, segment, classify; needs --save-plots): 0 for none, "
        "-1 for every incorrect image (does not change the metrics)",
    ),
    workers: int = typer.Option(4, help="Dataloader workers"),
    device: str = typer.Option("auto", help="Device"),
    project: str = typer.Option("runs/val", help="Output directory root"),
    name: str = typer.Option("exp", help="Experiment name"),
    exist_ok: bool = typer.Option(False, help="Reuse output directory"),
    allow_download_scripts: bool = typer.Option(
        False,
        "--allow-download-scripts",
        help="Allow embedded Python in dataset YAML download blocks",
    ),
    # Agent flags
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
    verbose: bool = typer.Option(True, help="Verbose output"),
    help_json: bool = typer.Option(
        False,
        "--help-json",
        is_eager=True,
        callback=help_json_callback,
        help="Dump command schema as JSON",
    ),
) -> None:
    """Evaluate a model on a dataset."""
    from libreyolo.utils.amp import normalize_amp_dtype
    from libreyolo.utils.general import increment_path

    out = OutputHandler(json_mode=json_output, quiet=quiet)
    try:
        imgsz = parse_imgsz_str(imgsz)
    except ValueError as exc:
        exit_with_error(out, "invalid_imgsz", str(exc))
    try:
        amp_dtype = normalize_amp_dtype(amp_dtype)
        if max_det < 1:
            raise ValueError(f"max_det must be >= 1, got {max_det}")
        if eval_max_det is not None and eval_max_det < 1:
            raise ValueError(
                f"eval_max_det must be >= 1, got {eval_max_det}"
            )
        if crop_pct is not None and not 0.0 < crop_pct <= 1.0:
            raise ValueError(f"crop_pct must be in (0, 1], got {crop_pct}")
    except ValueError as exc:
        exit_with_error(out, "config_type_error", str(exc))
    model_path = resolve_model_or_exit(out, model)

    if allow_download_scripts:
        out.warning(
            "Dataset download scripts are enabled. Embedded Python from the dataset YAML may execute locally."
        )

    # Load model
    loaded_model = load_model_or_exit(
        out, model=model, model_path=model_path, device=device
    )

    # crop_pct is classification eval preprocessing; say so rather than accept
    # it and change nothing (#878).
    if crop_pct is not None and getattr(loaded_model, "task", "detect") != "classify":
        out.warning(
            f"{getattr(loaded_model, 'FAMILY', 'This model')} is not a "
            "classification model and ignores crop_pct; it only affects the "
            "classification eval resize and center crop."
        )

    # Resolve save directory
    save_dir = str(increment_path(Path(project) / name, exist_ok=exist_ok, mkdir=True))

    # Run validation
    out.progress(f"Validating {model} on {data} ({split} split)...")
    try:
        metrics = loaded_model.val(
            data=data,
            batch=batch,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            workers=workers,
            allow_download_scripts=allow_download_scripts,
            device=device,
            split=split,
            classes=classes,
            save_json=save_json,
            save_plots=save_plots,
            verbose=verbose and not quiet,
            save_dir=save_dir,
            data_dir=data_dir,
            half=half,
            amp_dtype=amp_dtype,
            max_det=max_det,
            eval_max_det=eval_max_det,
            faster_coco_eval=faster_coco_eval,
            plot_samples=plot_samples,
            crop_pct=crop_pct,
            # Only when set: some families' val() reject unknown kwargs.
            **({"plot_errors": plot_errors} if plot_errors else {}),
        )
    except FileNotFoundError as e:
        exit_with_error(out, "data_not_found", str(e))
    except Exception as e:
        exit_stage_error(out, stage="Validation", detail=e)

    if getattr(loaded_model, "task", "detect") == "classify":
        top1 = metrics.get("metrics/accuracy_top1", 0.0)
        top5 = metrics.get("metrics/accuracy_top5", 0.0)
        data_out = {
            "model": model,
            "model_family": loaded_model.FAMILY,
            "data": data,
            "split": split,
            "device": str(loaded_model.device),
            "metrics": {
                "accuracy_top1": round(float(top1), 4),
                "accuracy_top5": round(float(top5), 4),
            },
        }
        if not json_output:
            data_out["_human_text"] = (
                f"Validating {loaded_model.FAMILY}-{loaded_model.size} "
                f"on {data} ({split}):\n"
                f"  top1: {float(top1):.4f}  top5: {float(top5):.4f}"
            )
        out.result(data_out)
        return

    if getattr(loaded_model, "task", "detect") == "point":
        precision = metrics.get("metrics/precision", 0.0)
        recall = metrics.get("metrics/recall", 0.0)
        f1 = metrics.get("metrics/f1", 0.0)
        mle = metrics.get("metrics/MLE", 0.0)
        mae = metrics.get("metrics/MAE", 0.0)
        rmse = metrics.get("metrics/RMSE", 0.0)
        sweep_key = next((k for k in metrics.keys() if k.startswith("metrics/mAP@[")), None)
        sweep_map = metrics.get(sweep_key, 0.0) if sweep_key else 0.0
        data_out = {
            "model": model,
            "model_family": loaded_model.FAMILY,
            "data": data,
            "split": split,
            "device": str(loaded_model.device),
            "metrics": {
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "MLE": round(float(mle), 4),
                "MAE": round(float(mae), 4),
                "RMSE": round(float(rmse), 4),
                "mAP_sweep": round(float(sweep_map), 4),
            },
        }
        if not json_output:
            data_out["_human_text"] = (
                f"Validating {loaded_model.FAMILY}-{loaded_model.size} on {data} ({split}):\n"
                f"  P: {precision:.4f}  R: {recall:.4f}  F1: {f1:.4f}\n"
                f"  MLE: {mle:.4f}  MAE: {mae:.4f}  RMSE: {rmse:.4f}\n"
                f"  Sweep mAP: {sweep_map:.4f}"
            )
        out.result(data_out)
        return

    task = getattr(loaded_model, "task", "detect")

    if task == "restore":
        psnr = metrics.get("metrics/PSNR", 0.0)
        ssim = metrics.get("metrics/SSIM", 0.0)
        data_out = {
            "model": model,
            "model_family": loaded_model.FAMILY,
            "data": data,
            "split": split,
            "device": str(loaded_model.device),
            "metrics": {
                "PSNR": round(float(psnr), 4),
                "SSIM": round(float(ssim), 4),
            },
        }
        if not json_output:
            data_out["_human_text"] = (
                f"Validating {loaded_model.FAMILY}-{loaded_model.size} "
                f"on {data} ({split}):\n"
                f"  PSNR: {float(psnr):.4f}  SSIM: {float(ssim):.4f}"
            )
        out.result(data_out)
        return

    if task == "depth":
        abs_rel = metrics.get("metrics/abs_rel", 0.0)
        rmse = metrics.get("metrics/rmse", 0.0)
        delta1 = metrics.get("metrics/delta1", 0.0)
        delta2 = metrics.get("metrics/delta2", 0.0)
        delta3 = metrics.get("metrics/delta3", 0.0)
        data_out = {
            "model": model,
            "model_family": loaded_model.FAMILY,
            "data": data,
            "split": split,
            "device": str(loaded_model.device),
            "metrics": {
                "abs_rel": round(float(abs_rel), 4),
                "rmse": round(float(rmse), 4),
                "delta1": round(float(delta1), 4),
                "delta2": round(float(delta2), 4),
                "delta3": round(float(delta3), 4),
            },
        }
        if not json_output:
            data_out["_human_text"] = (
                f"Validating {loaded_model.FAMILY}-{loaded_model.size} "
                f"on {data} ({split}):\n"
                f"  AbsRel: {float(abs_rel):.4f}  RMSE: {float(rmse):.4f}\n"
                f"  delta1: {float(delta1):.4f}  delta2: {float(delta2):.4f}  "
                f"delta3: {float(delta3):.4f}"
            )
        out.result(data_out)
        return

    if task == "semantic":
        miou = metrics.get("metrics/mIoU", 0.0)
        pixel_acc = metrics.get("metrics/pixel_accuracy", 0.0)
        data_out = {
            "model": model,
            "model_family": loaded_model.FAMILY,
            "data": data,
            "split": split,
            "device": str(loaded_model.device),
            "metrics": {
                "mIoU": round(float(miou), 4),
                "pixel_accuracy": round(float(pixel_acc), 4),
            },
        }
        if not json_output:
            data_out["_human_text"] = (
                f"Validating {loaded_model.FAMILY}-{loaded_model.size} "
                f"on {data} ({split}):\n"
                f"  mIoU: {float(miou):.4f}  "
                f"pixel accuracy: {float(pixel_acc):.4f}"
            )
        out.result(data_out)
        return

    # Extract metrics (keys like "metrics/mAP50", "metrics/mAP50-95")
    mAP50 = metrics.get("metrics/mAP50", 0.0)
    mAP50_95 = metrics.get("metrics/mAP50-95", metrics.get("metrics/mAP50_95", 0.0))
    precision = metrics.get(
        "metrics/precision",
        metrics.get("metrics/precision(M)", metrics.get("metrics/precision(B)", 0.0)),
    )
    recall = metrics.get(
        "metrics/recall",
        metrics.get("metrics/recall(M)", metrics.get("metrics/recall(B)", 0.0)),
    )

    data_out = {
        "model": model,
        "model_family": loaded_model.FAMILY,
        "data": data,
        "split": split,
        "device": str(loaded_model.device),
        # COCO eval backend provenance, e.g. "faster-coco-eval 1.7.2".
        "eval_backend": getattr(loaded_model, "last_eval_backend", None),
        "metrics": {
            "mAP50": round(mAP50, 4),
            "mAP50_95": round(mAP50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        },
    }
    box_metrics = _metric_group(metrics, "(B)")
    mask_metrics = _metric_group(metrics, "(M)")
    obb_metrics = _metric_group(metrics, "(OBB)")
    if box_metrics is not None:
        data_out["box_metrics"] = box_metrics
    if mask_metrics is not None:
        data_out["mask_metrics"] = mask_metrics
    if obb_metrics is not None:
        data_out["obb_metrics"] = obb_metrics

    if not json_output:
        human_text = (
            f"Validating {loaded_model.FAMILY}-{loaded_model.size} on {data} ({split}):\n"
            f"  mAP50: {mAP50:.4f}  mAP50-95: {mAP50_95:.4f}  "
            f"P: {precision:.4f}  R: {recall:.4f}"
        )
        if box_metrics is not None:
            human_text += (
                f"\n  Box mAP50: {box_metrics.get('mAP50', 0.0):.4f}  "
                f"Box mAP50-95: {box_metrics.get('mAP50_95', 0.0):.4f}"
            )
        if mask_metrics is not None:
            human_text += (
                f"\n  Mask mAP50: {mask_metrics.get('mAP50', 0.0):.4f}  "
                f"Mask mAP50-95: {mask_metrics.get('mAP50_95', 0.0):.4f}"
            )
        if obb_metrics is not None:
            human_text += (
                f"\n  OBB mAP50: {obb_metrics.get('mAP50', 0.0):.4f}  "
                f"OBB mAP50-95: {obb_metrics.get('mAP50_95', 0.0):.4f}"
            )
        data_out["_human_text"] = human_text

    out.result(data_out)
