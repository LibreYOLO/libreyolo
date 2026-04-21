"""Train command: train a model on a dataset."""

import time

import typer

from ..command_utils import exit_with_error, load_model_or_exit
from ..config import (
    RFDETR_UNSUPPORTED_PARAMS,
    apply_family_defaults,
    build_family_train_kwargs,
    detect_family_from_name,
)
from ..output import OutputHandler


def train_cmd(
    data: str = typer.Option(..., help="Path to dataset YAML (YOLO format, e.g. coco8.yaml)"),
    model: str = typer.Option("yolox-s", help="Model name or path to weights"),
    # Training
    epochs: int = typer.Option(300, help="Training epochs"),
    batch: int = typer.Option(16, help="Batch size per device"),
    imgsz: int = typer.Option(640, help="Training image size"),
    device: str = typer.Option("auto", help="Device: 0, cpu, mps, auto"),
    workers: int = typer.Option(4, help="Dataloader workers"),
    seed: int = typer.Option(0, help="Random seed"),
    resume: str = typer.Option("", help="Resume training: true, or path to checkpoint"),
    amp: bool = typer.Option(True, help="Automatic Mixed Precision"),
    pretrained: bool = typer.Option(True, help="Use pretrained weights"),
    # Optimizer
    optimizer: str = typer.Option("sgd", help="Optimizer: sgd, adam, adamw"),
    lr0: float = typer.Option(0.01, help="Initial learning rate"),
    momentum: float = typer.Option(0.937, help="SGD momentum / Adam beta1"),
    weight_decay: float = typer.Option(5e-4, help="L2 regularization"),
    nesterov: bool = typer.Option(True, help="Nesterov momentum"),
    # Scheduler
    scheduler: str = typer.Option("yoloxwarmcos", help="LR schedule type"),
    warmup_epochs: int = typer.Option(5, help="Warmup duration"),
    warmup_lr_start: float = typer.Option(0.0, help="Initial warmup LR"),
    min_lr_ratio: float = typer.Option(0.05, help="Minimum LR ratio"),
    # Augmentation
    mosaic: float = typer.Option(1.0, help="Mosaic probability"),
    mixup: float = typer.Option(1.0, help="Mixup probability"),
    hsv_prob: float = typer.Option(1.0, help="HSV jitter probability"),
    flip_prob: float = typer.Option(0.5, help="Horizontal flip probability"),
    degrees: float = typer.Option(10.0, help="Rotation +/- degrees"),
    translate: float = typer.Option(0.1, help="Translation ratio"),
    shear: float = typer.Option(2.0, help="Shear angle"),
    mosaic_scale: str = typer.Option("(0.1,2.0)", help="Mosaic scale range"),
    mixup_scale: str = typer.Option("(0.5,1.5)", help="Mixup scale range"),
    no_aug_epochs: int = typer.Option(
        15, help="Disable augmentation for final N epochs"
    ),
    # EMA
    ema: bool = typer.Option(True, help="Exponential Moving Average"),
    ema_decay: float = typer.Option(0.9998, help="EMA decay factor"),
    # Validation
    val: bool = typer.Option(True, help="Validate during training"),
    eval_interval: int = typer.Option(10, help="Validate every N epochs"),
    patience: int = typer.Option(50, help="Early stopping patience (0=disabled)"),
    # Output
    project: str = typer.Option("runs/train", help="Output directory root"),
    name: str = typer.Option("exp", help="Experiment name"),
    exist_ok: bool = typer.Option(False, help="Reuse existing output directory"),
    save_period: int = typer.Option(10, help="Save checkpoint every N epochs"),
    log_interval: int = typer.Option(10, help="Log loss every N batches"),
    allow_download_scripts: bool = typer.Option(
        False,
        "--allow-download-scripts",
        help="Allow embedded Python in dataset YAML download blocks",
    ),
    # Agent flags
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without executing"),
    help_json: bool = typer.Option(
        False,
        "--help-json",
        is_eager=True,
        callback=lambda ctx, param, v: _help_json(ctx, v),
        help="Dump command schema as JSON",
    ),
) -> None:
    """Train a detection model on a dataset."""
    import ast

    out = OutputHandler(json_mode=json_output, quiet=quiet)

    # Parse tuple strings
    try:
        mosaic_scale_val = (
            ast.literal_eval(mosaic_scale)
            if isinstance(mosaic_scale, str)
            else mosaic_scale
        )
        mixup_scale_val = (
            ast.literal_eval(mixup_scale)
            if isinstance(mixup_scale, str)
            else mixup_scale
        )
    except (ValueError, SyntaxError) as e:
        exit_with_error(out, "config_type_error", f"Invalid scale value: {e}")

    # Parse resume (can be "true"/"false" or a path)
    resume_val: bool | str = False
    if resume:
        if resume.lower() == "true":
            resume_val = True
        elif resume.lower() == "false":
            resume_val = False
        else:
            resume_val = resume

    # Detect model family
    family = detect_family_from_name(model)

    # All training params in CLI-facing names (single source of truth).
    # build_train_kwargs() maps these to TrainConfig field names automatically.
    params = {
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "workers": workers,
        "seed": seed,
        "resume": resume_val,
        "amp": amp,
        "optimizer": optimizer,
        "lr0": lr0,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "nesterov": nesterov,
        "scheduler": scheduler,
        "warmup_epochs": warmup_epochs,
        "warmup_lr_start": warmup_lr_start,
        "min_lr_ratio": min_lr_ratio,
        "mosaic": mosaic,
        "mixup": mixup,
        "hsv_prob": hsv_prob,
        "flip_prob": flip_prob,
        "degrees": degrees,
        "translate": translate,
        "shear": shear,
        "mosaic_scale": mosaic_scale_val,
        "mixup_scale": mixup_scale_val,
        "no_aug_epochs": no_aug_epochs,
        "ema": ema,
        "ema_decay": ema_decay,
        "eval_interval": eval_interval,
        "patience": patience,
        "project": project,
        "name": name,
        "exist_ok": exist_ok,
        "save_period": save_period,
        "log_interval": log_interval,
        "allow_download_scripts": allow_download_scripts,
    }
    if family:
        params = apply_family_defaults(params, family, "train")

    # RF-DETR: warn and ignore unsupported params
    rfdetr_warnings = []
    if family == "rfdetr":
        from ..config import is_user_provided

        for param_name in RFDETR_UNSUPPORTED_PARAMS:
            if is_user_provided(param_name):
                rfdetr_warnings.append(param_name)
        if rfdetr_warnings:
            out.progress(
                f"Warning: RF-DETR ignores these parameters: {', '.join(sorted(rfdetr_warnings))}"
            )

    # Dry run: validate and show resolved config
    if dry_run:
        data_out = {
            "valid": True,
            "mode": "train",
            "model_family": family or "auto-detect",
            "resolved_config": {
                "model": model,
                "data": data,
                "epochs": params["epochs"],
                "batch": params["batch"],
                "imgsz": params["imgsz"],
                "optimizer": params["optimizer"],
                "lr0": params["lr0"],
                "momentum": params["momentum"],
                "scheduler": params["scheduler"],
            },
        }
        if not json_output:
            import yaml

            data_out["_human_text"] = (
                f"Dry run — resolved config for {model}:\n"
                + yaml.dump(data_out["resolved_config"], default_flow_style=False)
            )
        out.result(data_out)
        return

    # Resolve CLI model name (yolox-s → LibreYOLOXs.pt)
    from ..config import resolve_model_name

    model_path = resolve_model_name(model)

    if allow_download_scripts:
        out.warning(
            "Dataset download scripts are enabled. Embedded Python from the dataset YAML may execute locally."
        )

    # Load model
    loaded_model = load_model_or_exit(
        out, model=model, model_path=model_path, device=device
    )

    # Build training kwargs, with family-specific translation where needed.
    train_kwargs = build_family_train_kwargs(params, family, model_path=model_path)
    train_kwargs["pretrained"] = pretrained  # Not in TrainConfig
    if family == "rfdetr":
        train_kwargs.pop("pretrained", None)
        if not val:
            from ..config import is_user_provided

            if is_user_provided("val"):
                out.progress(
                    "Warning: RF-DETR does not support disabling validation via val=false. Ignoring."
                )
    elif not val:
        train_kwargs["eval_interval"] = 0

    # Run training
    out.progress(f"Training {model} on {data} for {params['epochs']} epochs...")
    t0 = time.time()
    try:
        results = loaded_model.train(data=data, **train_kwargs)
    except FileNotFoundError as e:
        exit_with_error(
            out,
            "data_not_found",
            str(e),
            suggestion=f"Check that '{data}' exists and is a valid YOLO-format dataset YAML.",
        )
    except Exception as e:
        exit_with_error(out, "io_error", str(e))

    training_hours = (time.time() - t0) / 3600

    # Build output
    best_mAP50 = results.get("best_mAP50", None)
    best_mAP50_95 = results.get("best_mAP50_95", None)
    best_epoch = results.get("best_epoch", None)
    save_dir = results.get("save_dir") or results.get(
        "output_dir", f"{project}/{params['name']}"
    )
    best_weights = results.get("best_checkpoint") or f"{save_dir}/weights/best.pt"
    last_weights = results.get("last_checkpoint")
    if last_weights is None and loaded_model.FAMILY != "rfdetr":
        last_weights = f"{save_dir}/weights/last.pt"

    data_out = {
        "status": "complete",
        "model": model,
        "model_family": loaded_model.FAMILY,
        "data": data,
        "device": str(loaded_model.device),
        "epochs_completed": params["epochs"],
        "best_epoch": best_epoch,
        "best_metrics": (
            {"mAP50": best_mAP50, "mAP50_95": best_mAP50_95}
            if best_mAP50 is not None
            else None
        ),
        "best_weights": best_weights,
        "last_weights": last_weights,
        "training_time_hours": round(training_hours, 2),
        "save_dir": str(save_dir),
    }

    if not json_output:
        lines = [
            f"Training complete: {params['epochs']} epochs in {training_hours:.2f}h",
        ]
        if best_mAP50 is not None:
            lines.append(
                f"Best results at epoch {best_epoch}:\n"
                f"  mAP50: {best_mAP50:.4f}  mAP50-95: {best_mAP50_95:.4f}"
            )
        if best_weights:
            lines.append(f"Weights saved to: {best_weights}")
        else:
            lines.append(f"Artifacts saved to: {save_dir}")
        data_out["_human_text"] = "\n".join(lines)

    out.result(data_out)


def _help_json(ctx: typer.Context, value: bool) -> None:
    if not value:
        return
    from ..commands.special import _help_json_callback

    _help_json_callback(ctx, value)
