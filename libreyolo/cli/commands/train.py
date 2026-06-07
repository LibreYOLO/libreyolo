"""Train command: train a model on a dataset."""

from pathlib import Path
import time
from typing import Optional

import typer

from ..command_utils import (
    exit_stage_error,
    exit_with_error,
    get_loaded_model_family,
    get_user_provided_params,
    help_json_callback,
    load_model_or_exit,
    resolve_model_or_exit,
)
from ..config import (
    apply_family_defaults,
    build_family_train_kwargs,
    detect_family_from_model_ref,
    get_unsupported_train_params,
)
from ..output import OutputHandler
from ...training.freezing import normalize_freeze_selectors, parse_freeze_spec


_LORA_TRAIN_FAMILIES = {"rfdetr"}


def _model_ref_exists(model_path: str) -> bool:
    path = Path(model_path)
    if path.exists():
        return True
    return path.parent == Path(".") and (Path("weights") / path.name).exists()


def _create_explicit_task_train_model(
    *,
    family: str | None,
    model_path: str,
    task: str | None,
    resume: bool | str,
    device: str,
):
    """Instantiate task-specific train models that should start from architecture.

    Cross-task training can use detect checkpoints only as transfer weights.
    Create the requested architecture first so task-specific heads exist before
    training.
    """
    if family not in {"yolo9", "rfdetr"} or resume:
        return None

    from libreyolo.tasks import normalize_task

    if family == "yolo9":
        from libreyolo.models.yolo9.model import LibreYOLO9 as model_cls
    else:
        from libreyolo.models.rfdetr.model import LibreRFDETR as model_cls

    filename_task = model_cls.detect_task_from_filename(Path(model_path).name)
    train_task = normalize_task(task) if task is not None else filename_task
    if train_task is None:
        return None
    if task is None and filename_task == train_task and _model_ref_exists(model_path):
        return None

    size = model_cls.detect_size_from_filename(Path(model_path).name)
    if size is None:
        return None
    if family == "rfdetr" and train_task == "obb" and _model_ref_exists(model_path):
        return model_cls(
            model_path,
            size=size,
            task=train_task,
            device=device,
            allow_detect_to_obb_transfer=True,
        )
    extra = (
        {"allow_detect_to_obb_transfer": True}
        if family == "rfdetr" and train_task == "obb"
        else {}
    )
    return model_cls(None, size=size, task=train_task, device=device, **extra)


def _create_yolo9_obb_from_loaded_detect_model(loaded_model, device: str):
    """Switch an already-loaded YOLO9 detect checkpoint to OBB architecture."""
    if (
        get_loaded_model_family(loaded_model) != "yolo9"
        or getattr(loaded_model, "task", "detect") != "detect"
    ):
        return None

    from libreyolo.models.yolo9.model import LibreYOLO9

    size = getattr(loaded_model, "size", None)
    if size is None:
        return None
    return LibreYOLO9(None, size=size, task="obb", device=device)


def _create_rfdetr_obb_from_loaded_detect_model(
    loaded_model,
    *,
    model_path: str,
    device: str,
):
    """Switch an already-loaded RF-DETR detect checkpoint to OBB architecture."""
    if (
        get_loaded_model_family(loaded_model) != "rfdetr"
        or getattr(loaded_model, "task", "detect") != "detect"
    ):
        return None

    from libreyolo.models.rfdetr.model import LibreRFDETR

    return LibreRFDETR(
        model_path,
        size=getattr(loaded_model, "size", None),
        task="obb",
        device=device,
        allow_detect_to_obb_transfer=True,
    )


def _create_yolo9_task_from_loaded_model(loaded_model, task: str, device: str):
    if get_loaded_model_family(loaded_model) != "yolo9":
        return None

    from libreyolo.models.yolo9.model import LibreYOLO9

    size = getattr(loaded_model, "size", None)
    if size is None:
        return None
    return LibreYOLO9(None, size=size, task=task, device=device)


def _should_use_yolo9_path_as_transfer(model_path: str, task: str | None) -> bool:
    if task is None or not Path(model_path).exists():
        return False

    from libreyolo.models.yolo9.model import LibreYOLO9

    filename_task = LibreYOLO9.detect_task_from_filename(Path(model_path).name)
    return filename_task != task


def train_cmd(
    data: str = typer.Option(
        ..., help="Path to dataset YAML (YOLO format, e.g. coco8.yaml)"
    ),
    model: str = typer.Option("yolox-s", help="Model name or path to weights"),
    task: Optional[str] = typer.Option(
        None, help="Explicit task override: detect, segment, pose, classify, gaze, obb"
    ),
    # Training
    epochs: int = typer.Option(300, help="Training epochs"),
    batch: int = typer.Option(16, help="Batch size per device"),
    imgsz: int = typer.Option(640, help="Training image size"),
    device: str = typer.Option("auto", help="Device: 0, cpu, mps, auto"),
    workers: int = typer.Option(4, help="Dataloader workers"),
    cache: str = typer.Option(
        "false", help="Cache images to speed dataloading: ram, disk, true, false"
    ),
    seed: int = typer.Option(0, help="Random seed"),
    resume: str = typer.Option("", help="Resume training: true, or path to checkpoint"),
    amp: bool = typer.Option(True, help="Automatic Mixed Precision"),
    pretrained: bool = typer.Option(True, help="Use pretrained weights"),
    lora: bool = typer.Option(
        False,
        "--lora",
        help="Enable LoRA fine-tuning for supported transformer families",
    ),
    freeze: str = typer.Option(
        "",
        help="Freeze layers: int count, list of indices, or module name(s)",
    ),
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
    lr_drop: int = typer.Option(100, help="RF-DETR step LR drop epoch"),
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
    save_plots: bool = typer.Option(
        False, help="Save final validation plots during training"
    ),
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
        callback=help_json_callback,
        help="Dump command schema as JSON",
    ),
) -> None:
    """Train a detection model on a dataset."""
    import ast

    out = OutputHandler(json_mode=json_output, quiet=quiet)
    user_provided = get_user_provided_params()
    normalized_task = None
    if task is not None:
        from libreyolo.tasks import normalize_task

        try:
            normalized_task = normalize_task(task)
        except ValueError as e:
            exit_with_error(out, "config_type_error", str(e))

    # Parse tuple/list strings
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
        freeze_val = parse_freeze_spec(freeze)
        normalize_freeze_selectors(freeze_val)
    except (TypeError, ValueError, SyntaxError) as e:
        exit_with_error(out, "config_type_error", f"Invalid train option value: {e}")

    # Parse cache (can be "ram"/"disk" or a bool string)
    cache_val: bool | str = False
    cache_str = cache.strip().lower()
    if cache_str in ("ram", "disk"):
        cache_val = cache_str
    elif cache_str in ("true", "1", "yes"):
        cache_val = True

    # Parse resume (can be "true"/"false" or a path)
    resume_val: bool | str = False
    if resume:
        if resume.lower() == "true":
            resume_val = True
        elif resume.lower() == "false":
            resume_val = False
        else:
            resume_val = resume

    model_path = resolve_model_or_exit(out, model)
    family = detect_family_from_model_ref(model, model_path, inspect_checkpoint=dry_run)
    loaded_model = None
    train_pretrained = pretrained
    if family is None and not dry_run:
        loaded_model = load_model_or_exit(
            out, model=model, model_path=model_path, device=device
        )
        family = get_loaded_model_family(loaded_model)
    if loaded_model is None:
        loaded_model = _create_explicit_task_train_model(
            family=family,
            model_path=model_path,
            task=normalized_task,
            resume=resume_val,
            device=device,
        )
        if (
            loaded_model is not None
            and family == "yolo9"
            and train_pretrained is True
            and _should_use_yolo9_path_as_transfer(model_path, normalized_task)
        ):
            train_pretrained = model_path
    elif normalized_task is not None:
        loaded_task = getattr(loaded_model, "task", "detect")
        if loaded_task != normalized_task:
            replacement = None
            replacement = _create_yolo9_task_from_loaded_model(
                loaded_model,
                normalized_task,
                device=device,
            )
            if replacement is None and normalized_task == "obb":
                replacement = _create_yolo9_obb_from_loaded_detect_model(
                    loaded_model,
                    device=device,
                )
                if replacement is None:
                    replacement = _create_rfdetr_obb_from_loaded_detect_model(
                        loaded_model,
                        model_path=model_path,
                        device=device,
                    )
            if replacement is None:
                exit_with_error(
                    out,
                    "config_unsupported",
                    f"Loaded model task '{loaded_task}' does not match requested task "
                    f"'{normalized_task}'.",
                )
            loaded_model = replacement
            if train_pretrained is True and get_loaded_model_family(loaded_model) == "yolo9":
                train_pretrained = model_path

    # All training params in CLI-facing names (single source of truth).
    # build_train_kwargs() maps these to TrainConfig field names automatically.
    params = {
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "workers": workers,
        "cache": cache_val,
        "seed": seed,
        "resume": resume_val,
        "amp": amp,
        "lora": lora,
        "freeze": freeze_val,
        "optimizer": optimizer,
        "lr0": lr0,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "nesterov": nesterov,
        "scheduler": scheduler,
        "warmup_epochs": warmup_epochs,
        "warmup_lr_start": warmup_lr_start,
        "min_lr_ratio": min_lr_ratio,
        "lr_drop": lr_drop,
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
        "save_plots": save_plots,
        "patience": patience,
        "project": project,
        "name": name,
        "exist_ok": exist_ok,
        "save_period": save_period,
        "log_interval": log_interval,
        "allow_download_scripts": allow_download_scripts,
    }
    if family:
        params = apply_family_defaults(
            params, family, "train", user_provided=user_provided
        )

    if params["lora"] and family is not None and family not in _LORA_TRAIN_FAMILIES:
        exit_with_error(
            out,
            "config_unsupported",
            f"LoRA fine-tuning (lora=True) is not supported for {family}.",
            suggestion="Use an RF-DETR model or remove --lora.",
        )

    # RF-DETR: warn and ignore unsupported params
    rfdetr_warnings = []
    unsupported_params = get_unsupported_train_params(family)
    if unsupported_params:
        for param_name in unsupported_params:
            if param_name in user_provided:
                rfdetr_warnings.append(param_name)
        if rfdetr_warnings:
            out.progress(
                f"Warning: RF-DETR ignores these parameters: {', '.join(sorted(rfdetr_warnings))}"
            )

    # Dry run: validate and show resolved config
    if dry_run:
        resolved_config = {
            "model": model,
            "data": data,
            "epochs": params["epochs"],
            "batch": params["batch"],
            "imgsz": params["imgsz"],
            "optimizer": params["optimizer"],
            "lr0": params["lr0"],
            "momentum": params["momentum"],
            "scheduler": params["scheduler"],
        }
        if params.get("freeze") is not None:
            resolved_config["freeze"] = params["freeze"]
        if normalized_task is not None:
            resolved_config["task"] = normalized_task
        if family == "rfdetr":
            resolved_config = {
                "model": model,
                "data": data,
                "epochs": params["epochs"],
                "batch": params["batch"],
                "lr0": params["lr0"],
                "workers": params["workers"],
                "weight_decay": params["weight_decay"],
                "eval_interval": params["eval_interval"],
                "warmup_epochs": params["warmup_epochs"],
                "lr_drop": params["lr_drop"],
                "ema": params["ema"],
                "ema_decay": params["ema_decay"],
                "save_period": params["save_period"],
                "lora": params["lora"],
            }
            if params.get("freeze") is not None:
                resolved_config["freeze"] = params["freeze"]
            if normalized_task is not None:
                resolved_config["task"] = normalized_task

        data_out = {
            "valid": True,
            "mode": "train",
            "model_family": family or "auto-detect",
            "resolved_config": resolved_config,
        }
        if not json_output:
            import yaml

            data_out["_human_text"] = (
                f"Dry run — resolved config for {model}:\n"
                + yaml.dump(data_out["resolved_config"], default_flow_style=False)
            )
        out.result(data_out)
        return

    if allow_download_scripts:
        out.warning(
            "Dataset download scripts are enabled. Embedded Python from the dataset YAML may execute locally."
        )

    # Load model
    if loaded_model is None:
        load_kwargs = {
            "out": out,
            "model": model,
            "model_path": model_path,
            "device": device,
        }
        if normalized_task is not None:
            load_kwargs["task"] = normalized_task
        loaded_model = load_model_or_exit(**load_kwargs)
    loaded_family = get_loaded_model_family(loaded_model) or family

    # Build training kwargs, with family-specific translation where needed.
    train_kwargs = build_family_train_kwargs(
        params, family, model_path=model_path, user_provided=user_provided
    )
    train_kwargs["pretrained"] = train_pretrained  # Not in TrainConfig
    if family == "rfdetr":
        train_kwargs.pop("pretrained", None)
        if not val and "val" in user_provided:
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
        exit_stage_error(out, stage="Training", detail=e)

    training_hours = (time.time() - t0) / 3600

    # Build output
    best_mAP50 = results.get("best_mAP50", None)
    best_mAP50_95 = results.get("best_mAP50_95", None)
    best_epoch = results.get("best_epoch", None)
    save_dir = results.get("save_dir") or results.get(
        "output_dir", f"{project}/{params['name']}"
    )
    best_weights = results.get("best_checkpoint")
    last_weights = results.get("last_checkpoint")

    data_out = {
        "status": "complete",
        "model": model,
        "model_family": loaded_family,
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
