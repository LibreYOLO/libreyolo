"""Histogram detection training uses the shared numerical input contract."""

from pathlib import Path

from ..utils.event_histogram import (
    HistogramTransform,
    configure_input,
    validate_input_profile,
)


def setup_histogram_data(trainer, cfg):
    """Build a detection loader without RGB colour or compositing transforms."""
    from .dataset import YOLODataset, create_dataloader

    wrapper = trainer.wrapper_model
    configure_input(wrapper, cfg.get("input_profile"))
    profile = wrapper.input_profile
    wrapper.input_size = trainer.config.imgsz
    trainer.model = wrapper.model
    trainer.num_classes = int(cfg.get("nc") or len(cfg["names"]))
    trainer.config.num_classes = trainer.num_classes
    if trainer.is_distributed or trainer.config.batch == -1:
        raise ValueError(
            "Histogram training currently requires a fixed batch on one device"
        )
    if wrapper.FAMILY == "rfdetr":
        wrapper._validate_imgsz(trainer.config.imgsz)
    preproc = HistogramTransform(
        profile,
        wrapper.FAMILY,
        getattr(wrapper, "letterbox_pad", "topleft"),
        training=True,
        flip_prob=trainer.config.flip_prob,
        flipud=trainer.config.flipud,
        max_labels=getattr(trainer.config, "max_labels", 300),
    )
    dataset = YOLODataset(
        img_files=cfg.get("train_img_files", []),
        label_files=cfg.get("train_label_files"),
        img_size=trainer.input_size,
        preproc=preproc,
        num_classes=trainer.num_classes,
        single_cls=trainer.config.single_cls,
        input_profile=profile,
    )
    dataset.enable_image_cache(getattr(trainer.config, "cache", False))
    trainer.train_loader = create_dataloader(
        dataset,
        batch_size=trainer.config.batch,
        num_workers=trainer.config.workers,
        shuffle=True,
        pin_memory=trainer.device.type == "cuda",
        min_samples=int(getattr(trainer.config, "min_samples", 0) or 0),
        class_balanced=bool(getattr(trainer.config, "class_balanced", False)),
    )
    return dataset


def prepare_histogram_training(wrapper, args, kwargs):
    """Apply the supported event recipe before the family constructs its trainer."""
    from . import load_data_config

    # Only inspect YAML paths here; other tasks can accept directories/dicts.
    data = kwargs.get("data", args[0] if args else None)
    if not isinstance(data, (str, Path)) or Path(data).suffix.lower() not in {
        ".yaml",
        ".yml",
    }:
        return
    if not Path(data).is_file():
        return
    cfg = load_data_config(
        data, allow_scripts=kwargs.get("allow_download_scripts", False)
    )
    profile = validate_input_profile(
        cfg.get("input_profile"), family=wrapper.FAMILY, task=wrapper.task
    )
    if profile is None:
        if getattr(wrapper, "input_profile", None) is not None:
            raise ValueError(
                "Histogram model training requires matching input_profile in dataset YAML"
            )
        return
    current = getattr(wrapper, "input_profile", None)
    if current is not None and current != profile:
        raise ValueError("Dataset input_profile does not match the loaded model")
    if kwargs.get("batch", kwargs.get("batch_size")) == -1:
        raise ValueError("Histogram training requires a fixed positive batch")
    defaults = histogram_recipe_defaults(wrapper.FAMILY)
    for key, value in defaults.items():
        if key in kwargs and kwargs[key] != value:
            raise ValueError(
                f"{key} is not supported for histogram training; use {key}={value}"
            )
        kwargs[key] = value
    for key in ("distill_model", "lora", "quant"):
        if kwargs.get(key):
            raise ValueError(f"{key} is not supported for histogram training")


def histogram_recipe_defaults(family):
    defaults = dict.fromkeys(
        (
            "mosaic_prob",
            "mixup_prob",
            "hsv_prob",
            "degrees",
            "translate",
            "shear",
            "perspective",
            "copy_paste",
            "rot90",
        ),
        0.0,
    )
    defaults.update(cuda_graph=False, mosaic_scale=(1.0, 1.0), mixup_scale=(1.0, 1.0))
    if family == "rfdetr":
        defaults.update(
            crop_resize_prob=0.0, multi_scale=False, do_random_resize_via_padding=False
        )
    return defaults


def apply_histogram_cli_defaults(params, *, data, family, user_provided):
    """Keep CLI-derived defaults separate from explicitly requested options."""
    from ..cli.aliases import TRAIN_ALIASES
    from . import load_data_config

    if not isinstance(data, (str, Path)) or not Path(data).is_file():
        return False
    cfg = load_data_config(data)
    if cfg.get("input_profile") is None:
        return False
    validate_input_profile(cfg["input_profile"], family=family)
    defaults = histogram_recipe_defaults(family)
    for key in list(params):
        internal = TRAIN_ALIASES.get(key, key)
        if internal in defaults:
            if key in user_provided and params[key] != defaults[internal]:
                raise ValueError(f"{key} is not supported for histogram training")
            params[key] = defaults[internal]
    return True
