"""LoRA SFT trainer for the LibreVLM tier.

Deliberately NOT a ``BaseTrainer`` subclass: that chassis assumes stacked image
tensors, detection losses, EMA, and mAP validation, none of which apply to an
autoregressive fine-tune. This trainer keeps the repo's user-facing surface
(run directories, callbacks, loggers, results dict) and owns a compact loop:
cross-entropy on assistant tokens, gradient accumulation, cosine schedule with
warmup, best/last adapter checkpoints.

Checkpoints follow the directory contract in :mod:`.checkpoint`; best/last
selection uses validation loss (validation mAP lands with the tier's ``val()``
in a later iteration, as documented in the design docs).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ....data import load_data_config
from ....training.callbacks import (
    TrainCallbackList,
    TrainCallbacks,
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from ....training.loggers import resolve_loggers
from .checkpoint import is_vlm_checkpoint, save_vlm_checkpoint
from .collate import VLMChatCollator
from .data import VLMDetectDataset, resolve_split_source
from .recipes import VLMTrainRecipe, get_recipe
from .targets import FamilyFormat

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "VLM fine-tuning requires the 'vlm-train' extra. Install with:\n"
    "    pip install 'libreyolo[vlm-train]'"
)

__all__ = ["VLMDetectionTrainer", "VLMTrainConfig"]


@dataclass
class VLMTrainConfig:
    """Resolved VLM training configuration (user kwargs over recipe defaults)."""

    data: str = ""
    epochs: int = 10
    batch: int = 1
    accumulate: int = 8
    lr0: Optional[float] = None
    lora: bool = True
    output_dir: str = "runs/vlm/train"
    project: Optional[str] = None
    name: Optional[str] = None
    exist_ok: bool = False
    workers: int = 0
    seed: int = 0
    device: Optional[str] = None
    gradient_checkpointing: bool = True
    hflip: float = 0.5
    vram_check: bool = True
    resume: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _normalize_names(raw) -> Dict[int, str]:
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)):
        return {i: str(v) for i, v in enumerate(raw)}
    raise ValueError("Dataset YAML must define names as a list or an id mapping.")


class VLMDetectionTrainer:
    """Drive one LoRA (or full) detection fine-tune of a LibreVLM wrapper."""

    def __init__(
        self,
        wrapper,
        data: str,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ) -> None:
        if not data:
            raise ValueError("train() requires data=<dataset yaml>.")
        known = set(VLMTrainConfig.__dataclass_fields__) - {"data", "extra"}
        config_kwargs = {k: v for k, v in kwargs.items() if k in known}
        extra = {k: v for k, v in kwargs.items() if k not in known}
        if extra:
            logger.warning("Ignoring unknown train() kwargs: %s", sorted(extra))
        self.config = VLMTrainConfig(data=data, extra=extra, **config_kwargs)
        if self.config.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.config.epochs}")
        if self.config.batch < 1 or self.config.accumulate < 1:
            raise ValueError("batch and accumulate must both be >= 1.")
        self.wrapper = wrapper
        self.recipe: VLMTrainRecipe = get_recipe(wrapper.FAMILY)
        self.callbacks = TrainCallbackList(callbacks)
        for logger_cb in resolve_loggers(loggers):
            self.callbacks.append(logger_cb)
        if self.config.resume is True:
            # resume=True reads weights/last from this exact save_dir (see
            # _resolve_resume_dir); never let _resolve_save_dir() increment
            # away from it mid-resume.
            self.config.exist_ok = True
        self.save_dir = self._resolve_save_dir()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _resolve_save_dir(self) -> Path:
        cfg = self.config
        output = Path(cfg.output_dir)
        project = Path(cfg.project) if cfg.project else output.parent
        name = cfg.name if cfg.name else output.name
        run_dir = Path(project) / str(name)
        if run_dir.exists() and not cfg.exist_ok:
            base = run_dir
            counter = 2
            while run_dir.exists():
                run_dir = base.with_name(f"{base.name}{counter}")
                counter += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _resolve_device(self) -> torch.device:
        if self.config.device:
            return torch.device(self.config.device)
        return self.wrapper.device

    def _check_vram_for_full_ft(self, device: torch.device) -> None:
        if self.config.lora:
            return
        if device.type != "cuda":
            logger.warning(
                "Full fine-tuning on %s will be extremely slow; lora=True is "
                "the supported path on this hardware.",
                device.type,
            )
            return
        if not self.config.vram_check:
            return
        params = sum(p.numel() for p in self.wrapper.model.parameters())
        # Full FT with AdamW: weights + grads + two fp32 moments, before
        # activations. ~16 bytes/param is the optimistic floor in bf16.
        needed = params * 16
        total = torch.cuda.get_device_properties(device).total_memory
        if needed > total:
            raise RuntimeError(
                f"Full fine-tuning needs roughly {needed / 1e9:.0f} GB for "
                f"optimizer state alone; this GPU has {total / 1e9:.0f} GB. "
                "Use lora=True (the default), or pass vram_check=False to "
                "proceed anyway."
            )

    def _resolve_resume_dir(self) -> Optional[Path]:
        resume = self.config.resume
        if not resume:
            return None
        candidate = (
            self.save_dir / "weights" / "last" if resume is True else Path(resume)
        )
        if not is_vlm_checkpoint(candidate):
            raise FileNotFoundError(
                f"resume={resume!r} is not a VLM checkpoint directory "
                "(missing libreyolo_vlm.json)."
            )
        return candidate

    def _build_train_model(self, resume_dir: Optional[Path]):
        """Return the trainable module: a PeftModel (LoRA) or the base model."""
        if self.config.lora:
            try:
                from peft import LoraConfig, PeftModel, get_peft_model
            except ImportError as exc:
                raise ImportError(_INSTALL_HINT) from exc
            if resume_dir is not None:
                model = PeftModel.from_pretrained(
                    self.wrapper.model, str(resume_dir), is_trainable=True
                )
                logger.warning(
                    "Resumed adapter weights from %s (optimizer state starts "
                    "fresh).",
                    resume_dir,
                )
            else:
                lora_config = LoraConfig(
                    r=self.recipe.lora_r,
                    lora_alpha=self.recipe.lora_alpha,
                    lora_dropout=self.recipe.lora_dropout,
                    target_modules=self.recipe.target_modules,
                    bias="none",
                )
                model = get_peft_model(self.wrapper.model, lora_config)
            injected = [
                name
                for name, module in model.named_modules()
                if getattr(module, "lora_A", None) is not None
            ]
            if not injected:
                raise RuntimeError(
                    f"LoRA recipe for {self.wrapper.FAMILY!r} matched no "
                    "modules; the base model layout changed. This is a "
                    "LibreYOLO bug."
                )
            leaks = [
                name
                for name in injected
                if any(prefix in name for prefix in self.recipe.frozen_prefixes)
            ]
            if leaks:
                raise RuntimeError(
                    f"LoRA recipe for {self.wrapper.FAMILY!r} leaked adapters "
                    f"into frozen scope: {leaks[:3]}. This is a LibreYOLO bug."
                )
            logger.info("Injected LoRA adapters into %d modules.", len(injected))
            return model

        # Full fine-tune: freeze the recipe's frozen prefixes, train the rest.
        if resume_dir is not None:
            raise NotImplementedError(
                "resume= is only supported for LoRA training (lora=True)."
            )
        frozen = 0
        for name, param in self.wrapper.model.named_parameters():
            if any(name.startswith(prefix) for prefix in self.recipe.frozen_prefixes):
                param.requires_grad_(False)
                frozen += 1
        logger.info("Full fine-tune: froze %d frozen-scope parameters.", frozen)
        return self.wrapper.model

    def _build_dataloaders(self, data_cfg: Dict, names: Dict[int, str], fmt: FamilyFormat):
        cfg = self.config
        train_source = resolve_split_source(data_cfg, "train")
        if not train_source:
            raise ValueError(f"Dataset {cfg.data!r} has no train split.")
        val_source = resolve_split_source(data_cfg, "val")

        collator = VLMChatCollator(
            self.wrapper.processor, max_length_warn=self.recipe.max_length_warn
        )
        pin = self._resolve_device().type == "cuda"
        train_set = VLMDetectDataset(
            train_source,
            names,
            fmt,
            augment=cfg.hflip > 0,
            hflip_p=cfg.hflip,
            seed=cfg.seed,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch,
            shuffle=True,
            num_workers=cfg.workers,
            collate_fn=collator,
            generator=torch.Generator().manual_seed(cfg.seed),
            pin_memory=pin,
            persistent_workers=cfg.workers > 0,
        )
        val_loader = None
        if val_source:
            val_set = VLMDetectDataset(val_source, names, fmt, augment=False)
            val_loader = DataLoader(
                val_set,
                batch_size=cfg.batch,
                shuffle=False,
                num_workers=cfg.workers,
                collate_fn=collator,
                pin_memory=pin,
                persistent_workers=cfg.workers > 0,
            )
        return train_loader, val_loader

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        start_time = time.time()
        torch.manual_seed(cfg.seed)

        device = self._resolve_device()
        self._check_vram_for_full_ft(device)

        data_cfg = load_data_config(
            cfg.data, allow_scripts=bool(cfg.extra.get("allow_download_scripts", False))
        )
        names = _normalize_names(data_cfg.get("names"))
        # Vocabulary comes from the dataset; sticky on the wrapper from here on.
        self.wrapper.set_classes([names[i] for i in range(len(names))])
        fmt = FamilyFormat.from_model(self.wrapper)
        train_loader, val_loader = self._build_dataloaders(data_cfg, names, fmt)

        resume_dir = self._resolve_resume_dir()
        train_model = self._build_train_model(resume_dir)
        train_model.to(device)
        train_model.train()
        if cfg.gradient_checkpointing and hasattr(
            train_model, "gradient_checkpointing_enable"
        ):
            train_model.gradient_checkpointing_enable()
            if hasattr(train_model, "enable_input_require_grads"):
                train_model.enable_input_require_grads()

        lr0 = (
            cfg.lr0
            if cfg.lr0 is not None
            else (self.recipe.lr0 if cfg.lora else self.recipe.full_ft_lr0)
        )
        trainable = [p for p in train_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=lr0, weight_decay=self.recipe.weight_decay
        )
        steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.accumulate))
        total_steps = steps_per_epoch * cfg.epochs
        warmup_steps = max(1, int(total_steps * self.recipe.warmup_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.wrapper._model_dtype)
            if device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )

        weights_dir = self.save_dir / "weights"
        best_metric: Optional[float] = None
        best_epoch: Optional[int] = None
        final_loss = float("nan")
        completed_epochs = 0
        config_dump = {
            **asdict(cfg),
            "lr0": lr0,
            "resume": str(cfg.resume) if cfg.resume else None,
            "family": self.wrapper.FAMILY,
        }
        config_dump.pop("extra", None)

        self.callbacks.on_train_start(
            TrainStartEvent(
                start_epoch=1,
                total_epochs=cfg.epochs,
                model_family=self.wrapper.FAMILY,
                model_size=self.wrapper.size,
                task="detect",
                save_dir=str(self.save_dir),
                config=config_dump,
            )
        )

        try:
            for epoch in range(1, cfg.epochs + 1):
                epoch_start = time.time()
                running, seen = 0.0, 0
                optimizer.zero_grad(set_to_none=True)
                progress = tqdm(
                    train_loader,
                    desc=f"vlm train {epoch}/{cfg.epochs}",
                    unit="batch",
                    leave=False,
                )
                for step, batch in enumerate(progress, 1):
                    batch = self._to_device(batch, device)
                    with autocast_ctx:
                        loss = train_model(**batch).loss
                    (loss / cfg.accumulate).backward()
                    running += float(loss.detach())
                    seen += 1
                    if step % cfg.accumulate == 0 or step == len(train_loader):
                        torch.nn.utils.clip_grad_norm_(
                            trainable, self.recipe.clip_grad_norm
                        )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    progress.set_postfix(loss=f"{running / max(seen, 1):.4f}")

                train_loss = running / max(seen, 1)
                final_loss = train_loss
                val_metrics: Dict[str, float] = {}
                if val_loader is not None:
                    val_metrics["val/loss"] = self._eval_loss(
                        train_model, val_loader, device, autocast_ctx
                    )
                metric_name = "val/loss" if val_loader is not None else "train/loss"
                current = val_metrics.get("val/loss", train_loss)
                is_best = best_metric is None or current < best_metric
                if is_best:
                    best_metric = current
                    best_epoch = epoch

                metrics = {"train/loss": train_loss, "epoch": epoch, **val_metrics}
                save_vlm_checkpoint(
                    weights_dir / "last",
                    peft_model=train_model,
                    processor=self.wrapper.processor,
                    wrapper=self.wrapper,
                    metrics=metrics,
                )
                if is_best:
                    save_vlm_checkpoint(
                        weights_dir / "best",
                        peft_model=train_model,
                        processor=self.wrapper.processor,
                        wrapper=self.wrapper,
                        metrics=metrics,
                    )
                completed_epochs = epoch

                self.callbacks.on_train_epoch_end(
                    TrainEpochEvent(
                        epoch=epoch,
                        total_epochs=cfg.epochs,
                        model_family=self.wrapper.FAMILY,
                        model_size=self.wrapper.size,
                        task="detect",
                        save_dir=str(self.save_dir),
                        train_loss=train_loss,
                        train_loss_items={"loss": train_loss},
                        lr={"lr0": optimizer.param_groups[0]["lr"]},
                        val_metrics=val_metrics,
                        validated=val_loader is not None,
                        is_best=is_best,
                        current_metric=current,
                        current_metric_name=metric_name,
                        best_metric=best_metric,
                        best_metric_name=metric_name,
                        best_epoch=best_epoch,
                        epoch_seconds=time.time() - epoch_start,
                    )
                )
        except BaseException as exc:
            self.callbacks.on_train_exception(
                TrainExceptionEvent(
                    epoch=completed_epochs or None,
                    total_epochs=cfg.epochs,
                    model_family=self.wrapper.FAMILY,
                    model_size=self.wrapper.size,
                    task="detect",
                    save_dir=str(self.save_dir),
                    exception=exc,
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                    elapsed_seconds=time.time() - start_time,
                )
            )
            raise
        finally:
            self._restore_inference_model(train_model)

        results: Dict[str, Any] = {
            "save_dir": str(self.save_dir),
            "best": str(weights_dir / "best"),
            "last": str(weights_dir / "last"),
            "epochs": completed_epochs,
            "final_loss": final_loss,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "metric_name": "val/loss" if val_loader is not None else "train/loss",
        }
        self.callbacks.on_train_end(
            TrainEndEvent(
                total_epochs=cfg.epochs,
                completed_epochs=completed_epochs,
                model_family=self.wrapper.FAMILY,
                model_size=self.wrapper.size,
                task="detect",
                save_dir=str(self.save_dir),
                final_loss=final_loss,
                best_metric=best_metric,
                best_epoch=best_epoch,
                total_seconds=time.time() - start_time,
                results=results,
            )
        )
        return results

    # ------------------------------------------------------------------
    # Pieces
    # ------------------------------------------------------------------

    @staticmethod
    def _to_device(batch, device: torch.device):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _eval_loss(self, model, loader, device, autocast_ctx) -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch, device)
                with autocast_ctx:
                    total += float(model(**batch).loss.detach())
                count += 1
        model.train()
        return total / max(count, 1)

    def _restore_inference_model(self, train_model) -> None:
        """Leave the wrapper usable for predict() after training.

        The LoRA path trained a PeftModel around ``self.wrapper.model``;
        merging the trained adapter into dense weights hands the wrapper back a
        plain model whose ``predict()`` reflects the fine-tune. The full-FT
        path trained the wrapper's model in place, so only eval() is needed.
        """
        try:
            from peft import PeftModel
        except ImportError:
            self.wrapper.model.eval()
            return
        if isinstance(train_model, PeftModel):
            self.wrapper.model = train_model.merge_and_unload()
            logger.info("Merged trained LoRA adapters into the loaded model.")
        self.wrapper.model.eval()
