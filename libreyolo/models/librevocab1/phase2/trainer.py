"""LibreVocab1 Phase 2 standalone trainer.

Deliberately *not* a ``BaseTrainer`` subclass. The LibreYOLO base trainer is
shaped around fixed-vocab YOLO/DETR pipelines (single-image forward, integer
class targets, mosaic dataset). Open-vocab needs a ``(images, text_emb)``
forward and per-batch prompt vocabulary, neither of which fit BaseTrainer's
hooks cleanly. We bypass the abstraction for now and use a clean, testable
training loop that we can wrap in BaseTrainer later if the contract grows.

Surface:

    Phase2Trainer(model, criterion, train_loader,
                  optimizer=..., scheduler=..., config=...).fit(epochs)

The model must be callable as::

    out = model(batch["images"], batch["text_emb"])

and return ``{"pred_logits": (B, Q, K), "pred_boxes": (B, Q, 4)}``.

This contract is satisfied by ``LibreVocab1Phase2Network`` once the decoder
stub is replaced. For tonight's smoke tests we use a fake-decoder to verify
the trainer harness end to end.
"""

from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .loss import OpenVocabSetCriterion

logger = logging.getLogger(__name__)


@dataclass
class Phase2TrainResult:
    """Lightweight summary of a finished training run."""

    epochs_completed: int
    last_step: int
    last_loss: float
    epoch_losses: List[float] = field(default_factory=list)


class Phase2Trainer:
    """Open-vocab DETR training loop for LibreVocab1 Phase 2."""

    def __init__(
        self,
        model: nn.Module,
        criterion: OpenVocabSetCriterion,
        train_loader: Iterable[Dict[str, Any]],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str | torch.device = "cpu",
        clip_max_norm: float = 0.1,
        amp: bool = False,
        ema_decay: Optional[float] = 0.9999,
        log_interval: int = 10,
        on_step_end: Optional[Callable[["Phase2Trainer", Dict[str, Any]], None]] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.optimizer = optimizer or torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=1e-4,
            weight_decay=1e-4,
        )
        self.scheduler = scheduler
        self.device = torch.device(device) if isinstance(device, str) else device
        self.clip_max_norm = float(clip_max_norm) if clip_max_norm else 0.0
        self.amp_enabled = bool(amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler() if self.amp_enabled else None
        self.log_interval = int(log_interval)
        self.on_step_end = on_step_end

        self.global_step = 0
        self.epoch = 0
        self.last_loss = float("nan")
        self.epoch_losses: List[float] = []
        self.ema = (
            _ParamEMA(self.model, decay=ema_decay)
            if ema_decay is not None
            else None
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=False)
            elif k == "targets":
                out[k] = [
                    {kk: vv.to(self.device) if isinstance(vv, torch.Tensor) else vv
                     for kk, vv in t.items()}
                    for t in v
                ]
            else:
                out[k] = v
        return out

    def _forward_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        out = self.model(batch["images"], batch["text_emb"])
        return self.criterion(out, batch["targets"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Run one optimizer step on a single batch. Returns loss components."""
        self.model.train()
        batch = self._move_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.amp.autocast(device_type=self.device.type, dtype=torch.float16)
            if self.amp_enabled
            else nullcontext()
        )
        with autocast_ctx:
            loss_dict = self._forward_loss(batch)
            total = loss_dict["loss_cls"] + loss_dict["loss_bbox"] + loss_dict["loss_giou"]

        if not torch.isfinite(total):
            raise RuntimeError(
                f"non-finite loss at step {self.global_step}: "
                f"{ {k: float(v) for k, v in loss_dict.items()} }"
            )

        if self.scaler is not None:
            self.scaler.scale(total).backward()
            if self.clip_max_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.clip_max_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total.backward()
            if self.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.clip_max_norm,
                )
            self.optimizer.step()

        if self.scheduler is not None:
            # Schedulers in LibreYOLO use ``update_lr(global_step)``; native
            # PyTorch schedulers use ``step()``. Support both.
            if hasattr(self.scheduler, "update_lr"):
                self.scheduler.update_lr(self.global_step)
            else:
                self.scheduler.step()

        if self.ema is not None:
            self.ema.update(self.model)

        self.global_step += 1
        out = {k: float(v.detach()) for k, v in loss_dict.items()}
        out["loss_total"] = float(total.detach())
        self.last_loss = out["loss_total"]
        if self.on_step_end is not None:
            self.on_step_end(self, batch)
        return out

    def train_one_epoch(self) -> float:
        """Iterate the loader once. Returns mean total loss for the epoch."""
        epoch_total = 0.0
        n_steps = 0
        t0 = time.perf_counter()
        for batch in self.train_loader:
            losses = self.step(batch)
            epoch_total += losses["loss_total"]
            n_steps += 1
            if n_steps % self.log_interval == 0 or n_steps == 1:
                logger.info(
                    "epoch %d step %d/%d total=%.4f cls=%.4f bbox=%.4f giou=%.4f",
                    self.epoch,
                    n_steps,
                    getattr(self.train_loader, "__len__", lambda: 0)() or n_steps,
                    losses["loss_total"],
                    losses["loss_cls"],
                    losses["loss_bbox"],
                    losses["loss_giou"],
                )
        mean_loss = epoch_total / max(n_steps, 1)
        self.epoch_losses.append(mean_loss)
        logger.info(
            "epoch %d done: mean_loss=%.4f, took %.1fs (%d steps)",
            self.epoch, mean_loss, time.perf_counter() - t0, n_steps,
        )
        return mean_loss

    def fit(self, epochs: int) -> Phase2TrainResult:
        for self.epoch in range(self.epoch, self.epoch + epochs):
            self.train_one_epoch()
        return Phase2TrainResult(
            epochs_completed=epochs,
            last_step=self.global_step,
            last_loss=self.last_loss,
            epoch_losses=list(self.epoch_losses),
        )


# ---------------------------------------------------------------------------
# Param-EMA helper (no buffer copying; lighter than BaseTrainer's EMA)
# ---------------------------------------------------------------------------


class _ParamEMA:
    """Exponential moving average of trainable parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            buf = self.shadow.get(n)
            if buf is None:
                self.shadow[n] = p.detach().clone()
                continue
            buf.mul_(d).add_(p.detach(), alpha=1 - d)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {n: t.clone() for n, t in self.shadow.items()}


# ---------------------------------------------------------------------------
# Tiny helper to compose a default trainer from a Phase 2 model + COCO root
# ---------------------------------------------------------------------------


def build_phase2_trainer_from_config(
    config: Any,
    model: nn.Module,
    text_encode_fn: Optional[Callable[[List[str], torch.device], torch.Tensor]] = None,
) -> Phase2Trainer:
    """Convenience builder: wires dataset + collator + cache + criterion.

    The cache is built from ``config.text_emb_cache_path`` if it exists, else
    rebuilt via ``text_encode_fn`` (typically the model's
    ``backbone.encode_text``) over the dataset's full class-name list.

    Returns a ready-to-call ``Phase2Trainer`` with the loader and criterion
    already wired.
    """
    # Lazy imports so unit tests that don't need pycocotools are unaffected.
    from .data import (
        OpenVocabCOCODataset,
        OpenVocabCollator,
        TextEmbeddingCache,
    )
    from .matcher import OpenVocabHungarianMatcher

    if config.data_dir is None:
        raise ValueError("config.data_dir must be set for Phase 2 training")

    dataset = OpenVocabCOCODataset(
        data_dir=config.data_dir,
        json_file=getattr(config, "ann_file", "instances_train2017.json"),
        name=getattr(config, "img_subdir", "train2017"),
        input_size=config.imgsz,
    )

    cache: TextEmbeddingCache
    if config.text_emb_cache_path and _path_exists(config.text_emb_cache_path):
        cache = TextEmbeddingCache.load(config.text_emb_cache_path)
    else:
        if text_encode_fn is None:
            raise ValueError(
                "text_encode_fn must be supplied when no on-disk cache is "
                "available; usually pass model.backbone.encode_text"
            )
        cache = TextEmbeddingCache.build(
            names=dataset.class_names,
            encode_text_fn=text_encode_fn,
            prompt_template=config.text_prompt_template,
        )
        if config.text_emb_cache_path:
            cache.save(config.text_emb_cache_path)

    collator = OpenVocabCollator(
        text_emb_cache=cache,
        full_class_names=dataset.class_names,
        num_negatives=config.num_negatives,
        rng_seed=config.seed,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch,
        collate_fn=collator,
        num_workers=config.workers,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    matcher = OpenVocabHungarianMatcher(
        cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
    )
    criterion = OpenVocabSetCriterion(
        matcher,
        weight_class=config.loss_weight_class,
        weight_bbox=config.loss_weight_bbox,
        weight_giou=config.loss_weight_giou,
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.lr0,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    return Phase2Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        optimizer=optimizer,
        device=config.device,
        clip_max_norm=config.clip_max_norm,
        amp=config.amp,
        ema_decay=config.ema_decay if config.ema else None,
    )


def _path_exists(p: str) -> bool:
    import os
    return bool(p) and os.path.isfile(p)
