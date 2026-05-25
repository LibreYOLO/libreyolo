"""LibreRTDETRv4 — RT-DETRv4 student detectors."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from libreyolo.training.ddp_spawn import ddp_aware

from ...training.config import RTDETRv4Config
from ..dfine.model import LibreDFINE
from ..dfine.nn import LibreDFINEModel


_TRAIN_DEFAULTS = RTDETRv4Config()


class LibreRTDETRv4(LibreDFINE):
    FAMILY = "rtdetrv4"
    FILENAME_PREFIX = "LibreRTDETRv4"
    INPUT_SIZES = {"s": 640, "m": 640, "l": 640, "x": 640}
    TRAIN_CONFIG = RTDETRv4Config

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # Disambiguates from D-FINE via the factory's ``model_family`` metadata
        # gate; raw upstream ckpts still carry ``feature_projector`` keys.
        return LibreDFINE.can_load(weights_dict) or any(
            "feature_projector" in k for k in weights_dict
        )

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        detected = super().detect_size_from_filename(filename)
        if detected is not None:
            return detected
        m = re.search(r"rtv4(?:_hgnetv2)?_([smlx])(?:_|\.|$)", filename.lower())
        if m:
            return m.group(1)
        return None

    def _init_model(self) -> nn.Module:
        return LibreDFINEModel(
            config=self.size,
            nb_classes=self.nb_classes,
            eval_spatial_size=(self.input_size, self.input_size),
            activation="silu",
        )

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
        epochs: int = _TRAIN_DEFAULTS.epochs,
        batch: int = 16,
        imgsz: int = 640,
        lr0: float = _TRAIN_DEFAULTS.lr0,
        device: str = "",
        workers: int = 4,
        seed: int = 0,
        project: str = "runs/train",
        name: str = _TRAIN_DEFAULTS.name,
        exist_ok: bool = False,
        resume: bool = False,
        amp: bool = False,
        patience: int = 50,
        **kwargs,
    ) -> dict:
        from libreyolo.data import load_data_config

        from .trainer import RTDETRv4Trainer

        try:
            data_config = load_data_config(data, autodownload=True)
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        yaml_nc = data_config.get("nc")
        yaml_names = data_config.get("names")
        if yaml_nc is not None and yaml_nc != self.nb_classes:
            self._rebuild_for_new_classes(yaml_nc)
        if yaml_names is not None:
            if isinstance(yaml_names, list):
                yaml_names = {i: n for i, n in enumerate(yaml_names)}
            self.names = self._sanitize_names(yaml_names, self.nb_classes)

        if seed >= 0:
            import random

            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trainer = RTDETRv4Trainer(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=self.nb_classes,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            device=device if device else "auto",
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            amp=amp,
            patience=patience,
            **kwargs,
        )

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreRTDETRv4('path/to/last.pt'); model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))
            return trainer.train()

        results = trainer.train()

        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self.model_path = best_ckpt
            self._load_weights(best_ckpt)

        self.model.to(self.device)

        return results
