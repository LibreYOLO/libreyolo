"""RTDETRv4Trainer — fine-tuning trainer for RT-DETRv4 student detectors."""

from __future__ import annotations

from typing import Type

from ...training.config import RTDETRv4Config, TrainConfig
from ..deim.trainer import DEIMTrainer


class RTDETRv4Trainer(DEIMTrainer):
    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return RTDETRv4Config

    def get_model_family(self) -> str:
        return "rtdetrv4"

    def get_model_tag(self) -> str:
        return f"RTDETRv4-{self.config.size}"
