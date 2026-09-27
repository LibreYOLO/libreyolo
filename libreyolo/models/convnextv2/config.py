"""Supervised classification defaults; independent of the FCMAE recipe."""

from dataclasses import dataclass

from ..convnext.config import ConvNeXtConfig


@dataclass(kw_only=True)
class ConvNeXtV2Config(ConvNeXtConfig):
    size: str = "atto"
    lr0: float = 1e-4
