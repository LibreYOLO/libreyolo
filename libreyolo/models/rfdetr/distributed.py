"""Small distributed helpers used by the RF-DETR criterion."""

from __future__ import annotations

import torch


def is_dist_avail_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()
