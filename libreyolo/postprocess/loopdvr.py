"""LoopDVR postprocessing (detect / segment / pose) — DETR-style top-K, no NMS.

Standalone module under ``libreyolo.postprocess`` so backends and models share
the same decoding contract without importing ``libreyolo.models``.
"""

from __future__ import annotations

from .ec import (
    postprocess,
    postprocess_pose,
    postprocess_seg,
)

__all__ = [
    "postprocess",
    "postprocess_pose",
    "postprocess_seg",
]
