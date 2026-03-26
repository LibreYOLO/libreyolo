"""
LibreYOLO Knowledge Distillation Module.

Provides model-agnostic MGD and CWD feature-based distillation for all
LibreYOLO architectures (YOLOv9, YOLOX, RF-DETR, RTDETR, and future models).

Quick start::

    from libreyolo.distillation import Distiller
    from libreyolo.distillation.configs import get_distill_config

    # Get configs for teacher and student
    t_cfg = get_distill_config("yolo9", "c")
    s_cfg = get_distill_config("yolo9", "t")

    # Create distiller
    distiller = Distiller(
        teacher_model=teacher.model,
        student_model=student.model,
        teacher_config=t_cfg,
        student_config=s_cfg,
        loss_type="mgd",         # or "cwd"
    )

    # In training loop:
    distiller.teacher_forward(images)
    outputs = student_model(images, targets)
    distill_loss = distiller.compute_loss()
    total_loss = outputs["total_loss"] + distill_loss
    # ... backward, step, etc.
    distiller.step()  # clear for next batch

Available loss types:
    - ``"mgd"``: Masked Generative Distillation (ECCV 2022)
    - ``"cwd"``: Channel-Wise Knowledge Distillation (ICCV 2021)
"""

from .distiller import Distiller
from .losses import MGDLoss, CWDLoss, DISTILL_LOSSES
from .hooks import FeatureHookManager
from .configs import get_distill_config, list_supported

__all__ = [
    "Distiller",
    "MGDLoss",
    "CWDLoss",
    "DISTILL_LOSSES",
    "FeatureHookManager",
    "get_distill_config",
    "list_supported",
]
