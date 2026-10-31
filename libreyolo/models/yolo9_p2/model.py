"""LibreYOLO9P2 inference and training wrapper.

YOLOv9 small-object (P2) variant. Shares the backbone with standard YOLOv9
but surfaces its stride-4 feature map, extends the PAN neck one level down,
and detects on four scales with strides (4, 8, 16, 32) instead of three.

Base YOLOv9 detect checkpoints transfer almost entirely: the backbone and
the existing neck modules load as-is, and the three pretrained head towers
load after an index shift (checkpoint towers 0/1/2 map to P3/P4/P5 slots
1/2/3, leaving the new P2 tower at index 0 freshly initialized).

Color space: RGB 0-1 (same as standard YOLOv9).
Sizes: t / s.
"""

import re
from typing import Dict

import torch.nn as nn

from ..yolo9.model import LibreYOLO9
from .config import YOLO9P2Config
from .nn import LibreYOLO9P2Model

_HEAD_TOWER_RE = re.compile(r"^(head\.(?:anchor_convs|class_convs))\.(\d+)\.(.*)$")
_CLASS_TOWER_HIDDEN_RE = re.compile(r"^head\.class_convs\.\d+\.0\.conv\.weight$")


class LibreYOLO9P2(LibreYOLO9):
    """YOLOv9 model with an extra stride-4 detection scale for small objects.

    Args:
        model_path: Path to weights, pre-loaded state_dict, or None.
        size: Model size variant ("t", "s").
        reg_max: Regression max for DFL (default: 16).
        nb_classes: Number of classes (default: 80).
        device: Device for inference.

    Example::

        >>> model = LibreYOLO9P2("LibreYOLO9P2s-visdrone.pt", size="s")
        >>> detections = model(image_path, save=True)
    """

    FAMILY = "yolo9_p2"
    FILENAME_PREFIX = "LibreYOLO9P2"
    INPUT_SIZES = {"t": 640, "s": 640}
    SUPPORTED_TASKS = ("detect",)
    TASK_INPUT_SIZES = {"detect": INPUT_SIZES}
    TRAIN_CONFIG = YOLO9P2Config
    # Base yolo9 detect checkpoints are valid transfer sources (their tensors
    # are remapped by _prepare_state_dict).
    TRANSFER_COMPATIBLE_FAMILIES = ("yolo9",)
    # Published VisDrone research-preview checkpoint (10 aerial classes).
    WEIGHT_VARIANTS = ("visdrone",)

    # =====================================================================
    # Registry classmethods
    # =====================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Match checkpoints that contain the P2 neck extension keys.

        ``neck.elan_up3`` / ``neck.elan_down0`` exist only in this family.
        Registered before LibreYOLO9 in the registry; LibreYOLO9.can_load
        also explicitly rejects these keys so routing stays unambiguous.
        """
        return any(
            key.startswith("neck.elan_up3") or key.startswith("neck.elan_down0")
            for key in weights_dict
        )

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> dict | None:
        """Decline upstream layouts: no upstream P2 variant exists.

        Without this override the parent's numbered-layout converter would
        claim upstream YOLOv9 checkpoints for this family (it is registered
        first) and misattribute them to yolo9_p2.
        """
        return None

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> str | None:
        if cls.detect_variant_from_filename(filename) == "visdrone":
            return (
                f"{filename} is a RESEARCH PREVIEW trained on VisDrone2019-DET "
                "(AISKYEYE, Tianjin University). VisDrone is licensed "
                "CC BY-NC-SA 3.0: these weights are for NON-COMMERCIAL use "
                "only and are not covered by LibreYOLO's permissive license. "
                "The checkpoint detects the 10 VisDrone aerial classes (not "
                "COCO) and was trained/evaluated at imgsz=768."
            )
        return super().get_download_notice(filename, url)

    # =====================================================================
    # Model lifecycle
    # =====================================================================

    def _init_model(self) -> nn.Module:
        return LibreYOLO9P2Model(
            config=self.size,
            reg_max=self.reg_max,
            nb_classes=self.nb_classes,
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        layers = super()._get_available_layers()
        layers.update(
            {
                "neck_elan_up3": self.model.neck.elan_up3,
                "neck_elan_down0": self.model.neck.elan_down0,
            }
        )
        return layers

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        """Remap 3-scale YOLOv9 checkpoints onto the 4-scale layout.

        Keys are first upgraded to the current layout (legacy ``cv2``/``cv3``
        towers become ``anchor_convs``/``class_convs``). Base checkpoints (no
        P2 neck keys) then get their head tower indices shifted by +1 so the
        pretrained P3/P4/P5 towers land in slots 1/2/3 and the stride-4 tower
        at index 0 stays freshly initialized. P2 checkpoints pass through
        unchanged.
        """
        state_dict = super()._prepare_state_dict(state_dict)
        if any(key.startswith("neck.elan_up3") for key in state_dict):
            return state_dict

        remapped = {}
        for key, value in state_dict.items():
            match = _HEAD_TOWER_RE.match(key)
            if match:
                branch, index, rest = match.groups()
                key = f"{branch}.{int(index) + 1}.{rest}"
            remapped[key] = value
        return remapped

    def _align_class_towers_for_transfer(self, state_dict: dict) -> None:
        """Match COCO-width class towers before partial transfer loading.

        Same as the parent, but the checkpoint hidden width is read from the
        first class tower present in the (possibly index-shifted) state dict
        rather than the fixed index 0.
        """
        key = next(
            (k for k in state_dict if _CLASS_TOWER_HIDDEN_RE.match(k)), None
        )
        if key is None:
            return

        self._rebuild_class_towers(int(state_dict[key].shape[0]))

    # =====================================================================
    # Training
    # =====================================================================

    def _trainer_class(self):
        from .trainer import YOLO9P2Trainer

        return YOLO9P2Trainer
