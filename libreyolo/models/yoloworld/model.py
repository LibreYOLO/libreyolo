"""LibreYOLOWorld — user-facing wrapper (real YOLOv8-CSPDarknet + RepVL-PAN + BNContrastive)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from .nn import LibreYOLOWorldModel, CLIP_EMBED_DIM


class LibreYOLOWorld(BaseModel):
    """Open-vocabulary YOLO. Text-prompted detection.

    Architecture: YOLOv8-CSPDarknet backbone + frozen CLIP ViT-B/32 text encoder
    + RepVL-PAN neck (MaxSigmoid fusion) + BNContrastive head. Structurally
    compatible with `wondervictor/YOLO-World-V2.1` checkpoints.

    Args:
        model_path: Path to weights. None = random-init (for smoke tests).
        size: YOLOv8 size "s"/"m"/"l"/"x" (default "l" — matches Tencent's default).
        prompts: Initial list of class-name strings.
        imgsz: Input image size (default 640, must be divisible by 32).
        device: Inference device.

    Example::

        >>> model = LibreYOLOWorld(prompts=["person", "dog", "traffic cone"])
        >>> result = model("photo.jpg")
        >>> # result.boxes.cls are indices into the prompts list
    """

    FAMILY = "yoloworld"
    FILENAME_PREFIX = "LibreYOLOWorld"
    INPUT_SIZES = {"s": 640, "m": 640, "l": 640, "x": 640}
    WEIGHT_EXT = ".pt"

    # -----------------------------------------------------------------
    # Registry classmethods
    # -----------------------------------------------------------------

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return any(
            k.startswith("backbone.stem") or k.startswith("backbone.stage1")
            or k.startswith("neck.top_down_layer") or k.startswith("head.cls_contrasts")
            for k in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        # Look at backbone.stem.conv.weight (c1=3, c2 varies with size)
        key = "backbone.stem.conv.weight"
        if key not in weights_dict:
            return None
        out_ch = weights_dict[key].shape[0]
        # stem channels = _make_divisible(64 * widen). s=32, m=48, l=64, x=80
        for size, ch in (("s", 32), ("m", 48), ("l", 64), ("x", 80)):
            if out_ch == ch:
                return size
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # Open-vocab: class count is driven by runtime prompts.
        return None

    # -----------------------------------------------------------------
    # Init
    # -----------------------------------------------------------------

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        size: str = "l",
        prompts: Optional[List[str]] = None,
        imgsz: int = 640,
        reg_max: int = 16,
        device: str = "auto",
        **kwargs,
    ):
        self._imgsz = imgsz
        self._reg_max = reg_max
        self._size_override = size
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=len(prompts) if prompts else 1,
            device=device,
            **kwargs,
        )
        if model_path is not None and isinstance(model_path, str) and Path(model_path).exists():
            self._load_weights(model_path)
        self.set_prompts(prompts or ["object"])

    # -----------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------

    def _init_model(self) -> nn.Module:
        return LibreYOLOWorldModel(size=self._size_override, imgsz=self._imgsz, reg_max=self._reg_max)

    def _get_available_layers(self):
        return {
            "backbone": self.model.backbone,
            "text_encoder": self.model.text_encoder,
            "neck": self.model.neck,
            "head": self.model.head,
        }

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def set_prompts(self, prompts: Sequence[str]) -> None:
        self.model.set_prompts(prompts)
        self.nb_classes = len(prompts)

    @property
    def prompts(self) -> List[str]:
        return self.model.prompts

    # -----------------------------------------------------------------
    # Inference plumbing
    # -----------------------------------------------------------------

    @staticmethod
    def _get_preprocess_numpy():
        import numpy as np

        def _preprocess_numpy(arr: "np.ndarray", input_size: int = 640):
            import cv2
            img = cv2.resize(arr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
            img = img[:, :, ::-1]
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            return np.ascontiguousarray(img), 1.0

        return _preprocess_numpy

    def _preprocess(self, image, color_format: str = "auto", input_size: Optional[int] = None):
        from PIL import Image
        import numpy as np

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        assert isinstance(image, Image.Image)
        size = self._imgsz if input_size is None else input_size
        img = image.resize((size, size))
        arr = np.asarray(img).astype("float32") / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor, image, image.size, 1.0

    def _forward(self, input_tensor: torch.Tensor):
        return self.model(input_tensor)

    def _postprocess(
        self,
        output,
        conf_thres: float,
        iou_thres: float,
        original_size,
        max_det: int = 300,
        **kwargs,
    ):
        """Decode DFL bbox_dists + contrastive logits to per-image detections."""
        bbox_dists = output["bbox_dists"]  # list of 3: (B, 4*reg_max, H, W)
        cls_logits = output["cls_logits"]  # list of 3: (B, N_cls, H, W)
        strides = output["strides"]
        B = bbox_dists[0].shape[0]
        assert B == 1, "MVP postprocess assumes batch=1"

        reg_max = self._reg_max
        all_boxes: List[torch.Tensor] = []
        all_conf: List[torch.Tensor] = []
        all_cls: List[torch.Tensor] = []

        for dist, logits, stride in zip(bbox_dists, cls_logits, strides):
            _, _, H, W = dist.shape
            N = logits.shape[1]

            # DFL: softmax over reg_max bins, expected value via integration
            d = dist[0].reshape(4, reg_max, H, W)  # (4, reg_max, H, W)
            d = F.softmax(d, dim=1)
            bins = torch.arange(reg_max, dtype=d.dtype, device=d.device).view(1, reg_max, 1, 1)
            ltrb = (d * bins).sum(dim=1)  # (4, H, W) — distances in grid units

            # Grid cells (cx, cy) in input-image coordinates
            ys, xs = torch.meshgrid(
                torch.arange(H, dtype=d.dtype, device=d.device),
                torch.arange(W, dtype=d.dtype, device=d.device),
                indexing="ij",
            )
            cx = (xs + 0.5) * stride
            cy = (ys + 0.5) * stride

            # Convert ltrb distances (in grid units) to absolute xyxy
            l, t, r, b = ltrb[0], ltrb[1], ltrb[2], ltrb[3]
            x1 = cx - l * stride
            y1 = cy - t * stride
            x2 = cx + r * stride
            y2 = cy + b * stride

            # Class scores: sigmoid of contrastive logits
            scores = torch.sigmoid(logits[0])  # (N, H, W)

            # Per-location max over classes
            max_scores, max_cls = scores.max(dim=0)  # (H, W)
            mask = max_scores > conf_thres
            if mask.any():
                boxes = torch.stack(
                    [x1[mask], y1[mask], x2[mask], y2[mask]], dim=-1
                )  # (K, 4)
                all_boxes.append(boxes)
                all_conf.append(max_scores[mask])
                all_cls.append(max_cls[mask].float())

        if not all_boxes:
            return {"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)),
                    "classes": torch.zeros((0,)), "num_detections": 0}

        boxes_t = torch.cat(all_boxes, dim=0)
        conf_t = torch.cat(all_conf, dim=0)
        cls_t = torch.cat(all_cls, dim=0)

        # Simple class-agnostic NMS
        keep = _nms(boxes_t, conf_t, iou_thres)[:max_det]
        boxes_t = boxes_t[keep]
        conf_t = conf_t[keep]
        cls_t = cls_t[keep]

        # Rescale from imgsz coords to original image size
        ow, oh = original_size
        scale_x = ow / self._imgsz
        scale_y = oh / self._imgsz
        boxes_t = boxes_t.clone()
        boxes_t[:, [0, 2]] *= scale_x
        boxes_t[:, [1, 3]] *= scale_y

        return {
            "boxes": boxes_t.detach().cpu().float(),
            "scores": conf_t.detach().cpu().float(),
            "classes": cls_t.detach().cpu().float(),
            "num_detections": int(conf_t.numel()),
        }


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    """Class-agnostic NMS. Uses torchvision if available, else a simple fallback."""
    try:
        from torchvision.ops import nms
        return nms(boxes, scores, iou_thres)
    except Exception:
        # Greedy NMS fallback
        order = scores.argsort(descending=True)
        keep: List[int] = []
        while order.numel() > 0:
            i = int(order[0])
            keep.append(i)
            if order.numel() == 1:
                break
            xx1 = boxes[order[1:], 0].clamp(min=float(boxes[i, 0]))
            yy1 = boxes[order[1:], 1].clamp(min=float(boxes[i, 1]))
            xx2 = boxes[order[1:], 2].clamp(max=float(boxes[i, 2]))
            yy2 = boxes[order[1:], 3].clamp(max=float(boxes[i, 3]))
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            iou = inter / (area_i + area_o - inter + 1e-9)
            order = order[1:][iou <= iou_thres]
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
