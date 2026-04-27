"""Base class for LibreYOLO inference backends."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..models.yolo9.utils import preprocess_image
from ..models.yolonas.utils import preprocess_image as yolonas_preprocess_image
from ..models.yolox.utils import preprocess_image as yolox_preprocess_image
from ..utils.drawing import draw_boxes, draw_masks
from ..utils.general import COCO_CLASSES, get_safe_stem
from ..utils.image_loader import ImageLoader
from ..utils.results import Boxes, Masks, Results
from ..utils.video import collect_video_results, is_video_file, run_video_inference

logger = logging.getLogger(__name__)


def _nms_numpy(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45
) -> list:
    """Numpy-based Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return keep


def _is_nms_free_family(model_family: Optional[str]) -> bool:
    """Whether backend outputs should bypass generic NMS.

    DETR-style families already emit a ranked set prediction after top-k
    selection. Applying YOLO-style IoU suppression on top of that can remove
    valid detections and make exported runtimes diverge from native PyTorch.
    """
    return model_family in {"dfine", "rfdetr", "rtdetr"}


class BaseBackend(ABC):
    """Abstract base class for all inference backends.

    Subclasses must:
    1. Implement ``__init__`` to load the runtime-specific model, then call
       ``super().__init__(...)`` with the resolved common attributes.
    2. Implement ``_run_inference`` to execute the model and return raw outputs.
    """

    def __init__(
        self,
        *,
        model_path: str,
        nb_classes: int,
        device: str,
        imgsz: int,
        model_family: Optional[str],
        names: Dict[int, str],
    ):
        self.model_path = model_path
        self.nb_classes = nb_classes
        self.device = device
        self.imgsz = imgsz
        self.model_family = model_family
        self.names = names

    # =========================================================================
    # Abstract interface
    # =========================================================================

    @abstractmethod
    def _run_inference(self, blob: np.ndarray) -> list:
        """Run backend-specific inference.

        Args:
            blob: Preprocessed input array of shape ``(1, C, H, W)``.

        Returns:
            List of numpy arrays, one per model output tensor.
        """

    # =========================================================================
    # Preprocessing
    # =========================================================================

    def _preprocess(self, image, effective_imgsz, color_format):
        """Dispatch to model-family-specific preprocessing.

        Returns:
            Tuple of (input_tensor, original_img, original_size, ratio).
        """
        if self.model_family == "yolox":
            return yolox_preprocess_image(
                image, input_size=effective_imgsz, color_format=color_format
            )
        elif self.model_family == "yolonas":
            return yolonas_preprocess_image(
                image, input_size=effective_imgsz, color_format=color_format
            )
        elif self.model_family == "rfdetr":
            tensor, img, size = self._preprocess_rfdetr(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, 1.0
        elif self.model_family == "dfine":
            tensor, img, size = self._preprocess_dfine(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, 1.0
        elif self.model_family == "rtdetr":
            tensor, img, size = self._preprocess_rtdetr(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, 1.0
        else:
            tensor, img, size = preprocess_image(
                image, input_size=effective_imgsz, color_format=color_format
            )
            return tensor, img, size, 1.0

    @staticmethod
    def _preprocess_rfdetr(image, input_size, color_format):
        """RF-DETR preprocessing: direct resize + ImageNet normalization."""
        from ..models.rfdetr.utils import preprocess_numpy as rfdetr_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size  # (W, H)
        original_img = img.copy()

        img_chw, _ = rfdetr_preprocess_numpy(np.array(img), input_size)
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_dfine(image, input_size, color_format):
        """D-FINE preprocessing: plain resize + RGB + /255, no ImageNet norm."""
        from ..models.dfine.utils import preprocess_numpy as dfine_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()

        img_chw, _ = dfine_preprocess_numpy(np.array(img), input_size)
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0)

        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_rtdetr(image, input_size, color_format):
        """RT-DETR preprocessing: direct resize + normalize to [0,1]."""
        from ..models.rtdetr.utils import preprocess_numpy as rtdetr_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size  # (W, H)
        original_img = img.copy()

        img_chw, _ = rtdetr_preprocess_numpy(np.array(img), input_size)
        img_tensor = torch.from_numpy(img_chw)
        return img_tensor, original_img, original_size

    # =========================================================================
    # Output parsing
    # =========================================================================

    def _parse_outputs(
        self,
        all_outputs: list,
        effective_imgsz: int,
        original_size: tuple,
        conf: float,
        ratio: float = 1.0,
    ):
        """Parse raw outputs into (boxes_xyxy, scores, class_ids, masks_or_None)."""
        orig_w, orig_h = original_size

        if self.model_family == "yolox":
            boxes, scores, cls = self._parse_yolox(
                all_outputs, effective_imgsz, orig_w, orig_h, conf, ratio
            )
            return boxes, scores, cls, None
        elif self.model_family == "yolonas":
            boxes, scores, cls = self._parse_yolonas(
                all_outputs, orig_w, orig_h, conf, ratio=ratio
            )
            return boxes, scores, cls, None
        elif self.model_family == "rfdetr":
            return self._parse_rfdetr(all_outputs, orig_w, orig_h, conf)
        elif self.model_family == "dfine":
            boxes, scores, cls = self._parse_dfine(
                all_outputs, orig_w, orig_h, conf
            )
            return boxes, scores, cls, None
        elif self.model_family == "rtdetr":
            boxes, scores, cls = self._parse_rtdetr(all_outputs, orig_w, orig_h, conf)
            return boxes, scores, cls, None
        else:
            boxes, scores, cls = self._parse_yolo9(
                all_outputs, effective_imgsz, orig_w, orig_h, conf
            )
            return boxes, scores, cls, None

    def _parse_yolox(
        self, all_outputs, effective_imgsz, orig_w, orig_h, conf, ratio=1.0
    ):
        """Parse YOLOX output: (B, N, 5+nc) — cxcywh + objectness + class_scores."""
        outputs = all_outputs[0][0]  # (N, 5+nc)

        cx, cy, w, h = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]
        objectness = outputs[:, 4]
        class_scores = outputs[:, 5:]

        max_class_scores = np.max(class_scores, axis=1)
        max_scores = objectness * max_class_scores
        class_ids = np.argmax(class_scores, axis=1)

        mask = max_scores > conf
        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        max_scores, class_ids = max_scores[mask], class_ids[mask]

        if len(max_scores) == 0:
            return np.empty((0, 4)), max_scores, class_ids

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        boxes /= ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        return boxes, max_scores, class_ids

    def _parse_yolo9(self, all_outputs, effective_imgsz, orig_w, orig_h, conf):
        """Parse YOLO9 output: (B, 4+nc, N) — xyxy + class_scores."""
        outputs = all_outputs[0][0].T  # (N, 4+nc)

        boxes = outputs[:, :4]
        scores = outputs[:, 4:]

        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        mask = max_scores > conf
        boxes, max_scores, class_ids = boxes[mask], max_scores[mask], class_ids[mask]

        if len(boxes) == 0:
            return boxes, max_scores, class_ids

        scale_x = orig_w / effective_imgsz
        scale_y = orig_h / effective_imgsz
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        return boxes, max_scores, class_ids

    def _parse_yolonas(self, all_outputs, orig_w, orig_h, conf, ratio=1.0):
        """Parse YOLO-NAS output: [boxes(B,N,4), scores(B,N,nc)] in input pixels."""
        first = all_outputs[0][0]
        second = all_outputs[1][0]
        if first.shape[-1] == 4 and second.shape[-1] != 4:
            boxes = first
            scores = second
        elif second.shape[-1] == 4 and first.shape[-1] != 4:
            boxes = second
            scores = first
        else:
            boxes = first
            scores = second

        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        mask = max_scores > conf
        boxes, max_scores, class_ids = boxes[mask], max_scores[mask], class_ids[mask]

        if len(boxes) == 0:
            return boxes, max_scores, class_ids

        boxes = boxes.astype(np.float32, copy=True)
        boxes /= ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        return boxes, max_scores, class_ids

    def _parse_dfine(self, all_outputs, orig_w, orig_h, conf, max_det: int = 300):
        """Parse D-FINE outputs: pred_logits (B, Q, nc) + pred_boxes (B, Q, 4) cxcywh [0,1].

        Matches the upstream DFINEPostProcessor (use_focal_loss=True): sigmoid →
        topk over (queries × classes) flattened → labels = topk_idx % nc, query_idx
        = topk_idx // nc. No NMS (DETR set-prediction).
        """
        pred_logits = all_outputs[0][0]  # (Q, nc)
        pred_boxes = all_outputs[1][0]  # (Q, 4)

        Q, nc = pred_logits.shape
        prob = 1.0 / (1.0 + np.exp(-pred_logits.astype(np.float64)))
        prob = prob.astype(np.float32)

        flat = prob.reshape(-1)  # (Q * nc,)
        k = min(max_det, flat.size)
        # Top-k via argpartition (faster than full sort).
        idx = np.argpartition(-flat, k - 1)[:k]
        idx = idx[np.argsort(-flat[idx])]

        scores = flat[idx]
        query_idx = idx // nc
        class_ids = idx % nc

        # cxcywh -> xyxy in [0,1], then gather + scale.
        cx, cy, w, h = (
            pred_boxes[:, 0],
            pred_boxes[:, 1],
            pred_boxes[:, 2],
            pred_boxes[:, 3],
        )
        boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        boxes = boxes_xyxy[query_idx]

        boxes[:, [0, 2]] *= orig_w
        boxes[:, [1, 3]] *= orig_h
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        mask = scores > conf
        return boxes[mask], scores[mask], class_ids[mask].astype(np.int64)

    def _parse_rfdetr(self, all_outputs, orig_w, orig_h, conf):
        """Parse RF-DETR output: boxes (B,300,4) cxcywh [0,1] + logits (B,300,nc).

        For segmentation models a third output is present:
        masks (B,300,Hm,Wm) raw mask logits at model resolution.
        """
        boxes_raw = all_outputs[0][0]  # (300, 4) normalized cxcywh
        logits = all_outputs[1][0]  # (300, nc) raw logits
        raw_masks = all_outputs[2][0] if len(all_outputs) >= 3 else None

        scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64))).astype(np.float32)

        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        mask = max_scores > conf
        boxes_raw = boxes_raw[mask]
        max_scores, class_ids = max_scores[mask], class_ids[mask]
        if raw_masks is not None:
            raw_masks = raw_masks[mask]

        if len(boxes_raw) == 0:
            return boxes_raw, max_scores, class_ids, None

        # COCO 91→80 class mapping
        if logits.shape[1] == 91 and self.nb_classes == 80:
            from ..models.rfdetr.model import _COCO91_TO_COCO80

            mapped = np.array([_COCO91_TO_COCO80.get(int(c), -1) for c in class_ids])
            valid = mapped >= 0
            boxes_raw = boxes_raw[valid]
            max_scores = max_scores[valid]
            class_ids = mapped[valid]
            if raw_masks is not None:
                raw_masks = raw_masks[valid]

        if len(boxes_raw) == 0:
            return boxes_raw, max_scores, class_ids, None

        cx, cy, w, h = (
            boxes_raw[:, 0],
            boxes_raw[:, 1],
            boxes_raw[:, 2],
            boxes_raw[:, 3],
        )
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        # Resize and threshold masks to original image resolution
        masks_out = None
        if raw_masks is not None and len(raw_masks) > 0:
            masks_t = torch.from_numpy(raw_masks).unsqueeze(1).float()
            masks_t = F.interpolate(
                masks_t,
                size=(int(orig_h), int(orig_w)),
                mode="bilinear",
                align_corners=False,
            )
            masks_out = (masks_t[:, 0] > 0.0).numpy()  # (N, H, W)

        return boxes, max_scores, class_ids, masks_out

    def _parse_rtdetr(self, all_outputs, orig_w, orig_h, conf):
        """Parse RT-DETR output: pred_boxes (B,Q,4) cxcywh [0,1] + pred_logits (B,Q,C).

        RTDETR outputs are already in the correct class indices (no COCO 91->80 mapping needed).
        """
        # all_outputs order depends on ONNX output naming; try both orderings
        first = all_outputs[0][0]  # (Q, 4) or (Q, C)
        second = all_outputs[1][0]  # (Q, C) or (Q, 4)

        # Detect which is boxes and which is logits by shape
        if first.shape[1] == 4 and len(second.shape) == 2 and second.shape[1] != 4:
            boxes_raw = first  # (Q, 4) normalized cxcywh
            logits = second  # (Q, C) raw logits
        elif second.shape[1] == 4 and len(first.shape) == 2 and first.shape[1] != 4:
            boxes_raw = second
            logits = first
        else:
            # Fallback: assume pred_logits has more columns (num_classes typically > 4)
            if first.shape[1] > second.shape[1]:
                logits = first
                boxes_raw = second
            else:
                logits = second
                boxes_raw = first

        scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64))).astype(np.float32)

        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        mask = max_scores > conf
        boxes_raw = boxes_raw[mask]
        max_scores, class_ids = max_scores[mask], class_ids[mask]

        if len(boxes_raw) == 0:
            return boxes_raw, max_scores, class_ids

        cx, cy, w, h = (
            boxes_raw[:, 0],
            boxes_raw[:, 1],
            boxes_raw[:, 2],
            boxes_raw[:, 3],
        )
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        return boxes, max_scores, class_ids

    # =========================================================================
    # Result building
    # =========================================================================

    def _build_result(
        self,
        boxes: np.ndarray,
        max_scores: np.ndarray,
        class_ids: np.ndarray,
        *,
        masks: "np.ndarray | None" = None,
        orig_shape: Tuple[int, int],
        image_path,
        iou: float,
        classes: Optional[List[int]],
        max_det: int,
    ) -> Results:
        """Apply family-appropriate suppression/max_det/filtering and wrap."""
        if len(boxes) == 0:
            return Results(
                boxes=Boxes(
                    torch.zeros((0, 4), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.float32),
                ),
                orig_shape=orig_shape,
                path=str(image_path) if image_path else None,
                names=self.names,
            )

        if not _is_nms_free_family(self.model_family):
            keep = _nms_numpy(boxes, max_scores, iou)
            boxes, max_scores, class_ids = (
                boxes[keep],
                max_scores[keep],
                class_ids[keep],
            )
            if masks is not None:
                masks = masks[keep]

        if len(boxes) > max_det:
            top_indices = np.argsort(max_scores)[::-1][:max_det]
            boxes = boxes[top_indices]
            max_scores = max_scores[top_indices]
            class_ids = class_ids[top_indices]
            if masks is not None:
                masks = masks[top_indices]

        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        conf_t = torch.tensor(max_scores, dtype=torch.float32)
        cls_t = torch.tensor(class_ids, dtype=torch.float32)

        if classes is not None and len(boxes_t) > 0:
            cls_mask = torch.zeros(len(cls_t), dtype=torch.bool)
            for cid in classes:
                cls_mask |= cls_t == cid
            boxes_t = boxes_t[cls_mask]
            conf_t = conf_t[cls_mask]
            cls_t = cls_t[cls_mask]
            if masks is not None:
                masks = masks[cls_mask.numpy()]

        masks_obj = None
        if masks is not None and len(masks) > 0:
            masks_obj = Masks(torch.from_numpy(masks).bool(), orig_shape=orig_shape)

        return Results(
            boxes=Boxes(boxes_t, conf_t, cls_t),
            masks=masks_obj,
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
        )

    # =========================================================================
    # Save
    # =========================================================================

    def _save_annotated(self, result, original_img, image_path, output_path):
        """Save annotated image to disk."""
        annotated_img = original_img
        if len(result) > 0:
            if result.masks is not None:
                annotated_img = draw_masks(
                    annotated_img,
                    result.masks.data.numpy(),
                    result.boxes.cls.tolist(),
                )
            annotated_img = draw_boxes(
                annotated_img,
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
            )

        if output_path:
            final_path = Path(output_path)
            final_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            stem = get_safe_stem(image_path) if image_path else "inference"
            ext = Path(image_path).suffix if image_path else ".jpg"
            model_tag = Path(self.model_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_dir = Path("runs/detections")
            save_dir.mkdir(parents=True, exist_ok=True)
            final_path = save_dir / f"{stem}_{model_tag}_{timestamp}{ext}"

        annotated_img.save(final_path)
        result.saved_path = str(final_path)

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def build_names(nb_classes: int) -> Dict[int, str]:
        """Build a class names dict — COCO for 80 classes, generic otherwise."""
        if nb_classes == 80:
            return {i: n for i, n in enumerate(COCO_CLASSES)}
        return {i: f"class_{i}" for i in range(nb_classes)}

    # =========================================================================
    # Inference pipeline
    # =========================================================================

    def _predict_single(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        save: bool = False,
        output_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
    ) -> Results:
        """Run inference on a single image."""
        image_path = image if isinstance(image, (str, Path)) else None
        effective_imgsz = imgsz if imgsz is not None else self.imgsz

        input_tensor, original_img, original_size, ratio = self._preprocess(
            image, effective_imgsz, color_format
        )

        blob = input_tensor.numpy()

        all_outputs = self._run_inference(blob)

        boxes, max_scores, class_ids, masks = self._parse_outputs(
            all_outputs, effective_imgsz, original_size, conf, ratio=ratio
        )

        orig_w, orig_h = original_size
        orig_shape = (orig_h, orig_w)
        result = self._build_result(
            boxes,
            max_scores,
            class_ids,
            masks=masks,
            orig_shape=orig_shape,
            image_path=image_path,
            iou=iou,
            classes=classes,
            max_det=max_det,
        )

        if save:
            self._save_annotated(result, original_img, image_path, output_path)

        return result

    def _process_in_batches(
        self,
        image_paths: List[Path],
        batch: int = 1,
        save: bool = False,
        output_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
    ) -> List[Results]:
        """Process multiple images sequentially."""
        results = []
        for i in range(0, len(image_paths), batch):
            chunk = image_paths[i : i + batch]
            for path in chunk:
                results.append(
                    self._predict_single(
                        path,
                        save=save,
                        output_path=output_path,
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        classes=classes,
                        max_det=max_det,
                        color_format=color_format,
                    )
                )
        return results

    # =========================================================================
    # Public API
    # =========================================================================

    def __call__(
        self,
        source: Union[str, Path, Image.Image, np.ndarray, None] = None,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        save: bool = False,
        batch: int = 1,
        # video parameters
        stream: bool = False,
        vid_stride: int = 1,
        show: bool = False,
        output_path: str | None = None,
        color_format: str = "auto",
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        """Run inference on an image, directory, or video."""
        # Handle video input
        if is_video_file(source):
            gen = self._predict_video(
                source,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                save=save,
                show=show,
                vid_stride=vid_stride,
                output_path=output_path,
            )
            if stream:
                return gen
            return collect_video_results(gen, source, vid_stride)

        if isinstance(source, (str, Path)) and Path(source).is_dir():
            image_paths = ImageLoader.collect_images(source)
            if not image_paths:
                return []
            return self._process_in_batches(
                image_paths,
                batch=batch,
                save=save,
                output_path=output_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format=color_format,
            )

        return self._predict_single(
            source,
            save=save,
            output_path=output_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=classes,
            max_det=max_det,
            color_format=color_format,
        )

    def predict(
        self, *args, **kwargs
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        """Alias for __call__ method."""
        return self(*args, **kwargs)

    def _predict_video(
        self,
        source: Union[str, Path],
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        save: bool = False,
        show: bool = False,
        vid_stride: int = 1,
        output_path: Optional[str] = None,
    ) -> Generator[Results, None, None]:
        """Run inference on a video file, yielding per-frame Results."""
        effective_imgsz = imgsz if imgsz is not None else self.imgsz

        def predict_frame(pil_img):
            input_tensor, original_img, original_size, ratio = self._preprocess(
                pil_img, effective_imgsz, "rgb"
            )
            blob = input_tensor.numpy()
            all_outputs = self._run_inference(blob)
            boxes, max_scores, class_ids, masks = self._parse_outputs(
                all_outputs, effective_imgsz, original_size, conf, ratio=ratio
            )
            orig_w, orig_h = original_size
            return self._build_result(
                boxes,
                max_scores,
                class_ids,
                masks=masks,
                orig_shape=(orig_h, orig_w),
                image_path=str(source),
                iou=iou,
                classes=classes,
                max_det=max_det,
            )

        yield from run_video_inference(
            source,
            predict_frame,
            vid_stride=vid_stride,
            save=save,
            show=show,
            output_path=output_path,
        )
