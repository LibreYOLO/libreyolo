"""
Inference runner for LibreYOLO models.

Encapsulates all inference-related logic: single-image prediction,
tiled inference, batch processing, and result wrapping.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

from ...utils.drawing import draw_boxes, draw_masks, draw_tile_grid
from ...utils.general import get_safe_stem, get_slice_bboxes, nms, resolve_save_path
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.results import Boxes, Masks, Results

if TYPE_CHECKING:
    from .model import BaseModel


class InferenceRunner:
    """Handles all inference logic on behalf of a BaseModel instance."""

    def __init__(self, model: "BaseModel"):
        self.model = model

    def __call__(
        self,
        source: ImageInput | None = None,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        save: bool = False,
        batch: int = 1,
        # libreyolo-specific
        output_path: str | None = None,
        color_format: str = "auto",
        tiling: bool = False,
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None,
        **kwargs,
    ) -> Union[Results, List[Results]]:
        """
        Run inference on an image or directory.

        Args:
            source: Input image or directory path.
            conf: Confidence threshold.
            iou: IoU threshold for NMS.
            imgsz: Input size override (None = model default).
            classes: Filter to specific class IDs.
            max_det: Maximum detections per image.
            save: If True, saves annotated image.
            batch: Batch size for directory processing.
            output_path: Optional output path.
            color_format: Color format hint.
            tiling: Enable tiled inference for large images.
            overlap_ratio: Tile overlap ratio.
            output_file_format: Output format ("jpg", "png", "webp").
            **kwargs: Additional arguments for postprocessing.

        Returns:
            Results instance or list of Results.
        """
        if output_file_format is not None:
            output_file_format = output_file_format.lower().lstrip(".")
            if output_file_format not in ("jpg", "jpeg", "png", "webp"):
                raise ValueError(
                    f"Invalid output_file_format: {output_file_format}. "
                    "Must be one of: 'jpg', 'png', 'webp'"
                )

        # Handle directory input
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
                tiling=tiling,
                overlap_ratio=overlap_ratio,
                output_file_format=output_file_format,
                **kwargs,
            )

        # Use tiled inference if enabled
        if tiling:
            return self._predict_tiled(
                source,
                save=save,
                output_path=output_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format=color_format,
                overlap_ratio=overlap_ratio,
                output_file_format=output_file_format,
                **kwargs,
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
            output_file_format=output_file_format,
            **kwargs,
        )

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
        tiling: bool = False,
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None,
        **kwargs,
    ) -> List[Results]:
        """Process multiple images in batches."""
        results = []
        for i in range(0, len(image_paths), batch):
            chunk = image_paths[i : i + batch]
            for path in chunk:
                if tiling:
                    results.append(
                        self._predict_tiled(
                            path,
                            save=save,
                            output_path=output_path,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            classes=classes,
                            max_det=max_det,
                            color_format=color_format,
                            overlap_ratio=overlap_ratio,
                            output_file_format=output_file_format,
                            **kwargs,
                        )
                    )
                else:
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
                            output_file_format=output_file_format,
                            **kwargs,
                        )
                    )
        return results

    @staticmethod
    def _apply_classes_filter(
        boxes_t: torch.Tensor,
        conf_t: torch.Tensor,
        cls_t: torch.Tensor,
        classes: List[int],
        masks_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Filter detections to keep only the requested class IDs."""
        mask = torch.zeros(len(cls_t), dtype=torch.bool, device=cls_t.device)
        for cid in classes:
            mask |= cls_t == cid
        filtered_masks = masks_t[mask] if masks_t is not None else None
        return boxes_t[mask], conf_t[mask], cls_t[mask], filtered_masks

    def _wrap_results(
        self,
        detections: Dict,
        original_size: Tuple[int, int],
        image_path,
        classes: Optional[List[int]],
    ) -> Results:
        """Convert raw detection dict to a Results object.

        Args:
            detections: Dict with 'boxes', 'scores', 'classes', 'num_detections',
                and optionally 'masks'.
            original_size: (width, height) from preprocessing.
            image_path: Source path or None.
            classes: Optional class filter list.
        """
        masks_t = None

        if detections["num_detections"] == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            conf_t = torch.zeros((0,), dtype=torch.float32)
            cls_t = torch.zeros((0,), dtype=torch.float32)
        else:
            raw_boxes = detections["boxes"]
            if isinstance(raw_boxes, torch.Tensor):
                boxes_t = raw_boxes.float()
            else:
                boxes_t = torch.tensor(raw_boxes, dtype=torch.float32)

            raw_conf = detections["scores"]
            if isinstance(raw_conf, torch.Tensor):
                conf_t = raw_conf.float()
            else:
                conf_t = torch.tensor(raw_conf, dtype=torch.float32)

            raw_cls = detections["classes"]
            if isinstance(raw_cls, torch.Tensor):
                cls_t = raw_cls.float()
            else:
                cls_t = torch.tensor(raw_cls, dtype=torch.float32)

            raw_masks = detections.get("masks")
            if raw_masks is not None:
                if isinstance(raw_masks, torch.Tensor):
                    masks_t = raw_masks
                else:
                    masks_t = torch.tensor(raw_masks)

        # Apply class filter
        if classes is not None and len(boxes_t) > 0:
            boxes_t, conf_t, cls_t, masks_t = self._apply_classes_filter(
                boxes_t, conf_t, cls_t, classes, masks_t
            )

        # original_size from preprocess is (W, H); orig_shape is (H, W)
        orig_w, orig_h = original_size
        orig_shape = (orig_h, orig_w)

        masks_obj = None
        if masks_t is not None:
            masks_obj = Masks(masks_t, orig_shape)

        return Results(
            boxes=Boxes(boxes_t, conf_t, cls_t),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.model.names,
            masks=masks_obj,
        )

    def _predict_single(
        self,
        image: ImageInput,
        save: bool = False,
        output_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
        output_file_format: Optional[str] = None,
        **kwargs,
    ) -> Results:
        """Run inference on a single image."""
        image_path = image if isinstance(image, (str, Path)) else None

        # Resolve input size
        effective_imgsz = imgsz if imgsz is not None else self.model._get_input_size()

        # Preprocess
        input_tensor, original_img, original_size, ratio = self.model._preprocess(
            image, color_format, input_size=effective_imgsz
        )

        # Forward pass
        with torch.no_grad():
            output = self.model._forward(input_tensor.to(self.model.device))

        # Postprocess
        detections = self.model._postprocess(
            output, conf, iou, original_size, max_det=max_det, ratio=ratio, **kwargs
        )

        # Wrap into Results
        result = self._wrap_results(detections, original_size, image_path, classes)

        # Save annotated image
        if save:
            if len(result) > 0:
                annotated_img = original_img
                # Draw masks first (underneath boxes)
                if result.masks is not None:
                    masks_np = result.masks.data
                    if hasattr(masks_np, "numpy"):
                        masks_np = masks_np.cpu().numpy()
                    annotated_img = draw_masks(
                        annotated_img,
                        masks_np,
                        result.boxes.cls.tolist(),
                    )
                annotated_img = draw_boxes(
                    annotated_img,
                    result.boxes.xyxy.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.cls.tolist(),
                )
            else:
                annotated_img = original_img

            ext = output_file_format or "jpg"
            save_path = resolve_save_path(
                output_path,
                image_path,
                ext=ext,
            )
            annotated_img.save(save_path)
            result.saved_path = str(save_path)

        return result

    def _merge_tile_detections(
        self,
        boxes: List,
        scores: List,
        classes: List,
        iou_thres: float,
    ) -> Tuple[List, List, List]:
        """Merge detections from tiles using class-wise NMS."""
        if not boxes:
            return [], [], []

        boxes_t = torch.tensor(boxes, dtype=torch.float32, device=self.model.device)
        scores_t = torch.tensor(scores, dtype=torch.float32, device=self.model.device)
        classes_t = torch.tensor(classes, dtype=torch.int64, device=self.model.device)

        final_boxes, final_scores, final_classes = [], [], []

        for cls_id in torch.unique(classes_t):
            mask = classes_t == cls_id
            cls_boxes = boxes_t[mask]
            cls_scores = scores_t[mask]

            keep = nms(cls_boxes, cls_scores, iou_thres)

            final_boxes.extend(cls_boxes[keep].cpu().tolist())
            final_scores.extend(cls_scores[keep].cpu().tolist())
            final_classes.extend([cls_id.item()] * len(keep))

        return final_boxes, final_scores, final_classes

    def _predict_tiled(
        self,
        image: ImageInput,
        save: bool = False,
        output_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None,
        **kwargs,
    ) -> Results:
        """Run tiled inference on large images."""
        input_size = imgsz if imgsz is not None else self.model._get_input_size()
        img_pil = ImageLoader.load(image, color_format=color_format)
        orig_width, orig_height = img_pil.size
        image_path = image if isinstance(image, (str, Path)) else None

        # Skip tiling if image is small enough
        if orig_width <= input_size and orig_height <= input_size:
            return self._predict_single(
                image,
                save=save,
                output_path=output_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format=color_format,
                output_file_format=output_file_format,
                **kwargs,
            )

        # Get tile coordinates
        slices = get_slice_bboxes(
            orig_width, orig_height, slice_size=input_size, overlap_ratio=overlap_ratio
        )

        # Process tiles
        all_boxes, all_scores, all_classes = [], [], []
        tiles_data = []

        for idx, (x1, y1, x2, y2) in enumerate(slices):
            tile = img_pil.crop((x1, y1, x2, y2))

            if save:
                tiles_data.append(
                    {"index": idx, "coords": (x1, y1, x2, y2), "image": tile.copy()}
                )

            tile_result = self._predict_single(
                tile,
                save=False,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                **kwargs,
            )

            # Shift boxes to original coordinates
            if len(tile_result) > 0:
                tile_boxes = tile_result.boxes.xyxy.tolist()
                for box in tile_boxes:
                    shifted_box = [box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1]
                    all_boxes.append(shifted_box)
                all_scores.extend(tile_result.boxes.conf.tolist())
                all_classes.extend(tile_result.boxes.cls.tolist())

        # Merge detections
        final_boxes, final_scores, final_classes = self._merge_tile_detections(
            all_boxes, all_scores, all_classes, iou
        )

        # Build Results
        original_size = (orig_width, orig_height)
        detections = {
            "boxes": final_boxes,
            "scores": final_scores,
            "classes": final_classes,
            "num_detections": len(final_boxes),
        }
        result = self._wrap_results(detections, original_size, image_path, classes)

        # Attach tiling metadata as extra attributes
        result.tiled = True
        result.num_tiles = len(slices)

        # Save if requested
        if save:
            ext = output_file_format or "jpg"

            if isinstance(image_path, (str, Path)):
                stem = get_safe_stem(image_path)
            else:
                stem = "inference"
            model_tag = f"{self.model._get_model_name()}_{self.model.size}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            if output_path:
                base_path = Path(output_path)
                if base_path.suffix == "":
                    save_dir = base_path / f"{stem}_{model_tag}_{timestamp}"
                else:
                    save_dir = base_path.parent / f"{stem}_{model_tag}_{timestamp}"
            else:
                save_dir = (
                    Path("runs/tiled_detections") / f"{stem}_{model_tag}_{timestamp}"
                )

            save_dir.mkdir(parents=True, exist_ok=True)

            # Save tiles
            tiles_dir = save_dir / "tiles"
            tiles_dir.mkdir(parents=True, exist_ok=True)
            for tile_data in tiles_data:
                tile_filename = f"tile_{tile_data['index']:03d}.{ext}"
                tile_data["image"].save(tiles_dir / tile_filename)

            # Save annotated image
            if len(result) > 0:
                annotated_img = draw_boxes(
                    img_pil,
                    result.boxes.xyxy.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.cls.tolist(),
                )
            else:
                annotated_img = img_pil.copy()

            annotated_img.save(save_dir / f"final_image.{ext}")

            # Save grid visualization
            grid_img = draw_tile_grid(img_pil, slices)
            grid_path = save_dir / f"grid_visualization.{ext}"
            grid_img.save(grid_path)

            # Save metadata
            metadata = {
                "model": self.model._get_model_name(),
                "size": self.model.size,
                "image_source": str(image_path) if image_path else "PIL/numpy input",
                "original_size": [orig_width, orig_height],
                "num_tiles": len(slices),
                "tile_size": input_size,
                "overlap_ratio": overlap_ratio,
                "num_detections": len(result),
                "conf": conf,
                "iou": iou,
            }
            with open(save_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            result.saved_path = str(save_dir)
            result.tiles_path = str(tiles_dir)
            result.grid_path = str(grid_path)

        return result
