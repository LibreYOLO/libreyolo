"""Pose validator for LibreYOLO.

Computes COCO-style keypoint metrics (OKS-AP) via pycocotools'
``COCOeval(iouType='keypoints')``. Inputs can be either:

- Ultralytics YOLO-pose ``data.yaml`` with ``kpt_shape`` and YOLO labels.
- COCO keypoints JSON plus ``images_dir``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from libreyolo.data import get_img_files, img2label_paths, load_data_config
from libreyolo.data.pose_dataset import parse_yolo_pose_label_line

from .base import BaseValidator
from .config import ValidationConfig

logger = logging.getLogger(__name__)

_COCO17_OKS_SIGMAS = [
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
]

if TYPE_CHECKING:
    from libreyolo.models.base import BaseModel


class PoseValidator(BaseValidator):
    """YOLO-pose/COCO-keypoints OKS-AP validator."""

    task = "pose"

    def __init__(
        self,
        model: "BaseModel",
        config: Optional[ValidationConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(model, config, **kwargs)
        self._coco_gt = None
        self._image_records: List[dict] = []
        self._predictions: List[dict] = []
        self._num_keypoints: int | None = None
        self._category_id: int = 0

    # PoseValidator runs a per-image loop driven by COCO JSON, so it does not
    # use the BaseValidator dataloader-template path. The required hooks below
    # are intentionally no-ops and ``run()`` is overridden.

    def _setup_dataloader(self):
        return None

    def _init_metrics(self) -> None:
        self._predictions = []

    def _warmup_model(self, n_warmup: int = 1) -> None:
        return None

    def _preprocess_batch(self, batch):  # pragma: no cover - unused
        raise NotImplementedError("PoseValidator does not use a batch dataloader.")

    def _postprocess_predictions(self, preds, batch):  # pragma: no cover
        raise NotImplementedError("PoseValidator does not use a batch dataloader.")

    def _update_metrics(self, detections, targets, img_info, img_ids):  # pragma: no cover
        raise NotImplementedError("PoseValidator does not use a batch dataloader.")

    def _compute_metrics(self) -> Dict[str, float]:
        return {}

    def _print_results(self, metrics: Dict[str, float]) -> None:  # pragma: no cover
        return None

    # =========================================================================
    # Custom run loop
    # =========================================================================

    def run(self, **_kwargs) -> Dict[str, float]:
        try:
            from pycocotools.coco import COCO  # noqa: F401
            from pycocotools.cocoeval import COCOeval  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Pose validation requires pycocotools. "
                "Install with: pip install pycocotools"
            ) from exc

        self._setup_paths()
        self._load_coco_gt()

        self._predictions = []
        self.seen = 0
        self.speed = {
            "preprocess": 0.0,
            "inference": 0.0,
            "postprocess": 0.0,
            "total": 0.0,
        }

        total_start = time.time()
        self._predict_all()
        self.speed["total"] = time.time() - total_start

        metrics = self._evaluate_oks_ap()
        self.config.to_yaml(self.save_dir / "config.yaml")

        if self.seen > 0:
            metrics["speed/total_ms"] = self.speed["total"] / self.seen * 1000
            metrics["speed/total_s"] = self.speed["total"]
            metrics["speed/images_seen"] = self.seen

        if self.config.verbose:
            self._log_metrics(metrics)
        return metrics

    # =========================================================================
    # Setup
    # =========================================================================

    def _setup_paths(self) -> None:
        if self.config.save_dir:
            self.save_dir = Path(self.config.save_dir)
        else:
            model_tag = f"{self.model._get_model_name()}_{self.model.size}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.save_dir = Path("runs/val") / f"{model_tag}_{timestamp}_pose"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.config.keypoints_json:
            if not self.config.images_dir:
                raise ValueError(
                    "PoseValidator requires images_dir with keypoints_json."
                )
            self._kpts_json = Path(self.config.keypoints_json)
            self._images_dir = Path(self.config.images_dir)
            if not self._kpts_json.exists():
                raise FileNotFoundError(f"Annotations JSON not found: {self._kpts_json}")
            if not self._images_dir.is_dir():
                raise FileNotFoundError(f"Images dir not found: {self._images_dir}")
            return

        if not self.config.data:
            raise ValueError(
                "PoseValidator requires either Ultralytics YOLO-pose data.yaml "
                "via data=... or keypoints_json + images_dir."
            )
        self._kpts_json, self._images_dir = self._build_coco_gt_from_yolo()

    def _build_coco_gt_from_yolo(self) -> tuple[Path, Path | None]:
        from PIL import Image

        data_cfg = load_data_config(
            self.config.data,
            autodownload=True,
            allow_scripts=self.config.allow_download_scripts,
        )
        split = self.config.split
        img_files = data_cfg.get(f"{split}_img_files")
        label_files = data_cfg.get(f"{split}_label_files")
        if not img_files:
            split_value = data_cfg.get(split)
            if not split_value:
                raise FileNotFoundError(f"Dataset yaml has no {split!r} split")
            img_files = get_img_files(split_value)
            label_files = img2label_paths(img_files)

        kpt_shape = data_cfg.get("kpt_shape")
        if not kpt_shape:
            raise ValueError("YOLO-pose validation requires kpt_shape in data.yaml")
        num_keypoints = int(kpt_shape[0])
        keypoint_dim = int(kpt_shape[1]) if len(kpt_shape) > 1 else 3
        self._num_keypoints = num_keypoints

        names = data_cfg.get("names") or {0: "object"}
        if isinstance(names, dict):
            categories_source = [
                (int(idx), str(name))
                for idx, name in sorted(names.items(), key=lambda item: int(item[0]))
            ]
        else:
            categories_source = [(idx, str(name)) for idx, name in enumerate(names)]
        if not categories_source:
            categories_source = [(0, "object")]
        kpt_names = data_cfg.get("keypoints") or [
            f"keypoint_{i}" for i in range(num_keypoints)
        ]
        skeleton = data_cfg.get("skeleton", [])

        images, annotations = [], []
        ann_id = 1
        for img_id, img_path in enumerate([Path(p) for p in img_files]):
            with Image.open(img_path) as img:
                width, height = img.size
            images.append(
                {
                    "id": img_id,
                    "file_name": str(img_path),
                    "width": width,
                    "height": height,
                }
            )

            label_path = Path(label_files[img_id])
            if not label_path.exists():
                continue
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    cls_id, bbox, kpts = parse_yolo_pose_label_line(
                        parts, num_keypoints, keypoint_dim
                    )
                except ValueError:
                    continue
                cx, cy, bw, bh = bbox.astype(float)
                x = (cx - bw * 0.5) * width
                y = (cy - bh * 0.5) * height
                box_w = bw * width
                box_h = bh * height
                keypoints = kpts.astype(float).copy()
                keypoints[:, 0] *= width
                keypoints[:, 1] *= height
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(cls_id),
                        "bbox": [x, y, box_w, box_h],
                        "area": box_w * box_h,
                        "iscrowd": 0,
                        "keypoints": keypoints.reshape(-1).tolist(),
                        "num_keypoints": int((keypoints[:, 2] > 0).sum()),
                    }
                )
                ann_id += 1

        coco = {
            "info": {},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [
                {
                    "id": class_id,
                    "name": name,
                    "supercategory": "object",
                    "keypoints": list(kpt_names),
                    "skeleton": skeleton,
                }
                for class_id, name in categories_source
            ],
        }
        out = self.save_dir / "ground_truth_yolo_pose.json"
        with out.open("w") as f:
            json.dump(coco, f)
        return out, None

    def _load_coco_gt(self) -> None:
        from pycocotools.coco import COCO

        self._coco_gt = COCO(str(self._kpts_json))
        self._category_id = self._infer_category_id()
        self._image_records = self._coco_gt.loadImgs(self._coco_gt.getImgIds())
        if self._num_keypoints is None:
            cats = self._coco_gt.loadCats(self._coco_gt.getCatIds())
            keypoints = cats[0].get("keypoints", []) if cats else []
            self._num_keypoints = len(keypoints) or None

    def _infer_category_id(self) -> int:
        cats = self._coco_gt.loadCats(self._coco_gt.getCatIds())
        if not cats:
            return 0
        for cat in cats:
            if cat.get("name") == "person":
                return int(cat["id"])
        return int(cats[0]["id"])

    # =========================================================================
    # Inference loop
    # =========================================================================

    def _predict_all(self) -> None:
        verbose = self.config.verbose
        records = self._image_records
        pbar = tqdm(
            records,
            desc="Pose val",
            total=len(records),
            disable=not verbose or not sys.stderr.isatty(),
            file=sys.stderr,
        )
        for record in pbar:
            file_name = record["file_name"]
            image_id = int(record["id"])
            img_path = Path(file_name)
            if not img_path.is_absolute():
                img_path = self._images_dir / file_name
            if not img_path.exists():
                logger.warning("Skipping missing image: %s", img_path)
                continue
            self._predict_image(img_path, image_id)
            self.seen += 1

    def _predict_image(self, img_path: Path, image_id: int) -> None:
        result = self.model(
            str(img_path),
            conf=self.config.conf_thres,
            iou=self.config.iou_thres,
            imgsz=self.config.imgsz,
            max_det=self.config.max_det,
        )
        if result.keypoints is None or len(result) == 0:
            return

        kpts = result.keypoints.data
        scores = result.boxes.conf
        classes = getattr(result.boxes, "cls", None)
        # Convert to numpy / list for JSON friendliness.
        kpts_np = kpts.detach().cpu().numpy() if hasattr(kpts, "detach") else kpts
        scores_np = scores.detach().cpu().numpy() if hasattr(scores, "detach") else scores
        if classes is None:
            classes_np = np.full(len(scores_np), self._category_id)
        else:
            classes_np = (
                classes.detach().cpu().numpy()
                if hasattr(classes, "detach")
                else classes
            )

        for instance_kpts, score, cls_id in zip(kpts_np, scores_np, classes_np):
            flat = []
            for x, y, v in instance_kpts:
                flat.extend([float(x), float(y), float(v)])
            self._predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(cls_id),
                    "keypoints": flat,
                    "score": float(score),
                }
            )

    # =========================================================================
    # Evaluation
    # =========================================================================

    def _evaluate_oks_ap(self) -> Dict[str, float]:
        from pycocotools.cocoeval import COCOeval

        pred_path = self.save_dir / "predictions.json"
        with pred_path.open("w") as f:
            json.dump(self._predictions, f)

        if not self._predictions:
            logger.warning("No pose predictions produced; returning zero metrics.")
            return {
                "metrics/keypoints_mAP50-95": 0.0,
                "metrics/keypoints_mAP50": 0.0,
                "metrics/keypoints_mAP75": 0.0,
            }

        coco_dt = self._coco_gt.loadRes(str(pred_path))
        coco_eval = COCOeval(self._coco_gt, coco_dt, iouType="keypoints")
        sigmas = self._resolve_oks_sigmas()
        if sigmas is not None:
            coco_eval.params.kpt_oks_sigmas = np.asarray(sigmas, dtype=np.float64)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats  # length 10 for keypoints
        return {
            "metrics/keypoints_mAP50-95": float(stats[0]),
            "metrics/keypoints_mAP50": float(stats[1]),
            "metrics/keypoints_mAP75": float(stats[2]),
            "metrics/keypoints_mAP_M": float(stats[3]),
            "metrics/keypoints_mAP_L": float(stats[4]),
            "metrics/keypoints_AR50-95": float(stats[5]),
            "metrics/keypoints_AR50": float(stats[6]),
            "metrics/keypoints_AR75": float(stats[7]),
            "metrics/keypoints_AR_M": float(stats[8]),
            "metrics/keypoints_AR_L": float(stats[9]),
        }

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        for key in (
            "metrics/keypoints_mAP50-95",
            "metrics/keypoints_mAP50",
            "metrics/keypoints_mAP75",
        ):
            if key in metrics:
                logger.info("%s: %.4f", key, metrics[key])

    def _resolve_oks_sigmas(self) -> list[float] | None:
        if self.config.oks_sigmas:
            return list(self.config.oks_sigmas)
        if self._num_keypoints == 17:
            return _COCO17_OKS_SIGMAS
        if self._num_keypoints:
            return [1.0 / self._num_keypoints] * self._num_keypoints
        return None
