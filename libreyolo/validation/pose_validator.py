"""Pose validator for LibreYOLO.

Computes COCO-style keypoint metrics (OKS-AP) via pycocotools' ``COCOeval`` with
``iouType='keypoints'``.

The expected inputs are a COCO keypoints annotations JSON and a directory of
images whose names match ``file_name`` in the annotations. Pass them via:

- ``keypoints_json`` and ``images_dir`` on ``ValidationConfig``, or
- a ``data.yaml`` file containing both fields under ``val:`` (TBD; for now use
  the explicit-path form).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from tqdm import tqdm

from .base import BaseValidator
from .config import ValidationConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from libreyolo.models.base import BaseModel


class PoseValidator(BaseValidator):
    """COCO-keypoints OKS-AP validator (single-class, person-only)."""

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

        if not self.config.keypoints_json:
            raise ValueError(
                "PoseValidator requires ValidationConfig.keypoints_json — "
                "pass the path to a COCO keypoints annotation JSON."
            )
        if not self.config.images_dir:
            raise ValueError(
                "PoseValidator requires ValidationConfig.images_dir — "
                "pass the path to a directory containing the images referenced "
                "by file_name in the annotation JSON."
            )

        self._kpts_json = Path(self.config.keypoints_json)
        self._images_dir = Path(self.config.images_dir)
        if not self._kpts_json.exists():
            raise FileNotFoundError(f"Annotations JSON not found: {self._kpts_json}")
        if not self._images_dir.is_dir():
            raise FileNotFoundError(f"Images dir not found: {self._images_dir}")

    def _load_coco_gt(self) -> None:
        from pycocotools.coco import COCO

        self._coco_gt = COCO(str(self._kpts_json))
        # Restrict to the "person" category for keypoint AP.
        self._person_cat_id = self._infer_person_cat_id()
        self._image_records = self._coco_gt.loadImgs(self._coco_gt.getImgIds())

    def _infer_person_cat_id(self) -> int:
        cats = self._coco_gt.loadCats(self._coco_gt.getCatIds())
        for cat in cats:
            if cat.get("name") == "person":
                return int(cat["id"])
        # Fall back to the first cat if the name "person" is missing.
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
        # Convert to numpy / list for JSON friendliness.
        kpts_np = kpts.detach().cpu().numpy() if hasattr(kpts, "detach") else kpts
        scores_np = scores.detach().cpu().numpy() if hasattr(scores, "detach") else scores

        for instance_kpts, score in zip(kpts_np, scores_np):
            flat = []
            for x, y, v in instance_kpts:
                flat.extend([float(x), float(y), float(v)])
            self._predictions.append(
                {
                    "image_id": image_id,
                    "category_id": self._person_cat_id,
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
