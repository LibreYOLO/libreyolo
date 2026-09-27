"""COCO evaluator for LibreYOLO."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Env override for the COCO eval backend: "1"/"true"/"yes" forces the
# faster-coco-eval backend on, "0"/"false"/"no" forces it off, unset defers
# to the faster_coco_eval config flag. Useful for benchmark harnesses that
# cannot touch per-run configs.
FASTER_COCO_EVAL_ENV_VAR = "LIBREYOLO_FASTER_COCO_EVAL"

_warned_faster_unavailable = False


def _faster_coco_eval_env_override() -> Optional[bool]:
    value = os.environ.get(FASTER_COCO_EVAL_ENV_VAR)
    if value is None:
        return None
    return value.strip().lower() in ("1", "true", "yes", "on")


def resolve_faster_coco_eval(requested: bool) -> bool:
    """Decide whether to use the faster-coco-eval backend.

    The LIBREYOLO_FASTER_COCO_EVAL env var, when set, overrides `requested`.
    Returns False (stock pycocotools) if the package is not importable.
    """
    override = _faster_coco_eval_env_override()
    enabled = requested if override is None else override
    if not enabled:
        return False
    try:
        import faster_coco_eval  # noqa: F401
    except ImportError:
        global _warned_faster_unavailable
        if not _warned_faster_unavailable:
            logger.warning(
                "faster_coco_eval requested but not installed; falling back to "
                "pycocotools. Install with: pip install faster-coco-eval"
            )
            _warned_faster_unavailable = True
        return False
    return True


class COCOEvaluator:
    """
    COCO evaluation wrapper.

    Computes standard COCO metrics: AP (mAP@[0.5:0.95]), AP50, AP75,
    AP/AR by object size, and AR at different maxDets.
    """

    def __init__(
        self,
        coco_gt,
        iou_type: str = "bbox",
        label_to_category_id: Optional[Mapping[int, int]] = None,
        max_det: int = 100,
        faster_coco_eval: bool = False,
    ):
        if max_det < 1:
            raise ValueError(f"max_det must be >= 1, got {max_det}")
        self.coco_gt = coco_gt
        self.iou_type = iou_type
        self.max_det = int(max_det)
        self.faster_coco_eval = faster_coco_eval
        self.label_to_category_id = (
            {int(k): int(v) for k, v in label_to_category_id.items()}
            if label_to_category_id is not None
            else None
        )
        self.results = []
        self._img_ids = set()
        self._last_coco_eval = None
        # Provenance: backend actually used by the last compute() call.
        self.last_backend: Optional[str] = None

    def update(self, predictions: Dict, image_id: int):
        """
        Add predictions for an image.

        Args:
            predictions: Dict with boxes (xyxy), scores, classes.
            image_id: Image ID matching COCO API.
        """
        boxes = predictions["boxes"]
        scores = predictions["scores"]
        classes = predictions["classes"]
        masks = predictions.get("masks")

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(classes, torch.Tensor):
            classes = classes.cpu().numpy()
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        boxes = np.array(boxes) if not isinstance(boxes, np.ndarray) else boxes
        scores = np.array(scores) if not isinstance(scores, np.ndarray) else scores
        classes = np.array(classes) if not isinstance(classes, np.ndarray) else classes
        masks = np.array(masks) if masks is not None and not isinstance(masks, np.ndarray) else masks

        if self.iou_type == "segm" and masks is None:
            self._img_ids.add(image_id)
            return

        for idx, (box, score, label) in enumerate(zip(boxes, scores, classes)):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            label = int(label)
            category_id = (
                self.label_to_category_id.get(label, label)
                if self.label_to_category_id is not None
                else label
            )

            result = {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": [float(x1), float(y1), float(w), float(h)],  # COCO xywh
                "score": float(score),
            }
            if self.iou_type == "segm":
                mask = masks[idx]
                result["segmentation"] = self._encode_mask(mask)
                result["area"] = float((mask > 0).sum())
            self.results.append(result)

        self._img_ids.add(image_id)

    @staticmethod
    def _encode_mask(mask: np.ndarray) -> dict:
        """Encode a binary mask to JSON-safe COCO RLE."""
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            raise ImportError(
                "pycocotools not installed. Install with: pip install pycocotools"
            )

        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask for COCO RLE, got shape {mask.shape}")
        mask = (mask > 0).astype(np.uint8)
        rle = mask_utils.encode(np.asfortranarray(mask))
        counts = rle.get("counts")
        if isinstance(counts, bytes):
            rle["counts"] = counts.decode("ascii")
        rle["size"] = [int(mask.shape[0]), int(mask.shape[1])]
        return rle

    def compute(self, save_json: Optional[str] = None) -> Dict[str, float]:
        """
        Run COCO evaluation and return 12 standard metrics.

        Args:
            save_json: Optional path to save predictions in COCO JSON format.
                Written even when no predictions were accumulated.
        """
        if save_json:
            # Written first: an opted-in run must produce the file even when
            # there are no predictions or evaluation fails below, and loadRes
            # mutates the result dicts in place (adds id/area/segmentation).
            with open(save_json, "w") as f:
                json.dump(self.results, f, indent=2)
            logger.info("Saved predictions to %s", Path(save_json).resolve())

        if len(self.results) == 0:
            logger.warning("No predictions to evaluate")
            return self._empty_metrics()

        coco_eval = self._build_coco_eval()
        if self._img_ids:
            coco_eval.params.imgIds = sorted(self._img_ids)
        # Retain a real AR@100 compatibility slot while adding the requested
        # protocol cap. COCOeval supports an arbitrary maxDets axis, but its
        # stock summarize() hard-codes overall AP to maxDets=100. Metrics below
        # are therefore read directly from the accumulated arrays.
        coco_eval.params.maxDets = sorted({1, 10, 100, self.max_det})
        coco_eval.evaluate()
        coco_eval.accumulate()
        if self.max_det == 100:
            # Preserve the historical/default path literally. This makes the
            # default output subject to pycocotools' own summarize semantics.
            coco_eval.summarize()
        else:
            coco_eval.stats = self._standard_stats(coco_eval)
        self._last_coco_eval = coco_eval  # kept for per-class AP access

        # stats layout: [mAP, mAP50, mAP75, AP_s, AP_m, AP_l,
        #                AR1, AR10, AR@max_det, AR_s, AR_m, AR_l]
        #
        # NOTE: these are NOT a precision/recall pair at a fixed operating
        # point. ``precision`` here is the mean of the precision array at the
        # last maxDet over all IoU/recall/class bins == mAP@[.5:.95] (stats[0]),
        # and ``recall`` remains the historical AR@100 value. They are emitted
        # under the honest ``map_5095`` / ``ar_100`` keys below; the legacy
        # ``precision`` / ``recall`` keys are kept as aliases for backward
        # compatibility and must not be plotted as a distinct P/R.
        map_5095 = self._summarize_metric(
            coco_eval, ap=True, max_det=self.max_det, empty=0.0
        )
        ar_100 = self._summarize_metric(
            coco_eval, ap=False, max_det=100, empty=0.0
        )
        ar_max_det = float(coco_eval.stats[8])
        return {
            "max_det": float(self.max_det),
            "map_5095": map_5095,
            "ar_100": ar_100,
            "ar_max_det": ar_max_det,
            "precision": map_5095,  # alias (deprecated): == map_5095, not real P
            "recall": ar_100,  # alias (deprecated): == ar_100, not real R
            "mAP": float(coco_eval.stats[0]),
            "mAP50": float(coco_eval.stats[1]),
            "mAP75": float(coco_eval.stats[2]),
            "mAP_small": float(coco_eval.stats[3]),
            "mAP_medium": float(coco_eval.stats[4]),
            "mAP_large": float(coco_eval.stats[5]),
            "AR1": float(coco_eval.stats[6]),
            "AR10": float(coco_eval.stats[7]),
            "AR100": ar_100,
            "AR_max_det": ar_max_det,
            "AR_small": float(coco_eval.stats[9]),
            "AR_medium": float(coco_eval.stats[10]),
            "AR_large": float(coco_eval.stats[11]),
        }

    def _build_coco_eval(self):
        """Construct a COCOeval instance using the configured backend.

        With faster_coco_eval=True (or the LIBREYOLO_FASTER_COCO_EVAL env
        override) and the faster-coco-eval package installed, evaluation runs
        through its C++ backend, which is 10-50x faster on detection-dense
        datasets while producing metrics identical to pycocotools within
        float64 summation order (<= 1 ULP).
        """
        if resolve_faster_coco_eval(self.faster_coco_eval):
            import faster_coco_eval
            from faster_coco_eval import COCO as FasterCOCO
            from faster_coco_eval import COCOeval_faster

            gt_dataset = getattr(self.coco_gt, "dataset", None)
            if not gt_dataset or not gt_dataset.get("images"):
                # COCO-like GT objects (e.g. YOLOCocoAPI) that don't carry a
                # raw dataset dict: synthesize one from their index maps.
                gt_dataset = {
                    "images": list(self.coco_gt.imgs.values()),
                    "annotations": list(self.coco_gt.anns.values()),
                    "categories": list(self.coco_gt.cats.values()),
                }
            # use_deepcopy so backend-side mutations (e.g. segm polygon->RLE
            # conversion) never leak back into self.coco_gt.
            coco_gt = FasterCOCO(gt_dataset, use_deepcopy=True)
            coco_dt = coco_gt.loadRes(self.results)
            fce_version = getattr(
                getattr(faster_coco_eval, "version", None), "__version__", "?"
            )
            self.last_backend = f"faster-coco-eval {fce_version}"
            logger.info("COCO eval backend: %s", self.last_backend)
            return COCOeval_faster(coco_gt, coco_dt, self.iou_type)

        try:
            import pycocotools
            from pycocotools.coco import COCO  # noqa: F401
            from pycocotools.cocoeval import COCOeval
        except ImportError:
            raise ImportError(
                "pycocotools not installed. Install with: pip install pycocotools"
            )

        self.last_backend = (
            f"pycocotools {getattr(pycocotools, '__version__', '?')}"
        )
        logger.info("COCO eval backend: %s", self.last_backend)
        coco_dt = self.coco_gt.loadRes(self.results)
        return COCOeval(self.coco_gt, coco_dt, self.iou_type)

    def _empty_metrics(self) -> Dict[str, float]:
        """Return all-zero metrics dict."""
        return {
            "max_det": float(self.max_det),
            "map_5095": 0.0,
            "ar_100": 0.0,
            "ar_max_det": 0.0,
            "precision": 0.0,  # alias (deprecated): == map_5095, not real P
            "recall": 0.0,  # alias (deprecated): == ar_100, not real R
            "mAP": 0.0,
            "mAP50": 0.0,
            "mAP75": 0.0,
            "mAP_small": 0.0,
            "mAP_medium": 0.0,
            "mAP_large": 0.0,
            "AR1": 0.0,
            "AR10": 0.0,
            "AR100": 0.0,
            "AR_max_det": 0.0,
            "AR_small": 0.0,
            "AR_medium": 0.0,
            "AR_large": 0.0,
        }

    def reset(self):
        """Clear all accumulated results."""
        self.results = []
        self._img_ids = set()

    @staticmethod
    def _best_f1_threshold(
        scores: np.ndarray, tps: np.ndarray, npig: int
    ) -> Tuple[float, float]:
        """Sweep F1 over score thresholds with a single sort plus cumsum.

        ``scores`` and ``tps`` are aligned per-detection arrays (ignored
        detections must already be removed); ``npig`` is the number of
        non-ignored ground truths. F1 is evaluated once per distinct score,
        always at the last detection of a tie group, so detections sharing a
        score are included or excluded as a whole. Ties in F1 resolve to the
        highest threshold.

        Returns:
            ``(threshold, f1)``. A NaN pair when no threshold achieves
            F1 > 0: no predictions, no ground truth, or every prediction
            is a false positive.
        """
        nan = float("nan")
        scores = np.asarray(scores, dtype=np.float64)
        tps = np.asarray(tps, dtype=bool)
        if scores.size == 0 or npig <= 0:
            return nan, nan
        order = np.argsort(-scores, kind="stable")
        scores = scores[order]
        tp = tps[order].astype(np.float64)
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(1.0 - tp)
        # Candidate cut points: the last detection of every distinct score.
        cut = np.flatnonzero(np.diff(scores))
        cut = np.append(cut, scores.size - 1)
        # F1 = 2*tp / (tp + fp + npig) == 2PR / (P + R) with P = tp/(tp+fp),
        # R = tp/npig; the denominator is always > 0 at a detection index.
        f1 = 2.0 * tp_cum[cut] / (tp_cum[cut] + fp_cum[cut] + float(npig))
        best = int(np.argmax(f1))  # first max == highest threshold on ties
        if not f1[best] > 0.0:
            return nan, nan
        return float(scores[cut[best]]), float(f1[best])

    def best_conf_thresholds(
        self, iou_thr: float = 0.5
    ) -> Optional[Dict[str, object]]:
        """Per-class and micro-averaged global best confidence thresholds.

        Reads the per-image match results (``evalImgs``) of the last
        :meth:`compute` call, so no matching is re-run; with the
        faster-coco-eval backend, which keeps matching in C++, the greedy
        assignment is replayed on its retained IoU matrices instead (see
        :meth:`_per_class_match_arrays`). F1 is defined at IoU ``iou_thr``
        matching (0.50 by default; on the ``evalImgs`` path, if that
        threshold was not evaluated, the lowest evaluated one is used) over
        the "all" area range. Detections flagged as ignored by COCOeval,
        which includes detections matched to crowd/ignore ground truths,
        take part in the sweep as neither TP nor FP, and ignored ground
        truths do not count toward recall.

        Returns:
            None when no evaluation ran or the backend exposes no usable
            match source. Otherwise
            ``{"global": (thr, f1), "per_class": {label: (thr, f1)}}``
            where ``label`` is the model class index (category ids are
            mapped back through ``label_to_category_id``) and entries are
            NaN pairs for classes where no threshold reaches F1 > 0.
        """
        coco_eval = self._last_coco_eval
        if coco_eval is None:
            return None
        try:
            source = self._per_class_match_arrays(coco_eval, iou_thr)
            if source is None:
                return None
            category_to_label = (
                {v: k for k, v in self.label_to_category_id.items()}
                if self.label_to_category_id is not None
                else {}
            )

            per_class: Dict[int, Tuple[float, float]] = {}
            pooled_scores = []
            pooled_tps = []
            total_npig = 0
            for cat_id, (scores, tps, npig) in source.items():
                label = category_to_label.get(int(cat_id), int(cat_id))
                per_class[label] = self._best_f1_threshold(scores, tps, npig)
                pooled_scores.append(scores)
                pooled_tps.append(tps)
                total_npig += npig

            global_pair = self._best_f1_threshold(
                np.concatenate(pooled_scores)
                if pooled_scores
                else np.zeros(0, dtype=np.float64),
                np.concatenate(pooled_tps)
                if pooled_tps
                else np.zeros(0, dtype=bool),
                total_npig,
            )
            return {"global": global_pair, "per_class": per_class}
        except Exception as exc:  # backend without a usable match source
            logger.debug("Best-conf sweep unavailable: %s", exc)
            return None

    def _per_class_match_arrays(
        self, coco_eval, iou_thr: float
    ) -> Optional[Dict[int, Tuple[np.ndarray, np.ndarray, int]]]:
        """Per-category ``(scores, tps, npig)`` arrays for the F1 sweep.

        Uses the stock pycocotools ``evalImgs`` match results when the
        backend populates them. faster-coco-eval keeps matching in C++ and
        leaves ``evalImgs`` empty, but retains the Python-side IoU matrices
        (``ious``) and the prepared ``cocoGt``/``cocoDt`` annotations, so
        the greedy COCO assignment at ``iou_thr`` is replayed on those:
        the expensive IoU computation is reused, only the trivial matching
        loop is repeated.
        """
        eval_imgs = getattr(coco_eval, "evalImgs", None)
        if isinstance(eval_imgs, list) and eval_imgs:
            return self._match_arrays_from_eval_imgs(coco_eval, iou_thr)
        return self._match_arrays_from_ious(coco_eval, iou_thr)

    @staticmethod
    def _match_arrays_from_eval_imgs(
        coco_eval, iou_thr: float
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, int]]:
        """Extract sweep arrays from pycocotools-shaped ``evalImgs``."""
        params = coco_eval.params
        iou_thrs = np.asarray(params.iouThrs, dtype=np.float64)
        iou_matches = np.flatnonzero(np.isclose(iou_thrs, iou_thr))
        iou_index = int(iou_matches[0]) if iou_matches.size else 0
        area_index = list(params.areaRngLbl).index("all")
        n_area = len(params.areaRng)
        n_img = len(params.imgIds)
        eval_imgs = coco_eval.evalImgs

        out: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {}
        for k, cat_id in enumerate(params.catIds):
            scores_parts = []
            tp_parts = []
            npig = 0
            base = k * n_area * n_img + area_index * n_img
            for entry in eval_imgs[base : base + n_img]:
                if entry is None:
                    continue
                gt_ignore = np.asarray(entry["gtIgnore"])
                npig += int((gt_ignore == 0).sum())
                dt_scores = np.asarray(entry["dtScores"], dtype=np.float64)
                if dt_scores.size == 0:
                    continue
                keep = ~np.asarray(entry["dtIgnore"], dtype=bool)[iou_index]
                matches = np.asarray(entry["dtMatches"])[iou_index]
                scores_parts.append(dt_scores[keep])
                tp_parts.append(matches[keep] > 0)
            scores = (
                np.concatenate(scores_parts)
                if scores_parts
                else np.zeros(0, dtype=np.float64)
            )
            tps = (
                np.concatenate(tp_parts)
                if tp_parts
                else np.zeros(0, dtype=bool)
            )
            out[int(cat_id)] = (scores, tps, npig)
        return out

    @staticmethod
    def _match_arrays_from_ious(
        coco_eval, iou_thr: float
    ) -> Optional[Dict[int, Tuple[np.ndarray, np.ndarray, int]]]:
        """Replay COCO's greedy assignment on the retained IoU matrices.

        Original reimplementation (no upstream code copied) of the greedy
        matching rule COCO evaluation defines, at a single IoU threshold and
        the "all" area range: detections are visited in score order, each
        takes the best still-free ground truth above ``iou_thr``
        (crowd/ignored ground truths only when no normal one qualifies) and
        a detection matched to an ignored ground truth is dropped from the
        sweep (neither TP nor FP). Behavioral equivalence with the reference
        evaluator is pinned by the cross-backend parity tests in
        tests/unit/test_best_conf_threshold.py.
        """
        from collections import defaultdict

        ious_all = getattr(coco_eval, "ious", None)
        coco_gt = getattr(coco_eval, "cocoGt", None)
        coco_dt = getattr(coco_eval, "cocoDt", None)
        if ious_all is None or coco_gt is None or coco_dt is None:
            return None
        params = coco_eval.params
        max_det = int(sorted(params.maxDets)[-1])

        # One pass over each annotation index. Iterating anns in insertion
        # order preserves the per-image dataset order computeIoU consumed,
        # so the IoU matrix axes line up below.
        img_id_set = {int(i) for i in params.imgIds}
        gt_by_key = defaultdict(list)
        for ann in coco_gt.anns.values():
            key = (int(ann["image_id"]), int(ann["category_id"]))
            if key[0] in img_id_set:
                gt_by_key[key].append(ann)
        dt_by_key = defaultdict(list)
        for ann in coco_dt.anns.values():
            if ann.get("drop", False):
                continue
            key = (int(ann["image_id"]), int(ann["category_id"]))
            if key[0] in img_id_set:
                dt_by_key[key].append(ann)
        imgs_by_cat = defaultdict(set)
        for img_id, cat_id in list(gt_by_key) + list(dt_by_key):
            imgs_by_cat[cat_id].add(img_id)

        out: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {}
        for cat_id in params.catIds:
            scores_parts = []
            tp_parts = []
            npig = 0
            for img_id in sorted(imgs_by_cat.get(int(cat_id), ())):
                gt = gt_by_key.get((img_id, int(cat_id)), [])
                dt = dt_by_key.get((img_id, int(cat_id)), [])
                gt_ig = np.asarray(
                    [
                        g.get(
                            "_ignore",
                            1 if (g.get("ignore") or g.get("iscrowd")) else 0,
                        )
                        for g in gt
                    ],
                    dtype=np.int64,
                )
                npig += int((gt_ig == 0).sum())
                if not dt:
                    continue
                # Sort exactly like evaluateImg: ignored GTs last, detections
                # by descending score, both with a stable sort; truncate the
                # detections to the evaluated cap.
                gtind = np.argsort(gt_ig, kind="mergesort")
                gt_ig = gt_ig[gtind]
                crowd = np.asarray(
                    [int(gt[i].get("iscrowd", 0)) for i in gtind], dtype=np.int64
                )
                dt_scores_full = np.asarray(
                    [d["score"] for d in dt], dtype=np.float64
                )
                dtind = np.argsort(-dt_scores_full, kind="mergesort")[:max_det]
                dt_scores = dt_scores_full[dtind]
                # IoU rows are already in sorted-truncated detection order
                # (computeIoU sorts and truncates); columns follow raw GT
                # order and are permuted here to the sorted GT order.
                ious = np.asarray(ious_all[img_id, cat_id])
                if ious.size:
                    ious = ious[:, gtind]
                n_dt = len(dtind)
                tps = np.zeros(n_dt, dtype=bool)
                ignored = np.zeros(n_dt, dtype=bool)
                if len(gt) and ious.size:
                    gt_taken = np.zeros(len(gt), dtype=bool)
                    for dind in range(n_dt):
                        best = min(iou_thr, 1.0 - 1e-10)
                        m = -1
                        for gind in range(len(gt)):
                            if gt_taken[gind] and not crowd[gind]:
                                continue
                            if m > -1 and gt_ig[m] == 0 and gt_ig[gind] == 1:
                                break
                            if ious[dind, gind] < best:
                                continue
                            best = ious[dind, gind]
                            m = gind
                        if m == -1:
                            continue
                        gt_taken[m] = True
                        ignored[dind] = bool(gt_ig[m])
                        tps[dind] = not gt_ig[m]
                keep = ~ignored
                scores_parts.append(dt_scores[keep])
                tp_parts.append(tps[keep])
            scores = (
                np.concatenate(scores_parts)
                if scores_parts
                else np.zeros(0, dtype=np.float64)
            )
            tps = (
                np.concatenate(tp_parts)
                if tp_parts
                else np.zeros(0, dtype=bool)
            )
            out[int(cat_id)] = (scores, tps, npig)
        return out

    @staticmethod
    def _mean_valid(values: np.ndarray, *, empty: float = 0.0) -> float:
        """Mean over COCOeval arrays while ignoring absent -1 entries."""
        valid = values[values > -1]
        if valid.size == 0:
            return empty
        return float(valid.mean())

    def _summarize_metric(
        self,
        coco_eval,
        *,
        ap: bool,
        max_det: int,
        iou_thr: Optional[float] = None,
        area: str = "all",
        empty: float = -1.0,
    ) -> float:
        """Read one metric from COCOeval's accumulated precision/recall arrays."""
        params = coco_eval.params
        area_indices = [
            index for index, label in enumerate(params.areaRngLbl) if label == area
        ]
        max_det_indices = [
            index for index, value in enumerate(params.maxDets) if value == max_det
        ]
        if not area_indices or not max_det_indices:
            return -1.0

        if ap:
            values = coco_eval.eval["precision"]
            if iou_thr is not None:
                iou_indices = np.flatnonzero(np.isclose(params.iouThrs, iou_thr))
                values = values[iou_indices]
            values = values[:, :, :, area_indices, max_det_indices]
        else:
            values = coco_eval.eval["recall"]
            if iou_thr is not None:
                iou_indices = np.flatnonzero(np.isclose(params.iouThrs, iou_thr))
                values = values[iou_indices]
            values = values[:, :, area_indices, max_det_indices]
        return self._mean_valid(values, empty=empty)

    def _standard_stats(self, coco_eval) -> np.ndarray:
        """Build COCO's 12 detection metrics at the configured maximum."""
        max_det = self.max_det
        return np.asarray(
            [
                self._summarize_metric(coco_eval, ap=True, max_det=max_det),
                self._summarize_metric(
                    coco_eval, ap=True, max_det=max_det, iou_thr=0.5
                ),
                self._summarize_metric(
                    coco_eval, ap=True, max_det=max_det, iou_thr=0.75
                ),
                self._summarize_metric(
                    coco_eval, ap=True, max_det=max_det, area="small"
                ),
                self._summarize_metric(
                    coco_eval, ap=True, max_det=max_det, area="medium"
                ),
                self._summarize_metric(
                    coco_eval, ap=True, max_det=max_det, area="large"
                ),
                self._summarize_metric(coco_eval, ap=False, max_det=1),
                self._summarize_metric(coco_eval, ap=False, max_det=10),
                self._summarize_metric(coco_eval, ap=False, max_det=max_det),
                self._summarize_metric(
                    coco_eval, ap=False, max_det=max_det, area="small"
                ),
                self._summarize_metric(
                    coco_eval, ap=False, max_det=max_det, area="medium"
                ),
                self._summarize_metric(
                    coco_eval, ap=False, max_det=max_det, area="large"
                ),
            ],
            dtype=np.float64,
        )
