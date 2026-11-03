"""Optical flow validator for LibreYOLO.

Scores a flow model on consecutive-frame pairs with dense ground truth
(:class:`~libreyolo.data.flow_dataset.FlowDataset`), reusing the
:class:`BaseValidator` template (setup, iterate, finalize).

Each pair goes through the family's own ``predict`` so the prediction lands
on the original frame canvas exactly as users see it, and is compared with
the ground truth there. Metrics are averaged per pair, then over pairs:

- ``metrics/epe``: mean end-point error, the Euclidean distance in pixels
  between predicted and true vectors over valid pixels (lower is better).
- ``metrics/px1``, ``metrics/px3``: fraction of valid pixels whose error is
  under 1 px and under 3 px (higher is better).
- ``metrics/fl_all``: the KITTI outlier rate, the fraction of valid pixels
  whose error is over 3 px and over 5 percent of the ground-truth magnitude
  (lower is better).

``fitness`` is ``metrics/px1``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from ..data.flow_dataset import FlowDataset, flow_collate_fn, resolve_flow_data
from .base import BaseValidator

logger = logging.getLogger(__name__)

_METRIC_KEYS = ("metrics/epe", "metrics/px1", "metrics/px3", "metrics/fl_all")


def flow_pair_metrics(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> Dict[str, float]:
    """Per-pair flow metrics over valid pixels.

    ``pred`` and ``target`` are ``(H, W, 2)`` in pixels; ``valid`` is a
    boolean ``(H, W)`` mask. Returns an empty dict when no pixel is valid.
    """
    pred = pred.detach().cpu().float()
    target = target.detach().cpu().float()
    valid = valid.detach().cpu().bool()
    if pred.shape != target.shape:
        raise ValueError(
            f"Predicted flow shape {tuple(pred.shape)} does not match ground "
            f"truth shape {tuple(target.shape)}."
        )
    if int(valid.sum()) == 0:
        return {}
    error = torch.linalg.norm(pred - target, dim=-1)[valid].double()
    magnitude = torch.linalg.norm(target, dim=-1)[valid].double()
    outlier = (error > 3.0) & (error > 0.05 * magnitude)
    return {
        "metrics/epe": float(error.mean()),
        "metrics/px1": float((error < 1.0).double().mean()),
        "metrics/px3": float((error < 3.0).double().mean()),
        "metrics/fl_all": float(outlier.double().mean()),
    }


class FlowValidator(BaseValidator):
    """EPE, px1, px3 and Fl-all validator for the flow task."""

    task = "flow"

    def _setup_dataloader(self) -> DataLoader:
        if not self.config.data:
            raise ValueError("Flow validation requires data= (a dataset YAML).")
        data_config = resolve_flow_data(
            self.config.data,
            allow_scripts=getattr(self.config, "allow_download_scripts", False),
        )
        dataset = FlowDataset(data_config, split=self.config.split or "val")
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=flow_collate_fn,
        )

    def _warmup_model(self, n_warmup: int = 3) -> None:
        # Flow families run through predict(), which warms its own path; a
        # dummy single-image forward would not even match their input.
        return None

    def _init_metrics(self) -> None:
        self._metric_sums = {key: 0.0 for key in _METRIC_KEYS}
        self._pair_count = 0

    def _preprocess_batch(self, batch: Any) -> tuple:
        imgs_a, imgs_b, flows, valids, infos, ids = batch
        return imgs_a, (flows, valids), infos, ids

    def _predict_pair(self, info: Dict[str, Any]) -> torch.Tensor:
        """Run the family on one pair and return its ``(H, W, 2)`` flow."""
        results = self.model.predict([info["img_path"], info["img_path_next"]])
        if not isinstance(results, (list, tuple)) or not results:
            raise ValueError(
                "Flow validation expects predict([frame, next_frame]) to return "
                "a one-element list of Results."
            )
        flow = getattr(results[0], "flow", None)
        if flow is None:
            raise ValueError(
                f"{type(self.model).__name__} returned no Results.flow for a "
                "frame pair; flow validation needs a flow-task model."
            )
        return torch.as_tensor(flow.data)

    def _run_validation(self) -> None:
        self.model.model.eval()
        if self.config.augment:
            self._run_validation_augmented()
            return

        total_start = time.time()
        with torch.no_grad():
            for batch in self.dataloader:
                _, targets, infos, ids = self._preprocess_batch(batch)
                t_inference = time.time()
                preds = [self._predict_pair(info) for info in infos]
                self.speed["inference"] += time.time() - t_inference
                self._update_metrics(preds, targets, infos, ids)
                self.seen += len(infos)
        self.speed["total"] = time.time() - total_start

    def _postprocess_predictions(self, preds: Any, batch: Any) -> Any:
        return preds

    def _update_metrics(
        self, preds: Any, targets: Any, img_info: Any, img_ids: Any = None
    ) -> None:
        flows, valids = targets
        for index, pred in enumerate(preds):
            target = flows[index].permute(1, 2, 0)
            valid = valids[index]
            pair = flow_pair_metrics(pred, target, valid)
            if not pair:
                continue
            for key in _METRIC_KEYS:
                self._metric_sums[key] += pair[key]
            self._pair_count += 1

    def _compute_metrics(self) -> Dict[str, float]:
        if self._pair_count == 0:
            raise ValueError("Flow validation found no pair with a valid pixel.")
        count = float(self._pair_count)
        metrics = {key: self._metric_sums[key] / count for key in _METRIC_KEYS}
        metrics["fitness"] = metrics["metrics/px1"]
        return metrics

    def _print_results(self, metrics: Dict[str, float]) -> None:
        logger.info("=" * 50)
        logger.info("Optical Flow Validation Results")
        logger.info("=" * 50)
        logger.info("  EPE:    %.4f px", metrics.get("metrics/epe", 0.0))
        logger.info("  px1:    %.4f", metrics.get("metrics/px1", 0.0))
        logger.info("  px3:    %.4f", metrics.get("metrics/px3", 0.0))
        logger.info("  Fl-all: %.4f", metrics.get("metrics/fl_all", 0.0))
        logger.info("=" * 50)


__all__ = ["FlowValidator", "flow_pair_metrics"]
