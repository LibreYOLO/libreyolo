"""Validation result visualisations for LibreYOLO."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_COLOR_GT = (50, 200, 50)    # BGR green  — ground-truth
_COLOR_PRED = (30, 80, 220)  # BGR red-ish — predictions
_ALPHA_MASK = 0.35           # mask overlay opacity

# Grouped metric definitions for the bar chart (lookup key is lowercase)
_METRIC_GROUPS = [
    ("mAP",       "#4C8EDA",
     [("map50-95", "mAP50-95"), ("map50", "mAP50"), ("map75", "mAP75"),
      ("map_small", "S"), ("map_medium", "M"), ("map_large", "L")]),
    ("AR",        "#F5A623",
     [("ar1", "AR@1"), ("ar10", "AR@10"), ("ar100", "AR@100"),
      ("ar_small", "S"), ("ar_medium", "M"), ("ar_large", "L")]),
    ("Precision", "#6BAE6D",
     [("p50-95", "P@50-95"), ("p50", "P@50"), ("p75", "P@75")]),
    ("Recall",    "#A87DC8",
     [("r50-95", "R@50-95"), ("r50", "R@50"), ("r75", "R@75")]),
]


# ---------------------------------------------------------------------------
# IoU helper (numpy-only, no torch dependency)
# ---------------------------------------------------------------------------

def _box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Vectorised xyxy IoU → (M, N)."""
    a1 = (boxes1[:, 2] - boxes1[:, 0]).clip(0) * (boxes1[:, 3] - boxes1[:, 1]).clip(0)
    a2 = (boxes2[:, 2] - boxes2[:, 0]).clip(0) * (boxes2[:, 3] - boxes2[:, 1]).clip(0)
    ix1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    iy1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    ix2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    iy2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    union = a1[:, None] + a2[None, :] - inter
    return inter / np.maximum(union, 1e-7)


# ---------------------------------------------------------------------------
# Confusion-matrix accumulator
# ---------------------------------------------------------------------------

class ConfusionMatrix:
    """Accumulates per-image prediction/GT assignments for a normalised confusion matrix.

    The last row/column (index nc) represents the background class
    (missed detections / false positives).
    """

    def __init__(self, nc: int, iou_thres: float = 0.5, conf_thres: float = 0.15) -> None:
        self.nc = nc
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        # matrix[actual, predicted]; nc == background
        self.matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)

    def process_image(
        self,
        pred_boxes: np.ndarray,    # (N, 4) xyxy pixel coords
        pred_classes: np.ndarray,  # (N,) int
        pred_scores: np.ndarray,   # (N,) float
        gt_boxes: np.ndarray,      # (M, 4) xyxy pixel coords
        gt_classes: np.ndarray,    # (M,) int
    ) -> None:
        conf_mask = pred_scores >= self.conf_thres
        pred_boxes = pred_boxes[conf_mask]
        pred_classes = pred_classes[conf_mask]

        n_pred, n_gt = len(pred_boxes), len(gt_boxes)

        if n_pred == 0 and n_gt == 0:
            return
        if n_pred == 0:
            for gc in gt_classes:
                if int(gc) < self.nc:
                    self.matrix[int(gc), self.nc] += 1
            return
        if n_gt == 0:
            for pc in pred_classes:
                if int(pc) < self.nc:
                    self.matrix[self.nc, int(pc)] += 1
            return

        iou = _box_iou_numpy(gt_boxes, pred_boxes)  # (M, N)
        gt_idxs, pred_idxs = np.where(iou >= self.iou_thres)

        matched_gt: set = set()
        matched_pred: set = set()

        if len(gt_idxs):
            order = np.argsort(-iou[gt_idxs, pred_idxs])
            for gi, pi in zip(gt_idxs[order], pred_idxs[order]):
                gi, pi = int(gi), int(pi)
                if gi in matched_gt or pi in matched_pred:
                    continue
                matched_gt.add(gi)
                matched_pred.add(pi)
                gc_i, pc_i = int(gt_classes[gi]), int(pred_classes[pi])
                if gc_i < self.nc and pc_i < self.nc:
                    self.matrix[gc_i, pc_i] += 1

        for i, gc in enumerate(gt_classes):
            if i not in matched_gt and int(gc) < self.nc:
                self.matrix[int(gc), self.nc] += 1
        for j, pc in enumerate(pred_classes):
            if j not in matched_pred and int(pc) < self.nc:
                self.matrix[self.nc, int(pc)] += 1


# ---------------------------------------------------------------------------
# ValPlotter
# ---------------------------------------------------------------------------

class ValPlotter:
    """Saves validation result visualisations to a plots/ subdirectory."""

    # ------------------------------------------------------------------ #
    # Metrics summary — grouped subplots (mAP | AR | P/R)
    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_metrics_bar(
        metrics: Dict[str, float],
        save_path: Path,
        title: str = "Metrics",
    ) -> None:
        plt = ValPlotter._require_matplotlib()

        def _norm(k: str) -> str:
            return (
                k.lower()
                .replace("metrics/", "")
                .replace("(b)", "")
                .replace("(m)", "")
                .strip()
            )

        lookup = {_norm(k): float(v) for k, v in metrics.items() if "speed" not in k}

        # Build active groups (only those with at least one matching key)
        active = []
        for grp_name, grp_color, key_pairs in _METRIC_GROUPS:
            entries = [(lbl, lookup[lk]) for lk, lbl in key_pairs if lk in lookup]
            if entries:
                active.append((grp_name, grp_color, entries))

        if not active:
            return

        ncols = len(active)
        fig, axes = plt.subplots(
            1, ncols,
            figsize=(4.2 * ncols, max(3.5, max(len(e) for _, _, e in active) * 0.45 + 1.5)),
            squeeze=False,
        )
        fig.suptitle(title, fontsize=12, fontweight="bold")

        for col, (grp_name, color, entries) in enumerate(active):
            ax = axes[0, col]
            labels = [lbl for lbl, _ in entries]
            vals = [v for _, v in entries]
            y_pos = range(len(entries))

            ax.barh(y_pos, vals, color=color, edgecolor="white", linewidth=0.7, alpha=0.88)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlim(0, 1.0)
            ax.set_title(grp_name, fontsize=10, fontweight="bold")
            ax.set_xlabel("Score", fontsize=8)
            ax.grid(axis="x", alpha=0.2, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.invert_yaxis()

            for y, val in zip(y_pos, vals):
                ax.text(
                    min(val + 0.012, 0.98), y, f"{val:.3f}",
                    va="center", ha="left", fontsize=8,
                )

        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved metrics chart → %s", save_path)

    # ------------------------------------------------------------------ #
    # Per-class AP bar chart — sorted descending
    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_per_class_ap(
        coco_eval,
        class_names: List[str],
        save_path: Path,
        label: str = "Box",
    ) -> None:
        plt = ValPlotter._require_matplotlib()

        if coco_eval is None or not hasattr(coco_eval, "eval") or coco_eval.eval is None:
            logger.warning("COCOeval not available — skipping per-class AP chart")
            return

        precision = coco_eval.eval.get("precision")  # (T, R, K, A, M)
        if precision is None:
            return

        cat_ids = list(coco_eval.params.catIds)
        nc = len(cat_ids)
        ap_vals, ap_names = [], []
        for k in range(nc):
            p = precision[:, :, k, 0, -1]
            valid = p[p > -1]
            ap_vals.append(float(valid.mean()) if len(valid) else 0.0)
            name = class_names[k] if class_names and k < len(class_names) else f"cls{cat_ids[k]}"
            ap_names.append(name)

        if not ap_vals:
            return

        # Sort descending by AP
        order = np.argsort(ap_vals)[::-1]
        ap_vals = [ap_vals[i] for i in order]
        ap_names = [ap_names[i] for i in order]

        fig_h = max(4, nc * 0.32 + 1.5)
        fig, ax = plt.subplots(figsize=(7, fig_h))
        cmap = plt.get_cmap("RdYlGn")
        bars = ax.barh(
            range(nc), ap_vals,
            color=[cmap(v) for v in ap_vals],
            edgecolor="white", linewidth=0.5,
        )
        ax.set_yticks(range(nc))
        ax.set_yticklabels(ap_names, fontsize=max(6, 9 - nc // 15))
        ax.set_xlim(0, 1.08)
        ax.set_title(f"Per-class mAP50-95 ({label}) — sorted", fontsize=11, fontweight="bold")
        ax.set_xlabel("mAP50-95")
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.invert_yaxis()

        fs_val = max(6, 8 - nc // 20)
        for bar, val in zip(bars, ap_vals):
            ax.text(
                val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=fs_val,
            )

        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved per-class AP chart → %s", save_path)

    # ------------------------------------------------------------------ #
    # Per-class Recall bar chart — sorted descending
    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_per_class_recall(
        coco_eval,
        class_names: List[str],
        save_path: Path,
        label: str = "Box",
    ) -> None:
        plt = ValPlotter._require_matplotlib()

        if coco_eval is None or not hasattr(coco_eval, "eval") or coco_eval.eval is None:
            logger.warning("COCOeval not available — skipping per-class recall chart")
            return

        recall = coco_eval.eval.get("recall")  # (T, K, A, M)
        if recall is None:
            return

        cat_ids = list(coco_eval.params.catIds)
        nc = len(cat_ids)
        # IoU=0.5 (index 0), all areas (index 0), maxDet=100 (index -1)
        rec_vals, rec_names = [], []
        for k in range(nc):
            r = recall[0, k, 0, -1]
            rec_vals.append(float(r) if r > -1 else 0.0)
            name = class_names[k] if class_names and k < len(class_names) else f"cls{cat_ids[k]}"
            rec_names.append(name)

        if not rec_vals:
            return

        order = np.argsort(rec_vals)[::-1]
        rec_vals = [rec_vals[i] for i in order]
        rec_names = [rec_names[i] for i in order]

        fig_h = max(4, nc * 0.32 + 1.5)
        fig, ax = plt.subplots(figsize=(7, fig_h))
        cmap = plt.get_cmap("RdYlGn")
        bars = ax.barh(
            range(nc), rec_vals,
            color=[cmap(v) for v in rec_vals],
            edgecolor="white", linewidth=0.5,
        )
        ax.set_yticks(range(nc))
        ax.set_yticklabels(rec_names, fontsize=max(6, 9 - nc // 15))
        ax.set_xlim(0, 1.08)
        ax.set_title(f"Per-class Recall@IoU=0.5 ({label}) — sorted", fontsize=11, fontweight="bold")
        ax.set_xlabel("Recall")
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.invert_yaxis()

        fs_val = max(6, 8 - nc // 20)
        for bar, val in zip(bars, rec_vals):
            ax.text(
                val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=fs_val,
            )

        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved per-class recall chart → %s", save_path)

    # ------------------------------------------------------------------ #
    # Confusion matrix heatmap
    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_confusion_matrix(
        matrix: np.ndarray,
        class_names: List[str],
        save_path: Path,
        normalize: bool = True,
    ) -> None:
        plt = ValPlotter._require_matplotlib()

        nc = len(class_names)
        labels = list(class_names) + ["background"]

        disp = matrix.astype(float)
        if normalize:
            row_sums = disp.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            disp = disp / row_sums

        side = max(6, (nc + 1) * 0.5 + 1.5)
        fig, ax = plt.subplots(figsize=(side, side * 0.85))
        im = ax.imshow(disp, interpolation="nearest", cmap="Blues",
                       vmin=0, vmax=1.0 if normalize else None)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ticks = np.arange(nc + 1)
        fs = max(5, 9 - (nc + 1) // 8)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs)
        ax.set_yticklabels(labels, fontsize=fs)

        # Fixed threshold: 0.5 for normalised (values in [0,1]),
        # or 30% of max for raw counts.
        thresh = 0.5 if normalize else disp.max() * 0.3
        show_cells = (nc + 1) <= 30  # skip text for very large matrices
        fs_cell = max(4, 7 - (nc + 1) // 10)
        if show_cells:
            for i in range(nc + 1):
                for j in range(nc + 1):
                    v = disp[i, j]
                    txt = f"{v:.2f}" if normalize else str(int(matrix[i, j]))
                    ax.text(
                        j, i, txt,
                        ha="center", va="center",
                        color="white" if v > thresh else "black",
                        fontsize=fs_cell,
                    )

        ax.set_ylabel("Actual", fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10)
        suffix = " (normalised)" if normalize else ""
        ax.set_title(f"Confusion Matrix{suffix}", fontsize=11, fontweight="bold")
        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved confusion matrix → %s", save_path)

    # ------------------------------------------------------------------ #
    # Precision–Recall, P–Conf and R–Conf curves
    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_pr_curves(
        coco_eval,
        class_names: List[str],
        save_dir: Path,
        label: str = "box",
    ) -> None:
        """Save three plots to save_dir using pycocotools eval data (IoU = 0.5).

        Files produced:
          pr_curve_{label}.png
          precision_conf_{label}.png
          recall_conf_{label}.png

        Mean across classes is drawn as a thick black line.
        Per-class lines are thin; legend is shown only when nc ≤ 5.
        """
        plt = ValPlotter._require_matplotlib()

        if coco_eval is None or not hasattr(coco_eval, "eval") or coco_eval.eval is None:
            logger.warning("COCOeval not available — skipping PR curves")
            return

        prec_arr = coco_eval.eval.get("precision")  # (T, R, K, A, M)
        score_arr = coco_eval.eval.get("scores")     # (T, R, K, A, M)
        if prec_arr is None:
            return

        # IoU=0.5 (index 0), all areas (index 0), maxDet=100 (index -1)
        p_mat = prec_arr[0, :, :, 0, -1].copy().astype(float)   # (101, K)
        s_mat = (score_arr[0, :, :, 0, -1].copy().astype(float)
                 if score_arr is not None else None)

        p_mat[p_mat < 0] = np.nan
        if s_mat is not None:
            s_mat[s_mat < 0] = np.nan

        rec_thrs = np.linspace(0.0, 1.0, 101)
        nc = p_mat.shape[1]
        names = class_names or [str(i) for i in range(nc)]
        show_legend = nc <= 5

        cmap_cls = plt.get_cmap("tab20" if nc > 10 else "tab10")

        p_mean = np.nanmean(p_mat, axis=1)   # (101,)
        s_mean = (np.nanmean(s_mat, axis=1)  # (101,)
                  if s_mat is not None else None)

        save_dir.mkdir(parents=True, exist_ok=True)

        def _base_ax(xlabel: str, ylabel: str, ttl: str):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(ttl, fontsize=11, fontweight="bold")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.2, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            return fig, ax

        # -------- 1. Precision–Recall ----------------------------------------
        fig, ax = _base_ax(
            "Recall", "Precision",
            f"Precision–Recall ({label.upper()}, IoU=0.5)",
        )
        for k in range(nc):
            v = ~np.isnan(p_mat[:, k])
            if v.any():
                ax.plot(
                    rec_thrs[v], p_mat[v, k],
                    color=cmap_cls(k % 20 / 20),
                    linewidth=0.8, alpha=0.45,
                    label=names[k] if show_legend else None,
                )
        valid_m = ~np.isnan(p_mean)
        ax.plot(
            rec_thrs[valid_m], p_mean[valid_m],
            color="black", linewidth=2.5, zorder=5,
            label=f"mean  AP={np.nanmean(p_mean):.3f}",
        )
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(save_dir / f"pr_curve_{label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved PR curve → %s", save_dir / f"pr_curve_{label}.png")

        if s_mat is None:
            logger.warning("'scores' not in COCOeval — skipping P-conf and R-conf plots")
            return

        # -------- 2. Precision–Confidence ------------------------------------
        fig, ax = _base_ax(
            "Confidence threshold", "Precision",
            f"Precision vs Confidence ({label.upper()}, IoU=0.5)",
        )
        for k in range(nc):
            v = ~np.isnan(p_mat[:, k]) & ~np.isnan(s_mat[:, k])
            if v.any():
                s_k = s_mat[v, k]
                p_k = p_mat[v, k]
                order = np.argsort(s_k)
                ax.plot(
                    s_k[order], p_k[order],
                    color=cmap_cls(k % 20 / 20),
                    linewidth=0.8, alpha=0.45,
                    label=names[k] if show_legend else None,
                )
        v_m = ~np.isnan(p_mean) & ~np.isnan(s_mean)
        if v_m.any():
            om = np.argsort(s_mean[v_m])
            ax.plot(
                s_mean[v_m][om], p_mean[v_m][om],
                color="black", linewidth=2.5, zorder=5,
                label="mean",
            )
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(save_dir / f"precision_conf_{label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved P-conf curve → %s", save_dir / f"precision_conf_{label}.png")

        # -------- 3. Recall–Confidence ---------------------------------------
        fig, ax = _base_ax(
            "Confidence threshold", "Recall",
            f"Recall vs Confidence ({label.upper()}, IoU=0.5)",
        )
        for k in range(nc):
            v = ~np.isnan(s_mat[:, k])
            if v.any():
                s_k = s_mat[v, k]
                r_k = rec_thrs[v]
                order = np.argsort(s_k)
                ax.plot(
                    s_k[order], r_k[order],
                    color=cmap_cls(k % 20 / 20),
                    linewidth=0.8, alpha=0.45,
                    label=names[k] if show_legend else None,
                )
        v_m2 = ~np.isnan(s_mean)
        if v_m2.any():
            om2 = np.argsort(s_mean[v_m2])
            ax.plot(
                s_mean[v_m2][om2], rec_thrs[v_m2][om2],
                color="black", linewidth=2.5, zorder=5,
                label="mean",
            )
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(save_dir / f"recall_conf_{label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved R-conf curve → %s", save_dir / f"recall_conf_{label}.png")

    # ------------------------------------------------------------------ #
    # Single validation-sample image — GT left | Predictions right
    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_val_sample(
        img_bgr: np.ndarray,
        gt_boxes: np.ndarray,       # (M, 4) xyxy float
        gt_classes: np.ndarray,     # (M,) int
        pred_boxes: np.ndarray,     # (N, 4) xyxy float
        pred_classes: np.ndarray,   # (N,) int
        pred_scores: np.ndarray,    # (N,) float
        class_names: Optional[List[str]],
        save_path: Path,
        pred_masks: Optional[np.ndarray] = None,  # (N, H, W) original pixel space
        gt_masks: Optional[np.ndarray] = None,    # (M, H, W) original pixel space
        conf_thres: float = 0.25,
    ) -> None:
        """Save a side-by-side composite: Ground Truth (left) | Predictions (right)."""
        cv2 = ValPlotter._require_cv2()

        h, w = img_bgr.shape[:2]

        def _cls_name(c: int) -> str:
            if class_names and 0 <= c < len(class_names):
                return class_names[c]
            return str(c)

        def _put_label(canvas, text: str, x1: int, y: int, color: tuple, above: bool) -> None:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            if above:
                y0 = max(y - th - 3, 0)
                cv2.rectangle(canvas, (x1, y0), (x1 + tw + 4, y0 + th + 3), color, -1)
                cv2.putText(canvas, text, (x1 + 2, y0 + th),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                y1_end = min(y + th + 3, h - 1)
                cv2.rectangle(canvas, (x1, y), (x1 + tw + 4, y1_end), color, -1)
                cv2.putText(canvas, text, (x1 + 2, y + th),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # ---- Left panel: Ground Truth ----
        img_gt = img_bgr.copy()
        if gt_masks is not None and len(gt_masks):
            overlay = img_gt.copy()
            for m in gt_masks:
                if m.ndim != 2:
                    m = m[0]
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                overlay[m > 0] = _COLOR_GT
            cv2.addWeighted(overlay, _ALPHA_MASK, img_gt, 1 - _ALPHA_MASK, 0, img_gt)
        for i, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(img_gt, (x1, y1), (x2, y2), _COLOR_GT, 2)
            _put_label(img_gt, _cls_name(int(gt_classes[i])), x1, y1, _COLOR_GT, above=True)

        # ---- Right panel: Predictions ----
        img_pred = img_bgr.copy()

        if pred_masks is not None and len(pred_masks):
            overlay = img_pred.copy()
            for idx, mask in enumerate(pred_masks):
                if idx < len(pred_scores) and pred_scores[idx] < conf_thres:
                    continue
                m = mask if mask.ndim == 2 else mask[0]
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                overlay[m > 0] = _COLOR_PRED
            cv2.addWeighted(overlay, _ALPHA_MASK, img_pred, 1 - _ALPHA_MASK, 0, img_pred)

        for i, box in enumerate(pred_boxes):
            if pred_scores[i] < conf_thres:
                continue
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(img_pred, (x1, y1), (x2, y2), _COLOR_PRED, 2)
            text = f"{_cls_name(int(pred_classes[i]))} {pred_scores[i]:.2f}"
            _put_label(img_pred, text, x1, y2, _COLOR_PRED, above=False)

        # ---- Header bar with panel labels ----
        header_h = 26
        header = np.full((header_h, w * 2 + 4, 3), 35, dtype=np.uint8)
        cv2.putText(header, "Ground Truth", (8, header_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, _COLOR_GT, 1, cv2.LINE_AA)
        cv2.putText(header, "Predictions", (w + 12, header_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, _COLOR_PRED, 1, cv2.LINE_AA)

        # ---- Compose ----
        divider = np.full((h, 4, 3), 90, dtype=np.uint8)
        row = np.hstack([img_gt, divider, img_pred])
        combined = np.vstack([header, row])

        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), combined)
        logger.info("Saved val sample → %s", save_path)

    # ------------------------------------------------------------------ #
    # Dependency helpers (lazy — not imported at module level)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _require_matplotlib():
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
            return plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for save_plots. "
                "Install with: pip install matplotlib"
            )

    @staticmethod
    def _require_cv2():
        try:
            import cv2  # noqa: PLC0415
            return cv2
        except ImportError:
            raise ImportError(
                "opencv-python is required for sample image plots. "
                "Install with: pip install opencv-python"
            )
