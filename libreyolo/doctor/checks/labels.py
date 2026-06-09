"""Label-content checks (``labels.*``) for detection boxes.

All boxes in the snapshot are normalized ``cls, cx, cy, w, h`` rows; checks
here mirror exactly what training would consume.
"""

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..config import DoctorConfig
from ..report import Finding, Severity
from ..snapshot import DatasetSnapshot, SplitSnapshot
from . import register


def _per_split(snap: DatasetSnapshot) -> Iterator[SplitSnapshot]:
    yield from snap.splits


# IoU pair-finding builds (n, n) matrices; cap n so a single densely
# annotated image cannot exhaust memory (coverage beyond the cap is skipped).
_MAX_IOU_BOXES = 1500


def _plausible(boxes: np.ndarray, eps: float) -> np.ndarray:
    """Rows whose coords lie in [0, 1]. Geometry checks skip the rest:
    implausible rows are already reported by labels.coords_out_of_range."""
    coords = boxes[:, 1:5]
    return ((coords >= -eps) & (coords <= 1 + eps)).all(axis=1)


@register("labels.syntax")
def check_syntax(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    for split in _per_split(snap):
        bad: list[tuple[Path, str]] = []
        for r in split.records:
            for issue in r.label_issues:
                bad.append((r.label_path, f"line {issue.line_no}: {issue.reason}"))
        if bad:
            sample = "; ".join(f"{p.name} ({why})" for p, why in bad[:3])
            yield Finding(
                "labels.syntax",
                Severity.ERROR,
                f"{len(bad)} malformed label line(s), e.g. {sample}.",
                split=split.name,
                paths=_unique([p for p, _ in bad], cfg.max_examples),
                count=len(bad),
            )


@register("labels.polygon_line")
def check_polygon_line(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    """>5-field rows train fine (polygon extent becomes the box) but usually
    mean a segmentation export ended up in a detection dataset."""
    for split in _per_split(snap):
        offenders = [r.label_path for r in split.records if r.polygon_lines]
        n_rows = sum(r.polygon_lines for r in split.records)
        if n_rows:
            yield Finding(
                "labels.polygon_line",
                Severity.INFO,
                f"{n_rows} label line(s) have more than 5 fields and are read "
                "as polygons (the box is derived from their extent), exactly "
                "as training does. If this is most of the dataset, it is "
                "probably a segmentation export.",
                split=split.name,
                paths=offenders[: cfg.max_examples],
                count=n_rows,
            )


@register("labels.class_out_of_range")
def check_class_out_of_range(
    snap: DatasetSnapshot, cfg: DoctorConfig
) -> Iterator[Finding]:
    if snap.nc is None:
        return
    nc = snap.nc
    for split in _per_split(snap):
        offenders = []
        n_rows = 0
        for r in split.records:
            if not r.boxes.shape[0]:
                continue
            mask = (r.boxes[:, 0] < 0) | (r.boxes[:, 0] >= nc)
            if mask.any():
                offenders.append(r.label_path)
                n_rows += int(mask.sum())
        if offenders:
            yield Finding(
                "labels.class_out_of_range",
                Severity.ERROR,
                f"{n_rows} box(es) use class ids outside [0, {nc}).",
                split=split.name,
                paths=offenders[: cfg.max_examples],
                count=n_rows,
            )


@register("labels.coords_out_of_range")
def check_coords_out_of_range(
    snap: DatasetSnapshot, cfg: DoctorConfig
) -> Iterator[Finding]:
    eps = cfg.coord_tolerance
    for split in _per_split(snap):
        raw_offenders: list[Path] = []
        raw_rows = 0
        spill_offenders: list[Path] = []
        spill_rows = 0
        for r in split.records:
            if not r.boxes.shape[0]:
                continue
            coords = r.boxes[:, 1:5]
            raw_mask = ((coords < -eps) | (coords > 1 + eps)).any(axis=1)
            if raw_mask.any():
                raw_offenders.append(r.label_path)
                raw_rows += int(raw_mask.sum())
            cx, cy, w, h = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
            tol = cfg.box_spill_tolerance
            spill_mask = (
                (cx - w / 2 < -tol)
                | (cy - h / 2 < -tol)
                | (cx + w / 2 > 1 + tol)
                | (cy + h / 2 > 1 + tol)
            ) & ~raw_mask
            if spill_mask.any():
                spill_offenders.append(r.label_path)
                spill_rows += int(spill_mask.sum())
        if raw_offenders:
            yield Finding(
                "labels.coords_out_of_range",
                Severity.ERROR,
                f"{raw_rows} box(es) have coordinates outside [0, 1]; "
                "labels may be in pixel coordinates instead of normalized.",
                split=split.name,
                paths=raw_offenders[: cfg.max_examples],
                count=raw_rows,
            )
        if spill_offenders:
            yield Finding(
                "labels.coords_out_of_range",
                Severity.WARNING,
                f"{spill_rows} box(es) extend past the image edge "
                "(center +/- size/2 spills outside [0, 1]).",
                split=split.name,
                paths=spill_offenders[: cfg.max_examples],
                count=spill_rows,
            )


@register("labels.degenerate_box")
def check_degenerate_box(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    for split in _per_split(snap):
        offenders = []
        n_rows = 0
        for r in split.records:
            if not r.boxes.shape[0]:
                continue
            mask = (r.boxes[:, 3] <= 0) | (r.boxes[:, 4] <= 0)
            if mask.any():
                offenders.append(r.label_path)
                n_rows += int(mask.sum())
        if offenders:
            yield Finding(
                "labels.degenerate_box",
                Severity.ERROR,
                f"{n_rows} box(es) have zero or negative width/height.",
                split=split.name,
                paths=offenders[: cfg.max_examples],
                count=n_rows,
            )


@register("labels.tiny_object")
def check_tiny_object(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    """Boxes that shrink below ~cfg.tiny_box_px pixels at the training size."""
    for split in _per_split(snap):
        offenders = []
        n_rows = 0
        total = 0
        for r in split.records:
            if not r.boxes.shape[0]:
                continue
            total += r.boxes.shape[0]
            w, h = r.boxes[:, 3], r.boxes[:, 4]
            if r.scan is not None and r.scan.ok and r.scan.width and r.scan.height:
                # Exact letterbox scale: long side maps to imgsz.
                scale = cfg.imgsz / max(r.scan.width, r.scan.height)
                px_w = w * r.scan.width * scale
                px_h = h * r.scan.height * scale
            else:
                px_w = w * cfg.imgsz
                px_h = h * cfg.imgsz
            mask = (
                (np.minimum(px_w, px_h) < cfg.tiny_box_px)
                & (w > 0)
                & (h > 0)
                & _plausible(r.boxes, cfg.coord_tolerance)
            )
            if mask.any():
                offenders.append(r.label_path)
                n_rows += int(mask.sum())
        if n_rows:
            pct = 100.0 * n_rows / max(total, 1)
            yield Finding(
                "labels.tiny_object",
                Severity.WARNING,
                f"{n_rows} box(es) ({pct:.1f}%) are smaller than "
                f"{cfg.tiny_box_px:g} px at imgsz={cfg.imgsz}; they contribute "
                "almost nothing to training.",
                split=split.name,
                paths=offenders[: cfg.max_examples],
                count=n_rows,
                details={"imgsz": cfg.imgsz, "threshold_px": cfg.tiny_box_px},
            )


@register("labels.huge_box")
def check_huge_box(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    for split in _per_split(snap):
        offenders = []
        n_rows = 0
        for r in split.records:
            if not r.boxes.shape[0]:
                continue
            area = r.boxes[:, 3] * r.boxes[:, 4]
            mask = (area > cfg.huge_box_area) & _plausible(r.boxes, cfg.coord_tolerance)
            if mask.any():
                offenders.append(r.label_path)
                n_rows += int(mask.sum())
        if offenders:
            yield Finding(
                "labels.huge_box",
                Severity.WARNING,
                f"{n_rows} box(es) cover more than "
                f"{cfg.huge_box_area:.0%} of the image.",
                split=split.name,
                paths=offenders[: cfg.max_examples],
                count=n_rows,
            )


@register("labels.extreme_aspect")
def check_extreme_aspect(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    for split in _per_split(snap):
        offenders = []
        n_rows = 0
        for r in split.records:
            if not r.boxes.shape[0]:
                continue
            w, h = r.boxes[:, 3], r.boxes[:, 4]
            threshold = cfg.extreme_box_aspect
            if r.scan is not None and r.scan.ok and r.scan.width and r.scan.height:
                w = w * r.scan.width
                h = h * r.scan.height
            else:
                # Normalized ratio differs from the pixel ratio by the image
                # aspect (typically <= 2:1); double the threshold so --fast
                # never over-reports, at the cost of missing borderline cases.
                threshold = cfg.extreme_box_aspect * 2
            valid = (w > 0) & (h > 0)
            ratio = np.where(
                valid, np.maximum(w, h) / np.where(valid, np.minimum(w, h), 1), 0
            )
            mask = (
                valid & (ratio > threshold) & _plausible(r.boxes, cfg.coord_tolerance)
            )
            if mask.any():
                offenders.append(r.label_path)
                n_rows += int(mask.sum())
        if offenders:
            reported = (
                cfg.extreme_box_aspect
                if snap.images_scanned
                else cfg.extreme_box_aspect * 2
            )
            yield Finding(
                "labels.extreme_aspect",
                Severity.WARNING,
                f"{n_rows} box(es) have an aspect ratio beyond "
                f"{reported:g}:1 (often annotation slips).",
                split=split.name,
                paths=offenders[: cfg.max_examples],
                count=n_rows,
            )


@register("labels.duplicate_box")
def check_duplicate_box(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    for split in _per_split(snap):
        offenders = []
        n_pairs = 0
        for r in split.records:
            boxes = r.boxes[_plausible(r.boxes, cfg.coord_tolerance)]
            if boxes.shape[0] < 2:
                continue
            pairs = _same_class_high_iou(boxes[:_MAX_IOU_BOXES], cfg.duplicate_box_iou)
            if pairs:
                offenders.append(r.label_path)
                n_pairs += pairs
        if offenders:
            yield Finding(
                "labels.duplicate_box",
                Severity.WARNING,
                f"{n_pairs} pair(s) of same-class boxes overlap with IoU > "
                f"{cfg.duplicate_box_iou:g} (likely double annotations).",
                split=split.name,
                paths=offenders[: cfg.max_examples],
                count=n_pairs,
            )


@register("labels.crowded_image")
def check_crowded_image(snap: DatasetSnapshot, cfg: DoctorConfig) -> Iterator[Finding]:
    for split in _per_split(snap):
        counts = np.array([r.boxes.shape[0] for r in split.records])
        if counts.size < 20 or counts.max() < cfg.crowded_min_objects:
            continue
        threshold = max(float(np.percentile(counts, 99)) * 3, cfg.crowded_min_objects)
        crowded = [r for r in split.records if r.boxes.shape[0] > threshold]
        if crowded:
            yield Finding(
                "labels.crowded_image",
                Severity.INFO,
                f"{len(crowded)} image(s) have an unusually high object count "
                f"(> {threshold:.0f}); worth a manual look.",
                split=split.name,
                paths=[r.path for r in crowded[: cfg.max_examples]],
                count=len(crowded),
            )


@register("labels.identical_files")
def check_identical_files(
    snap: DatasetSnapshot, cfg: DoctorConfig
) -> Iterator[Finding]:
    for split in _per_split(snap):
        groups: dict[str, list[Path]] = defaultdict(list)
        for r in split.records:
            if r.label_digest is not None and r.boxes.shape[0]:
                groups[r.label_digest].append(r.label_path)
        suspicious = [
            paths
            for paths in groups.values()
            if len(paths) >= cfg.identical_label_files
        ]
        if suspicious:
            biggest = max(suspicious, key=len)
            yield Finding(
                "labels.identical_files",
                Severity.WARNING,
                f"{len(suspicious)} group(s) of byte-identical non-empty label "
                f"files (largest: {len(biggest)} files) - often a copy-paste "
                "export error.",
                split=split.name,
                paths=biggest[: cfg.max_examples],
                count=sum(len(g) for g in suspicious),
            )


def _same_class_high_iou(boxes: np.ndarray, iou_thr: float) -> int:
    """Count same-class box pairs with IoU above the threshold."""
    cls = boxes[:, 0]
    cx, cy, w, h = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    areas = np.maximum(w, 0) * np.maximum(h, 0)

    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = areas[:, None] + areas[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)

    same_class = cls[:, None] == cls[None, :]
    upper = np.triu(np.ones_like(iou, dtype=bool), k=1)
    return int((same_class & upper & (iou > iou_thr)).sum())


def _unique(paths: list[Path], limit: int) -> list[Path]:
    seen: list[Path] = []
    for p in paths:
        if p not in seen:
            seen.append(p)
        if len(seen) >= limit:
            break
    return seen
