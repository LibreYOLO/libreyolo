"""Orchestration for LibreDoctor: build the snapshot, run the checks."""

import logging
from typing import Iterable, Optional

from .checks import needs_image_scan, select_checks
from .config import DoctorConfig, NotADetectionDatasetError
from .report import Report, sorted_findings
from .snapshot import build_snapshot, detect_non_detection, scan_images

logger = logging.getLogger(__name__)


def diagnose(
    data: str,
    *,
    imgsz: Optional[int] = None,
    fast: bool = False,
    skip: Iterable[str] = (),
    only: Iterable[str] = (),
    config: Optional[DoctorConfig] = None,
    progress: bool = True,
    autodownload: bool = False,
) -> Report:
    """Run dataset health checks and return a :class:`Report`.

    Args:
        data: Dataset YAML name or path (same resolution as train/val).
        imgsz: Target training size for pixel-based thresholds (default 640).
        fast: Skip the image-decoding pass (no corruption/duplicate/leakage
            checks); label and balance checks still run.
        skip: Check ids or families to exclude (e.g. ``["images.near_duplicates"]``).
        only: Restrict to these check ids or families.
        config: Full threshold control; overrides ``imgsz``.
        progress: Show a progress bar during the image scan.
        autodownload: Allow the dataset YAML's URL download (never scripts).

    Raises:
        DoctorError: The dataset could not be scanned at all.
        NotADetectionDatasetError: The dataset is pose/segment/obb shaped.
    """
    cfg = config or DoctorConfig()
    if config is None and imgsz is not None:
        cfg.imgsz = imgsz

    selected, skipped = select_checks(skip=skip, only=only)

    snapshot = build_snapshot(data, autodownload=autodownload)

    suspected = detect_non_detection(snapshot)
    if suspected is not None:
        raise NotADetectionDatasetError(suspected)

    if fast:
        for cid in sorted(selected):
            if needs_image_scan(cid):
                selected.pop(cid)
                skipped.append(cid)
    elif any(needs_image_scan(cid) for cid in selected):
        scan_images(snapshot, workers=cfg.workers, progress=progress)

    findings = []
    for cid, fn in selected.items():
        try:
            findings.extend(fn(snapshot, cfg))
        except Exception:
            logger.exception("doctor check %s failed; skipping it", cid)
            skipped.append(cid)

    return Report(
        findings=sorted_findings(findings),
        stats=snapshot.stats(),
        skipped_checks=sorted(skipped),
    )
