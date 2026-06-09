"""LibreDoctor — opt-in dataset health checks.

Verify a YOLO detection dataset before training::

    from libreyolo import doctor
    report = doctor.diagnose("data.yaml", imgsz=640)
    if report.errors:
        ...

Or from the CLI: ``libreyolo doctor data.yaml``. See
``docs/libredoctor_design.md`` for the check catalog.
"""

from .config import (
    DatasetNotFoundError,
    DoctorConfig,
    DoctorError,
    NotADetectionDatasetError,
    UnknownCheckError,
)
from .report import Finding, Report, Severity
from .runner import diagnose

__all__ = [
    "diagnose",
    "DatasetNotFoundError",
    "DoctorConfig",
    "DoctorError",
    "NotADetectionDatasetError",
    "UnknownCheckError",
    "Finding",
    "Report",
    "Severity",
]
