"""Configuration and errors for LibreDoctor."""

from dataclasses import dataclass
from typing import Optional


class DoctorError(Exception):
    """Doctor could not run (bad YAML, unreadable dataset, bad selection)."""


class NotADetectionDatasetError(DoctorError):
    """Format guard: the dataset does not look like a YOLO detection dataset."""

    def __init__(self, suspected: str) -> None:
        self.suspected = suspected
        super().__init__(
            f"This looks like a {suspected} dataset. "
            "LibreDoctor currently supports YOLO detection datasets only "
            "(label lines of the form 'class cx cy w h')."
        )


class UnknownCheckError(DoctorError):
    """A --skip/--only selector matched no registered check."""


@dataclass
class DoctorConfig:
    """Thresholds for doctor checks. CLI exposes only imgsz; the Python API
    accepts a full instance for fine-grained control."""

    # Target training size, used to express "tiny object" in on-target pixels.
    imgsz: int = 640

    # labels.*
    tiny_box_px: float = 3.0  # min box side at imgsz below which a box is useless
    huge_box_area: float = 0.95  # fraction of image area
    extreme_box_aspect: float = 50.0  # w/h or h/w beyond this is a sliver
    duplicate_box_iou: float = 0.95  # same-class IoU above this is a double annotation
    box_spill_tolerance: float = 1e-3  # allowed overhang of cx +/- w/2 past [0, 1]
    coord_tolerance: float = 1e-6  # float slack for raw coords in [0, 1]
    crowded_min_objects: int = 50  # never flag images below this many objects
    identical_label_files: int = 5  # byte-identical label files to warrant a warning

    # balance.*
    few_instances: int = 10
    imbalance_warn_ratio: float = 100.0  # max/min instance ratio
    background_warn_ratio: float = 0.5  # fraction of images with no labels
    split_skew_points: float = 0.10  # train-vs-val class share difference

    # images.*
    min_image_side: int = 32
    extreme_image_aspect: float = 20.0
    near_duplicate_distance: int = 4  # max dHash Hamming distance
    uniform_pixel_range: int = 2  # max-min on a tiny grayscale thumb

    # reporting / execution
    max_examples: int = 20  # offending paths stored per finding
    workers: Optional[int] = None  # image-scan threads (None = auto)
