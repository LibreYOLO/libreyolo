"""Shared COCO-style pose metadata and OKS helpers.

Clean-room metadata facts come from the public COCO keypoints specification.
The helper centralizes values already used by in-repo pose implementations so
new families do not duplicate COCO-17 constants.
"""

from __future__ import annotations

COCO17_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO17_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

COCO17_SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]

COCO17_OKS_SIGMAS = [
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
]


def default_oks_sigmas(num_keypoints: int) -> list[float]:
    """Return COCO-17 OKS sigmas or a uniform fallback for custom skeletons."""
    num_keypoints = int(num_keypoints)
    if num_keypoints == 17:
        return list(COCO17_OKS_SIGMAS)
    if num_keypoints < 1:
        raise ValueError(f"num_keypoints must be >= 1, got {num_keypoints}")
    return [1.0 / num_keypoints] * num_keypoints


def keypoints_per_class(data_cfg: dict, nc: int, num_keypoints: int) -> list[int]:
    """Return the keypoint count of each dataset class.

    Every class shares the ``kpt_shape`` skeleton of ``num_keypoints`` rows. A
    ``kpt_names`` mapping keyed by class index or class name narrows a class to
    the first ``len(names)`` rows, and an empty list declares a class without
    keypoints. Classes not listed keep all ``num_keypoints`` rows; a
    ``kpt_names`` list, or no ``kpt_names``, applies to every class.
    """
    nc = int(nc)
    num_keypoints = int(num_keypoints)
    counts = [num_keypoints] * nc
    kpt_names = data_cfg.get("kpt_names")
    if not isinstance(kpt_names, dict):
        return counts

    names = data_cfg.get("names") or {}
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    name_to_idx = {str(name): int(idx) for idx, name in names.items()}

    for key, class_kpt_names in kpt_names.items():
        if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
            class_idx = int(key)
        elif str(key) in name_to_idx:
            class_idx = name_to_idx[str(key)]
        else:
            raise ValueError(
                f"kpt_names key {key!r} is neither a class index nor a class name"
            )
        if not 0 <= class_idx < nc:
            raise ValueError(
                f"kpt_names key {key!r} is outside the dataset classes 0..{nc - 1}"
            )
        if class_kpt_names is None:
            class_kpt_names = []
        if not isinstance(class_kpt_names, (list, tuple)):
            raise ValueError(
                f"kpt_names[{key!r}] must be a list of keypoint names, "
                f"got {type(class_kpt_names).__name__}"
            )
        if len(class_kpt_names) > num_keypoints:
            raise ValueError(
                f"kpt_names[{key!r}] lists {len(class_kpt_names)} keypoints but "
                f"kpt_shape allows {num_keypoints}"
            )
        counts[class_idx] = len(class_kpt_names)
    return counts
