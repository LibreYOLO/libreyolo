"""YOLO-format pose-estimation dataset for LibreYOLO.

Reads Ultralytics-style YOLO pose labels: one object per line as

    class cx cy w h  kx1 ky1 v1  kx2 ky2 v2  ...  kxK kyK vK

with ``cx, cy, w, h`` and every ``kx, ky`` normalized to ``[0, 1]`` and ``v``
the per-keypoint visibility flag (``0`` absent, ``1`` labelled-but-occluded,
``2`` visible). The keypoint count ``K`` and the horizontal-flip permutation
come from ``kpt_shape`` / ``flip_idx`` in the dataset ``data.yaml``.

The dataset hands the raw BGR image plus normalized labels to a ``preproc``
transform, which performs letterboxing / augmentation and returns the padded
``(max_labels, 5 + 3K)`` target slab the YOLO-NAS pose loss expects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def parse_yolo_pose_label_line(
    parts: Sequence[str], num_keypoints: int, keypoint_dim: int = 3
):
    """Parse one YOLO pose label line into ``(cls, bbox, keypoints)``.

    Args:
        parts: Whitespace-split tokens of the line.
        num_keypoints: Expected keypoint count ``K``.
        keypoint_dim: Number of values per keypoint in the label file. YOLO pose
            YAML uses ``kpt_shape: [K, 2]`` for xy-only labels and
            ``kpt_shape: [K, 3]`` for xyv labels.

    Returns:
        Tuple of ``(cls_id: int, bbox: (4,) cxcywh float32,
        keypoints: (K, 3) float32)`` — all coordinates normalized. xy-only
        labels are promoted to xyv with visibility ``2``.

    Raises:
        ValueError: If the line does not have exactly ``5 + keypoint_dim*K`` fields.
    """
    if keypoint_dim not in (2, 3):
        raise ValueError(f"Unsupported keypoint_dim {keypoint_dim}; expected 2 or 3")
    expected = 5 + keypoint_dim * num_keypoints
    if len(parts) != expected:
        raise ValueError(
            f"Expected {expected} fields for a {num_keypoints}-keypoint pose "
            f"label, got {len(parts)}"
        )
    cls_id = int(float(parts[0]))
    bbox = np.array(parts[1:5], dtype=np.float32)
    keypoints = np.array(parts[5:], dtype=np.float32).reshape(
        num_keypoints, keypoint_dim
    )
    if keypoint_dim == 2:
        visibility = np.full((num_keypoints, 1), 2.0, dtype=np.float32)
        keypoints = np.concatenate([keypoints, visibility], axis=1)
    return cls_id, bbox, keypoints


class YOLOPoseDataset(Dataset):
    """YOLO-format keypoint dataset.

    Each item is ``(image, target, img_info, index)`` where ``image`` and
    ``target`` are produced by ``preproc``. ``target`` is the padded
    ``(max_labels, 5 + 3K)`` slab; ``img_info`` is the original ``(h, w)``.
    """

    def __init__(
        self,
        img_files: Sequence[Path],
        num_keypoints: int,
        label_files: Optional[Sequence[Path]] = None,
        img_size: Tuple[int, int] = (640, 640),
        preproc=None,
        keypoint_dim: int = 3,
        decode_scale: int = 1,
    ):
        if num_keypoints < 1:
            raise ValueError(f"num_keypoints must be >= 1, got {num_keypoints}")
        if keypoint_dim not in (2, 3):
            raise ValueError(f"keypoint_dim must be 2 or 3, got {keypoint_dim}")

        self.num_keypoints = num_keypoints
        self.keypoint_dim = keypoint_dim
        self.img_size = img_size
        self._input_dim = img_size
        self.preproc = preproc
        if decode_scale not in (1, 2, 4, 8):
            raise ValueError(f"decode_scale must be one of 1, 2, 4, 8; got {decode_scale}")
        self.decode_scale = decode_scale

        self.img_files = [Path(f) for f in img_files]
        if label_files is not None:
            self.label_files = [Path(f) for f in label_files]
        else:
            from .utils import img2label_paths

            self.label_files = img2label_paths(self.img_files)

        if len(self.img_files) == 0:
            raise ValueError("YOLOPoseDataset: no images found")
        if len(self.img_files) != len(self.label_files):
            raise ValueError(
                "YOLOPoseDataset: img_files and label_files length mismatch"
            )

        self.labels = self._load_all_labels()
        n_obj = sum(lbl[0].shape[0] for lbl in self.labels)
        logger.info(
            "YOLOPoseDataset: %d images, %d objects, %d keypoints/object",
            len(self.img_files),
            n_obj,
            num_keypoints,
        )
        if n_obj == 0:
            logger.warning("YOLOPoseDataset: no pose labels found in any file")

    def _load_all_labels(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        labels = []
        bad_lines = 0
        for label_file in self.label_files:
            cls_list, box_list, kpt_list = [], [], []
            if label_file.exists():
                with open(label_file, "r") as fh:
                    for line in fh:
                        parts = line.split()
                        if not parts:
                            continue
                        try:
                            cls_id, bbox, kpts = parse_yolo_pose_label_line(
                                parts, self.num_keypoints, self.keypoint_dim
                            )
                        except ValueError:
                            bad_lines += 1
                            continue
                        cls_list.append(cls_id)
                        box_list.append(bbox)
                        kpt_list.append(kpts)
            if box_list:
                labels.append(
                    (
                        np.stack(box_list).astype(np.float32),
                        np.array(cls_list, dtype=np.float32),
                        np.stack(kpt_list).astype(np.float32),
                    )
                )
            else:
                labels.append(
                    (
                        np.zeros((0, 4), dtype=np.float32),
                        np.zeros((0,), dtype=np.float32),
                        np.zeros((0, self.num_keypoints, 3), dtype=np.float32),
                    )
                )
        if bad_lines:
            logger.warning(
                "YOLOPoseDataset: skipped %d label line(s) with a field count "
                "that does not match %d keypoints",
                bad_lines,
                self.num_keypoints,
            )
        return labels

    def __len__(self) -> int:
        return len(self.img_files)

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value):
        self._input_dim = value

    def load_image(self, index: int) -> np.ndarray:
        flags = {
            1: cv2.IMREAD_COLOR,
            2: cv2.IMREAD_REDUCED_COLOR_2,
            4: cv2.IMREAD_REDUCED_COLOR_4,
            8: cv2.IMREAD_REDUCED_COLOR_8,
        }[self.decode_scale]
        img = cv2.imread(str(self.img_files[index]), flags)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {self.img_files[index]}")
        return img

    def __getitem__(self, index: int):
        img = self.load_image(index)
        h, w = img.shape[:2]
        bboxes_norm, cls, kpts_norm = self.labels[index]

        if self.preproc is not None:
            img, target = self.preproc(
                img,
                bboxes_norm.copy(),
                cls.copy(),
                kpts_norm.copy(),
                self.input_dim,
            )
        else:
            target = (bboxes_norm, cls, kpts_norm)
        return img, target, (h, w), index


def pose_collate_fn(batch):
    """Collate ``YOLOPoseDataset`` items into batched tensors.

    Returns ``(imgs, targets, img_infos, img_ids)`` where ``imgs`` is
    ``(B, 3, H, W)`` and ``targets`` is ``(B, max_labels, 5 + 3K)``.
    """
    imgs, targets, img_infos, img_ids = zip(*batch)
    imgs = torch.from_numpy(np.stack(imgs))
    targets = torch.from_numpy(np.stack(targets))
    return imgs, targets, img_infos, img_ids
