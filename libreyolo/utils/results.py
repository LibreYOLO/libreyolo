"""
Ultralytics-compatible Results and Boxes classes for LibreYOLO.

Provides structured access to detection results via attribute-style API:
    result.boxes.xyxy   # (N, 4) tensor of boxes
    result.boxes.conf   # (N,) tensor of confidences
    result.boxes.cls    # (N,) tensor of class IDs
    result.boxes.xywh   # (N, 4) center-x, center-y, width, height
    result.boxes.data   # (N, 6) combined [xyxy, conf, cls]
"""

from typing import Dict, Optional, Tuple

import torch


class Boxes:
    """
    Wraps detection tensors for a single image.

    Args:
        boxes: (N, 4) tensor in xyxy format.
        conf: (N,) tensor of confidence scores.
        cls: (N,) tensor of class IDs.
    """

    def __init__(
        self,
        boxes: torch.Tensor,
        conf: torch.Tensor,
        cls: torch.Tensor,
    ):
        self._boxes = boxes
        self._conf = conf
        self._cls = cls

    @property
    def xyxy(self) -> torch.Tensor:
        """(N, 4) boxes in x1, y1, x2, y2 format."""
        return self._boxes

    @property
    def conf(self) -> torch.Tensor:
        """(N,) confidence scores."""
        return self._conf

    @property
    def cls(self) -> torch.Tensor:
        """(N,) class IDs."""
        return self._cls

    @property
    def xywh(self) -> torch.Tensor:
        """(N, 4) boxes in center-x, center-y, width, height format."""
        b = self._boxes
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=1)

    @property
    def data(self) -> torch.Tensor:
        """(N, 6) combined tensor: [x1, y1, x2, y2, conf, cls]."""
        return torch.cat(
            [self._boxes, self._conf.unsqueeze(1), self._cls.unsqueeze(1)],
            dim=1,
        )

    def cpu(self) -> "Boxes":
        """Return a copy with all tensors on CPU."""
        return Boxes(
            self._boxes.cpu(),
            self._conf.cpu(),
            self._cls.cpu(),
        )

    def numpy(self) -> "Boxes":
        """Return a copy backed by numpy arrays (moves to CPU first)."""
        cpu = self.cpu()
        return Boxes(
            cpu._boxes.numpy(),
            cpu._conf.numpy(),
            cpu._cls.numpy(),
        )

    def __len__(self) -> int:
        return self._boxes.shape[0]

    def __repr__(self) -> str:
        return (
            f"Boxes(n={len(self)}, "
            f"xyxy={tuple(self._boxes.shape)}, "
            f"conf={tuple(self._conf.shape)}, "
            f"cls={tuple(self._cls.shape)})"
        )


class Results:
    """
    Single-image detection result, matching the Ultralytics Results API.

    Args:
        boxes: Boxes instance containing detections.
        orig_shape: (height, width) of the original image.
        path: Source image path (or None).
        names: Dict mapping class ID -> class name.
    """

    def __init__(
        self,
        boxes: Boxes,
        orig_shape: Tuple[int, int],
        path: Optional[str] = None,
        names: Optional[Dict[int, str]] = None,
    ):
        self.boxes = boxes
        self.orig_shape = orig_shape
        self.path = path
        self.names = names or {}

    def cpu(self) -> "Results":
        """Return a copy with all tensors on CPU."""
        return Results(
            boxes=self.boxes.cpu(),
            orig_shape=self.orig_shape,
            path=self.path,
            names=self.names,
        )

    def __len__(self) -> int:
        return len(self.boxes)

    def __repr__(self) -> str:
        return (
            f"Results("
            f"path='{self.path}', "
            f"orig_shape={self.orig_shape}, "
            f"boxes={self.boxes})"
        )
