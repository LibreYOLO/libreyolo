"""
Results, Boxes, and Masks classes for LibreYOLO.

Provides structured access to detection and segmentation results:
    result.boxes.xyxy   # (N, 4) tensor of boxes
    result.boxes.conf   # (N,) tensor of confidences
    result.boxes.cls    # (N,) tensor of class IDs
    result.boxes.xywh   # (N, 4) center-x, center-y, width, height
    result.boxes.data   # (N, 6) combined [xyxy, conf, cls]
    result.masks.data   # (N, H, W) tensor of instance masks
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
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


class Masks:
    """
    Wraps instance segmentation mask tensors for a single image.

    Args:
        masks: (N, H, W) tensor of binary instance masks at original image resolution.
        orig_shape: (height, width) of the original image.
    """

    def __init__(
        self,
        masks: Union[torch.Tensor, np.ndarray],
        orig_shape: Tuple[int, int],
    ):
        self._masks = masks
        self.orig_shape = orig_shape

    @property
    def data(self) -> Union[torch.Tensor, np.ndarray]:
        """(N, H, W) instance masks."""
        return self._masks

    @property
    def xy(self) -> List[np.ndarray]:
        """List of (M_i, 2) contour arrays in pixel coordinates, one per mask."""
        return self._masks_to_contours(normalize=False)

    @property
    def xyn(self) -> List[np.ndarray]:
        """List of (M_i, 2) contour arrays in normalized [0, 1] coordinates."""
        return self._masks_to_contours(normalize=True)

    def _masks_to_contours(self, normalize: bool) -> List[np.ndarray]:
        """Extract contour polygons from binary masks."""
        import cv2

        masks_np = self._masks
        if isinstance(masks_np, torch.Tensor):
            masks_np = masks_np.cpu().numpy()
        masks_np = masks_np.astype(np.uint8)

        h, w = self.orig_shape
        contours_list = []
        for mask in masks_np:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                # take the largest contour
                c = max(contours, key=cv2.contourArea).squeeze(1).astype(np.float64)
                if normalize:
                    c[:, 0] /= w
                    c[:, 1] /= h
                contours_list.append(c)
            else:
                contours_list.append(np.empty((0, 2), dtype=np.float64))
        return contours_list

    def cpu(self) -> "Masks":
        """Return a copy with all tensors on CPU."""
        if isinstance(self._masks, torch.Tensor):
            return Masks(self._masks.cpu(), self.orig_shape)
        return self

    def numpy(self) -> "Masks":
        """Return a copy backed by numpy arrays."""
        if isinstance(self._masks, torch.Tensor):
            return Masks(self._masks.cpu().numpy(), self.orig_shape)
        return self

    def __len__(self) -> int:
        return self._masks.shape[0]

    def __repr__(self) -> str:
        return (
            f"Masks(n={len(self)}, "
            f"shape={tuple(self._masks.shape)}, "
            f"orig_shape={self.orig_shape})"
        )


class Results:
    """
    Single-image result for detection and/or instance segmentation.

    Args:
        boxes: Boxes instance containing detections.
        orig_shape: (height, width) of the original image.
        path: Source image path (or None).
        names: Dict mapping class ID -> class name.
        masks: Optional Masks instance for segmentation results.
        track_id: Optional (N,) tensor of integer track IDs from a tracker.
    """

    def __init__(
        self,
        boxes: Boxes,
        orig_shape: Tuple[int, int],
        path: Optional[str] = None,
        names: Optional[Dict[int, str]] = None,
        masks: Optional[Masks] = None,
        track_id: Optional[torch.Tensor] = None,
        frame_idx: Optional[int] = None,
    ):
        self.boxes = boxes
        self.orig_shape = orig_shape
        self.path = path
        self.names = names or {}
        self.masks = masks
        self.track_id = track_id
        self.frame_idx = frame_idx

    def cpu(self) -> "Results":
        """Return a copy with all tensors on CPU."""
        return Results(
            boxes=self.boxes.cpu(),
            orig_shape=self.orig_shape,
            path=self.path,
            names=self.names,
            masks=self.masks.cpu() if self.masks is not None else None,
            track_id=self.track_id.cpu() if self.track_id is not None else None,
            frame_idx=self.frame_idx,
        )

    def __len__(self) -> int:
        return len(self.boxes)

    def __repr__(self) -> str:
        parts = [
            f"path='{self.path}'",
            f"orig_shape={self.orig_shape}",
            f"boxes={self.boxes}",
        ]
        if self.masks is not None:
            parts.append(f"masks={self.masks}")
        if self.track_id is not None:
            parts.append(f"track_ids={len(self.track_id)}")
        if self.frame_idx is not None:
            parts.append(f"frame_idx={self.frame_idx}")
        return f"Results({', '.join(parts)})"
