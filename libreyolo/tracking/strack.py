"""Single track representation for ByteTrack."""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from .kalman_filter import KalmanFilterXYAH


class TrackState(IntEnum):
    """Track lifecycle states."""

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """A single tracked object managed by a Kalman filter.

    Stores the Kalman state, track metadata, and provides coordinate
    conversions between xyxy, xyah, and tlwh formats.
    """

    def __init__(self, xyxy: np.ndarray, score: float, cls: int, detection_index: int):
        self.mean: np.ndarray = np.zeros(8, dtype=np.float64)
        self.covariance: np.ndarray = np.eye(8, dtype=np.float64)

        self._xyxy = xyxy.astype(np.float64)
        self.score = score
        self.cls = int(cls)
        self.detection_index = detection_index

        self.track_id: int = 0
        self.state: TrackState = TrackState.New
        self.is_activated: bool = False

        self.frame_id: int = 0
        self.start_frame: int = 0
        self._hits: int = 0

    # ------------------------------------------------------------------
    # Coordinate conversions
    # ------------------------------------------------------------------

    @staticmethod
    def xyxy_to_xyah(xyxy: np.ndarray) -> np.ndarray:
        """Convert (x1, y1, x2, y2) to (cx, cy, aspect_ratio, height)."""
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        cx = xyxy[0] + w / 2
        cy = xyxy[1] + h / 2
        a = w / max(h, 1e-6)
        return np.array([cx, cy, a, h], dtype=np.float64)

    @property
    def xyah(self) -> np.ndarray:
        """Current (cx, cy, aspect_ratio, height) from Kalman state."""
        return self.mean[:4].copy()

    @property
    def tlwh(self) -> np.ndarray:
        """Current (top, left, width, height) from Kalman state."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # width = aspect * height
        ret[:2] -= ret[2:] / 2  # center to top-left
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        """Current (x1, y1, x2, y2) from Kalman state."""
        tlwh = self.tlwh
        return np.array(
            [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------

    def activate(self, kf: KalmanFilterXYAH, frame_id: int, track_id: int):
        """Initialize this track from its first detection."""
        self.track_id = track_id
        measurement = self.xyxy_to_xyah(self._xyxy)
        self.mean, self.covariance = kf.initiate(measurement)

        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self._hits = 1

    def re_activate(
        self,
        kf: KalmanFilterXYAH,
        new_detection: STrack,
        frame_id: int,
        new_id: bool = False,
        new_track_id: int = 0,
    ):
        """Re-activate a lost track with a new detection."""
        measurement = self.xyxy_to_xyah(new_detection._xyxy)
        # If covariance has degenerated (NaN/Inf from repeated prediction
        # without updates), reinitialize from the new measurement.
        if np.any(np.isnan(self.covariance)) or np.any(np.isinf(self.covariance)):
            self.mean, self.covariance = kf.initiate(measurement)
        else:
            self.mean, self.covariance = kf.update(
                self.mean, self.covariance, measurement
            )

        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_detection.score
        self.cls = new_detection.cls
        self.detection_index = new_detection.detection_index
        self._hits += 1

        if new_id:
            self.track_id = new_track_id

    def update(self, kf: KalmanFilterXYAH, new_detection: STrack, frame_id: int):
        """Update a matched track with a new detection."""
        measurement = self.xyxy_to_xyah(new_detection._xyxy)
        self.mean, self.covariance = kf.update(self.mean, self.covariance, measurement)

        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_detection.score
        self.cls = new_detection.cls
        self.detection_index = new_detection.detection_index
        self._hits += 1

    def predict(self, kf: KalmanFilterXYAH):
        """Predict the next state using the Kalman filter."""
        # Zero velocity for lost tracks to prevent unbounded drift.
        if self.state != TrackState.Tracked:
            self.mean[7] = 0  # zero height velocity
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)

    def mark_lost(self):
        """Transition to Lost state."""
        self.state = TrackState.Lost

    def mark_removed(self):
        """Transition to Removed state."""
        self.state = TrackState.Removed
