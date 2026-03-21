"""Tracking configuration for ByteTrack."""

import warnings
from dataclasses import dataclass, fields


@dataclass(kw_only=True)
class TrackConfig:
    """Configuration for the ByteTrack multi-object tracker.

    Args:
        track_high_thresh: Minimum confidence for first association stage.
        track_low_thresh: Minimum confidence for second association (low-conf recovery).
        new_track_thresh: Minimum confidence to initialize a new track.
        match_thresh: IoU cost threshold for first association.
        track_buffer: Frames to keep lost tracks before removal.
        frame_rate: Video frame rate (used to scale track_buffer).
        fuse_score: Fuse detection score with IoU for first association.
        minimum_consecutive_frames: Frames a track must be matched before it is confirmed.
    """

    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    match_thresh: float = 0.8
    track_buffer: int = 30
    frame_rate: int = 30
    fuse_score: bool = True
    minimum_consecutive_frames: int = 1

    def __post_init__(self):
        if self.frame_rate <= 0:
            raise ValueError(f"frame_rate must be > 0, got {self.frame_rate}")
        if not (0 <= self.track_high_thresh <= 1):
            raise ValueError(f"track_high_thresh must be in [0, 1], got {self.track_high_thresh}")
        if not (0 <= self.track_low_thresh <= 1):
            raise ValueError(f"track_low_thresh must be in [0, 1], got {self.track_low_thresh}")
        if not (0 <= self.new_track_thresh <= 1):
            raise ValueError(f"new_track_thresh must be in [0, 1], got {self.new_track_thresh}")
        if not (0 <= self.match_thresh <= 1):
            raise ValueError(f"match_thresh must be in [0, 1], got {self.match_thresh}")
        if self.track_buffer < 0:
            raise ValueError(f"track_buffer must be >= 0, got {self.track_buffer}")
        if self.minimum_consecutive_frames < 1:
            raise ValueError(f"minimum_consecutive_frames must be >= 1, got {self.minimum_consecutive_frames}")
        if self.track_high_thresh < self.track_low_thresh:
            raise ValueError(
                f"track_high_thresh ({self.track_high_thresh}) must be >= "
                f"track_low_thresh ({self.track_low_thresh})"
            )

    @classmethod
    def from_kwargs(cls, **kwargs) -> "TrackConfig":
        """Construct config, warning on unknown keys."""
        valid = {f.name for f in fields(cls)}
        unknown = set(kwargs) - valid
        if unknown:
            warnings.warn(
                f"Unknown tracking config keys (ignored): {sorted(unknown)}",
                stacklevel=2,
            )
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        return cls(**filtered)
