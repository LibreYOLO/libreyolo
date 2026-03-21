"""Multi-object tracking for LibreYOLO."""

from .config import TrackConfig
from .tracker import ByteTracker

__all__ = ["ByteTracker", "TrackConfig"]
