"""Public interface for user-supplied tracking algorithms."""

from typing import Protocol

from PIL import Image

from ..utils.results import Results


class Tracker(Protocol):
    """Structural interface accepted by ``model.track(tracker=instance)``.

    No inheritance or registration is required. Configure the instance yourself.
    Each track() iteration resets it once; use separate instances per camera.
    """

    def reset(self) -> None:
        """Clear all state for a new sequence."""
        ...

    def update(self, results: Results, image: Image.Image | None = None) -> Results:
        """Associate one frame, including frames with no detections.

        ``image`` is the original RGB PIL frame. Return Results whose integer
        ``track_id`` tensor/array has one entry per box, including an empty
        vector for empty output. Keep masks/keypoints aligned when selecting or
        reordering rows (``results[indices]``). track() also exposes these IDs
        through ``result.boxes.id``. Use the same backend/device as the boxes.
        """
        ...
