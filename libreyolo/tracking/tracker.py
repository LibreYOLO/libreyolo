"""ByteTrack multi-object tracker.

Implements the BYTE association method from:
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box
    (Zhang et al., ECCV 2022)
"""

from __future__ import annotations

import numpy as np
import torch

from ..utils.results import Boxes, Results
from .config import TrackConfig
from .kalman_filter import KalmanFilterXYAH
from .matching import fuse_score, iou_distance, linear_assignment
from .strack import STrack, TrackState


class ByteTracker:
    """Multi-object tracker using the ByteTrack algorithm.

    Args:
        config: Tracking configuration. If None, uses default TrackConfig.
        **kwargs: Forwarded to TrackConfig.from_kwargs when config is None.

    Example::

        tracker = ByteTracker()
        for frame in frames:
            result = model(frame, conf=0.1)
            tracked = tracker.update(result)
            print(tracked.track_id)
    """

    def __init__(self, config: TrackConfig | None = None, **kwargs):
        self.config = config or TrackConfig.from_kwargs(**kwargs)
        self._id_count: int = 0
        self._frame_id: int = 0
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self.kalman_filter = KalmanFilterXYAH()

    def _next_id(self) -> int:
        self._id_count += 1
        return self._id_count

    def reset(self):
        """Clear all tracks and reset the ID counter."""
        self._id_count = 0
        self._frame_id = 0
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()

    def update(self, results: Results) -> Results:
        """Run one frame of tracking.

        Takes detection results and returns new Results with track IDs
        assigned. Only confirmed, currently tracked objects are returned.

        Args:
            results: Detection results from any detector.

        Returns:
            New Results with ``track_id`` attribute set as an (N,) int tensor.
        """
        self._frame_id += 1
        cfg = self.config

        # ------------------------------------------------------------------
        # 1. Extract detections (torch → numpy boundary)
        # ------------------------------------------------------------------
        boxes_np = results.boxes.xyxy.cpu().numpy().astype(np.float64)
        scores_np = results.boxes.conf.cpu().numpy().astype(np.float64)
        classes_np = results.boxes.cls.cpu().numpy().astype(np.float64)

        # Filter below track_low_thresh and build STrack candidates.
        keep = scores_np >= cfg.track_low_thresh
        boxes_np = boxes_np[keep]
        scores_np = scores_np[keep]
        classes_np = classes_np[keep]
        # Map back to original detection indices for result slicing.
        original_indices = np.where(keep)[0]

        # Split into high / low confidence.
        high_mask = scores_np >= cfg.track_high_thresh
        low_mask = ~high_mask

        high_dets = [
            STrack(boxes_np[i], scores_np[i], classes_np[i], int(original_indices[i]))
            for i in np.where(high_mask)[0]
        ]
        low_dets = [
            STrack(boxes_np[i], scores_np[i], classes_np[i], int(original_indices[i]))
            for i in np.where(low_mask)[0]
        ]

        high_bboxes = boxes_np[high_mask] if len(high_dets) > 0 else np.empty((0, 4))
        low_bboxes = boxes_np[low_mask] if len(low_dets) > 0 else np.empty((0, 4))
        high_scores = scores_np[high_mask] if len(high_dets) > 0 else np.empty(0)

        # ------------------------------------------------------------------
        # 2. Predict existing tracks
        # ------------------------------------------------------------------
        for t in self.tracked_stracks:
            t.predict(self.kalman_filter)
        for t in self.lost_stracks:
            t.predict(self.kalman_filter)

        # Split tracked into confirmed and unconfirmed.
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked_stracks = [t for t in self.tracked_stracks if t.is_activated]

        # Pool for first association: confirmed tracked + lost.
        strack_pool = _joint_stracks(tracked_stracks, self.lost_stracks)

        # ------------------------------------------------------------------
        # 3. Stage 1: high-confidence detections ↔ track pool
        # ------------------------------------------------------------------
        cost = iou_distance(strack_pool, high_bboxes)
        if cfg.fuse_score and len(high_scores) > 0:
            cost = fuse_score(cost, high_scores)
        matches, u_track, u_det_high = linear_assignment(cost, cfg.match_thresh)

        for m in matches:
            track = strack_pool[m[0]]
            det = high_dets[m[1]]
            if track.state == TrackState.Tracked:
                track.update(self.kalman_filter, det, self._frame_id)
            else:
                track.re_activate(self.kalman_filter, det, self._frame_id)

        # ------------------------------------------------------------------
        # 4. Stage 2: low-confidence detections ↔ remaining tracked (NOT lost)
        # ------------------------------------------------------------------
        remaining_tracked = [
            strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked
        ]
        cost2 = iou_distance(remaining_tracked, low_bboxes)
        matches2, u_track2, _ = linear_assignment(cost2, 0.5)

        for m in matches2:
            track = remaining_tracked[m[0]]
            det = low_dets[m[1]]
            track.update(self.kalman_filter, det, self._frame_id)

        # Mark unmatched tracked as lost.
        for i in u_track2:
            t = remaining_tracked[i]
            if t.state != TrackState.Lost:
                t.mark_lost()

        # ------------------------------------------------------------------
        # 5. Stage 3: remaining high-conf ↔ unconfirmed tracks
        # ------------------------------------------------------------------
        remaining_high_dets = [high_dets[i] for i in u_det_high]
        remaining_high_bboxes = (
            np.array([d._xyxy for d in remaining_high_dets])
            if remaining_high_dets
            else np.empty((0, 4))
        )
        cost3 = iou_distance(unconfirmed, remaining_high_bboxes)
        matches3, u_unconf, u_det_final = linear_assignment(cost3, 0.7)

        for m in matches3:
            track = unconfirmed[m[0]]
            det = remaining_high_dets[m[1]]
            track.update(self.kalman_filter, det, self._frame_id)

        # Remove unmatched unconfirmed tracks.
        for i in u_unconf:
            unconfirmed[i].mark_removed()

        # ------------------------------------------------------------------
        # 6. Initialize new tracks from remaining unmatched high-conf detections
        # ------------------------------------------------------------------
        for i in u_det_final:
            det = remaining_high_dets[i]
            if det.score >= cfg.new_track_thresh:
                det.activate(self.kalman_filter, self._frame_id, self._next_id())
                if cfg.minimum_consecutive_frames > 1:
                    # Start as unconfirmed — needs more consecutive matches.
                    det.is_activated = False

        # ------------------------------------------------------------------
        # 7. Handle lost tracks: mark expired as removed
        # ------------------------------------------------------------------
        max_time_lost = int(cfg.track_buffer * cfg.frame_rate / 30)
        for t in self.lost_stracks:
            if self._frame_id - t.frame_id > max_time_lost:
                t.mark_removed()

        # ------------------------------------------------------------------
        # 8. Update track lists
        # ------------------------------------------------------------------
        # Collect all tracks that are now Tracked (from any source).
        all_candidates = strack_pool + unconfirmed + high_dets + low_dets
        new_tracked = [t for t in all_candidates if t.state == TrackState.Tracked]
        # Deduplicate by track_id (keep first seen = the updated one).
        seen_ids: set[int] = set()
        deduped_tracked: list[STrack] = []
        for t in new_tracked:
            if t.track_id not in seen_ids:
                seen_ids.add(t.track_id)
                deduped_tracked.append(t)
        self.tracked_stracks = deduped_tracked

        # Lost: anything from strack_pool or old lost list that is still Lost.
        tracked_ids = {t.track_id for t in self.tracked_stracks}
        self.lost_stracks = [
            t
            for t in strack_pool + self.lost_stracks
            if t.state == TrackState.Lost and t.track_id not in tracked_ids
        ]
        # Deduplicate lost list.
        seen_lost: set[int] = set()
        deduped_lost: list[STrack] = []
        for t in self.lost_stracks:
            if t.track_id not in seen_lost:
                seen_lost.add(t.track_id)
                deduped_lost.append(t)
        self.lost_stracks = deduped_lost

        # Remove duplicates between tracked and lost.
        self.tracked_stracks, self.lost_stracks = _remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        # Prune removed list.
        self.removed_stracks = [
            t for t in self.removed_stracks if t.state == TrackState.Removed
        ]
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-500:]

        # ------------------------------------------------------------------
        # 9. Build output Results
        # ------------------------------------------------------------------
        output_stracks = [
            t
            for t in self.tracked_stracks
            if t.is_activated and t._hits >= cfg.minimum_consecutive_frames
        ]

        if len(output_stracks) == 0:
            empty_boxes = Boxes(
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
            )
            return Results(
                boxes=empty_boxes,
                orig_shape=results.orig_shape,
                path=results.path,
                names=results.names,
                track_id=torch.zeros((0,), dtype=torch.int64),
            )

        # Slice original detections by detection_index.
        indices = [t.detection_index for t in output_stracks]
        out_boxes = results.boxes.xyxy[indices]
        out_conf = results.boxes.conf[indices]
        out_cls = results.boxes.cls[indices]
        track_ids = torch.tensor(
            [t.track_id for t in output_stracks], dtype=torch.int64
        )

        return Results(
            boxes=Boxes(out_boxes, out_conf, out_cls),
            orig_shape=results.orig_shape,
            path=results.path,
            names=results.names,
            track_id=track_ids,
        )


# --------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------


def _joint_stracks(a: list[STrack], b: list[STrack]) -> list[STrack]:
    """Merge two track lists, deduplicating by track_id."""
    seen = {}
    for t in a:
        seen[t.track_id] = t
    for t in b:
        if t.track_id not in seen:
            seen[t.track_id] = t
    return list(seen.values())


def _remove_duplicate_stracks(
    tracked: list[STrack], lost: list[STrack]
) -> tuple[list[STrack], list[STrack]]:
    """Remove duplicates between tracked and lost lists.

    When two tracks overlap (IoU > 0.85), keep the one with more frames.
    """
    if not tracked or not lost:
        return tracked, lost

    from .matching import bbox_iou_batch

    t_bboxes = np.array([t.xyxy for t in tracked], dtype=np.float64)
    l_bboxes = np.array([t.xyxy for t in lost], dtype=np.float64)
    iou = bbox_iou_batch(t_bboxes, l_bboxes)

    remove_tracked = set()
    remove_lost = set()
    for ti, li in zip(*np.where(iou > 0.85)):
        t_age = tracked[ti].frame_id - tracked[ti].start_frame
        l_age = lost[li].frame_id - lost[li].start_frame
        if t_age >= l_age:
            remove_lost.add(li)
        else:
            remove_tracked.add(ti)

    kept_tracked = [t for i, t in enumerate(tracked) if i not in remove_tracked]
    kept_lost = [t for i, t in enumerate(lost) if i not in remove_lost]
    return kept_tracked, kept_lost
