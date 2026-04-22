"""End-to-end smoke test for the ByteTrack pipeline.

Exercises the full detection → tracking integration without loading a real
detector or video — we feed synthetic detection Results directly to
ByteTracker.update() and assert the standard tracker properties:

  - a single moving object keeps its track ID across frames,
  - two co-existing objects get distinct, stable IDs,
  - a track survives a short occlusion and is recovered,
  - a disappeared track is eventually removed,
  - the tracker's internal state resets cleanly.

CPU-only, <60s. No network, no real weights, no video files.

Run: pytest tests/smoke/test_tracking_smoke.py -v -m smoke
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.tracking import ByteTracker, TrackConfig
from libreyolo.utils.results import Boxes, Results


pytestmark = pytest.mark.smoke


def _make_frame(boxes_xyxy: list[list[float]], scores: list[float], cls: list[int]) -> Results:
    """Construct a single-frame Results object from raw lists."""
    if len(boxes_xyxy) == 0:
        xyxy = torch.zeros((0, 4), dtype=torch.float32)
        conf = torch.zeros((0,), dtype=torch.float32)
        c = torch.zeros((0,), dtype=torch.float32)
    else:
        xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32)
        conf = torch.tensor(scores, dtype=torch.float32)
        c = torch.tensor(cls, dtype=torch.float32)
    return Results(
        boxes=Boxes(boxes=xyxy, conf=conf, cls=c),
        orig_shape=(480, 640),
        names={0: "obj", 1: "obj2"},
    )


def _extract_track_ids(tracked: Results) -> list[int]:
    if tracked.track_id is None:
        return []
    return tracked.track_id.cpu().tolist()


def test_single_moving_object_keeps_same_id():
    """One object sliding right across 5 frames gets one track with a stable ID."""
    tracker = ByteTracker(minimum_consecutive_frames=1, track_high_thresh=0.3, new_track_thresh=0.3)
    ids: list[int] = []
    for dx in range(0, 100, 20):  # 5 frames, object moves 20 px / frame
        frame = _make_frame([[100 + dx, 150, 200 + dx, 250]], scores=[0.9], cls=[0])
        tracked = tracker.update(frame)
        frame_ids = _extract_track_ids(tracked)
        assert len(frame_ids) == 1, f"expected 1 track, got {len(frame_ids)}"
        ids.append(frame_ids[0])
    assert len(set(ids)) == 1, f"track ID should stay stable, got {ids}"


def test_two_coexisting_objects_get_distinct_stable_ids():
    tracker = ByteTracker(minimum_consecutive_frames=1, track_high_thresh=0.3, new_track_thresh=0.3)
    ids_a: list[int] = []
    ids_b: list[int] = []
    for dx in range(0, 60, 15):
        frame = _make_frame(
            [[100 + dx, 100, 180 + dx, 180], [400 - dx, 300, 480 - dx, 380]],
            scores=[0.9, 0.85],
            cls=[0, 0],
        )
        tracked = tracker.update(frame)
        pairs = sorted(zip(tracked.boxes.xyxy[:, 0].tolist(), _extract_track_ids(tracked)))
        assert len(pairs) == 2, f"expected 2 tracks, got {pairs}"
        # The leftmost-starting box (moving right) gets track A, the one moving left gets B
        ids_a.append(pairs[0][1])
        ids_b.append(pairs[1][1])
    assert len(set(ids_a)) == 1 and len(set(ids_b)) == 1, (
        f"IDs should be stable per track; got A={ids_a} B={ids_b}"
    )
    assert ids_a[0] != ids_b[0], "two co-existing tracks must get distinct IDs"


def test_track_survives_short_occlusion():
    """Detection drops for 2 frames; the tracker should reuse the same ID when the object reappears nearby."""
    tracker = ByteTracker(
        minimum_consecutive_frames=1,
        track_high_thresh=0.3,
        new_track_thresh=0.3,
        track_buffer=30,  # generous buffer so a 2-frame gap is well within
    )

    # Frame 0: track established
    f0 = _make_frame([[100, 100, 200, 200]], scores=[0.9], cls=[0])
    tracked0 = tracker.update(f0)
    id0 = _extract_track_ids(tracked0)[0]

    # Frame 1: same track, slight motion
    f1 = _make_frame([[110, 102, 210, 202]], scores=[0.9], cls=[0])
    _ = tracker.update(f1)

    # Frames 2 & 3: detector drops the object (occlusion) — empty frame
    tracker.update(_make_frame([], [], []))
    tracker.update(_make_frame([], [], []))

    # Frame 4: object reappears where we'd expect it (Kalman extrapolation assists)
    f4 = _make_frame([[140, 110, 240, 210]], scores=[0.9], cls=[0])
    tracked4 = tracker.update(f4)
    id4_list = _extract_track_ids(tracked4)
    # After the short occlusion, the original track should recover with the same ID
    assert len(id4_list) == 1, f"expected 1 track recovered, got {id4_list}"
    assert id4_list[0] == id0, (
        f"track ID should recover after short occlusion: started={id0}, recovered={id4_list[0]}"
    )


def test_disappeared_track_is_removed_after_buffer():
    """A track lost for more than track_buffer frames should be dropped from the lost pool."""
    tracker = ByteTracker(
        minimum_consecutive_frames=1,
        track_high_thresh=0.3,
        new_track_thresh=0.3,
        track_buffer=3,  # very short buffer so we can check removal quickly
    )
    f0 = _make_frame([[100, 100, 200, 200]], scores=[0.9], cls=[0])
    _ = tracker.update(f0)
    assert len(tracker.tracked_stracks) == 1

    # 6 empty frames — way past the buffer (scaled by frame_rate internally, but 6 > 3)
    for _ in range(10):
        tracker.update(_make_frame([], [], []))

    # A new object appears — it should get a NEW id, not recover the old one
    f_new = _make_frame([[100, 100, 200, 200]], scores=[0.9], cls=[0])
    tracked = tracker.update(f_new)
    new_id = _extract_track_ids(tracked)[0]
    # With track_buffer=3 and 10 empty frames, the old track must be removed
    assert new_id > 1, (
        f"after long absence, a re-detection should be a NEW track, got id={new_id}"
    )


def test_reset_clears_all_state():
    tracker = ByteTracker(minimum_consecutive_frames=1, track_high_thresh=0.3, new_track_thresh=0.3)
    _ = tracker.update(_make_frame([[100, 100, 200, 200]], [0.9], [0]))
    assert tracker._id_count > 0
    assert tracker._frame_id > 0
    assert len(tracker.tracked_stracks) >= 1

    tracker.reset()

    assert tracker._id_count == 0
    assert tracker._frame_id == 0
    assert tracker.tracked_stracks == []
    assert tracker.lost_stracks == []
    assert tracker.removed_stracks == []

    # After reset, a new detection restarts ID counting from 1
    tracked = tracker.update(_make_frame([[50, 50, 100, 100]], [0.9], [0]))
    assert _extract_track_ids(tracked) == [1], "reset tracker should start IDs from 1"


def test_config_low_thresh_detections_also_used():
    """Verify ByteTrack's second association stage: a track first seen at high conf
    can be associated to a subsequent low-conf detection (the whole point of ByteTrack).
    """
    tracker = ByteTracker(
        minimum_consecutive_frames=1,
        track_high_thresh=0.6,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
    )

    # Frame 0: high-confidence detection creates track
    id0 = _extract_track_ids(tracker.update(
        _make_frame([[100, 100, 200, 200]], [0.9], [0])
    ))[0]

    # Frame 1: same location but LOW conf — should keep the track alive, not start a new one
    tracked1 = tracker.update(
        _make_frame([[105, 103, 205, 203]], [0.25], [0])
    )
    ids1 = _extract_track_ids(tracked1)
    assert ids1 == [id0], (
        "ByteTrack's second-association should recover the track via the low-conf "
        f"detection; expected [{id0}], got {ids1}"
    )
