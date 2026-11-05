"""Unit tests for the optical flow frame-window iterator."""

from __future__ import annotations

import pytest

from libreyolo.models.base.flow_sequence import (
    FLOW_MIN_FRAMES_MESSAGE,
    iter_flow_windows,
    validate_flow_context,
)

pytestmark = pytest.mark.unit


def test_pairs_without_context():
    windows = list(iter_flow_windows(["a", "b", "c"]))
    assert windows == [(0, ["a", "b"]), (1, ["b", "c"])]


def test_two_frames_give_one_pair():
    assert list(iter_flow_windows(["a", "b"])) == [(0, ["a", "b"])]


def test_context_pads_both_ends():
    windows = list(iter_flow_windows(["a", "b", "c"], before=1, after=1))
    assert windows == [(0, ["a", "a", "b", "c"]), (1, ["a", "b", "c", "c"])]
    for _, window in windows:
        assert len(window) == 4


def test_pair_sits_at_the_before_offset():
    frames = list(range(6))
    for before, after in ((0, 2), (2, 0), (2, 3)):
        windows = list(iter_flow_windows(frames, before=before, after=after))
        assert [t for t, _ in windows] == list(range(5))
        for t, window in windows:
            assert len(window) == before + 2 + after
            assert window[before] == t
            assert window[before + 1] == t + 1
            # Padded neighbours never leave the sequence.
            assert window[0] == max(0, t - before)
            assert window[-1] == min(5, t + 1 + after)


def test_after_context_longer_than_sequence_still_yields_every_pair():
    windows = list(iter_flow_windows(["a", "b"], before=0, after=3))
    assert windows == [(0, ["a", "b", "b", "b", "b"])]


def test_lazy_emission_order():
    seen = []

    def frames():
        for name in "abcd":
            seen.append(name)
            yield name

    generator = iter_flow_windows(frames(), before=0, after=1)
    t, window = next(generator)
    assert (t, window) == (0, ["a", "b", "c"])
    # Pair 0 needed frame 2 and nothing further.
    assert seen == ["a", "b", "c"]


def test_single_frame_raises():
    with pytest.raises(ValueError, match="at least two frames"):
        list(iter_flow_windows(["only"]))


def test_empty_source_raises():
    with pytest.raises(ValueError, match=FLOW_MIN_FRAMES_MESSAGE[:20]):
        list(iter_flow_windows([]))


@pytest.mark.parametrize("context", [(1,), (1, 2, 3), "ab", None, (-1, 0), (0, True), (0.5, 0)])
def test_invalid_context_rejected(context):
    with pytest.raises((TypeError, ValueError), match="FLOW_CONTEXT"):
        validate_flow_context(context)


def test_valid_context_returns_ints():
    assert validate_flow_context((1, 2)) == (1, 2)
    assert validate_flow_context([0, 0]) == (0, 0)
