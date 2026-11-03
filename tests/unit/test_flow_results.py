"""Unit tests for the FlowField payload, its drawing helpers and Results.flow."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import FlowField
from libreyolo.utils.drawing import draw_flow, draw_results, flow_to_rgb
from libreyolo.utils.results import Results

pytestmark = pytest.mark.unit


def _translation_flow(h, w, dx, dy):
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[..., 0] = dx
    flow[..., 1] = dy
    return flow


class TestFlowField:
    def test_requires_hw2(self):
        with pytest.raises(ValueError, match=r"\(H, W, 2\)"):
            FlowField(np.zeros((4, 4), dtype=np.float32))
        with pytest.raises(ValueError, match=r"\(H, W, 2\)"):
            FlowField(np.zeros((4, 4, 3), dtype=np.float32))
        with pytest.raises(TypeError):
            FlowField([[[0.0, 0.0]]])

    def test_orig_shape_must_match_canvas(self):
        with pytest.raises(ValueError, match="does not match"):
            FlowField(np.zeros((4, 6, 2), dtype=np.float32), (4, 5))
        field = FlowField(np.zeros((4, 6, 2), dtype=np.float32))
        assert field.orig_shape == (4, 6)

    def test_confidence_shape_and_dtype(self):
        data = np.zeros((3, 5, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="confidence"):
            FlowField(data, confidence=np.ones((3, 4)))
        field = FlowField(data, confidence=np.ones((3, 5), dtype=np.float64))
        assert field.data.dtype == np.float32
        assert field.confidence.dtype == np.float32
        assert FlowField(data).confidence is None

    def test_casts_torch_to_float32(self):
        field = FlowField(torch.ones((2, 2, 2), dtype=torch.float64))
        assert field.data.dtype == torch.float32

    def test_magnitude_angle_and_stats(self):
        flow = _translation_flow(2, 3, 3.0, 4.0)
        field = FlowField(flow)
        assert np.allclose(field.magnitude, 5.0)
        assert np.allclose(field.angle, np.arctan2(4.0, 3.0))
        assert field.max_magnitude == pytest.approx(5.0)
        assert field.mean_magnitude == pytest.approx(5.0)
        assert np.allclose(field.dx, 3.0)
        assert np.allclose(field.dy, 4.0)

        torch_field = FlowField(torch.from_numpy(flow))
        assert torch.allclose(torch_field.magnitude, torch.full((2, 3), 5.0))

    def test_stats_ignore_nonfinite(self):
        flow = _translation_flow(2, 2, 1.0, 0.0)
        flow[0, 0] = (np.nan, np.inf)
        field = FlowField(flow)
        assert field.max_magnitude == pytest.approx(1.0)
        assert field.mean_magnitude == pytest.approx(1.0)

    def test_moves_carry_confidence(self):
        field = FlowField(
            torch.zeros((2, 2, 2)), confidence=torch.ones((2, 2))
        )
        moved = field.numpy()
        assert isinstance(moved.data, np.ndarray)
        assert isinstance(moved.confidence, np.ndarray)
        back = moved.cpu()
        assert back.confidence.shape == (2, 2)
        assert field.to("cpu").confidence.shape == (2, 2)

    def test_indexing_keeps_the_field_intact(self):
        field = FlowField(np.zeros((3, 4, 2), dtype=np.float32), confidence=np.ones((3, 4)))
        sliced = field[0]
        assert sliced.data.shape == (3, 4, 2)
        assert sliced.confidence.shape == (3, 4)
        assert len(field) == 1

    def test_to_rgb_shape_dtype_and_zero_is_white(self):
        field = FlowField(np.zeros((5, 7, 2), dtype=np.float32))
        rgb = field.to_rgb()
        assert rgb.shape == (5, 7, 3)
        assert rgb.dtype == np.uint8
        assert (rgb == 255).all()

    def test_to_rgb_direction_changes_hue_and_fixed_scale_is_stable(self):
        right = FlowField(_translation_flow(2, 2, 1.0, 0.0)).to_rgb()
        down = FlowField(_translation_flow(2, 2, 0.0, 1.0)).to_rgb()
        assert not np.array_equal(right, down)
        short = FlowField(_translation_flow(2, 2, 1.0, 0.0))
        long = FlowField(_translation_flow(2, 2, 4.0, 0.0))
        # Per-image normalisation saturates both; a shared scale separates them.
        assert np.array_equal(short.to_rgb(), long.to_rgb())
        assert not np.array_equal(
            short.to_rgb(max_magnitude=4.0), long.to_rgb(max_magnitude=4.0)
        )

    def test_to_rgb_nonfinite_is_black(self):
        flow = _translation_flow(2, 2, 1.0, 1.0)
        flow[1, 1] = (np.nan, 0.0)
        rgb = flow_to_rgb(flow)
        assert (rgb[1, 1] == 0).all()
        assert (rgb[0, 0] != 0).any()

    def test_warp_identity_on_zero_flow(self):
        image = (np.arange(4 * 6 * 3) % 251).reshape(4, 6, 3).astype(np.uint8)
        warped = FlowField(np.zeros((4, 6, 2), dtype=np.float32)).warp(image)
        assert warped.dtype == np.uint8
        assert np.array_equal(warped, image)
        gray = image[..., 0]
        assert np.array_equal(FlowField(np.zeros((4, 6, 2), dtype=np.float32)).warp(gray), gray)

    def test_warp_recovers_a_translated_frame(self):
        rng = np.random.default_rng(0)
        first = rng.integers(0, 255, size=(8, 10), dtype=np.uint8)
        # The second frame is the first shifted two pixels to the right.
        second = np.zeros_like(first)
        second[:, 2:] = first[:, :-2]
        flow = _translation_flow(8, 10, 2.0, 0.0)
        warped = FlowField(flow).warp(second)
        assert np.array_equal(warped[:, :-2], first[:, :-2])

    def test_warp_rejects_wrong_canvas(self):
        with pytest.raises(ValueError, match="original canvas"):
            FlowField(np.zeros((4, 6, 2), dtype=np.float32)).warp(np.zeros((5, 6)))


class TestFlowResults:
    def _result(self, h=6, w=8, confidence=False):
        flow = _translation_flow(h, w, 1.5, -0.5)
        conf = np.full((h, w), 0.5, dtype=np.float32) if confidence else None
        return Results(
            boxes=None,
            orig_shape=(h, w),
            flow=FlowField(flow, confidence=conf),
            names={0: "flow"},
            path="frame.jpg",
        )

    def test_results_flow_slot(self):
        result = self._result()
        assert result.boxes is None
        assert len(result) == 1
        assert isinstance(result.flow, FlowField)
        assert "flow" in Results._keys
        assert "flow=FlowField" in repr(result)

    def test_summary_and_json(self):
        result = self._result(confidence=True)
        rows = result.summary(decimals=3)
        assert rows == [
            {
                "name": "flow",
                "shape": [6, 8, 2],
                "max_magnitude": pytest.approx(1.581, abs=1e-3),
                "mean_magnitude": pytest.approx(1.581, abs=1e-3),
                "has_confidence": True,
            }
        ]
        parsed = json.loads(result.to_json())
        assert parsed[0]["name"] == "flow"
        assert self._result().summary()[0]["has_confidence"] is False

    def test_moves_and_indexing_preserve_flow(self):
        result = self._result(confidence=True)
        numpy_result = result.numpy()
        assert isinstance(numpy_result.flow.data, np.ndarray)
        assert numpy_result.flow.confidence is not None
        assert result[0].flow.data.shape == (6, 8, 2)
        assert result.cpu().flow.orig_shape == (6, 8)

    def test_update_rewraps_to_canvas(self):
        result = self._result()
        result.update(flow=FlowField(np.zeros((6, 8, 2), dtype=np.float32)))
        assert result.flow.max_magnitude == 0.0
        with pytest.raises(ValueError, match="does not match"):
            result.update(flow=FlowField(np.zeros((3, 3, 2), dtype=np.float32)))

    def test_constructor_rejects_mismatched_canvas(self):
        with pytest.raises(ValueError, match="does not match"):
            Results(
                boxes=None,
                orig_shape=(6, 8),
                flow=FlowField(np.zeros((3, 3, 2), dtype=np.float32)),
            )

    def test_plot_returns_colour_wheel_image(self):
        result = self._result()
        rendered = result.plot()
        assert isinstance(rendered, Image.Image)
        assert rendered.size == (8, 6)
        assert np.array_equal(np.asarray(rendered), result.flow.to_rgb())
        bgr = result.plot(pil=False)
        assert isinstance(bgr, np.ndarray)
        assert bgr.shape == (6, 8, 3)

    def test_plot_arrows_draw_on_the_frame_not_the_wheel(self):
        result = self._result(h=24, w=32)
        frame = np.full((24, 32, 3), (20, 30, 40), dtype=np.uint8)
        with_arrows = np.asarray(result.plot(image=frame, arrows=True, arrow_step=8))
        assert with_arrows.shape == (24, 32, 3)
        yellow = np.all(with_arrows == (255, 255, 0), axis=-1)
        assert yellow.any()
        untouched = np.all(with_arrows == (20, 30, 40), axis=-1)
        assert untouched.sum() > yellow.sum()
        wheel = result.flow.to_rgb()
        assert not np.array_equal(with_arrows, wheel)
        # arrow_scale stretches the drawing only; the field is untouched.
        longer = np.asarray(result.plot(image=frame, arrows=True, arrow_step=8, arrow_scale=3.0))
        assert np.all(longer == (255, 255, 0), axis=-1).sum() > yellow.sum()
        assert result.flow.max_magnitude == pytest.approx(np.hypot(1.5, 0.5))

    def test_plot_warp_view_needs_next_frame(self):
        result = self._result()
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="next_frame"):
            result.plot(image=frame, warp=True)

    def test_plot_warp_view_shows_both_frames(self):
        h, w = 6, 8
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[2, 3] = (200, 0, 0)
        shifted = np.roll(frame, 2, axis=1)  # the red pixel moved 2 px right
        flow = _translation_flow(h, w, 2.0, 0.0)
        result = Results(
            boxes=None,
            orig_shape=(h, w),
            flow=FlowField(flow, next_frame=shifted),
            names={0: "flow"},
        )
        panels = np.asarray(result.plot(image=frame, warp=True))
        assert panels.shape == (h, 2 * w, 3)
        assert np.array_equal(panels[:, :w], frame)
        # Warping frame t+1 back with the true flow reproduces frame t.
        assert np.array_equal(panels[:, w:], frame)
        both = np.asarray(result.plot(image=frame, warp=True, arrows=True, arrow_step=2))
        assert np.all(both[:, :w] == (255, 255, 0), axis=-1).any()
        assert np.array_equal(both[:, w:], frame)

    def test_next_frame_is_validated_and_carried(self):
        flow = _translation_flow(3, 4, 1.0, 0.0)
        with pytest.raises(ValueError, match="next_frame"):
            FlowField(flow, next_frame=np.zeros((3, 5, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="next_frame"):
            FlowField(flow, next_frame=np.zeros((3, 4), dtype=np.uint8))
        field = FlowField(torch.from_numpy(flow), next_frame=torch.zeros((3, 4, 3), dtype=torch.uint8))
        assert isinstance(field.next_frame, np.ndarray)
        assert field.next_frame.dtype == np.uint8
        assert field.numpy().next_frame.shape == (3, 4, 3)
        assert field.cpu().next_frame.shape == (3, 4, 3)
        assert field.to("cpu").next_frame.shape == (3, 4, 3)
        assert field[0].next_frame.shape == (3, 4, 3)
        assert "next_frame=yes" in repr(field)
        assert "next_frame=no" in repr(FlowField(flow))
        result = Results(boxes=None, orig_shape=(3, 4), flow=field, names={0: "flow"})
        assert result.flow.next_frame.shape == (3, 4, 3)
        result.update(flow=FlowField(flow, next_frame=np.ones((3, 4, 3), dtype=np.uint8)))
        assert int(result.flow.next_frame.max()) == 1

    def test_save_round_trips_flo_and_npy(self, tmp_path):
        from libreyolo.data.flow_dataset import read_flo

        flow = _translation_flow(5, 7, 1.25, -2.5)
        field = FlowField(flow, confidence=np.ones((5, 7), dtype=np.float32))
        flo = field.save(tmp_path / "nested" / "a.flo")
        assert flo.exists()
        assert np.array_equal(read_flo(flo), flow)
        npy = field.save(tmp_path / "a.npy")
        loaded = np.load(npy)
        assert loaded.dtype == np.float32 and np.array_equal(loaded, flow)
        with pytest.raises(ValueError, match=r"\.flo or \.npy"):
            field.save(tmp_path / "a.png")

    def test_plot_save(self, tmp_path):
        out = tmp_path / "flow.png"
        self._result().plot(filename=str(out))
        assert out.exists()
        assert Image.open(out).size == (8, 6)

    def test_draw_results_dispatches_flow(self):
        result = self._result()
        canvas = Image.new("RGB", (8, 6), color=(0, 0, 0))
        rendered = draw_results(result, canvas)
        assert rendered.size == (8, 6)
        assert np.array_equal(np.asarray(rendered), result.flow.to_rgb())

    def test_draw_flow_resizes_to_the_canvas(self):
        rendered = draw_flow(Image.new("RGB", (16, 12)), _translation_flow(6, 8, 1.0, 0.0))
        assert rendered.size == (16, 12)

    def test_draw_flow_arrows_skip_still_pixels(self):
        from libreyolo.utils.drawing import draw_flow_arrows

        frame = Image.new("RGB", (32, 24), color=(5, 5, 5))
        still = draw_flow_arrows(frame, np.zeros((24, 32, 2), dtype=np.float32), step=4)
        assert np.array_equal(np.asarray(still), np.asarray(frame))
        tiny = draw_flow_arrows(frame, _translation_flow(24, 32, 0.4, 0.4), step=4)
        assert np.array_equal(np.asarray(tiny), np.asarray(frame))
        moving = draw_flow(frame, _translation_flow(24, 32, 3.0, 0.0), arrows=True, step=4)
        assert moving.size == (32, 24)
        assert np.all(np.asarray(moving) == (255, 255, 0), axis=-1).any()
        # The frame passed in is never modified in place.
        assert np.array_equal(np.asarray(frame), np.full((24, 32, 3), 5, dtype=np.uint8))
