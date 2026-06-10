"""Unit tests for the standalone fusion ops in libreyolo.ops."""

import pytest
import torch

from libreyolo.ops import FUSIONS, nms_fusion, wbf_seeded, weighted_boxes_fusion

pytestmark = pytest.mark.unit

WBF_VARIANTS = [weighted_boxes_fusion, wbf_seeded]
ALL_OPS = [weighted_boxes_fusion, wbf_seeded, nms_fusion]


def _pair():
    """Two overlapping same-class boxes from two models (IoU = 0.833)."""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 12.0]])
    scores = torch.tensor([0.9, 0.7])
    labels = torch.tensor([0, 0])
    model_ids = torch.tensor([0, 1])
    return boxes, scores, labels, model_ids


class TestSharedBehavior:
    @pytest.mark.parametrize("op", ALL_OPS)
    def test_empty_inputs(self, op):
        fb, fs, fl = op(
            torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long), num_models=2,
        )
        assert fb.shape == (0, 4) and fs.shape == (0,) and fl.shape == (0,)
        assert fb.dtype == torch.float32 and fl.dtype == torch.int64

    @pytest.mark.parametrize("op", ALL_OPS)
    def test_output_types_and_ordering(self, op):
        boxes = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]]
        )
        fb, fs, fl = op(
            boxes, torch.tensor([0.3, 0.8]), torch.tensor([0, 1]),
            torch.tensor([0, 1]), num_models=2,
        )
        assert fb.dtype == torch.float32
        assert fs.dtype == torch.float32
        assert fl.dtype == torch.int64
        assert torch.all(fs[:-1] >= fs[1:]), "outputs must be sorted by score desc"

    @pytest.mark.parametrize("op", ALL_OPS)
    def test_list_inputs_accepted(self, op):
        fb, fs, fl = op(
            [[0.0, 0.0, 10.0, 10.0]], [0.9], [0], [0], num_models=1,
        )
        assert fb.shape == (1, 4)

    @pytest.mark.parametrize("op", ALL_OPS)
    def test_skip_box_thr(self, op):
        boxes = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]
        )
        fb, fs, fl = op(
            boxes, torch.tensor([0.9, 0.05]), torch.tensor([0, 0]),
            torch.tensor([0, 1]), num_models=2, skip_box_thr=0.1,
        )
        assert fb.shape[0] == 1

    @pytest.mark.parametrize("op", ALL_OPS)
    def test_classes_never_merge(self, op):
        # Identical coordinates, different labels: two outputs.
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
        fb, fs, fl = op(
            boxes, torch.tensor([0.9, 0.8]), torch.tensor([0, 1]),
            torch.tensor([0, 1]), num_models=2,
        )
        assert fb.shape[0] == 2
        assert set(fl.tolist()) == {0, 1}

    @pytest.mark.parametrize("op", ALL_OPS)
    def test_shape_validation(self, op):
        with pytest.raises(ValueError, match=r"\(N, 4\)"):
            op(torch.zeros((2, 3)), torch.zeros(2), torch.zeros(2), torch.zeros(2))
        with pytest.raises(ValueError, match="scores"):
            op(torch.zeros((2, 4)), torch.zeros(3), torch.zeros(2), torch.zeros(2))

    @pytest.mark.parametrize("op", ALL_OPS)
    def test_weight_validation(self, op):
        boxes, scores, labels, model_ids = _pair()
        with pytest.raises(ValueError, match="one entry per model"):
            op(boxes, scores, labels, model_ids, weights=[1.0], num_models=2)
        with pytest.raises(ValueError, match="positive"):
            op(boxes, scores, labels, model_ids, weights=[1.0, -1.0])
        with pytest.raises(ValueError, match="model_ids contains index"):
            op(boxes, scores, labels, model_ids, num_models=1)


class TestWeightedBoxesFusion:
    @pytest.mark.parametrize("op", WBF_VARIANTS)
    def test_pair_fuses_to_paper_values(self, op):
        boxes, scores, labels, model_ids = _pair()
        fb, fs, fl = op(boxes, scores, labels, model_ids, num_models=2)
        assert fb.shape[0] == 1
        # Confidence-weighted coordinates: y2 = (0.9*10 + 0.7*12) / 1.6
        expected = torch.tensor([[0.0, 0.0, 10.0, 10.875]])
        assert torch.allclose(fb, expected, atol=1e-5)
        # avg score 0.8 rescaled by min(2, 2)/2 = 1
        assert torch.allclose(fs, torch.tensor([0.8]), atol=1e-5)
        assert fl.tolist() == [0]

    @pytest.mark.parametrize("op", WBF_VARIANTS)
    def test_conf_type_max(self, op):
        boxes, scores, labels, model_ids = _pair()
        _, fs, _ = op(boxes, scores, labels, model_ids, num_models=2, conf_type="max")
        assert torch.allclose(fs, torch.tensor([0.9]), atol=1e-5)

    @pytest.mark.parametrize("op", WBF_VARIANTS)
    def test_conf_type_validation(self, op):
        boxes, scores, labels, model_ids = _pair()
        with pytest.raises(ValueError, match="conf_type"):
            op(boxes, scores, labels, model_ids, conf_type="median")

    @pytest.mark.parametrize("op", WBF_VARIANTS)
    def test_solo_box_score_rescaled(self, op):
        # A box only one of two models found: score halves (min(1, 2)/2).
        fb, fs, fl = op(
            torch.tensor([[50.0, 50.0, 60.0, 60.0]]), torch.tensor([0.8]),
            torch.tensor([3]), torch.tensor([0]), num_models=2,
        )
        assert torch.allclose(fs, torch.tensor([0.4]), atol=1e-5)
        assert fl.tolist() == [3]

    @pytest.mark.parametrize("op", WBF_VARIANTS)
    def test_model_weights_shift_coordinates_and_scores(self, op):
        boxes, scores, labels, model_ids = _pair()
        fb, fs, _ = op(
            boxes, scores, labels, model_ids, weights=[3.0, 1.0], num_models=2,
        )
        # Coordinates weighted by fs*w: y2 = (0.9*3*10 + 0.7*1*12) / 3.4
        assert torch.allclose(fb[0, 3], torch.tensor(35.4 / 3.4), atol=1e-4)
        # Score = sum(fs*w)/sum(w) = 3.4/4, rescale min(4, 4)/4 = 1.
        assert torch.allclose(fs, torch.tensor([0.85]), atol=1e-5)

    @pytest.mark.parametrize("op", WBF_VARIANTS)
    def test_weighted_solo_rescale(self, op):
        # Solo box from the w=3 model in a [3, 1] ensemble: rescale 3/4.
        _, s_strong, _ = op(
            torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([0.8]),
            torch.tensor([0]), torch.tensor([0]), weights=[3.0, 1.0], num_models=2,
        )
        _, s_weak, _ = op(
            torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([0.8]),
            torch.tensor([0]), torch.tensor([1]), weights=[3.0, 1.0], num_models=2,
        )
        assert torch.allclose(s_strong, torch.tensor([0.6]), atol=1e-5)
        assert torch.allclose(s_weak, torch.tensor([0.2]), atol=1e-5)

    @pytest.mark.parametrize("op", WBF_VARIANTS)
    def test_scale_invariance(self, op):
        boxes, scores, labels, model_ids = _pair()
        b_px, s_px, l_px = op(boxes, scores, labels, model_ids, num_models=2)
        b_n, s_n, l_n = op(boxes / 1000.0, scores, labels, model_ids, num_models=2)
        assert torch.allclose(b_px / 1000.0, b_n, atol=1e-7)
        assert torch.allclose(s_px, s_n)
        assert torch.equal(l_px, l_n)

    @pytest.mark.parametrize("op", WBF_VARIANTS)
    def test_min_votes_drops_solo_keeps_pair(self, op):
        boxes = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 12.0], [50.0, 50.0, 60.0, 60.0]]
        )
        scores = torch.tensor([0.9, 0.7, 0.95])
        labels = torch.tensor([0, 0, 0])
        model_ids = torch.tensor([0, 1, 0])
        fb, fs, fl = op(boxes, scores, labels, model_ids, num_models=2, min_votes=2)
        assert fb.shape[0] == 1
        assert torch.allclose(fb[0, 3], torch.tensor(10.875), atol=1e-4)

    @pytest.mark.parametrize("op", WBF_VARIANTS)
    def test_votes_count_models_not_boxes(self, op):
        # Two boxes from the SAME model in one cluster is still one vote.
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 11.0]])
        fb, _, _ = op(
            boxes, torch.tensor([0.9, 0.85]), torch.tensor([0, 0]),
            torch.tensor([0, 0]), num_models=2, min_votes=2,
        )
        assert fb.shape[0] == 0

    @pytest.mark.parametrize("op", WBF_VARIANTS)
    def test_models_per_label_caps_votes(self, op):
        # Class 1 is known to a single model: min_votes=2 must not erase it.
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 1])
        model_ids = torch.tensor([0, 0])
        mpl = torch.tensor([2, 1])
        fb, fs, fl = op(
            boxes, scores, labels, model_ids,
            num_models=2, min_votes=2, models_per_label=mpl,
        )
        assert fl.tolist() == [1]

    @pytest.mark.parametrize("op", WBF_VARIANTS)
    def test_models_per_label_too_short_raises(self, op):
        boxes, scores, labels, model_ids = _pair()
        with pytest.raises(ValueError, match="models_per_label"):
            op(
                boxes, scores, labels + 5, model_ids,
                num_models=2, models_per_label=torch.tensor([2]),
            )

    def test_variants_agree_on_unambiguous_clusters(self):
        boxes = torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 12.0],
                [100.0, 100.0, 120.0, 120.0],
                [101.0, 101.0, 120.0, 121.0],
                [300.0, 300.0, 310.0, 310.0],
            ]
        )
        scores = torch.tensor([0.9, 0.7, 0.8, 0.75, 0.6])
        labels = torch.tensor([0, 0, 1, 1, 0])
        model_ids = torch.tensor([0, 1, 0, 1, 1])
        out_a = weighted_boxes_fusion(boxes, scores, labels, model_ids, num_models=2)
        out_b = wbf_seeded(boxes, scores, labels, model_ids, num_models=2)
        for seq, par in zip(out_a, out_b):
            assert torch.allclose(seq.float(), par.float(), atol=1e-5)


class TestNmsFusion:
    def test_keeps_top_box_unchanged(self):
        boxes, scores, labels, model_ids = _pair()
        fb, fs, fl = nms_fusion(boxes, scores, labels, model_ids, num_models=2)
        assert fb.shape[0] == 1
        assert torch.equal(fb[0], boxes[0])
        assert torch.allclose(fs, torch.tensor([0.9]))

    def test_weights_pick_winner_but_scores_stay_original(self):
        boxes, scores, labels, model_ids = _pair()
        fb, fs, _ = nms_fusion(
            boxes, scores, labels, model_ids, weights=[1.0, 10.0], num_models=2,
        )
        assert torch.equal(fb[0], boxes[1]), "weighted ranking should pick model 1"
        assert torch.allclose(fs, torch.tensor([0.7])), "scores must stay calibrated"

    def test_min_votes_rejected(self):
        boxes, scores, labels, model_ids = _pair()
        with pytest.raises(ValueError, match="min_votes"):
            nms_fusion(boxes, scores, labels, model_ids, num_models=2, min_votes=2)

    def test_negative_coordinates_stay_class_separated(self):
        # The class-offset trick needs the internal non-negative shift.
        boxes = torch.tensor([[-20.0, -20.0, -10.0, -10.0], [-20.0, -20.0, -10.0, -10.0]])
        fb, _, fl = nms_fusion(
            boxes, torch.tensor([0.9, 0.8]), torch.tensor([0, 1]),
            torch.tensor([0, 1]), num_models=2,
        )
        assert fb.shape[0] == 2
        assert torch.allclose(fb, boxes[torch.tensor([0, 1])])


class TestRegistry:
    def test_registry_contents(self):
        assert FUSIONS["wbf"] is weighted_boxes_fusion
        assert FUSIONS["wbf_seeded"] is wbf_seeded
        assert FUSIONS["nms"] is nms_fusion


class TestFuzz:
    @pytest.mark.parametrize("op", ALL_OPS)
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_random_inputs_hold_invariants(self, op, seed):
        gen = torch.Generator().manual_seed(seed)
        n = 200
        xy = torch.rand((n, 2), generator=gen) * 500
        wh = torch.rand((n, 2), generator=gen) * 100 + 1
        boxes = torch.cat([xy, xy + wh], dim=1)
        scores = torch.rand(n, generator=gen)
        labels = torch.randint(0, 5, (n,), generator=gen)
        model_ids = torch.randint(0, 3, (n,), generator=gen)

        fb, fs, fl = op(boxes, scores, labels, model_ids, num_models=3)
        assert fb.shape[0] <= n
        assert torch.isfinite(fb).all() and torch.isfinite(fs).all()
        assert (fs >= 0).all() and (fs <= 1).all()
        assert set(fl.tolist()) <= set(labels.tolist())
        assert torch.all(fs[:-1] >= fs[1:])
        # Fused coordinates stay inside the convex hull of the inputs.
        assert fb.min() >= boxes.min() - 1e-4
        assert fb.max() <= boxes.max() + 1e-4

    @pytest.mark.parametrize("op", WBF_VARIANTS)
    def test_min_votes_monotonic(self, op):
        gen = torch.Generator().manual_seed(7)
        n = 120
        xy = torch.rand((n, 2), generator=gen) * 200
        wh = torch.rand((n, 2), generator=gen) * 60 + 5
        boxes = torch.cat([xy, xy + wh], dim=1)
        scores = torch.rand(n, generator=gen)
        labels = torch.randint(0, 3, (n,), generator=gen)
        model_ids = torch.randint(0, 3, (n,), generator=gen)
        counts = [
            op(boxes, scores, labels, model_ids, num_models=3, min_votes=v)[0].shape[0]
            for v in (1, 2, 3)
        ]
        assert counts[0] >= counts[1] >= counts[2]
