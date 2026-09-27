"""Sync-reduction parity and regression tests for the EC decoder (issue #763 item 4).

Measured attribution before this work (2 training-like forward+loss steps of the
EC detection path, CPU, Python-level counter over item/tolist/cpu/numpy/nonzero):

    17 sync calls per step, all attributed to libreyolo/models/dfine/
        (matcher.py cpu+numpy: 12, loss.py tolist+item: 3, denoising.py
        nonzero: 1, plus box_ops.py bool asserts: 11 not counted above),
    0 sync calls attributed to libreyolo/models/ec/ files.

The handoff's "~9 candidate sites in ec/decoder.py" were textual grep matches
(``int(`` on Python scalars, a deploy-only ``.item()`` in weighting_function,
a pose-only cache-miss memoisation); none fire on the detection training path.
The changes covered here instead remove ec/decoder.py's per-step host costs:
training-time anchor regeneration (host grids + H2D copy every step) and the
per-forward recomputation of the frozen ``weighting_function`` project vector.
Both are cached; the tests prove bitwise parity against the original control
flow, kept as private reference copies below.
"""

from __future__ import annotations

import collections
import traceback

import pytest
import torch

import libreyolo.models.ec.decoder as ec_decoder_mod
from libreyolo.models.dfine.matcher import HungarianMatcher
from libreyolo.models.ec.decoder import ECPoseTransformer, ECTransformer
from libreyolo.models.ec.loss import ECCriterion
from libreyolo.models.ec.utils import weighting_function

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Private reference copies of the ORIGINAL (pre-cache) control flow.
# ---------------------------------------------------------------------------


def _reference_det_training_anchors(self, spatial_shapes, memory):
    """Original ECTransformer._get_decoder_input training branch."""
    return self._generate_anchors(spatial_shapes, device=memory.device)


def _reference_pose_training_anchors(self, spatial_shapes, memory):
    """Original ECPoseTransformer.forward training branch."""
    return self._generate_anchors(
        spatial_shapes, device=memory.device, dtype=memory.dtype
    )


def _reference_weighting_function(module, reg_max, up, reg_scale):
    """Original per-forward weighting_function call (no cache)."""
    return weighting_function(reg_max, up, reg_scale)


class _original_control_flow:
    """Context manager that restores the pre-cache decoder behavior."""

    def __enter__(self):
        self._det = ECTransformer._get_training_anchors
        self._pose = ECPoseTransformer._get_training_anchors
        self._weight = ec_decoder_mod._cached_weighting_function
        ECTransformer._get_training_anchors = _reference_det_training_anchors
        ECPoseTransformer._get_training_anchors = _reference_pose_training_anchors
        ec_decoder_mod._cached_weighting_function = _reference_weighting_function
        return self

    def __exit__(self, *exc):
        ECTransformer._get_training_anchors = self._det
        ECPoseTransformer._get_training_anchors = self._pose
        ec_decoder_mod._cached_weighting_function = self._weight
        return False


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _make_det_decoder(seed=0):
    torch.manual_seed(seed)
    return ECTransformer(
        num_classes=4,
        hidden_dim=64,
        num_queries=30,
        feat_channels=(64, 64, 64),
        dim_feedforward=128,
        num_layers=2,
        num_points=(3, 6, 3),
        eval_idx=-1,
        num_denoising=10,
        reg_max=32,
        reg_scale=4.0,
        eval_spatial_size=[64, 64],
    )


def _make_criterion():
    matcher = HungarianMatcher(
        weight_dict={"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
    )
    return ECCriterion(
        matcher=matcher,
        weight_dict={
            "loss_mal": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
        },
        losses=["mal", "boxes", "local"],
        num_classes=4,
        alpha=0.75,
        gamma=2.0,
        reg_max=32,
    )


def _make_feats(bs=2, seed=1):
    torch.manual_seed(seed)
    return [
        torch.randn(bs, 64, 8, 8),
        torch.randn(bs, 64, 4, 4),
        torch.randn(bs, 64, 2, 2),
    ]


def _targets_empty():
    return [
        {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)},
        {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)},
    ]


def _targets_single():
    return [
        {
            "labels": torch.tensor([1], dtype=torch.long),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        },
        {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)},
    ]


def _targets_many():
    g = torch.Generator().manual_seed(3)
    return [
        {
            "labels": torch.tensor([0, 1, 2], dtype=torch.long),
            "boxes": torch.rand(3, 4, generator=g) * 0.4 + 0.2,
        },
        {
            "labels": torch.tensor([3, 0, 1, 2, 3], dtype=torch.long),
            "boxes": torch.rand(5, 4, generator=g) * 0.4 + 0.2,
        },
    ]


def _assert_tree_equal(a, b, path="out"):
    assert type(a) is type(b), f"{path}: type {type(a)} vs {type(b)}"
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), f"{path}: tensors differ"
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), f"{path}: keys differ"
        for k in a:
            _assert_tree_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_tree_equal(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a} != {b}"


# ---------------------------------------------------------------------------
# Fixed-input forward parity
# ---------------------------------------------------------------------------


def test_det_training_forward_parity_cold_vs_warm_vs_reference():
    """Decoder outputs must be bitwise identical with a cold cache, a warm
    cache, and the original regenerate-every-step control flow."""
    dec = _make_det_decoder()
    dec.train()
    ref = _make_det_decoder(seed=99)
    ref.load_state_dict(dec.state_dict())
    ref.train()

    feats = _make_feats()
    targets = _targets_many()

    torch.manual_seed(7)
    out_cold = dec(feats, targets=targets)
    torch.manual_seed(7)
    out_warm = dec(feats, targets=targets)
    with _original_control_flow():
        torch.manual_seed(7)
        out_ref = ref(feats, targets=targets)

    _assert_tree_equal(out_cold, out_warm)
    _assert_tree_equal(out_cold, out_ref)


def test_det_eval_dynamic_size_forward_parity():
    """eval_spatial_size=None routes eval through the training-anchor cache;
    outputs must match the original per-forward regeneration exactly."""
    torch.manual_seed(0)
    dec = ECTransformer(
        num_classes=4,
        hidden_dim=64,
        num_queries=30,
        feat_channels=(64, 64, 64),
        dim_feedforward=128,
        num_layers=2,
        num_points=(3, 6, 3),
        eval_idx=-1,
        num_denoising=10,
        reg_max=32,
        reg_scale=4.0,
        eval_spatial_size=None,
    ).eval()

    feats = _make_feats(bs=1)
    with torch.no_grad():
        out_cold = dec(feats)
        out_warm = dec(feats)
        with _original_control_flow():
            out_ref = dec(feats)
    _assert_tree_equal(out_cold, out_warm)
    _assert_tree_equal(out_cold, out_ref)


def test_cached_anchors_equal_fresh_generation():
    dec = _make_det_decoder()
    dec.train()
    memory = torch.zeros(2, 84, 64)
    shapes = [[8, 8], [4, 4], [2, 2]]
    a_new, m_new = dec._get_training_anchors(shapes, memory)
    a_ref, m_ref = _reference_det_training_anchors(dec, shapes, memory)
    assert torch.equal(a_new, a_ref)
    assert torch.equal(m_new, m_ref)
    # warm lookup returns the identical cached tensors
    a_again, _ = dec._get_training_anchors(shapes, memory)
    assert a_again is a_new


def test_cached_weighting_function_tracks_parameter_updates():
    dec = _make_det_decoder()
    inner = dec.decoder
    p1 = ec_decoder_mod._cached_weighting_function(
        inner, inner.reg_max, dec.up, dec.reg_scale
    )
    assert torch.equal(p1, weighting_function(inner.reg_max, dec.up, dec.reg_scale))
    p2 = ec_decoder_mod._cached_weighting_function(
        inner, inner.reg_max, dec.up, dec.reg_scale
    )
    assert p2 is p1
    # in-place parameter update (what load_state_dict does) must invalidate
    with torch.no_grad():
        dec.reg_scale.copy_(torch.tensor([8.0]))
    p3 = ec_decoder_mod._cached_weighting_function(
        inner, inner.reg_max, dec.up, dec.reg_scale
    )
    assert not torch.equal(p3, p1)
    assert torch.equal(p3, weighting_function(inner.reg_max, dec.up, dec.reg_scale))


# ---------------------------------------------------------------------------
# Training-path loss parity (denoising active)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "targets_fn", [_targets_empty, _targets_single, _targets_many]
)
def test_det_training_loss_parity(targets_fn):
    """EC detection loss values bitwise identical to the original control flow
    on fixed inputs, across empty / single / many target configurations."""
    dec = _make_det_decoder()
    dec.train()
    ref = _make_det_decoder(seed=99)
    ref.load_state_dict(dec.state_dict())
    ref.train()
    crit = _make_criterion()
    crit.train()

    feats = _make_feats()
    targets = targets_fn()

    torch.manual_seed(11)
    losses_cold = crit(dec(feats, targets=targets), targets)
    torch.manual_seed(11)
    losses_warm = crit(dec(feats, targets=targets), targets)
    with _original_control_flow():
        torch.manual_seed(11)
        losses_ref = crit(ref(feats, targets=targets), targets)

    assert losses_cold.keys() == losses_ref.keys()
    for k in losses_cold:
        assert torch.equal(losses_cold[k], losses_warm[k]), k
        assert torch.equal(losses_cold[k], losses_ref[k]), k


def test_pose_training_forward_parity():
    def make():
        torch.manual_seed(0)
        return ECPoseTransformer(
            hidden_dim=64,
            nhead=4,
            num_queries=10,
            num_decoder_layers=2,
            dim_feedforward=128,
            num_keypoints=17,
            eval_spatial_size=None,
        )

    def make_targets():
        g = torch.Generator().manual_seed(5)
        out = []
        for n in (2, 1):
            xy = torch.rand(n, 34, generator=g)
            vis = torch.randint(0, 3, (n, 17), generator=g).float()
            out.append(
                {
                    "labels": torch.zeros(n, dtype=torch.long),
                    "boxes": torch.rand(n, 4, generator=g) * 0.4 + 0.2,
                    "keypoints": torch.cat([xy, vis], dim=1),
                    "area": torch.rand(n, generator=g) * 100,
                    "orig_size": torch.tensor([64, 64]),
                }
            )
        return out

    dec = make()
    dec.train()
    ref = make()
    ref.load_state_dict(dec.state_dict())
    ref.train()

    torch.manual_seed(2)
    feats = [
        torch.randn(2, 64, 8, 8),
        torch.randn(2, 64, 4, 4),
        torch.randn(2, 64, 2, 2),
    ]
    samples = torch.randn(2, 3, 64, 64)
    targets = make_targets()

    torch.manual_seed(21)
    out_cold = dec(feats, targets=targets, samples=samples)
    torch.manual_seed(21)
    out_warm = dec(feats, targets=targets, samples=samples)
    with _original_control_flow():
        torch.manual_seed(21)
        out_ref = ref(feats, targets=targets, samples=samples)

    _assert_tree_equal(out_cold, out_warm)
    _assert_tree_equal(out_cold, out_ref)


# ---------------------------------------------------------------------------
# Sync-count regression
# ---------------------------------------------------------------------------


class _SyncCounter:
    """Counts Python-level item/tolist/cpu/numpy/nonzero calls per source file."""

    KINDS = ("item", "tolist", "cpu", "numpy", "nonzero")

    def __init__(self):
        self.counts = collections.Counter()
        self._orig = {}

    def _record(self):
        for fr in reversed(traceback.extract_stack()):
            f = fr.filename.replace("\\", "/")
            if "libreyolo/" in f and "test_ec_sync_reduction" not in f:
                self.counts[f.split("libreyolo/", 1)[-1]] += 1
                return
        self.counts["<outside>"] += 1

    def __enter__(self):
        rec = self._record
        for name in ("item", "tolist", "cpu", "numpy", "nonzero"):
            orig = getattr(torch.Tensor, name)
            self._orig[name] = orig

            def make(orig):
                def fn(self_, *a, **k):
                    rec()
                    return orig(self_, *a, **k)

                return fn

            setattr(torch.Tensor, name, make(orig))
        self._orig["torch.nonzero"] = torch.nonzero
        orig_nz = torch.nonzero

        def t_nonzero(*a, **k):
            rec()
            return orig_nz(*a, **k)

        torch.nonzero = t_nonzero
        return self

    def __exit__(self, *exc):
        for name in ("item", "tolist", "cpu", "numpy", "nonzero"):
            setattr(torch.Tensor, name, self._orig[name])
        torch.nonzero = self._orig["torch.nonzero"]
        return False

    def total(self, prefix):
        return sum(c for f, c in self.counts.items() if f.startswith(prefix))


def test_ec_decoder_sync_count_regression():
    """The EC decoder path must issue no Python-level syncs from ec/ files.

    Before this work: 17 sync calls per detection training step, all in
    libreyolo/models/dfine/ (matcher/loss/denoising); 0 in libreyolo/models/ec/.
    This test pins the ec/ contribution at 0 for the full forward and bounds
    the whole decoder forward (which still crosses dfine denoising) at 2 to
    leave headroom for the dfine-side work landing separately.
    """
    dec = _make_det_decoder()
    dec.train()
    feats = _make_feats()
    targets = _targets_many()

    torch.manual_seed(7)
    dec(feats, targets=targets)  # warm caches; not counted

    with _SyncCounter() as counter:
        torch.manual_seed(7)
        dec(feats, targets=targets)

    ec_count = counter.total("models/ec/")
    forward_count = sum(
        c for f, c in counter.counts.items() if f != "<outside>"
    )
    assert ec_count == 0, f"ec/ sync sites regressed: {counter.counts}"
    assert forward_count <= 2, f"decoder forward sync budget: {counter.counts}"
