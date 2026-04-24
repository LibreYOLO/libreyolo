"""Loss-level parity between LibreDFINE and upstream D-FINE.

Builds both ``DFINECriterion`` instances with identical matcher + weight_dict
+ losses, feeds the same fixed outputs+targets, and asserts every loss key
matches within 1e-5.

Skipped if the reference repo is missing.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.unit

from libreyolo.models.dfine.loss import DFINECriterion
from libreyolo.models.dfine.matcher import HungarianMatcher

_DFINE_REF_PATH = Path(
    os.environ.get(
        "LIBREYOLO_DFINE_REF_PATH",
        "/Users/xuban.ceccon/dfine-libreyolo-review/D-FINE",
    )
)


def _load_ref_loss_modules():
    """Import the upstream criterion + matcher as standalone modules.

    Sidesteps the reference's heavyweight package imports (tensorboard, yaml
    workspace, etc.) by loading individual files directly.
    """
    dfine_src = _DFINE_REF_PATH / "src"
    if not dfine_src.is_dir():
        pytest.skip(f"D-FINE reference not at {_DFINE_REF_PATH}")

    pkg_name = "_dfine_ref_loss"
    pkg = sys.modules.get(pkg_name)
    if pkg is None:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(dfine_src)]
        sys.modules[pkg_name] = pkg

    # Stub the @register decorator the reference uses.
    core_mod_name = f"{pkg_name}.core"
    if core_mod_name not in sys.modules:
        core_mod = types.ModuleType(core_mod_name)
        core_mod.__path__ = [str(dfine_src / "core")]

        def register(*a, **kw):
            def deco(cls):
                return cls

            return deco

        core_mod.register = register
        sys.modules[core_mod_name] = core_mod
        setattr(pkg, "core", core_mod)

    # Stub misc.dist_utils — only is_dist_available_and_initialized + get_world_size
    # are referenced by the criterion.
    misc_pkg_name = f"{pkg_name}.misc"
    dist_utils_name = f"{pkg_name}.misc.dist_utils"
    if misc_pkg_name not in sys.modules:
        misc_mod = types.ModuleType(misc_pkg_name)
        misc_mod.__path__ = [str(dfine_src / "misc")]
        sys.modules[misc_pkg_name] = misc_mod
        setattr(pkg, "misc", misc_mod)
    if dist_utils_name not in sys.modules:
        dist_mod = types.ModuleType(dist_utils_name)

        def is_dist_available_and_initialized():
            return False

        def get_world_size():
            return 1

        dist_mod.is_dist_available_and_initialized = is_dist_available_and_initialized
        dist_mod.get_world_size = get_world_size
        sys.modules[dist_utils_name] = dist_mod
        setattr(sys.modules[misc_pkg_name], "dist_utils", dist_mod)

    def _load(modpath: str, relpath: str):
        fq = f"{pkg_name}.{modpath}"
        if fq in sys.modules:
            return sys.modules[fq]
        # Ensure parent packages exist.
        parts = modpath.split(".")
        parent = pkg_name
        for part in parts[:-1]:
            qual = f"{parent}.{part}"
            if qual not in sys.modules:
                sub = types.ModuleType(qual)
                sub.__path__ = [str((dfine_src / relpath).parent.parent)]
                sys.modules[qual] = sub
                setattr(sys.modules[parent], part, sub)
            parent = qual
        spec = importlib.util.spec_from_file_location(fq, dfine_src / relpath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[fq] = module
        if "." in modpath:
            setattr(sys.modules[parent], modpath.rsplit(".", 1)[-1], module)
        spec.loader.exec_module(module)
        return module

    _load("zoo.dfine.box_ops", "zoo/dfine/box_ops.py")
    _load("zoo.dfine.dfine_utils", "zoo/dfine/dfine_utils.py")
    matcher_mod = _load("zoo.dfine.matcher", "zoo/dfine/matcher.py")
    criterion_mod = _load("zoo.dfine.dfine_criterion", "zoo/dfine/dfine_criterion.py")

    return matcher_mod.HungarianMatcher, criterion_mod.DFINECriterion


def _make_synthetic_inputs(num_classes=80, num_queries=300, batch=2, reg_max=32):
    """Construct a deterministic outputs dict + targets list shaped like a real
    forward pass in training mode.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")
    hidden_dim = 128

    def make_outputs(seed_offset):
        torch.manual_seed(seed_offset)
        return {
            "pred_logits": torch.randn(batch, num_queries, num_classes, device=device),
            "pred_boxes": torch.rand(batch, num_queries, 4, device=device),
            "pred_corners": torch.randn(
                batch, num_queries, 4 * (reg_max + 1), device=device
            ),
            "ref_points": torch.rand(batch, num_queries, 4, device=device),
            "up": torch.tensor([0.5], device=device),
            "reg_scale": torch.tensor([4.0], device=device),
        }

    main = make_outputs(0)

    # Two aux layers.
    main["aux_outputs"] = [make_outputs(1 + i) for i in range(2)]
    # Pre-output (first decoder layer).
    main["pre_outputs"] = {
        "pred_logits": torch.randn(batch, num_queries, num_classes, device=device),
        "pred_boxes": torch.rand(batch, num_queries, 4, device=device),
    }
    # Encoder aux outputs.
    main["enc_aux_outputs"] = [
        {
            "pred_logits": torch.randn(batch, num_queries, num_classes, device=device),
            "pred_boxes": torch.rand(batch, num_queries, 4, device=device),
        }
    ]
    main["enc_meta"] = {"class_agnostic": False}

    # Add teacher_corners to main + each aux for the DDF branch (D-FINE wires
    # this in `_set_aux_loss2`; emulate the same layout).
    teacher_corners = main["pred_corners"].clone()
    teacher_logits = main["pred_logits"].clone()
    for entry in [main] + main["aux_outputs"]:
        entry["teacher_corners"] = teacher_corners
        entry["teacher_logits"] = teacher_logits

    # Two images, each with a couple of GT boxes.
    targets = [
        {
            "labels": torch.tensor([3, 17, 56], dtype=torch.int64, device=device),
            "boxes": torch.tensor(
                [
                    [0.30, 0.30, 0.20, 0.20],
                    [0.60, 0.50, 0.10, 0.10],
                    [0.50, 0.80, 0.15, 0.10],
                ],
                device=device,
            ),
        },
        {
            "labels": torch.tensor([1, 25], dtype=torch.int64, device=device),
            "boxes": torch.tensor(
                [
                    [0.40, 0.40, 0.30, 0.20],
                    [0.70, 0.20, 0.05, 0.05],
                ],
                device=device,
            ),
        },
    ]

    return main, targets


def test_loss_parity():
    RefHungarianMatcher, RefDFINECriterion = _load_ref_loss_modules()

    weight_dict = {
        "loss_vfl": 1.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
        "loss_fgl": 0.15,
        "loss_ddf": 1.5,
    }
    matcher_cost_dict = {"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0}

    ours_matcher = HungarianMatcher(
        weight_dict=matcher_cost_dict, use_focal_loss=True, alpha=0.25, gamma=2.0
    )
    theirs_matcher = RefHungarianMatcher(
        weight_dict=matcher_cost_dict, use_focal_loss=True, alpha=0.25, gamma=2.0
    )

    ours = DFINECriterion(
        matcher=ours_matcher,
        weight_dict=weight_dict,
        losses=["vfl", "boxes", "local"],
        alpha=0.75,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
    )
    theirs = RefDFINECriterion(
        matcher=theirs_matcher,
        weight_dict=weight_dict,
        losses=["vfl", "boxes", "local"],
        alpha=0.75,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
    )

    outputs1, targets = _make_synthetic_inputs()

    # The criterion mutates outputs (it adds "is_dn" / "up"/"reg_scale" to aux);
    # build a fresh copy for the second run.
    import copy as _copy

    outputs2 = _copy.deepcopy(outputs1)

    losses_ours = ours(outputs1, targets)
    losses_theirs = theirs(outputs2, targets)

    assert set(losses_ours.keys()) == set(losses_theirs.keys()), (
        f"Key mismatch:\n  ours - theirs = {set(losses_ours) - set(losses_theirs)}\n"
        f"  theirs - ours = {set(losses_theirs) - set(losses_ours)}"
    )

    diffs = {}
    for k in sorted(losses_ours.keys()):
        a = float(losses_ours[k].item())
        b = float(losses_theirs[k].item())
        diffs[k] = abs(a - b)
        assert abs(a - b) < 1e-5, f"{k}: ours={a:.6f} theirs={b:.6f} (|Δ|={abs(a-b):.2e})"

    # Also check the aggregated weighted sum matches.
    sum_ours = sum(v.item() for v in losses_ours.values())
    sum_theirs = sum(v.item() for v in losses_theirs.values())
    assert abs(sum_ours - sum_theirs) < 1e-4, (
        f"sum: ours={sum_ours:.4f} theirs={sum_theirs:.4f}"
    )
