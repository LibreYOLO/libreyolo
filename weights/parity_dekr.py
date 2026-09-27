"""Exact raw-parity gate for the LibreYOLO DEKR port.

Compares ``LibreDEKRModel`` against the *unmodified* upstream DEKR source. The
upstream file is executed verbatim under a minimal shim for the SuperGradients
framework plumbing it imports, so the architecture under comparison is literally
upstream's rather than a transcription of it.

Usage::

    # 1. Fetch the pinned upstream model source (not vendored: it is only an
    #    oracle, never a runtime dependency).
    curl -sL -o /tmp/dekr_hrnet.py https://raw.githubusercontent.com/Deci-AI/\
super-gradients/63de22c404d5740f34f7706c302b37fce3c8fe5d/src/super_gradients/\
training/models/pose_estimation_models/dekr_hrnet.py

    # 2. Fetch the released checkpoint.
    curl -sL -o /tmp/dekr.pth https://d2gjn4b69gu75n.cloudfront.net/models/\
dekr_w32_no_dc_coco_pose.pth

    DEKR_UPSTREAM_SOURCE=/tmp/dekr_hrnet.py DEKR_OFFICIAL_CKPT=/tmp/dekr.pth \
        python weights/parity_dekr.py

Requires ``max_abs_diff == 0.0`` on every probed stage and on both final
outputs, for zeros, seeded noise, batch 2 and a rectangular input.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libreyolo.models.dekr.nn import DEKR_W32_NO_DC_SPEC, LibreDEKRModel  # noqa: E402
from libreyolo.models.dekr.utils import (  # noqa: E402
    strip_module_prefix,
    unwrap_dekr_checkpoint,
)

PROBES = (
    "bn2",
    "layer1",
    "stage2",
    "stage3",
    "stage4",
    "transition_heatmap",
    "transition_offset",
    "head_heatmap.0",
    "offset_feature_layers.0",
    "offset_final_layer.16",
)


class AttrDict(dict):
    """Attribute + item access, mirroring the omegaconf config SuperGradients passes."""

    def __init__(self, mapping=None, **kwargs):
        super().__init__()
        for key, value in dict(mapping or {}, **kwargs).items():
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def to_dict(self):
        return dict(self)


def _install_stubs() -> None:
    """Stub only the framework plumbing, never the model code under test."""

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
        return mod

    def decorator(*_args, **_kwargs):
        return lambda obj: obj

    for pkg in (
        "super_gradients",
        "super_gradients.common",
        "super_gradients.common.decorators",
        "super_gradients.common.factories",
        "super_gradients.common.registry",
        "super_gradients.common.abstractions",
        "super_gradients.training",
        "super_gradients.training.models",
        "super_gradients.training.models.pose_estimation_models",
        "super_gradients.training.utils",
        "super_gradients.training.utils.media",
        "super_gradients.training.pipelines",
        "super_gradients.training.processing",
        "super_gradients.module_interfaces",
    ):
        module(pkg)

    module("super_gradients.common.decorators.factory_decorator", resolve_param=decorator)
    module("super_gradients.common.factories.processing_factory", ProcessingFactory=object)
    module("super_gradients.common.registry.registry", register_model=decorator)
    module(
        "super_gradients.common.object_names",
        Models=types.SimpleNamespace(DEKR_CUSTOM="a", DEKR_W32_NO_DC="b"),
    )
    module(
        "super_gradients.common.abstractions.abstract_logger",
        get_logger=lambda *a, **k: types.SimpleNamespace(
            error=lambda *a, **k: None, warning=lambda *a, **k: None
        ),
    )
    sys.modules["super_gradients.module_interfaces"].HasPredict = object
    module("super_gradients.training.utils.predict", ImagesPoseEstimationPrediction=object)
    module("super_gradients.training.models.sg_module", SgModule=nn.Module)
    module("super_gradients.training.models.arch_params_factory", get_arch_params=lambda n: {})
    module("super_gradients.training.pipelines.pipelines", PoseEstimationPipeline=object)
    module(
        "super_gradients.training.processing.processing",
        Processing=object,
        ComposeProcessing=object,
        KeypointsAutoPadding=object,
    )
    utils = sys.modules["super_gradients.training.utils"]
    utils.HpmStruct = AttrDict
    utils.DEKRPoseEstimationDecodeCallback = object
    utils.get_param = lambda p, n, d=None: (getattr(p, n, d) if getattr(p, n, d) is not None else d)
    module("super_gradients.training.utils.media.image", ImageSource=object)


def build_upstream(source_path: str, num_keypoints: int = 17) -> nn.Module:
    _install_stubs()
    source = Path(source_path).read_text(encoding="utf-8")
    upstream = types.ModuleType("upstream_dekr_hrnet")
    upstream.__file__ = source_path
    exec(compile(source, source_path, "exec"), upstream.__dict__)
    model = upstream.DEKRPoseEstimationModel(
        AttrDict({"SPEC": DEKR_W32_NO_DC_SPEC, "num_classes": num_keypoints, "in_channels": 3})
    )
    assert isinstance(model.heatmap_activation, nn.Identity), (
        "upstream must be built with HEATMAP_APPLY_SIGMOID=False (raw logits)"
    )
    return model


def main() -> None:
    source_path = os.environ.get("DEKR_UPSTREAM_SOURCE")
    checkpoint = os.environ.get("DEKR_OFFICIAL_CKPT")
    if not source_path or not checkpoint:
        raise SystemExit(
            "Set DEKR_UPSTREAM_SOURCE (pinned upstream dekr_hrnet.py) and "
            "DEKR_OFFICIAL_CKPT (dekr_w32_no_dc_coco_pose.pth). See the module "
            "docstring for the exact curl commands."
        )

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = strip_module_prefix(unwrap_dekr_checkpoint(raw))
    print(f"native entries: {len(state)}")

    upstream = build_upstream(source_path).eval()
    upstream.load_state_dict(state, strict=True)
    ours = LibreDEKRModel(num_keypoints=17).eval()
    ours.load_state_dict(state, strict=True)
    print("strict load into both models: OK")

    assert set(upstream.state_dict()) == set(ours.state_dict()), "key sets differ"
    print(f"state-dict key sets identical ({len(state)} entries)")

    captured: dict[str, dict[str, torch.Tensor]] = {"upstream": {}, "ours": {}}

    def attach(model, tag):
        modules = dict(model.named_modules())
        for name in PROBES:
            def hook(_module, _inputs, output, name=name, tag=tag):
                value = output[0] if isinstance(output, (list, tuple)) else output
                captured[tag][name] = value.detach().clone()

            modules[name].register_forward_hook(hook)

    attach(upstream, "upstream")
    attach(ours, "ours")

    cases = {
        "zeros": torch.zeros(1, 3, 640, 640),
        "seeded_randn": torch.randn(1, 3, 640, 640, generator=torch.Generator().manual_seed(0)),
        "batch2": torch.randn(2, 3, 640, 640, generator=torch.Generator().manual_seed(1)),
        "rect_512x384": torch.randn(1, 3, 512, 384, generator=torch.Generator().manual_seed(2)),
    }

    failures = []
    for name, x in cases.items():
        captured["upstream"].clear()
        captured["ours"].clear()
        with torch.no_grad():
            up_heatmap, up_offsets = upstream(x)
            our_heatmap, our_offsets = ours(x)

        for probe in PROBES:
            diff = (captured["upstream"][probe] - captured["ours"][probe]).abs().max().item()
            if diff != 0.0:
                failures.append(f"{name}/{probe}={diff}")

        heatmap_diff = (up_heatmap - our_heatmap).abs().max().item()
        offset_diff = (up_offsets - our_offsets).abs().max().item()
        ok = heatmap_diff == 0.0 and offset_diff == 0.0
        print(
            f"{'OK  ' if ok else 'FAIL'} {name:14s} in={tuple(x.shape)} "
            f"heatmap={tuple(our_heatmap.shape)} offsets={tuple(our_offsets.shape)} "
            f"max_abs_diff(heatmap)={heatmap_diff} max_abs_diff(offsets)={offset_diff}"
        )
        if not ok:
            failures.append(f"{name}/final heatmap={heatmap_diff} offsets={offset_diff}")

    if failures:
        raise SystemExit(f"PARITY FAILED: {failures}")
    print("\nALL PARITY CHECKS PASSED (max_abs_diff == 0.0 everywhere)")


if __name__ == "__main__":
    main()
