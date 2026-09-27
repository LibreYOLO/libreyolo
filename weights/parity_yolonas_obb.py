"""Inference parity: LibreYOLO YOLO-NAS-R (OBB) vs the pinned upstream head.

The oracle is upstream's *actual* source, not a re-description of it: this
script executes the two pinned SuperGradients files

    src/super_gradients/training/models/detection_models/yolo_nas_r/
        yolo_nas_r_dfl_head.py
        yolo_nas_r_ndfl_heads.py

from https://github.com/Deci-AI/super-gradients (Apache-2.0) at commit
``69141b55c1161d939939a270523a7eca5a645f72`` with the handful of
SuperGradients framework symbols they import replaced by minimal stubs
(a registry decorator, a module factory, ``width_multiplier``, and the
detection ``YoloNASDFLHead`` the rotated head subclasses -- the latter taken
from LibreYOLO, whose YOLO-NAS detect port is already parity-proven).
Installing SuperGradients itself is not possible on this toolchain (its
``onnx`` build dependency needs cmake), so the files are run directly.

The backbone and neck are byte-identical between the detect and rotated
arch_params, so they are LibreYOLO's own and are shared by both sides of the
comparison; what is under test is the new rotated head and its decode.

Usage:
    export YOLONAS_R_UPSTREAM_SRC=/path/with/the/two/pinned/py/files
    export YOLONAS_R_OFFICIAL_CKPT_DIR=/path/with/yolo_nas_r_*_dota2.pth
    python weights/parity_yolonas_obb.py
"""

from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libreyolo.models.yolonas.nn import (  # noqa: E402
    LibreYOLONASOBBModel,
    YoloNASDFLHead,
    width_multiplier,
)
from libreyolo.models.yolonas.utils import unwrap_yolonas_checkpoint  # noqa: E402

UPSTREAM_SRC = os.environ.get("YOLONAS_R_UPSTREAM_SRC")
CKPT_DIR = os.environ.get("YOLONAS_R_OFFICIAL_CKPT_DIR")

UPSTREAM_COMMIT = "69141b55c1161d939939a270523a7eca5a645f72"
HEAD_INTER_CHANNELS = (128, 256, 512)
HEAD_STRIDES = (8, 16, 32)
WIDTH_MULT = {"s": 0.5, "m": 0.75, "l": 1.0}


def _install_super_gradients_stubs() -> None:
    """Minimal stand-ins for the framework symbols the pinned files import."""

    def module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    for name in (
        "super_gradients",
        "super_gradients.common",
        "super_gradients.common.factories",
        "super_gradients.common.registry",
        "super_gradients.module_interfaces",
        "super_gradients.modules",
        "super_gradients.modules.base_modules",
        "super_gradients.modules.utils",
        "super_gradients.training",
        "super_gradients.training.models",
        "super_gradients.training.models.detection_models",
        "super_gradients.training.models.detection_models.yolo_nas",
        "super_gradients.training.utils",
        "omegaconf",
    ):
        module(name)

    sys.modules["super_gradients.common.registry"].register_detection_module = (
        lambda *a, **k: lambda cls: cls
    )

    class SupportsReplaceNumClasses:  # noqa: D401 - marker interface upstream
        pass

    sys.modules[
        "super_gradients.module_interfaces"
    ].SupportsReplaceNumClasses = SupportsReplaceNumClasses

    class BaseDetectionModule(nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.in_channels = in_channels

    sys.modules[
        "super_gradients.modules.base_modules"
    ].BaseDetectionModule = BaseDetectionModule
    sys.modules["super_gradients.modules.utils"].width_multiplier = width_multiplier
    sys.modules[
        "super_gradients.training.models.detection_models.yolo_nas"
    ].YoloNASDFLHead = YoloNASDFLHead

    class HpmStruct(dict):
        pass

    sys.modules["super_gradients.training.utils"].HpmStruct = HpmStruct
    sys.modules["super_gradients.training.utils"].torch_version_is_greater_or_equal = (
        lambda *_: True
    )
    sys.modules["omegaconf"].DictConfig = dict

    # The detection-module factory: upstream builds each rotated head through
    # it, so the stub just applies the injected params to the head class.
    factory_mod = module("super_gradients.common.factories.detection_modules_factory")

    class DetectionModulesFactory:
        head_cls = None  # set after the upstream module is executed

        @staticmethod
        def insert_module_param(cfg, name, value):
            cfg = dict(cfg)
            cfg[name] = value
            return cfg

        def get(self, cfg):
            return DetectionModulesFactory.head_cls(**cfg)

    factory_mod.DetectionModulesFactory = DetectionModulesFactory
    sys.modules[
        "super_gradients.common.factories"
    ].detection_modules_factory = factory_mod
    return DetectionModulesFactory


def _load_upstream_modules(src_dir: Path):
    factory_cls = _install_super_gradients_stubs()

    def run(filename: str, module_name: str) -> types.ModuleType:
        path = src_dir / filename
        if not path.exists():
            raise SystemExit(f"Missing pinned upstream file: {path}")
        mod = types.ModuleType(module_name)
        mod.__file__ = str(path)
        sys.modules[module_name] = mod
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), mod.__dict__)
        return mod

    dfl = run(
        "yolo_nas_r_dfl_head.py",
        "super_gradients.training.models.detection_models.yolo_nas_r.yolo_nas_r_dfl_head",
    )
    factory_cls.head_cls = dfl.YoloNASRDFLHead
    ndfl = run(
        "yolo_nas_r_ndfl_heads.py",
        "super_gradients.training.models.detection_models.yolo_nas_r.yolo_nas_r_ndfl_heads",
    )
    return ndfl.YoloNASRNDFLHeads


def _build_upstream_heads(heads_cls, size: str, in_channels, num_classes: int):
    heads_list = [
        {
            "inter_channels": inter,
            "width_mult": WIDTH_MULT[size],
            "first_conv_group_size": 0,
            "stride": stride,
        }
        for inter, stride in zip(HEAD_INTER_CHANNELS, HEAD_STRIDES)
    ]
    heads = heads_cls(
        num_classes=num_classes,
        in_channels=list(in_channels),
        heads_list=heads_list,
        reg_max=16,
    )
    # arch_params ship bn_eps: 1e-3 / bn_momentum: 0.03 and upstream applies
    # them model-wide after assembly. Built standalone here, the heads would
    # keep torch's 1e-5 default and the comparison would measure the harness
    # rather than the port.
    for m in heads.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
    return heads


def main() -> None:
    if not UPSTREAM_SRC or not CKPT_DIR:
        raise SystemExit(
            "Set YOLONAS_R_UPSTREAM_SRC (pinned upstream .py files, commit "
            f"{UPSTREAM_COMMIT}) and YOLONAS_R_OFFICIAL_CKPT_DIR "
            "(yolo_nas_r_{s,m,l}_dota2.pth)."
        )

    heads_cls = _load_upstream_modules(Path(UPSTREAM_SRC))
    torch.manual_seed(0)
    failures = []

    for size in ("s", "m", "l"):
        ckpt_path = Path(CKPT_DIR) / f"yolo_nas_r_{size}_dota2.pth"
        state = dict(
            unwrap_yolonas_checkpoint(
                torch.load(ckpt_path, map_location="cpu", weights_only=False)
            )
        )

        ours = LibreYOLONASOBBModel(config=size, nb_classes=18).eval()
        result = ours.load_state_dict(state, strict=True)
        assert not getattr(result, "missing_keys", []), result

        upstream_heads = _build_upstream_heads(
            heads_cls, size, ours.neck.out_channels, 18
        ).eval()
        head_state = {
            k[len("heads.") :]: v for k, v in state.items() if k.startswith("heads.")
        }
        missing, unexpected = upstream_heads.load_state_dict(head_state, strict=False)
        if missing or unexpected:
            failures.append(f"size={size} upstream head load: {missing} / {unexpected}")
            continue

        x = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            feats = ours.neck(ours.backbone(x))
            up_logits = upstream_heads(feats)
            up = up_logits.as_decoded()
            (our_boxes, our_scores), _raw = ours.heads(feats)

        box_diff = (up.boxes_cxcywhr - our_boxes).abs().max().item()
        score_diff = (up.scores - our_scores).abs().max().item()
        angle_max = our_boxes[..., 4].max().item()
        angle_min = our_boxes[..., 4].min().item()
        ok = box_diff == 0.0 and score_diff == 0.0
        print(
            f"size={size}: boxes max_abs_diff={box_diff} "
            f"scores max_abs_diff={score_diff} "
            # Upstream's docstring says [-3*pi/4, pi/4], but its own code is
            # ``(sigmoid(x) - 0.25) * pi``, i.e. (-pi/4, 3*pi/4). The port
            # reproduces the code, not the comment.
            f"angle_range=[{angle_min:.4f}, {angle_max:.4f}] "
            f"(upstream code range ({-math.pi / 4:.4f}, {3 * math.pi / 4:.4f})) "
            f"-> {'OK' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(f"size={size}: boxes={box_diff} scores={score_diff}")

    if failures:
        raise SystemExit("PARITY FAILED:\n" + "\n".join(failures))
    print("All sizes: exact parity (max_abs_diff == 0) vs pinned upstream head.")


if __name__ == "__main__":
    main()
