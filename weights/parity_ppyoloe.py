"""Developer-only exact-parity harness for the PP-YOLOE port.

Cross-loads a released checkpoint into the pinned source implementation and
into ``LibrePPYOLOEModel``, then asserts ``max_abs_diff == 0`` on decoded
boxes, class scores, the training-form raw tuple, and the anchor tensors.

The source implementation is not a runtime dependency of LibreYOLO. This
script fetches the four architecture files from the pinned commit into a
scratch directory and imports them with the framework's registry/factory
plumbing stubbed out, so the compared code is the upstream code.

Usage::

    export PPYOLOE_OFFICIAL_CKPT_DIR=/path/with/ppyoloe_{s,m,l,x}_coco.pth
    python weights/parity_ppyoloe.py            # all four sizes
    python weights/parity_ppyoloe.py s m        # a subset

Exits non-zero unless every required tensor is exactly equal.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import torch
from torch import nn

SOURCE_REVISION = "63de22c404d5740f34f7706c302b37fce3c8fe5d"
_RAW = f"https://raw.githubusercontent.com/Deci-AI/super-gradients/{SOURCE_REVISION}/src/super_gradients"

_FILES = {
    "conv_bn_act_block.py": "modules/conv_bn_act_block.py",
    "repvgg_block.py": "modules/repvgg_block.py",
    "se_blocks.py": "modules/se_blocks.py",
    "bbox_utils.py": "training/utils/bbox_utils.py",
    "csp_resnet.py": "training/models/detection_models/csp_resnet.py",
    "pan.py": "training/models/detection_models/pp_yolo_e/pan.py",
    "pp_yolo_head.py": "training/models/detection_models/pp_yolo_e/pp_yolo_head.py",
}

MULTS = {"s": (0.33, 0.50), "m": (0.67, 0.75), "l": (1.00, 1.00), "x": (1.33, 1.25)}


def _fetch_sources(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, remote in _FILES.items():
        target = dest / name
        if target.exists():
            continue
        with urllib.request.urlopen(f"{_RAW}/{remote}") as response:
            target.write_bytes(response.read())


def _install_stubs() -> None:
    """Provide the super_gradients plumbing the four files import."""
    import contextlib

    def module(name):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    for name in [
        "super_gradients",
        "super_gradients.common",
        "super_gradients.common.registry",
        "super_gradients.common.registry.registry",
        "super_gradients.common.decorators",
        "super_gradients.common.decorators.factory_decorator",
        "super_gradients.common.factories",
        "super_gradients.common.factories.activations_type_factory",
        "super_gradients.common.environment",
        "super_gradients.common.environment.ddp_utils",
        "super_gradients.module_interfaces",
        "super_gradients.modules",
        "super_gradients.modules.utils",
        "super_gradients.modules.weight_replacement_utils",
        "super_gradients.training",
        "super_gradients.training.models",
        "super_gradients.training.models.detection_models",
        "super_gradients.training.models.detection_models.pp_yolo_e",
        "super_gradients.training.utils",
        "super_gradients.training.utils.distributed_training_utils",
        "super_gradients.training.utils.utils",
        "super_gradients.training.utils.version_utils",
    ]:
        module(name)

    def passthrough(*_a, **_k):
        return lambda obj: obj

    registry = sys.modules["super_gradients.common.registry.registry"]
    registry.register_detection_module = passthrough
    registry.register_model = passthrough
    sys.modules["super_gradients.common.decorators.factory_decorator"].resolve_param = passthrough
    sys.modules["super_gradients.common.factories.activations_type_factory"].ActivationsTypeFactory = object
    sys.modules["super_gradients.module_interfaces"].SupportsReplaceInputChannels = type(
        "SupportsReplaceInputChannels", (), {}
    )
    sys.modules["super_gradients.modules.utils"].autopad = lambda k, p=None: p
    sys.modules["super_gradients.modules.weight_replacement_utils"].replace_conv2d_input_channels = None

    @contextlib.contextmanager
    def wait_for_the_master(*_a, **_k):
        yield

    sys.modules["super_gradients.training.utils.distributed_training_utils"].wait_for_the_master = wait_for_the_master
    sys.modules["super_gradients.common.environment.ddp_utils"].get_local_rank = lambda: 0
    utils = sys.modules["super_gradients.training.utils.utils"]
    utils.infer_model_device = lambda m: next(m.parameters()).device
    utils.infer_model_dtype = lambda m: next(m.parameters()).dtype
    sys.modules["super_gradients.training.utils.version_utils"].torch_version_is_greater_or_equal = lambda *_a: True


def _load(directory: Path, filename: str, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, directory / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_upstream(directory: Path, size: str, num_classes: int = 80) -> nn.Module:
    _install_stubs()
    conv_bn_act = _load(directory, "conv_bn_act_block.py", "super_gradients.modules.conv_bn_act_block")
    repvgg = _load(directory, "repvgg_block.py", "super_gradients.modules.repvgg_block")
    se_blocks = _load(directory, "se_blocks.py", "super_gradients.modules.se_blocks")

    modules_pkg = sys.modules["super_gradients.modules"]
    modules_pkg.ConvBNAct = conv_bn_act.ConvBNAct
    modules_pkg.Conv = conv_bn_act.Conv
    modules_pkg.RepVGGBlock = repvgg.RepVGGBlock
    modules_pkg.EffectiveSEBlock = se_blocks.EffectiveSEBlock

    bbox_utils = _load(directory, "bbox_utils.py", "super_gradients.training.utils.bbox_utils")
    sys.modules["super_gradients.training.utils"].bbox_utils = bbox_utils

    csp = _load(directory, "csp_resnet.py", "super_gradients.training.models.detection_models.csp_resnet")
    pan = _load(directory, "pan.py", "super_gradients.training.models.detection_models.pp_yolo_e.pan")
    head = _load(directory, "pp_yolo_head.py", "super_gradients.training.models.detection_models.pp_yolo_e.pp_yolo_head")

    depth_mult, width_mult = MULTS[size]

    class UpstreamPPYoloE(nn.Module):
        """``PPYoloE.__init__`` / ``forward`` without the SG base classes."""

        def __init__(self):
            super().__init__()
            self.backbone = csp.CSPResNetBackbone(
                layers=[3, 6, 6, 3],
                channels=[64, 128, 256, 512, 1024],
                activation=nn.SiLU,
                return_idx=[1, 2, 3],
                use_large_stem=True,
                use_alpha=False,
                pretrained_weights=None,
                depth_mult=depth_mult,
                width_mult=width_mult,
            )
            self.neck = pan.PPYoloECSPPAN(
                in_channels=[256, 512, 1024],
                out_channels=[768, 384, 192],
                activation=nn.SiLU,
                block_num=3,
                stage_num=1,
                spp=True,
                depth_mult=depth_mult,
                width_mult=width_mult,
            )
            self.head = head.PPYOLOEHead(
                in_channels=[768, 384, 192],
                activation=nn.SiLU,
                fpn_strides=[32, 16, 8],
                grid_cell_scale=5.0,
                grid_cell_offset=0.5,
                reg_max=16,
                eval_size=None,
                width_mult=width_mult,
                num_classes=num_classes,
            )

        def forward(self, x):
            return self.head(self.neck(self.backbone(x)))

    return UpstreamPPYoloE()


def main() -> int:
    ckpt_dir = os.environ.get("PPYOLOE_OFFICIAL_CKPT_DIR")
    if not ckpt_dir:
        raise SystemExit(
            "Set PPYOLOE_OFFICIAL_CKPT_DIR to the directory holding "
            "ppyoloe_{s,m,l,x}_coco.pth"
        )
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from libreyolo.models.ppyoloe.convert import convert_upstream, unwrap_ppyoloe_checkpoint
    from libreyolo.models.ppyoloe.nn import LibrePPYOLOEModel

    sizes = sys.argv[1:] or list(MULTS)
    sources = Path(tempfile.gettempdir()) / f"ppyoloe_parity_{SOURCE_REVISION[:8]}"
    _fetch_sources(sources)

    torch.manual_seed(0)
    probes = {
        "zeros": torch.zeros(1, 3, 640, 640),
        "randn": torch.randn(1, 3, 640, 640),
        "batch2": torch.randn(2, 3, 640, 640),
        "nonsquare": torch.randn(1, 3, 640, 960),
    }

    failures = []
    for size in sizes:
        raw = torch.load(
            Path(ckpt_dir) / f"ppyoloe_{size}_coco.pth", map_location="cpu", weights_only=False
        )
        state = convert_upstream(unwrap_ppyoloe_checkpoint(raw))

        upstream = build_upstream(sources, size)
        upstream.load_state_dict(state, strict=True)
        upstream.eval()

        ours = LibrePPYOLOEModel(size=size, nb_classes=80)
        ours.load_state_dict(state, strict=True)
        ours.eval()

        for name, x in probes.items():
            with torch.no_grad():
                (up_boxes, up_scores), up_raw = upstream(x)
                (our_boxes, our_scores), our_raw = ours(x)
            checks = {
                "boxes": (up_boxes - our_boxes).abs().max().item(),
                "scores": (up_scores - our_scores).abs().max().item(),
                "cls_logits": (up_raw[0] - our_raw[0]).abs().max().item(),
                "reg_distri": (up_raw[1] - our_raw[1]).abs().max().item(),
                "anchors": (up_raw[2] - our_raw[2]).abs().max().item(),
                "anchor_points": (up_raw[3] - our_raw[3]).abs().max().item(),
                "strides": (up_raw[5] - our_raw[5]).abs().max().item(),
            }
            bad = {k: v for k, v in checks.items() if v != 0.0}
            print(f"[{size}/{name}] {checks}")
            if bad:
                failures.append((size, name, bad))

        upstream.train()
        ours.train()
        with torch.no_grad():
            up_train = upstream(probes["batch2"])
            our_train = ours(probes["batch2"])
        train_diff = max((a - b).abs().max().item() for a, b in zip(up_train[:4], our_train[:4]))
        counts_match = list(up_train[4]) == list(our_train[4])
        print(f"[{size}/train-form] max_abs_diff={train_diff} num_anchors_match={counts_match}")
        if train_diff != 0.0 or not counts_match:
            failures.append((size, "train-form", train_diff))

    if failures:
        print("PARITY FAIL", failures)
        return 1
    print("PARITY PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
