"""Convert open-mmlab RTMDet checkpoints to LibreRTMDet.

Usage::

    python weights/convert_rtmdet_weights.py \\
        /path/to/rtmdet_tiny_8xb32-300e_coco_*.pth \\
        weights/LibreRTMDett.pt --size t

The upstream `.pth` is mmengine-pickled and contains EMA weights. We:

1. Load with stub modules so we don't need mmengine/mmcv installed.
2. Prefer ``ema_state_dict`` over ``state_dict`` (EMA params are what mmdet uses
   for evaluation; the published checkpoints expose both).
3. Drop ``data_preprocessor.*`` and ``num_batches_tracked`` keys we don't need.
4. Rename ``bbox_head.`` -> ``head.`` so it matches our nn.py attribute names.
5. With ``share_conv=True``, upstream stores the same conv weight at all 3 levels
   (cls_convs.0/1/2[i].conv.weight all hold identical values). After loading
   into our model, the aliasing in nn.py means later writes overwrite earlier
   ones with the same value, which is fine. We keep all redundant entries to
   avoid having to know the aliasing structure inside the converter.
6. Wrap with ``wrap_libreyolo_checkpoint`` and write atomically.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Any

# Stub mmengine/mmcv before torch.load so unpickling succeeds without the
# upstream packages installed.
class _ModuleProxy(types.ModuleType):
    def __init__(self, name: str):
        super().__init__(name)
        self.__path__ = []

    def __getattr__(self, attr: str):
        if attr.startswith("__"):
            raise AttributeError(attr)
        cls = type(attr, (), {})
        cls.__module__ = self.__name__
        setattr(self, attr, cls)
        return cls


_STUB_NAMES = [
    "mmengine",
    "mmengine.config",
    "mmengine.config.config",
    "mmengine.fileio",
    "mmengine.logging",
    "mmengine.logging.message_hub",
    "mmengine.logging.history_buffer",
    "mmengine.utils",
    "mmengine.utils.path",
    "mmengine.runner",
    "mmengine.dataset",
    "mmcv",
    "mmcv.transforms",
    "mmcv.transforms.base",
    "mmdet",
    "mmdet.structures",
    "mmdet.datasets",
]
for _name in _STUB_NAMES:
    if _name not in sys.modules:
        sys.modules[_name] = _ModuleProxy(_name)

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weights._conversion_utils import (  # noqa: E402
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)
from libreyolo.models.rtmdet.nn import LibreRTMDetModel  # noqa: E402


_DROP_PREFIXES = ("data_preprocessor.",)


def _select_state_dict(ckpt: dict) -> dict:
    """Prefer EMA weights, fall back to plain state_dict."""
    if "ema_state_dict" in ckpt:
        sd = ckpt["ema_state_dict"]
        # mmengine ExpMomentumEMA prefixes module params with "module."
        return {k[len("module.") :]: v for k, v in sd.items() if k.startswith("module.")}
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def _remap_keys(sd: dict) -> dict:
    """Apply the bbox_head -> head rename and drop data_preprocessor."""
    out = {}
    for k, v in sd.items():
        if any(k.startswith(p) for p in _DROP_PREFIXES):
            continue
        new_key = k
        if k.startswith("bbox_head."):
            new_key = "head." + k[len("bbox_head.") :]
        out[new_key] = v
    return out


def convert(
    input_path: str | Path,
    output_path: str | Path,
    size: str,
    nc: int = 80,
) -> Path:
    print(f"[convert_rtmdet] loading {input_path}")
    ckpt: Any = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected checkpoint layout: {type(ckpt)}")

    raw_sd = _select_state_dict(ckpt)
    print(f"[convert_rtmdet] source has {len(raw_sd)} parameters")

    libre_sd = _remap_keys(raw_sd)

    # Sanity: build a fresh model and check what loads cleanly.
    model = LibreRTMDetModel(size=size, nc=nc)
    incompat = model.load_state_dict(libre_sd, strict=False)
    missing = [k for k in incompat.missing_keys if not k.endswith("num_batches_tracked")]
    unexpected = [
        k for k in incompat.unexpected_keys if not k.endswith("num_batches_tracked")
    ]
    if missing:
        print(f"[convert_rtmdet] WARNING: {len(missing)} missing keys (showing 10):")
        for k in missing[:10]:
            print(f"    - {k}")
    if unexpected:
        print(f"[convert_rtmdet] WARNING: {len(unexpected)} unexpected keys (showing 10):")
        for k in unexpected[:10]:
            print(f"    + {k}")
    if not missing and not unexpected:
        print(f"[convert_rtmdet] clean load, all {len(libre_sd)} keys matched")

    libre_sd = model.state_dict()  # canonical: post-load (handles aliasing)

    wrapped = wrap_libreyolo_checkpoint(
        libre_sd,
        model_family="rtmdet",
        size=size,
        nc=nc,
        task="detect",
        supported_tasks=("detect",),
        default_task="detect",
    )

    out = Path(output_path)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.rename(out)
    print(f"[convert_rtmdet] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="upstream rtmdet *.pth path")
    parser.add_argument("output", help="output LibreRTMDet*.pt path")
    parser.add_argument(
        "--size",
        required=True,
        choices=["t", "s", "m", "l", "x"],
        help="model size",
    )
    parser.add_argument("--nc", type=int, default=80, help="number of classes")
    args = parser.parse_args()

    convert(args.input, args.output, size=args.size, nc=args.nc)


if __name__ == "__main__":
    main()
