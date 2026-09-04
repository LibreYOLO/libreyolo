"""Developer-only U-Net parity harness: LibreYOLO port vs pinned upstream mmseg.

Builds the real ``EncoderDecoder`` from the pinned mmsegmentation config
(``unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py`` at commit
``b040e147adfa``), loads the official Cityscapes checkpoint strictly into both
graphs, and asserts bit-identical logits on shared inputs. Exits non-zero on
any mismatch.

The oracle is not a LibreYOLO dependency. Run this in a separate environment
that has ``mmengine``, ``mmcv`` (``mmcv-lite`` is enough: the UNet + FCN path
never calls a compiled op) and the pinned mmsegmentation checkout installed::

    UNET_MMSEG_ROOT=/path/to/mmsegmentation \
    UNET_OFFICIAL_CKPT=/path/to/fcn_unet_s5-d16_..._6860854e.pth \
    UNET_PARITY_IMAGES=/path/a.png:/path/b.jpg \
    python weights/parity_unet.py

Gates:

* eval main logits and eval auxiliary logits: ``max_abs_diff == 0.0`` on zeros,
  uniform noise, and every image fixture, at the 1024x2048 evaluation canvas
  and at the 512x1024 training crop;
* every encoder stage, every decoder stage and the head features: ``0.0``;
* train-mode (batch statistics) main + auxiliary logits with dropout disabled
  on both sides: ``0.0``;
* end-to-end ``mmseg.apis.inference_model`` vs ``LibreUNet.predict`` on a
  Cityscapes-aspect image: identical class-id maps.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

from _conversion_utils import add_repo_root_to_path
from convert_unet_weights import SOURCE_DIGEST, convert, sha256

add_repo_root_to_path()

from libreyolo.models.unet.convert import convert_upstream  # noqa: E402
from libreyolo.models.unet.nn import SIZE_CONFIGS, LibreUNetNet  # noqa: E402

MMSEG_ROOT = os.environ.get("UNET_MMSEG_ROOT")
CKPT_PATH = os.environ.get("UNET_OFFICIAL_CKPT")
IMAGE_PATHS = [p for p in os.environ.get("UNET_PARITY_IMAGES", "").split(":") if p]
CONFIG_REL = "configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py"
EVAL_CANVAS = (1024, 2048)
TRAIN_CROP = tuple(SIZE_CONFIGS["s"]["train_crop"])


def _stub_mmcv_ext_if_missing() -> None:
    """``mmcv-lite`` ships no ``mmcv._ext``; mmseg imports it eagerly through
    ``mmseg.utils.mask_classification`` even though UNet never uses it."""
    if importlib.util.find_spec("mmcv._ext") is not None:
        return

    class _Ext(types.ModuleType):
        def __getattr__(self, name):  # noqa: D401
            if name.startswith("__"):
                raise AttributeError(name)

            def _unavailable(*_args, **_kwargs):
                raise RuntimeError(f"mmcv._ext.{name} is unavailable in this oracle")

            return _unavailable

    stub = _Ext("mmcv._ext")
    stub.__spec__ = importlib.machinery.ModuleSpec("mmcv._ext", loader=None)
    sys.modules["mmcv._ext"] = stub


def _build_upstream(state: dict):
    _stub_mmcv_ext_if_missing()
    from mmengine.config import Config
    from mmengine.model import revert_sync_batchnorm
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(Path(MMSEG_ROOT) / CONFIG_REL))
    model = MODELS.build(cfg.model)
    # The upstream CPU path (tools/test.py, mmseg.apis) does the same revert.
    model = revert_sync_batchnorm(model)
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    return cfg, model.eval()


def _source_state(path: Path) -> dict:
    digest = sha256(path)
    if digest != SOURCE_DIGEST:
        raise SystemExit(f"Digest mismatch for {path}: {digest} != {SOURCE_DIGEST}.")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return raw["state_dict"]


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.double() - b.double()).abs().max().item()


def _check(label: str, a: torch.Tensor, b: torch.Tensor, failures: list) -> None:
    if tuple(a.shape) != tuple(b.shape):
        failures.append(f"{label}: shape {tuple(a.shape)} != {tuple(b.shape)}")
        print(f"  {label}: SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
        return
    diff = _max_abs_diff(a, b)
    status = "OK" if diff == 0.0 else "FAIL"
    print(f"  {label}: max_abs_diff={diff:.3e}  {status}")
    if diff != 0.0:
        failures.append(f"{label}: max_abs_diff={diff}")


def _load_rgb(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def _image_fixture(path: str, canvas: tuple[int, int]) -> torch.Tensor:
    from libreyolo.models.unet.utils import preprocess_numpy

    chw, _ = preprocess_numpy(_load_rgb(path), canvas)
    return torch.from_numpy(chw).unsqueeze(0)


def _fixtures(canvas: tuple[int, int]) -> dict[str, torch.Tensor]:
    height, width = canvas
    torch.manual_seed(0)
    fixtures = {
        "zeros": torch.zeros(1, 3, height, width),
        "rand": torch.rand(1, 3, height, width),
    }
    for path in IMAGE_PATHS:
        fixtures[Path(path).stem] = _image_fixture(path, canvas)
    return fixtures


def run_graph_parity(upstream, ours: LibreUNetNet, canvas: tuple[int, int], failures: list) -> None:
    print(f"\n=== eval graph parity at {canvas[0]}x{canvas[1]} ===")
    upstream.eval()
    ours.eval()
    for name, x01 in _fixtures(canvas).items():
        # Ours standardizes internally from [0, 1] RGB. Feed upstream the
        # identical standardized tensor so any diff is architecture, not the
        # normalization constants (those are checked end to end below).
        x_norm = ours._normalize(x01)
        with torch.no_grad():
            up_feats = upstream.extract_feat(x_norm)
            up_main = upstream.decode_head.forward(up_feats)
            up_aux = upstream.auxiliary_head.forward(up_feats)
            our_main = ours(x01)
            our_feats = ours.backbone(ours._normalize(x01))
            our_aux = ours.auxiliary_head(our_feats)
        _check(f"{name}/main_logits", our_main, up_main, failures)
        _check(f"{name}/aux_logits", our_aux, up_aux, failures)

    # Stage probes on the seeded tensor localize a logit mismatch.
    x01 = _fixtures(canvas)["rand"]
    x_norm = ours._normalize(x01)
    with torch.no_grad():
        up_x, our_x = x_norm, x_norm
        for index, (up_stage, our_stage) in enumerate(zip(upstream.backbone.encoder, ours.backbone.encoder)):
            up_x, our_x = up_stage(up_x), our_stage(our_x)
            _check(f"encoder/stage{index}", our_x, up_x, failures)
        up_dec = upstream.backbone(x_norm)
        our_dec = ours.backbone(x_norm)
        for index, (a, b) in enumerate(zip(our_dec, up_dec)):
            _check(f"backbone_out/{index}", a, b, failures)
        _check(
            "decode_head/convs",
            ours.decode_head.convs(our_dec[4]),
            upstream.decode_head.convs(upstream.decode_head._transform_inputs(up_dec)),
            failures,
        )


def run_train_parity(upstream, ours: LibreUNetNet, failures: list) -> None:
    height, width = TRAIN_CROP
    print(f"\n=== train-mode parity (batch stats, dropout off) at {height}x{width} ===")
    upstream.train()
    ours.train()
    for model in (upstream, ours):
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout2d):
                module.eval()
    torch.manual_seed(1)
    x01 = torch.rand(2, 3, height, width)
    x_norm = ours._normalize(x01)
    with torch.no_grad():
        up_feats = upstream.extract_feat(x_norm)
        up_main = upstream.decode_head.forward(up_feats)
        up_aux = upstream.auxiliary_head.forward(up_feats)
        our_main, our_aux = ours(x01)
        up_aux = torch.nn.functional.interpolate(
            up_aux, size=(height, width), mode="bilinear", align_corners=False
        )
    _check("train/main_logits", our_main, up_main, failures)
    _check("train/aux_logits", our_aux, up_aux, failures)
    upstream.eval()
    ours.eval()


def run_end_to_end(cfg, ckpt_path: Path, failures: list) -> None:
    """Official ``inference_model`` pipeline vs ``LibreUNet.predict``."""
    from mmseg.apis import inference_model, init_model

    from libreyolo.models.unet.model import LibreUNet

    print("\n=== end-to-end: mmseg.apis.inference_model vs LibreUNet.predict ===")
    upstream = init_model(cfg, str(ckpt_path), device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        converted = Path(tmp) / "LibreUNets-sem.pt"
        convert(str(ckpt_path), str(converted))
        ours = LibreUNet(str(converted), device="cpu")
        for path in IMAGE_PATHS:
            rgb = _load_rgb(path)
            h, w = rgb.shape[:2]
            # Upstream's whole-image test pipeline only accepts canvases
            # divisible by the encoder stride; only Cityscapes-aspect frames
            # go through the official API unchanged.
            if (h * 2048) != (w * 1024):
                print(f"  {Path(path).name}: skipped e2e (aspect {w}x{h} is not 2:1)")
                continue
            with torch.no_grad():
                up_pred = inference_model(upstream, path).pred_sem_seg.data[0].cpu()
                our_pred = ours.predict(rgb, verbose=False).semantic_mask.data.cpu()
            agree = (up_pred == our_pred).double().mean().item()
            status = "OK" if agree == 1.0 else "FAIL"
            print(f"  {Path(path).name}: class-map agreement={agree:.6f}  {status}")
            if agree != 1.0:
                failures.append(f"e2e/{Path(path).name}: agreement={agree}")


def main() -> int:
    if not MMSEG_ROOT or not CKPT_PATH:
        raise SystemExit("Set UNET_MMSEG_ROOT and UNET_OFFICIAL_CKPT.")
    ckpt_path = Path(CKPT_PATH)
    state = _source_state(ckpt_path)
    cfg, upstream = _build_upstream(state)

    ours = LibreUNetNet(size="s", num_classes=19)
    missing, unexpected = ours.load_state_dict(convert_upstream(state), strict=True)
    assert not missing and not unexpected
    ours.eval()

    failures: list[str] = []
    for canvas in (TRAIN_CROP, EVAL_CANVAS):
        run_graph_parity(upstream, ours, canvas, failures)
    run_train_parity(upstream, ours, failures)
    run_end_to_end(cfg, ckpt_path, failures)

    print("\n" + "=" * 60)
    if failures:
        print(f"PARITY FAILED ({len(failures)} mismatches):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PARITY OK: max_abs_diff == 0.0 for every probe; end-to-end class maps identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
