"""Axelera export never modifies the user's model object (regression gate, 2026-11-11).

Every Axelera graph transform (raw-head wrappers, output windows, caps, equalization, BatchNorm folds,
per-family residual transforms) must work on a copy. This runs the real ``export(format="axelera")``
path on random-init models with only the SDK steps stubbed (``AxeleraExporter._preflight``: support tier +
SDK check; ``export_axelera``: quantize/compile), then checks that the user's model keeps identical
weights, module structure, training flags, requires_grad flags and parameter objects.
"""

import pytest

pytestmark = pytest.mark.unit

CASES = {
    "yolonas_detect": ("LibreYOLONAS", dict(size='s', nb_classes=3, task='detect')),
    "yolonas_pose": ("LibreYOLONAS", dict(size='m', nb_classes=1, task='pose')),
    "yolonas_obb": ("LibreYOLONAS", dict(size='s', nb_classes=3, task='obb')),
    "yolo9": ("LibreYOLO9", dict(size='t', nb_classes=3)),
    "yolox": ("LibreYOLOX", dict(size='t', nb_classes=3)),
    "yolo3": ("LibreYOLO3", dict(size='t', nb_classes=3)),
    "yolo4": ("LibreYOLO4", dict(size='t', nb_classes=3)),
    "yolo7": ("LibreYOLO7", dict(size='b', nb_classes=3)),
    "ppyoloe": ("LibrePPYOLOE", dict(size='l', nb_classes=3)),
    "resnet": ("LibreResNet", dict(size='18', nb_classes=3)),
    "mobilenetv4": ("LibreMobileNetV4", dict(size='m', nb_classes=3)),
    "fcn": ("LibreFCN", dict(size='r50', nb_classes=3)),
    "efficientnetv2": ("LibreEfficientNetV2", dict(size='b0', nb_classes=3)),
    "picodet": ("LibrePICODET", dict(size='s', nb_classes=3)),
    "quicksrnet": ("LibreQuickSRNet", dict(size='m2')),
    "midas": ("LibreMiDaS", dict(size='s')),
    "deeplabv3": ("LibreDeepLabv3", dict(size='mv3', nb_classes=3)),
    "rtmdet": ("LibreRTMDet", dict(size='t', nb_classes=3)),
    "rtmdet_seg": ("LibreRTMDet", dict(size='t', nb_classes=3, task='segment')),
}

SUPPORT_KEYS = {
    "yolonas_detect": ('yolonas', 'detect'),
    "yolonas_pose": ('yolonas', 'pose'),
    "yolonas_obb": ('yolonas', 'obb'),
    "yolo9": ('yolo9', 'detect'),
    "yolox": ('yolox', 'detect'),
    "yolo3": ('yolo3', 'detect'),
    "yolo4": ('yolo4', 'detect'),
    "yolo7": ('yolo7', 'detect'),
    "ppyoloe": ('ppyoloe', 'detect'),
    "resnet": ('resnet', 'classify'),
    "mobilenetv4": ('mobilenetv4', 'classify'),
    "fcn": ('fcn', 'semantic'),
    "efficientnetv2": ('efficientnetv2', 'classify'),
    "picodet": ('picodet', 'detect'),
    "quicksrnet": ('quicksrnet', 'restore'),
    "midas": ('midas', 'depth'),
    "deeplabv3": ('deeplabv3', 'semantic'),
    "rtmdet": ('rtmdet', 'detect'),
    "rtmdet_seg": ('rtmdet', 'segment'),
}

#: Families whose model code needs an optional extra (skip, as the rest of the suite does, when it is missing).
OPTIONAL_DEPS = {"midas": "timm"}

#: Support tiers whose exports users can run (accuracy is a label, not a gate: "available" also exports).
EXPORTABLE_TIERS = ("validated", "available")


class _Compiled(Exception):
    """Raised by the stubbed SDK compile step: everything before it ran for real."""


def _dataset(tmp_path, n=4):
    import numpy as np
    from PIL import Image

    images = tmp_path / "data" / "images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        Image.fromarray(rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)).save(images / f"{i}.jpg")
    yaml_path = tmp_path / "data" / "data.yaml"
    yaml_path.write_text(
        f"path: {tmp_path / 'data'}\ntrain: images\nval: images\nnc: 3\nnames: [a, b, c]\n"
    )
    return yaml_path


def _state(net):
    return {
        "weights": {k: v.detach().clone() for k, v in net.state_dict().items()},
        "structure": repr(net),
        "training": [(n, m.training) for n, m in net.named_modules()],
        "requires_grad": [(n, p.requires_grad) for n, p in net.named_parameters()],
        "params": [id(p) for p in net.parameters()],
    }


@pytest.mark.parametrize("case", sorted(CASES))
def test_axelera_export_leaves_the_users_model_untouched(case, tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    import torch

    import libreyolo
    from libreyolo.export import axelera as ax
    from libreyolo.export import exporter as ex

    if case in OPTIONAL_DEPS:
        pytest.importorskip(OPTIONAL_DEPS[case])
    cls_name, kwargs = CASES[case]
    from libreyolo.export.support import get_support

    family, task = SUPPORT_KEYS[case]
    if getattr(get_support(family, task, "axelera"), "tier", None) not in EXPORTABLE_TIERS:
        pytest.skip(f"{family} {task} is not an exportable Axelera family in this checkout")
    torch.manual_seed(0)
    model = getattr(libreyolo, cls_name)(None, device="cpu", **kwargs)
    net = model.model
    before = _state(net)
    calls = []

    def fake_preflight(self, **_kwargs):
        self._devkit_version = "test"
        self._compiler_overrides = {}

    def fake_export_axelera(*args, **kw):
        calls.append(kw)
        raise _Compiled()

    monkeypatch.setattr(ex.AxeleraExporter, "_preflight", fake_preflight)
    monkeypatch.setattr(ax, "export_axelera", fake_export_axelera)
    with pytest.raises(Exception) as info:
        model.export(
            format="axelera",
            data=str(_dataset(tmp_path)),
            imgsz=64,
            output_path=str(tmp_path / "m_axelera_model"),
        )
    assert calls, f"export failed before the compile step: {info.value!r}"
    after = _state(net)
    assert model.model is net
    assert after["structure"] == before["structure"]
    assert after["weights"].keys() == before["weights"].keys()
    changed = [k for k in before["weights"] if not torch.equal(before["weights"][k], after["weights"][k])]
    assert not changed, f"export changed the user's weights: {changed[:5]}"
    assert after["training"] == before["training"]
    assert after["requires_grad"] == before["requires_grad"]
    assert after["params"] == before["params"]
