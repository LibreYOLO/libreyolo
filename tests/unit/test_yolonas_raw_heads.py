"""YOLO-NAS raw-head export contract: fusion, graph cut and host decode parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.export.exporter import (
    _check_yolonas_fusion_parity,
    _fused_yolonas_copy,
    _YOLONASExportWrapper,
    _YOLONASRawHeadsWrapper,
)
from libreyolo.postprocess.yolonas_raw import (
    YOLONAS_RAW_LAYOUT_BY_TASK,
    decode_raw_heads,
    raw_head_output_names,
)

pytestmark = pytest.mark.unit

# (task, size, nb_classes) per task: the smallest size each task ships.
_TASKS = (("detect", "s", 5), ("pose", "n", 1), ("obb", "s", 4))


def _build(task: str, size: str, nb_classes: int, seed: int = 0):
    from libreyolo import LibreYOLONAS

    torch.manual_seed(seed)
    model = LibreYOLONAS(None, size=size, nb_classes=nb_classes, task=task, device="cpu")
    net = model.model.eval()
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        # Non-trivial BatchNorm statistics so fusion is really exercised, and
        # spread-out prediction weights so the DFL softmax is not near-uniform.
        for module in net.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                n = module.num_features
                module.running_mean.copy_(torch.randn(n, generator=generator) * 0.1)
                module.running_var.copy_(torch.rand(n, generator=generator) + 0.5)
                module.weight.copy_(torch.rand(n, generator=generator) + 0.5)
                module.bias.copy_(torch.randn(n, generator=generator) * 0.1)
        for name, module in net.named_modules():
            if name.endswith(("reg_pred", "cls_pred", "pose_pred", "offset_pred", "rot_pred")):
                module.weight.copy_(torch.randn(module.weight.shape, generator=generator) * 0.05)
                module.bias.copy_(torch.randn(module.bias.shape, generator=generator))
    return model, net


@pytest.mark.parametrize(("task", "size", "nb_classes"), _TASKS)
def test_calibration_preprocess_matches_inference_preprocess(task, size, nb_classes):
    model, _ = _build(task, size, nb_classes)
    rng = np.random.default_rng(3)
    image = Image.fromarray(rng.integers(0, 256, size=(50, 70, 3), dtype=np.uint8))
    imgsz = 96

    calibration, calibration_ratio = model._get_preprocess_numpy()(np.asarray(image), imgsz)
    inference, _, _, inference_ratio = model._preprocess(image, input_size=imgsz)

    inference = inference.detach().cpu().numpy() if torch.is_tensor(inference) else inference
    np.testing.assert_array_equal(calibration[None], inference)
    assert calibration.dtype == np.float32
    assert calibration_ratio == pytest.approx(inference_ratio)


@pytest.mark.parametrize(("task", "size", "nb_classes"), _TASKS)
def test_fused_copy_matches_and_leaves_model_untouched(task, size, nb_classes):
    _, net = _build(task, size, nb_classes)
    fused = _fused_yolonas_copy(net, full_fusion=True)

    worst = _check_yolonas_fusion_parity(net, fused, task, (64, 64))
    assert worst < 1.0

    def blocks(module):
        return [m for m in module.modules() if type(m).__name__ == "QARepVGGBlock"]

    assert blocks(fused) and all(block.fully_fused for block in blocks(fused))
    assert all(not block.fully_fused and hasattr(block, "branch_3x3") for block in blocks(net))


def test_fusion_parity_check_rejects_a_changed_network():
    _, net = _build("detect", "s", 5)
    fused = _fused_yolonas_copy(net)
    with torch.no_grad():
        fused.heads.head1.cls_pred.bias.add_(1.0)
    with pytest.raises(AssertionError, match="fusion changed raw output"):
        _check_yolonas_fusion_parity(net, fused, "detect", (64, 64))


@pytest.mark.parametrize(("task", "size", "nb_classes"), _TASKS)
def test_raw_wrapper_output_contract(task, size, nb_classes):
    _, net = _build(task, size, nb_classes)
    wrapper = _YOLONASRawHeadsWrapper(net, task).eval()
    imgsz = 64
    with torch.no_grad():
        outputs = wrapper(torch.rand(1, 3, imgsz, imgsz))

    layout = YOLONAS_RAW_LAYOUT_BY_TASK[task]
    assert wrapper.output_layout == layout
    assert wrapper.strides == (8, 16, 32)
    assert wrapper.reg_max == 16
    assert wrapper.output_names == raw_head_output_names(layout)
    assert len(outputs) == len(wrapper.output_names)

    bins = wrapper.reg_max + 1
    keypoints = 17
    channels = {
        "detect": {"reg": 4 * bins, "cls": nb_classes},
        "pose": {"reg": 4 * bins, "cls": nb_classes + keypoints, "pose": 2 * keypoints},
        "obb": {"reg": 2 * bins, "cls": nb_classes, "offset": 2, "rot": 1},
    }[task]
    for name, output in zip(wrapper.output_names, outputs):
        level, part = name.split("_")
        stride = 2 ** int(level[1:])
        assert tuple(output.shape) == (1, channels[part], imgsz // stride, imgsz // stride)


def test_raw_output_names_are_level_major():
    assert raw_head_output_names("yolonas_raw_heads_v1") == [
        "p3_reg", "p3_cls", "p4_reg", "p4_cls", "p5_reg", "p5_cls",
    ]
    assert raw_head_output_names("yolonas_obb_raw_heads_v1")[:4] == [
        "p3_reg", "p3_cls", "p3_offset", "p3_rot",
    ]


def test_raw_wrapper_rejects_unsupported_task():
    _, net = _build("detect", "s", 5)
    with pytest.raises(ValueError, match="raw-head export supports tasks"):
        _YOLONASRawHeadsWrapper(net, "segment")


_DECODE_CASES = tuple(
    (task, size, nb_classes, imgsz)
    for task, size, nb_classes in _TASKS
    for imgsz in (160, 320)
)


@pytest.mark.parametrize(("task", "size", "nb_classes", "imgsz"), _DECODE_CASES)
def test_host_decode_matches_in_graph_decode(task, size, nb_classes, imgsz):
    _, net = _build(task, size, nb_classes)
    sample = torch.rand(2, 3, imgsz, imgsz, generator=torch.Generator().manual_seed(7))
    wrapper = _YOLONASRawHeadsWrapper(net, task).eval()
    with torch.no_grad():
        expected = _YOLONASExportWrapper(net).eval()(sample)
        raw = wrapper(sample)

    layout = YOLONAS_RAW_LAYOUT_BY_TASK[task]
    decoded = decode_raw_heads(
        layout, [t.numpy() for t in raw], strides=(8, 16, 32), reg_max=16,
        output_windows=wrapper.output_windows,
    )

    assert len(decoded) == len(expected)
    for actual, reference in zip(decoded, expected):
        reference = reference.numpy()
        assert actual.shape == reference.shape
        assert actual.dtype == np.float32
        # 1e-4 absolute, plus 1e-6 relative for pixel values: two float32
        # implementations of the DFL sum differ by a couple of ULPs at
        # ~1000 px (the in-graph decode itself is ~1.2e-4 px off float64).
        # Windowed outputs (pose) also round-trip through the window map in
        # float32, which the DFL exponent amplifies to ~1e-3 px at ~450 px.
        windowed = bool(wrapper.output_windows)
        np.testing.assert_allclose(
            actual, reference, rtol=5e-6 if windowed else 1e-6, atol=5e-4 if windowed else 1e-4
        )


def test_decode_rejects_wrong_output_count_and_layout():
    outputs = [np.zeros((1, 68, 4, 4), np.float32), np.zeros((1, 3, 4, 4), np.float32)]
    with pytest.raises(ValueError, match="expects 6 outputs"):
        decode_raw_heads("yolonas_raw_heads_v1", outputs)
    with pytest.raises(ValueError, match="Unknown YOLO-NAS raw output layout"):
        decode_raw_heads("yolonas_raw_heads_v9", outputs)


def test_decode_rejects_mismatched_strides():
    levels = []
    for size in (8, 4, 4):  # third level should be 2x2 for strides 8/16/32
        levels += [np.zeros((1, 68, size, size), np.float32), np.zeros((1, 3, size, size), np.float32)]
    with pytest.raises(ValueError, match="do not match strides"):
        decode_raw_heads("yolonas_raw_heads_v1", levels)


def test_relu6_window_maps_onto_zero_six_and_clamps():
    from libreyolo.export.exporter import _relu6_window
    from libreyolo.postprocess.yolonas_raw import unwindow_raw_heads

    conv = torch.nn.Conv2d(1, 1, 1)
    with torch.no_grad():
        conv.weight.fill_(1.0)
        conv.bias.zero_()
    logits = torch.linspace(-40.0, 40.0, 81).reshape(1, 1, 9, 9)
    with torch.no_grad():
        windowed = _relu6_window(conv, logits, -8.0, 16.0)
    assert float(windowed.min()) == 0.0 and float(windowed.max()) == 6.0
    expected = ((logits.clamp(-8.0, 16.0) + 8.0) * 6.0 / 24.0).numpy()
    np.testing.assert_allclose(windowed.numpy(), expected, atol=1e-5)

    # Host side: the cls part of a detect-shaped output list maps back.
    outputs = [np.zeros((1, 68, 9, 9), np.float32), windowed.numpy()] * 3
    restored = unwindow_raw_heads("yolonas_raw_heads_v1", outputs, {"cls": (-8.0, 16.0)})
    np.testing.assert_allclose(restored[1], logits.clamp(-8.0, 16.0).numpy(), atol=1e-5)
    assert restored[0] is outputs[0]  # parts without a window pass through
    assert restored[1] is not outputs[1]  # and inputs are never modified
    np.testing.assert_array_equal(outputs[1], windowed.numpy())


def test_pose_raw_outputs_are_windowed_by_default():
    from libreyolo.postprocess.yolonas_raw import (
        YOLONAS_RAW_OUTPUT_WINDOWS_BY_TASK,
        unwindow_raw_heads,
    )

    _, net = _build("pose", "n", 1)
    windows = YOLONAS_RAW_OUTPUT_WINDOWS_BY_TASK["pose"]
    windowed = _YOLONASRawHeadsWrapper(net, "pose").eval()
    plain = _YOLONASRawHeadsWrapper(net, "pose", output_windows={}).eval()
    assert windowed.output_windows == windows and plain.output_windows == {}
    sample = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        encoded = [t.numpy() for t in windowed(sample)]
        raw = [t.numpy() for t in plain(sample)]

    restored = unwindow_raw_heads("yolonas_pose_raw_heads_v1", encoded, windows)
    for name, enc, back, ref in zip(windowed.output_names, encoded, restored, raw):
        part = name.split("_")[1]
        if part in windows:
            assert enc.min() >= 0.0 and enc.max() <= 6.0
            ref = np.clip(ref, *windows[part])
        np.testing.assert_allclose(back, ref, rtol=1e-5, atol=1e-4)


def test_output_windows_are_validated():
    from libreyolo.postprocess.yolonas_raw import check_output_windows

    assert check_output_windows("yolonas_raw_heads_v1", None) == {}
    assert check_output_windows("yolonas_raw_heads_v1", {"cls": [-8, 8]}) == {"cls": (-8.0, 8.0)}
    with pytest.raises(ValueError, match="does not have"):
        check_output_windows("yolonas_raw_heads_v1", {"pose": (-1, 1)})
    with pytest.raises(ValueError, match="low < high"):
        check_output_windows("yolonas_raw_heads_v1", {"cls": (8, -8)})
    with pytest.raises(ValueError, match=r"\[low, high\]"):
        check_output_windows("yolonas_raw_heads_v1", {"cls": 8})
    _, net = _build("detect", "s", 5)
    with pytest.raises(ValueError, match="pose heads only"):
        _YOLONASRawHeadsWrapper(net, "detect", output_windows={"cls": (-8, 8)})


def test_pose_head_equalization_is_exact_and_evens_out_channel_ranges():
    from libreyolo.export.exporter import _equalize_yolonas_pose_heads

    _, net = _build("pose", "n", 1)
    fused = _fused_yolonas_copy(net)
    before = {k: v.clone() for k, v in fused.state_dict().items()}
    sample = torch.rand(1, 3, 96, 96, generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        reference = _YOLONASRawHeadsWrapper(fused, "pose").eval()(sample)
    for samples in ([sample.numpy(), torch.rand(1, 3, 96, 96).numpy()], None):  # measured, then BN estimate
        _equalize_yolonas_pose_heads(fused, samples)
        with torch.no_grad():
            equalized = _YOLONASRawHeadsWrapper(fused, "pose").eval()(sample)
        for a, b in zip(reference, equalized):
            np.testing.assert_allclose(b.numpy(), a.numpy(), rtol=1e-4, atol=1e-4)
    changed = [k for k, v in fused.state_dict().items() if not torch.equal(v, before[k])]
    assert any("pose_pred" in k for k in changed)
    assert all(k.split(".")[2] in {"pose_convs", "pose_pred"} for k in changed)
    # Detect and OBB models pass through untouched.
    _, detect = _build("detect", "s", 5)
    state = {k: v.clone() for k, v in detect.state_dict().items()}
    _equalize_yolonas_pose_heads(detect)
    assert all(torch.equal(v, state[k]) for k, v in detect.state_dict().items())


def test_pose_input_bound_caps_large_activations_only():
    from libreyolo.export.exporter import _bound_yolonas_pose_inputs

    _, net = _build("pose", "n", 1)
    fused = _fused_yolonas_copy(net)
    samples = [torch.rand(1, 3, 96, 96, generator=torch.Generator().manual_seed(i)).numpy() for i in range(3)]
    sample = torch.from_numpy(samples[0])
    with torch.no_grad():
        reference = [t.numpy() for t in _YOLONASRawHeadsWrapper(fused, "pose", output_windows={}).eval()(sample)]
    caps = _bound_yolonas_pose_inputs(fused, samples, percentile=99.5)
    # Only the class tower is capped (the keypoint tower's large values carry detail).
    assert sorted(caps) == ["head1.cls_convs", "head2.cls_convs", "head3.cls_convs"]
    assert all(cap > 0 for cap in caps.values())
    for i in (1, 2, 3):
        head = getattr(fused.heads, f"head{i}")
        assert isinstance(head.cls_convs[-1].seq.act, torch.nn.ReLU6)
        assert not isinstance(head.pose_convs[-1].seq.act, torch.nn.ReLU6)
    with torch.no_grad():
        bounded = [t.numpy() for t in _YOLONASRawHeadsWrapper(fused, "pose", output_windows={}).eval()(sample)]
    # Class outputs move only slightly (0.5% of activations are capped); the
    # keypoint and box branches are untouched.
    for name, a, b in zip(raw_head_output_names("yolonas_pose_raw_heads_v1"), reference, bounded):
        if name.endswith("cls"):
            assert np.median(np.abs(a - b)) < 1e-3 * (np.abs(a).max() + 1)
        else:
            np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-5)

