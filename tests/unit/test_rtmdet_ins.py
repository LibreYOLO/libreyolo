"""RTMDet-Ins instance-segmentation unit tests."""

from __future__ import annotations

import pytest
import torch

from libreyolo import LibreRTMDet, LibreYOLO
from libreyolo.models.rtmdet.nn import LibreRTMDetModel, RTMDetInsSepBNHead
from libreyolo.models.rtmdet.utils import postprocess
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint


pytestmark = pytest.mark.unit


def _minimal_state(*, segment: bool) -> dict:
    state = {
        "backbone.stem.0.conv.weight": torch.zeros(12, 3, 3, 3),
        "head.rtm_cls.0.weight": torch.zeros(2, 96, 1, 1),
        "head.rtm_reg.0.weight": torch.zeros(4, 96, 1, 1),
    }
    if segment:
        state["head.rtm_kernel.0.weight"] = torch.zeros(169, 96, 1, 1)
        state["head.mask_head.projection.weight"] = torch.zeros(8, 96, 1, 1)
    return state


def test_rtmdet_ins_suffix_task_and_download_url():
    assert LibreRTMDet.SUPPORTED_TASKS == ("detect", "segment")
    assert LibreRTMDet.detect_task_from_filename("LibreRTMDetm-seg.pt") == "segment"
    assert LibreRTMDet.detect_checkpoint_task(_minimal_state(segment=True)) == "segment"
    assert LibreRTMDet.get_download_url("LibreRTMDetm-seg.pt") == (
        "https://huggingface.co/LibreYOLO/LibreRTMDetm-seg/resolve/main/"
        "LibreRTMDetm-seg.pt"
    )


def test_rtmdet_ins_forward_shapes_and_tower_aliasing():
    model = LibreRTMDetModel(size="t", nc=3, enable_mask_head=True).eval()
    assert isinstance(model.head, RTMDetInsSepBNHead)
    assert model.head.num_gen_params == 169
    for level in range(3):
        assert model.head.reg_convs[level] is model.head.cls_convs[level]

    with torch.no_grad():
        cls, reg, kernels, mask_feat = model(torch.zeros(1, 3, 64, 64))
    assert [item.shape[-2:] for item in cls] == [(8, 8), (4, 4), (2, 2)]
    assert [item.shape[1] for item in reg] == [4, 4, 4]
    assert [item.shape[1] for item in kernels] == [169, 169, 169]
    assert mask_feat.shape == (1, 8, 8, 8)


@pytest.mark.parametrize("size", ["t", "s", "m", "l", "x"])
def test_rtmdet_ins_all_sizes_use_upstream_relu_box_decode(size):
    """RTMDet-Ins never uses detection's m/l/x exponential box decode."""
    model = LibreRTMDetModel(size=size, nc=1, enable_mask_head=True)
    assert isinstance(model.head, RTMDetInsSepBNHead)
    assert not hasattr(model.head, "exp_on_reg")


def test_rtmdet_ins_postprocess_decodes_masks_after_nms():
    cls = tuple(torch.full((1, 1, size, size), -20.0) for size in (2, 1, 1))
    reg = tuple(torch.full((1, 4, size, size), 4.0) for size in (2, 1, 1))
    kernels = tuple(torch.zeros(1, 169, size, size) for size in (2, 1, 1))
    cls[0][0, 0, 1, 1] = 10.0
    # The final generated parameter is the output-layer bias. A positive value
    # makes the survivor mask unambiguously foreground after sigmoid.
    kernels[0][0, -1, 1, 1] = 10.0
    mask_feat = torch.zeros(1, 8, 2, 2, dtype=torch.float16)

    result = postprocess(
        (cls, reg, kernels, mask_feat),
        conf_thres=0.25,
        iou_thres=0.6,
        input_size=16,
        original_size=(24, 12),
        ratio=2 / 3,
        max_det=100,
    )

    assert result["num_detections"] == 1
    assert result["masks"].dtype == torch.bool
    assert result["masks"].shape == (1, 12, 24)
    assert result["masks"].all()


def test_rtmdet_ins_factory_resolves_segment_checkpoint(tmp_path):
    checkpoint = wrap_libreyolo_checkpoint(
        _minimal_state(segment=True),
        model_family="rtmdet",
        size="t",
        task="segment",
        nc=2,
    )
    path = tmp_path / "LibreRTMDett-seg.pt"
    torch.save(checkpoint, path)

    model = LibreYOLO(str(path), device="cpu")
    assert isinstance(model, LibreRTMDet)
    assert model.task == "segment"
    assert isinstance(model.model.head, RTMDetInsSepBNHead)


def test_rtmdet_ins_rejects_seg_checkpoint_for_default_task(tmp_path):
    checkpoint = wrap_libreyolo_checkpoint(
        _minimal_state(segment=True),
        model_family="rtmdet",
        size="t",
        task="segment",
        nc=2,
    )
    path = tmp_path / "seg-checkpoint.pt"
    torch.save(checkpoint, path)

    with pytest.raises(RuntimeError, match="task='segment'"):
        LibreRTMDet(str(path), size="t", nb_classes=2, device="cpu")


def test_rtmdet_ins_rejects_detect_checkpoint_for_segment_task(tmp_path):
    checkpoint = wrap_libreyolo_checkpoint(
        _minimal_state(segment=False),
        model_family="rtmdet",
        size="t",
        task="detect",
        nc=2,
    )
    path = tmp_path / "detect-checkpoint.pt"
    torch.save(checkpoint, path)

    with pytest.raises(RuntimeError, match="task='detect'"):
        LibreRTMDet(
            str(path),
            size="t",
            nb_classes=2,
            device="cpu",
            task="segment",
        )


def test_rtmdet_ins_training_and_export_are_explicitly_unsupported():
    wrapper = LibreRTMDet(None, size="t", nb_classes=2, device="cpu", task="segment")
    with pytest.raises(NotImplementedError, match="training"):
        wrapper.train(data="unused.yaml")

    wrapper.model.head.export = True
    with pytest.raises(NotImplementedError, match="export"):
        wrapper.model(torch.zeros(1, 3, 64, 64))


def test_rtmdet_ins_postprocess_clamps_boxes_to_both_frames():
    """Boxes must be clamped in-place to the input frame and, after the ratio
    inverse, to the original frame (a fancy-indexed clamp_ silently no-ops)."""
    cls = tuple(torch.full((1, 1, size, size), -20.0) for size in (2, 1, 1))
    reg = tuple(torch.full((1, 4, size, size), 40.0) for size in (2, 1, 1))
    kernels = tuple(torch.zeros(1, 169, size, size) for size in (2, 1, 1))
    cls[0][0, 0, 1, 1] = 10.0
    kernels[0][0, -1, 1, 1] = 10.0
    mask_feat = torch.zeros(1, 8, 2, 2)

    result = postprocess(
        (cls, reg, kernels, mask_feat),
        conf_thres=0.25,
        iou_thres=0.6,
        input_size=16,
        original_size=(24, 12),
        ratio=2 / 3,
        max_det=100,
    )

    assert result["num_detections"] == 1
    # Raw decode is (-32, -32, 48, 48): input-frame clamp gives (0, 0, 16, 16),
    # the ratio inverse gives (0, 0, 24, 24), and the original-frame clamp
    # caps y2 at the 12-pixel image height.
    assert result["boxes"].tolist() == [[0.0, 0.0, 24.0, 12.0]]


def test_rtmdet_ins_postprocess_respects_max_det_above_100():
    # 150 classes on one location: class-aware NMS keeps every candidate, so
    # the only cap is ``max_det``.
    num_classes = 150
    cls = tuple(torch.full((1, num_classes, size, size), -20.0) for size in (2, 1, 1))
    reg = tuple(torch.full((1, 4, size, size), 4.0) for size in (2, 1, 1))
    kernels = tuple(torch.zeros(1, 169, size, size) for size in (2, 1, 1))
    cls[0][0, :, 1, 1] = 10.0
    mask_feat = torch.zeros(1, 8, 2, 2)

    def run(max_det):
        return postprocess(
            (cls, reg, kernels, mask_feat),
            conf_thres=0.25,
            iou_thres=0.6,
            input_size=16,
            original_size=(16, 16),
            max_det=max_det,
        )

    assert run(1000)["num_detections"] == num_classes
    assert run(20)["num_detections"] == 20
