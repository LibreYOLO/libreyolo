"""RF-DETR AutoBatch probes the largest multi-scale canvas with the real loss."""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.unit


def _write_yaml(tmp_path, body="nc: 2\nnames: [a, b]\n"):
    for split in ("train", "val"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
    path = tmp_path / "data.yaml"
    path.write_text(f"path: {tmp_path}\ntrain: train/images\nval: val/images\n{body}")
    return path


def _trainer(tmp_path, task="detect", size="n", body="nc: 2\nnames: [a, b]\n", **overrides):
    from libreyolo.models.rfdetr.model import LibreRFDETR
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    model = LibreRFDETR(size=size, task=task, device="cpu", _scratch_init=True)
    trainer = RFDETRTrainer(
        model.model,
        wrapper_model=model,
        data=str(_write_yaml(tmp_path, body)),
        epochs=1,
        imgsz=overrides.pop("imgsz", model.input_size),
        device="cpu",
        size=size,
        **overrides,
    )
    trainer.on_setup()
    return trainer


def test_probe_uses_the_largest_multi_scale_canvas(tmp_path):
    trainer = _trainer(tmp_path)

    probe = trainer.autobatch_probe()

    # n trains imgsz=384 with expanded scales up to (384 // 32 + 5) * 32.
    assert trainer.config.imgsz == 384
    assert probe["imgsz"] == 544
    assert probe["imgsz"] == max(trainer._multi_scale_scales())
    assert probe["step"] is not None


def test_probe_keeps_imgsz_without_multi_scale(tmp_path):
    trainer = _trainer(tmp_path, multi_scale=False)

    assert trainer.autobatch_probe()["imgsz"] == 384


@pytest.mark.parametrize(
    ("task", "size", "body", "side"),
    [
        ("detect", "n", "nc: 2\nnames: [a, b]\n", 64),
        ("segment", "n", "nc: 2\nnames: [a, b]\n", 48),
        ("obb", "n", "nc: 2\nnames: [a, b]\n", 64),
        ("pose", "x", "nc: 1\nnames: [person]\nkpt_shape: [17, 3]\n", 48),
    ],
)
def test_probe_step_backpropagates_the_training_loss(tmp_path, task, size, body, side):
    trainer = _trainer(tmp_path, task=task, size=size, body=body)
    trainer.criterion.distributed_normalize = True
    inner = trainer.model.model

    loss = trainer.autobatch_probe()["step"](torch.zeros(2, 3, side, side))
    loss.backward()

    assert loss.ndim == 0 and math.isfinite(float(loss))
    # The whole loss ran: classification and box heads both receive gradients.
    assert inner.class_embed.weight.grad is not None
    assert inner.bbox_embed.layers[-1].weight.grad is not None
    assert trainer.criterion.distributed_normalize is True


def test_probe_follows_the_dataset_label_density(tmp_path):
    trainer = _trainer(tmp_path)
    for i in range(20):
        (tmp_path / "train" / "images" / f"{i}.jpg").touch()
        (tmp_path / "train" / "labels" / f"{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n" * (i + 1))

    # 95th percentile of 1..20 instances per image.
    assert trainer._probe_instances_per_image(max_labels=300) == 19
    assert trainer._probe_instances_per_image(max_labels=10) == 10


def test_paired_iou_matches_the_pairwise_diagonal():
    from libreyolo.models.rfdetr import box_ops

    generator = torch.Generator().manual_seed(0)
    src = box_ops.box_cxcywh_to_xyxy(torch.rand(64, 4, generator=generator)).requires_grad_()
    tgt = box_ops.box_cxcywh_to_xyxy(torch.rand(64, 4, generator=generator))

    iou = box_ops.paired_box_iou(src, tgt)[0]
    assert torch.equal(iou, torch.diag(box_ops.box_iou(src, tgt)[0]))
    paired = box_ops.paired_generalized_box_iou(src, tgt)
    reference = torch.diag(box_ops.generalized_box_iou(src, tgt))
    assert torch.equal(paired, reference)
    (paired_grad,) = torch.autograd.grad(paired.sum(), src)
    (reference_grad,) = torch.autograd.grad(reference.sum(), src)
    assert torch.equal(paired_grad, reference_grad)


def test_losses_never_build_pairwise_iou_over_matched_pairs(monkeypatch):
    """Matched-pair IoU/GIoU must be O(pairs): with 13 query groups, an NxN
    matrix over every matched pair grows with (13 * batch * targets)^2."""
    from libreyolo.models.rfdetr import box_ops
    from libreyolo.models.rfdetr.model import LibreRFDETR

    net = LibreRFDETR(size="n", nb_classes=2, device="cpu", _scratch_init=True).model
    net.train()
    criterion, _ = net.build_criterion_and_postprocess()
    targets = [
        {"labels": torch.zeros(8, dtype=torch.long), "boxes": torch.full((8, 4), 0.25)}
        for _ in range(2)
    ]
    pairwise_calls = []
    for name in ("box_iou", "generalized_box_iou"):
        original = getattr(box_ops, name)

        def spy(boxes1, boxes2, _original=original, _name=name):
            pairwise_calls.append((_name, len(boxes1), len(boxes2)))
            return _original(boxes1, boxes2)

        monkeypatch.setattr(box_ops, name, spy)

    losses = criterion(net(torch.zeros(2, 3, 64, 64), targets=targets), targets)
    sum(v for k, v in losses.items() if k in criterion.weight_dict).backward()

    # The matcher legitimately compares every query with every target
    # (queries x targets); the losses compare matched pairs (pairs x pairs).
    matched = criterion.group_detr * 2 * 8
    assert pairwise_calls
    assert [call for call in pairwise_calls if call[1] == call[2] == matched] == []
