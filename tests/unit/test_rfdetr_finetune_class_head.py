"""RF-DETR fine-tuning keeps pretrained class logits aligned with dataset classes."""

from __future__ import annotations

import copy
import logging

import pytest
import torch

from libreyolo.utils.coco import COCO91_CATEGORY_IDS
from libreyolo.utils.general import COCO_CLASSES

pytestmark = pytest.mark.unit

COCO_NAMES = dict(enumerate(COCO_CLASSES))


def pretrained_class_rows(*args):
    from libreyolo.models.rfdetr.trainer import pretrained_class_rows as rows

    return rows(*args)


def test_coco_names_map_to_coco_category_id_columns():
    rows = pretrained_class_rows(91, COCO_NAMES, COCO_NAMES, 80)
    assert rows == [*COCO91_CATEGORY_IDS, 0]


def test_coco_subset_in_any_order_maps_by_name():
    dataset = {0: "umbrella", 1: "Person", 2: "traffic_light"}
    rows = pretrained_class_rows(91, COCO_NAMES, dataset, 3)
    assert rows == [28, 1, 10, 0]


def test_contiguous_head_maps_by_name_and_keeps_last_column():
    checkpoint = {0: "cat", 1: "dog", 2: "bird"}
    assert pretrained_class_rows(4, checkpoint, {0: "bird", 1: "cat"}, 2) == [2, 0, 3]


@pytest.mark.parametrize(
    "dataset",
    [{0: "helmet"}, {0: "person", 1: "helmet"}, {}],
)
def test_unmatched_names_fall_back(dataset):
    assert pretrained_class_rows(91, COCO_NAMES, dataset, max(1, len(dataset))) is None


def _write_yaml(tmp_path, names):
    for split in ("train", "val"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
    path = tmp_path / "data.yaml"
    listed = ", ".join(f'"{n}"' for n in names)
    path.write_text(
        f"path: {tmp_path}\ntrain: train/images\nval: val/images\n"
        f"nc: {len(names)}\nnames: [{listed}]\n"
    )
    return path


@pytest.fixture(scope="module", params=["detect", "segment"])
def _coco_model(request):
    """A scratch RF-DETR given the released 91-wide COCO head; row i holds value i."""
    from libreyolo.models.rfdetr.model import LibreRFDETR

    model = LibreRFDETR(size="n", task=request.param, device="cpu", _scratch_init=True)
    inner = model.model.model
    inner.reinitialize_detection_head(91)
    model.model.nb_classes = 90
    model.model.args.num_classes = 90
    assert model.nb_classes == 80 and model.names[0] == "person"
    heads = [inner.class_embed, *inner.transformer.enc_out_class_embed]
    with torch.no_grad():
        for head in heads:
            head.weight.copy_(torch.arange(91.0)[:, None].expand_as(head.weight))
            head.bias.copy_(torch.arange(91.0))
    return model


@pytest.fixture
def coco_model(_coco_model):
    return copy.deepcopy(_coco_model)


def _setup_trainer(model, data):
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    trainer = RFDETRTrainer(
        model.model,
        wrapper_model=model,
        data=str(data),
        epochs=1,
        imgsz=192,
        device="cpu",
        size="n",
    )
    trainer.on_setup()
    return trainer


def _head_rows(linear):
    return linear.bias.detach().round().long().tolist()


def test_finetune_on_coco_named_dataset_keeps_pretrained_rows(tmp_path, coco_model):
    model = coco_model
    data = _write_yaml(tmp_path, ["umbrella", "person", "toothbrush"])

    trainer = _setup_trainer(model, data)

    inner = trainer.model.model
    expected = [28, 1, 90, 0]
    assert _head_rows(inner.class_embed) == expected
    assert inner.class_embed.weight[:, 0].round().long().tolist() == expected
    for enc in inner.transformer.enc_out_class_embed:
        assert _head_rows(enc) == expected
    assert trainer.model.nb_classes == 3
    assert trainer.criterion.num_classes == inner.class_embed.out_features
    assert model.names == {0: "umbrella", 1: "person", 2: "toothbrush"}


def test_finetune_on_full_coco_dataset_maps_every_class(tmp_path, coco_model):
    model = coco_model
    data = _write_yaml(tmp_path, COCO_CLASSES)

    trainer = _setup_trainer(model, data)

    assert _head_rows(trainer.model.model.class_embed) == [*COCO91_CATEGORY_IDS, 0]
    assert trainer.model.nb_classes == 80


def test_resuming_a_mapped_coco_head_leaves_it_unchanged(tmp_path, coco_model):
    """A training checkpoint already holds the mapped ``nc + 1`` head (1.5.0
    resized it too), so setting up its resume must not remap it again."""
    data = _write_yaml(tmp_path, COCO_CLASSES)
    trainer = _setup_trainer(coco_model, data)
    trained = _head_rows(trainer.model.model.class_embed)

    resumed = _setup_trainer(coco_model, data)

    assert resumed.model.model.class_embed.out_features == 81
    assert _head_rows(resumed.model.model.class_embed) == trained


def test_finetune_on_other_classes_resizes_and_says_so(tmp_path, caplog, coco_model):
    model = coco_model
    data = _write_yaml(tmp_path, ["helmet", "vest"])

    with caplog.at_level(logging.WARNING, logger="libreyolo.models.rfdetr.trainer"):
        trainer = _setup_trainer(model, data)

    assert trainer.model.model.class_embed.out_features == 3
    assert trainer.model.nb_classes == 2
    assert "do not correspond to the dataset classes" in caplog.text
    assert "from scratch" not in caplog.text
