"""Hermetic event histogram input/label/checkpoint contracts."""

import numpy as np
import pytest
import torch
import yaml

from libreyolo.utils.event_histogram import (
    HistogramTransform,
    check_dataset_profile,
    configure_input,
    load_histogram,
    predict_histogram,
    preprocess_histogram,
    validate_input_profile,
    visualize_histogram,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def profile():
    return {
        "format": "event_histogram",
        "layout": "HWC",
        "polarity": "positive_negative",
        "encoding": "counts",
        "scale": 16.0,
        "window_us": 40000,
    }


@pytest.fixture
def sample():
    a = np.zeros((20, 30, 2), dtype=np.float32)
    a[5:15, 10:20, 0] = 8
    a[5:15, 20:25, 1] = 32
    return a


@pytest.fixture
def dataset(tmp_path, sample, profile):
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        np.save(tmp_path / "images" / split / "a.npy", sample)
        (tmp_path / "labels" / split / "a.txt").write_text(
            "0 0.5 0.5 0.3333333333 0.5\n"
        )
    path = tmp_path / "data.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "person"},
                "input_profile": profile,
            }
        )
    )
    return path


@pytest.mark.parametrize(
    "key,value",
    [
        ("format", "rgb"),
        ("layout", "CHW"),
        ("polarity", "negative_positive"),
        ("encoding", "normalized"),
        ("scale", 0),
        ("scale", float("nan")),
        ("scale", 1e-100),
        ("scale", 1e100),
        ("scale", True),
        ("window_us", 0),
        ("window_us", 40.0),
    ],
)
def test_profile_rejects_ambiguous_contract(profile, key, value):
    profile[key] = value
    with pytest.raises(ValueError):
        validate_input_profile(profile)


def test_complete_profile_required(profile):
    del profile["window_us"]
    with pytest.raises(ValueError, match="exactly"):
        validate_input_profile(profile)


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((2, 20, 30)),
        np.zeros((20, 30, 3)),
        np.zeros((0, 30, 2)),
        np.full((2, 2, 2), -1),
        np.full((2, 2, 2), np.inf),
        np.full((2, 2, 2), np.nan),
        np.full((2, 2, 2), 1j),
        np.zeros((2, 2, 2), dtype=object),
    ],
)
def test_malformed_arrays_rejected(array):
    with pytest.raises(ValueError):
        load_histogram(array)


def test_load_npy_and_visualization_are_separate(tmp_path, sample):
    path = tmp_path / "a.npy"
    np.save(path, sample)
    np.testing.assert_array_equal(load_histogram(path), sample)
    preview = visualize_histogram(path, scale=16)
    assert preview.shape == (20, 30, 3) and preview.dtype == np.uint8
    np.testing.assert_array_equal(preview[10, 15], [128, 0, 0])
    np.testing.assert_array_equal(preview[10, 22], [0, 0, 255])
    assert sample[10, 22, 1] == 32
    with pytest.raises(ValueError, match=".npy"):
        load_histogram(tmp_path / "a.png")


@pytest.mark.parametrize(
    "family,pad", [("yolo9", "topleft"), ("yolo9", "center"), ("rfdetr", "topleft")]
)
def test_train_val_predict_geometry_agree(profile, sample, family, pad):
    targets = np.array([[10, 5, 20, 15, 0]], dtype=np.float32)
    val = HistogramTransform(profile, family, pad)
    train = HistogramTransform(profile, family, pad, training=True)
    vp, vt = val(sample, targets, (32, 32))
    tp, tt = train(sample, targets, (32, 32))
    pp, geom = preprocess_histogram(sample, profile, 32, family, pad)
    np.testing.assert_array_equal(vp, pp)
    np.testing.assert_array_equal(tp, pp)
    assert pp.shape == (2, 32, 32) and 0 <= pp.min() <= pp.max() <= 1
    box = vt[0, :4]
    if family == "yolo9":
        np.testing.assert_allclose(tt[0, 1:] * 32, box, atol=1e-5)
        r, dx, dy = val.letterbox_scale(20, 30, 32)
        np.testing.assert_allclose(
            (box - [dx, dy, dx, dy]) / r, targets[0, :4], atol=1e-5
        )
        if pad == "topleft":
            assert np.count_nonzero(pp[:, geom[2] :]) == 0
    else:
        np.testing.assert_allclose(tt[0, 1:3], (box[:2] + box[2:]) / 2)
        np.testing.assert_allclose(tt[0, 3:5], box[2:] - box[:2])


def test_flips_preserve_polarity_and_transform_labels(profile, sample):
    targets = np.array([[10, 5, 25, 15, 0]], dtype=np.float32)
    plain = HistogramTransform(profile, "rfdetr", training=True)
    flipped = HistogramTransform(profile, "rfdetr", training=True, flip_prob=1)
    p, pt = plain(sample, targets, (20, 30))
    f, ft = flipped(sample, targets, (20, 30))
    np.testing.assert_array_equal(f, p[:, :, ::-1])
    assert ft[0, 1] == 30 - pt[0, 1]
    np.testing.assert_array_equal(ft[0, 2:], pt[0, 2:])


def test_dataset_npy_labels_and_coco_geometry(dataset, profile):
    from libreyolo.data import get_img_files, load_data_config
    from libreyolo.data.dataset import YOLODataset
    from libreyolo.data.yolo_coco_api import YOLOCocoAPI

    cfg = load_data_config(str(dataset))
    assert len(cfg["train_img_files"]) == 1
    assert (
        get_img_files(dataset.parent / "images/train") == []
    )  # RGB discovery unchanged
    ds = YOLODataset(
        img_files=cfg["train_img_files"],
        img_size=(32, 32),
        preproc=HistogramTransform(profile, "yolo9"),
        input_profile=profile,
    )
    image, target, size, _ = ds[0]
    assert image.shape == (2, 32, 32) and size == (20, 30)
    np.testing.assert_allclose(target[0, :4] / (32 / 30), [10, 5, 20, 15], atol=1e-5)
    api = YOLOCocoAPI(
        images_dir=None,
        labels_dir=None,
        image_files=cfg["train_img_files"],
        class_names=["person"],
    )
    assert api.imgs[0]["width"] == 30
    np.testing.assert_allclose(api.anns[1]["bbox"], [10, 5, 10, 10], atol=1e-5)


def test_yolo9_checkpoint_reload_and_stem_transfer(tmp_path, profile, sample):
    from libreyolo import LibreYOLO, LibreYOLO9

    model = LibreYOLO9._from_scratch(size="t", nb_classes=1, device="cpu")
    old = model.model.backbone.conv0.conv.weight.detach().clone()
    configure_input(model, profile, initialization="rgb_mean")
    torch.testing.assert_close(
        model.model.backbone.conv0.conv.weight,
        old.mean(1, keepdim=True).repeat(1, 2, 1, 1) * 1.5,
    )
    model.model.eval()
    path = model.save(tmp_path / "histogram.pt")
    reloaded = LibreYOLO(path, device="cpu")
    assert reloaded.input_profile == profile
    assert reloaded.input_initialization == "rgb_mean"
    a, *_ = predict_histogram(model, sample, 64)
    b, *_ = predict_histogram(reloaded, sample, 64)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(
        reloaded.model.backbone.conv0.conv.weight,
        model.model.backbone.conv0.conv.weight,
    )
    with pytest.raises(ValueError, match="match"):
        check_dataset_profile(reloaded, {})
    with pytest.raises(ValueError, match="color_format"):
        predict_histogram(model, sample, 64, "rgb")


def test_checkpoint_schema_rejects_missing_initialization(profile):
    from libreyolo.utils.serialization import (
        CheckpointMetadataError,
        wrap_libreyolo_checkpoint,
    )

    with pytest.raises(CheckpointMetadataError, match="input_initialization"):
        wrap_libreyolo_checkpoint(
            {},
            model_family="yolo9",
            size="t",
            task="detect",
            nc=1,
            imgsz=64,
            input_profile=profile,
        )


def test_histogram_training_rejects_ignored_options(dataset, profile):
    from types import SimpleNamespace

    from libreyolo.data.event_histogram import prepare_histogram_training

    wrapper = SimpleNamespace(FAMILY="yolo9", task="detect")
    with pytest.raises(ValueError, match="hsv_prob"):
        prepare_histogram_training(wrapper, (), {"data": str(dataset), "hsv_prob": 1})
    kwargs = {"data": str(dataset)}
    prepare_histogram_training(wrapper, (), kwargs)
    assert kwargs["hsv_prob"] == kwargs["mosaic_prob"] == kwargs["mixup_prob"] == 0
    wrapper.FAMILY = "yolox"
    with pytest.raises(ValueError, match="YOLO9 and RF-DETR"):
        prepare_histogram_training(wrapper, (), {"data": str(dataset)})


def test_rfdetr_checkpoint_and_class_rebuild_preserve_stem(tmp_path, profile):
    pytest.importorskip("transformers")
    from libreyolo import LibreRFDETR, LibreYOLO

    model = LibreRFDETR._from_scratch(size="n", nb_classes=1, device="cpu")
    configure_input(model, profile)
    state = model.model.state_dict()
    key = next(k for k in state if k.endswith("patch_embeddings.projection.weight"))
    weight = state[key].clone()
    assert weight.shape[1] == 2
    model._rebuild_for_new_classes(2)
    torch.testing.assert_close(model.model.state_dict()[key], weight)
    path = model.save(tmp_path / "event-detector.pt")
    loaded = LibreYOLO(path, device="cpu")
    assert loaded.input_profile == profile and loaded.nb_classes == 2
    torch.testing.assert_close(loaded.model.state_dict()[key], weight)


def test_backend_center_padding_undo_matches_canvas(profile):
    from types import SimpleNamespace

    from libreyolo.backends.base import BaseBackend

    # Original 30x20 on a 32x32 canvas: 32x21 resized, five rows above.
    g = preprocess_histogram(np.zeros((20, 30, 2)), profile, 32, "yolo9", "center")[1]
    sx, sy, dx, dy = g[4:]
    box = np.array(
        [10 * sx + dx, 5 * sy + dy, 20 * sx + dx, 15 * sy + dy, 0.9], np.float32
    )
    backend = SimpleNamespace(
        task="detect",
        model_family="yolo9",
        input_profile=profile,
        letterbox_pad="center",
    )
    boxes, scores, classes = BaseBackend._parse_yolo9(
        backend, [box[None, :, None]], 32, 30, 20, 0.1
    )
    np.testing.assert_allclose(boxes[0], [10, 5, 20, 15], atol=1e-5)
    assert len(scores) == len(classes) == 1


def test_histogram_target_padding_is_not_a_class_zero_object(profile, sample):
    transform = HistogramTransform(profile, "yolo9", training=True)
    _, targets = transform(sample, np.array([[1, 1, 10, 10, 0]], np.float32), (32, 32))
    assert targets[0, 0] == 0
    assert (targets[1:, 0] == -1).all()


@pytest.mark.parametrize("explicit", [set(), {"mosaic"}, {"hsv_prob"}])
def test_cli_histogram_defaults_preserve_explicitness(dataset, explicit):
    from libreyolo.data.event_histogram import apply_histogram_cli_defaults

    params = {"mosaic": 1.0, "hsv_prob": 1.0, "mosaic_scale": (0.5, 1.5)}
    if explicit:
        with pytest.raises(ValueError, match="not supported"):
            apply_histogram_cli_defaults(
                params, data=str(dataset), family="yolo9", user_provided=explicit
            )
    else:
        assert apply_histogram_cli_defaults(
            params, data=str(dataset), family="yolo9", user_provided=set()
        )
        assert params["mosaic"] == params["hsv_prob"] == 0
        assert params["mosaic_scale"] == (1.0, 1.0)


def test_rgb_scratch_clears_previous_input_profile(profile):
    from libreyolo import LibreYOLO9

    model = LibreYOLO9._from_scratch(size="t", nb_classes=1, device="cpu")
    configure_input(model, profile)
    model._reset_for_scratch()
    assert model.input_profile is None
    assert model.model.backbone.conv0.conv.in_channels == 3


def test_resume_rejects_different_count_scale_before_loading(tmp_path, profile):
    from types import SimpleNamespace

    from libreyolo.models.yolo9.trainer import YOLO9Trainer
    from libreyolo.training.trainer import BaseTrainer
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    other = dict(profile, scale=8.0)
    path = tmp_path / "last.pt"
    torch.save(
        wrap_libreyolo_checkpoint(
            {},
            model_family="yolo9",
            size="t",
            task="detect",
            nc=1,
            imgsz=64,
            input_profile=other,
            input_initialization="random",
        ),
        path,
    )
    trainer = object.__new__(YOLO9Trainer)
    trainer.device = torch.device("cpu")
    trainer.wrapper_model = SimpleNamespace(input_profile=profile)
    with pytest.raises(ValueError, match="match"):
        BaseTrainer.resume(trainer, str(path))


def test_histogram_training_respects_classes_subset(tmp_path, sample, profile):
    """setup_histogram_data must thread classes= through to its YOLODataset
    the same way the regular (non-histogram) _setup_data path does -- it
    computes _class_remap from data_cfg but was dropping it on the floor
    before delegating, so a dataset using input_profile silently trained on
    every class regardless of classes=.
    """
    from libreyolo import LibreYOLO9
    from libreyolo.models.yolo9.trainer import YOLO9Trainer

    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        np.save(tmp_path / "images" / split / "a.npy", sample)
        (tmp_path / "labels" / split / "a.txt").write_text(
            "0 0.5 0.5 0.3333333333 0.5\n1 0.2 0.2 0.1 0.1\n"
        )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val",
                "nc": 2,
                "names": {0: "person", 1: "car"},
                "input_profile": profile,
            }
        )
    )

    wrapper = LibreYOLO9._from_scratch(size="t", nb_classes=2, device="cpu")
    trainer = YOLO9Trainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="t",
        num_classes=2,
        data=str(data_yaml),
        classes=[0],  # excludes label 1 ("car")
        epochs=1,
        batch=1,
        imgsz=32,
        device="cpu",
        amp=False,
        ema=False,
        workers=0,
        eval_interval=-1,
    )

    trainer._setup_data()

    # nc stays the full declared count -- classes= only filters the loss.
    assert trainer.num_classes == 2
    dataset = trainer.train_loader.dataset
    labels = dataset.annotations[0][0]
    assert sorted(labels[:, 4].tolist()) == [0.0]


def test_histogram_disk_cache_does_not_create_extra_training_samples(dataset, profile):
    from libreyolo.data import load_data_config
    from libreyolo.data.dataset import YOLODataset

    cfg = load_data_config(str(dataset))
    source = cfg["train_img_files"][0]
    values = np.full((20, 30, 2), 1000.25, dtype=np.float32)
    np.save(source, values)
    ds = YOLODataset(
        img_files=[source],
        img_size=(32, 32),
        input_profile=profile,
        preproc=HistogramTransform(profile, "yolo9"),
    )
    ds.enable_image_cache("disk")
    np.testing.assert_array_equal(ds.load_image(0), values)
    assert ds._resize_decoded(values).dtype == np.float32
    assert ds._resize_decoded(values).max() == 1000.25
    assert len(load_data_config(str(dataset))["train_img_files"]) == 1
    assert list(source.parent.glob("*.npy")) == [source]


@pytest.mark.parametrize("output", ["previews", "explicit.jpg"])
def test_backend_histogram_preview_uses_an_image_extension(tmp_path, profile, output):
    from types import SimpleNamespace

    from PIL import Image

    from libreyolo.backends.base import BaseBackend
    from libreyolo.utils.results import Boxes, Results

    backend = SimpleNamespace(
        input_profile=profile, names={0: "person"}, model_path="model.onnx"
    )
    result = Results(
        orig_shape=(4, 4),
        boxes=Boxes(
            torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0), orig_shape=(4, 4)
        ),
        names={0: "person"},
    )
    BaseBackend._save_annotated(
        backend,
        result,
        Image.new("RGB", (4, 4)),
        str(tmp_path / "events.npy"),
        str(tmp_path / output),
    )
    expected = (
        tmp_path / output
        if output.endswith(".jpg")
        else tmp_path / output / "events.png"
    )
    assert expected.is_file()
    with Image.open(expected) as image:
        assert image.size == (4, 4)
