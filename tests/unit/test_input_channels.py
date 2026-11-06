"""Grayscale ``channels: 1`` input contract (Ultralytics-compatible dataset key)."""

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from libreyolo.utils.input_channels import (
    channels_metadata,
    configure_channels,
    dataset_channels,
    dataset_input_channels,
    find_input_convs,
    fold_input_channels,
    to_gray3,
    validate_channels,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def color_dataset(tmp_path):
    cv2 = pytest.importorskip("cv2")
    img = np.zeros((32, 48, 3), dtype=np.uint8)
    img[..., 0] = 200  # blue
    img[8:24, 12:36, 2] = 255  # red box
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        cv2.imwrite(str(tmp_path / "images" / split / "a.png"), img)
        (tmp_path / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    path = tmp_path / "data.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "box"},
                "channels": 1,
            }
        )
    )
    return path, img


@pytest.mark.parametrize("value", [0, 2, 4, "1", 1.0, True, None])
def test_validate_channels_rejects_everything_but_1_and_3(value):
    with pytest.raises(ValueError, match="channels"):
        validate_channels(value)


def test_dataset_channels_reads_key_and_leaves_event_histograms_alone():
    assert dataset_channels({}) is None
    assert dataset_channels({"channels": 1}) == 1
    assert dataset_channels({"channels": 3}) == 3
    # channels: 2 belongs to event histograms, which declare an input_profile.
    assert dataset_channels({"channels": 2, "input_profile": {"format": "x"}}) is None
    with pytest.raises(ValueError):
        dataset_channels({"channels": 2})


def test_to_gray3_replicates_luma():
    cv2 = pytest.importorskip("cv2")
    img = np.random.default_rng(0).integers(0, 255, (5, 7, 3), dtype=np.uint8)
    out = to_gray3(img)
    assert out.shape == (5, 7, 3) and out.dtype == np.uint8
    expected = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for c in range(3):
        np.testing.assert_array_equal(out[..., c], expected)


def test_dataset_converts_after_the_cache(color_dataset):
    from libreyolo.data import load_data_config
    from libreyolo.data.dataset import YOLODataset

    path, img = color_dataset
    cfg = load_data_config(str(path), autodownload=False)
    assert cfg["channels"] == 1
    rgb = YOLODataset(
        img_files=cfg["train_img_files"],
        label_files=cfg["train_label_files"],
        img_size=(32, 48),
    )
    with dataset_input_channels(1):
        gray = YOLODataset(
            img_files=cfg["train_img_files"],
            label_files=cfg["train_label_files"],
            img_size=(32, 48),
        )
    gray.enable_image_cache("ram")
    np.testing.assert_array_equal(rgb.load_image(0), img)
    out = gray.load_image(0)
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])
    np.testing.assert_array_equal(out, to_gray3(img))
    # The RAM cache holds the colour decode; conversion happens on read.
    np.testing.assert_array_equal(gray._ram_cache[0], img)
    resized = gray.load_resized_img(0)
    assert np.array_equal(resized[..., 0], resized[..., 2])


class _Stem(nn.Module):
    """A RepVGG-like stem: two parallel convolutions read the image."""

    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(3, 4, 3, padding=1)
        self.b = nn.Conv2d(3, 4, 1)
        self.c = nn.Conv2d(4, 2, 1)

    def forward(self, x):
        return self.c(self.a(x) + self.b(x))


class _Focus(nn.Module):
    """YOLOX-style space-to-depth: the first convolution reads 4 RGB groups."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(12, 4, 3, padding=1)

    def forward(self, x):
        tl, tr = x[..., ::2, ::2], x[..., ::2, 1::2]
        bl, br = x[..., 1::2, ::2], x[..., 1::2, 1::2]
        return self.conv(torch.cat([tl, bl, tr, br], 1))


class _Wrapper:
    def __init__(self, net):
        self.model = net
        self.model_path = "weights.pt"

    def _get_input_size(self):
        return 16

    def _forward(self, x):
        return self.model(x)


@pytest.mark.parametrize("net_cls", [_Stem, _Focus])
def test_gray_model_matches_rgb_model_on_gray_images(net_cls):
    torch.manual_seed(0)
    wrapper = _Wrapper(net_cls().eval())
    x1 = torch.rand(2, 1, 16, 16)
    with torch.no_grad():
        ref = wrapper._forward(x1.repeat(1, 3, 1, 1))
    configure_channels(wrapper, 1)
    assert wrapper.input_channels == 1
    with torch.no_grad():
        torch.testing.assert_close(wrapper._forward(x1), ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            wrapper._forward(x1.repeat(1, 3, 1, 1)), ref, rtol=1e-5, atol=1e-5
        )
    convs = [m for m in wrapper.model.modules() if type(m).__name__ == "GrayInputConv2d"]
    assert len(convs) == (2 if net_cls is _Stem else 1)
    configure_channels(wrapper, 1)  # idempotent
    assert channels_metadata(wrapper) == {"channels": 1}


def test_rgb_model_metadata_is_unchanged():
    wrapper = _Wrapper(_Stem())
    assert channels_metadata(wrapper) == {}
    configure_channels(wrapper, 3)
    assert not hasattr(wrapper, "input_channels")
    assert [n for n, _ in find_input_convs(wrapper)] == ["a", "b"]


def test_fold_input_channels():
    blob = np.random.default_rng(0).random((2, 3, 4, 4)).astype(np.float32)
    folded = fold_input_channels(blob, 1)
    assert folded.shape == (2, 1, 4, 4) and folded.dtype == np.float32
    np.testing.assert_allclose(folded[:, 0], blob.mean(1), rtol=1e-6)
    assert fold_input_channels(blob, 3) is blob


def test_yolo9_checkpoint_round_trip_and_prediction(tmp_path):
    from libreyolo import LibreYOLO, LibreYOLO9

    model = LibreYOLO9._from_scratch(size="t", nb_classes=1, device="cpu")
    model.input_size = 64
    configure_channels(model, 1)
    model.model.eval()
    assert model.model.backbone.conv0.conv.weight.shape[1] == 1
    path = model.save(tmp_path / "gray.pt")
    assert torch.load(path, map_location="cpu", weights_only=False)["channels"] == 1
    reloaded = LibreYOLO(path, device="cpu")
    assert reloaded.input_channels == 1
    torch.testing.assert_close(
        reloaded.model.backbone.conv0.conv.weight,
        model.model.backbone.conv0.conv.weight,
    )
    image = np.random.default_rng(0).integers(0, 255, (40, 40, 3), dtype=np.uint8)
    tensor, *_ = reloaded._preprocess_predict(image, input_size=64)
    assert torch.equal(tensor[:, 0], tensor[:, 1]) and torch.equal(tensor[:, 1], tensor[:, 2])


def test_rgb_checkpoint_has_no_channels_key(tmp_path):
    from libreyolo import LibreYOLO9

    model = LibreYOLO9._from_scratch(size="t", nb_classes=1, device="cpu")
    path = model.save(tmp_path / "rgb.pt")
    assert "channels" not in torch.load(path, map_location="cpu", weights_only=False)


def test_class_rebuild_and_scratch_reset_keep_contract():
    from libreyolo import LibreYOLO9

    model = LibreYOLO9._from_scratch(size="t", nb_classes=1, device="cpu")
    model.input_size = 64
    configure_channels(model, 1)
    weight = model.model.backbone.conv0.conv.weight.detach().clone()
    model._rebuild_for_new_classes(2)
    assert model.input_channels == 1
    torch.testing.assert_close(model.model.backbone.conv0.conv.weight, weight)
    model._reset_for_scratch()
    assert getattr(model, "input_channels", 3) == 3
    assert model.model.backbone.conv0.conv.in_channels == 3


def test_training_prep_reads_dataset_yaml(color_dataset):
    from libreyolo import LibreYOLO9
    from libreyolo.utils.input_channels import prepare_channels_training

    path, _ = color_dataset
    model = LibreYOLO9._from_scratch(size="t", nb_classes=1, device="cpu")
    model.input_size = 64
    prepare_channels_training(model, (), {"data": str(path)})
    assert model.input_channels == 1
    rgb_yaml = path.with_name("rgb.yaml")
    cfg = yaml.safe_load(path.read_text())
    cfg["channels"] = 3
    rgb_yaml.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="channels: 3"):
        prepare_channels_training(model, (), {"data": str(rgb_yaml)})


def test_gray_onnx_export_consumes_one_channel(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from libreyolo import LibreYOLO, LibreYOLO9

    model = LibreYOLO9._from_scratch(size="t", nb_classes=1, device="cpu")
    model.input_size = 64
    configure_channels(model, 1)
    model.model.eval()
    path = model.export(format="onnx", imgsz=64, output_path=str(tmp_path / "gray.onnx"))
    graph = onnx.load(path)
    dims = [d.dim_value or d.dim_param for d in graph.graph.input[0].type.tensor_type.shape.dim]
    assert dims[1] == 1
    assert {p.key: p.value for p in graph.metadata_props}["channels"] == "1"
    backend = LibreYOLO(path, device="cpu")
    assert backend.input_channels == 1
    image = np.random.default_rng(0).integers(0, 255, (40, 40, 3), dtype=np.uint8)
    backend.predict(image, conf=0.5)


def test_pose_dataset_converts_images(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from libreyolo.data.pose_dataset import YOLOPoseDataset

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[..., 2] = 255
    cv2.imwrite(str(tmp_path / "images" / "a.png"), img)
    (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 0.5 0.5 0.5 0.5 2\n")
    files = [tmp_path / "images" / "a.png"]
    rgb = YOLOPoseDataset(files, num_keypoints=1, img_size=(32, 32))
    with dataset_input_channels(1):
        gray = YOLOPoseDataset(files, num_keypoints=1, img_size=(32, 32))
    np.testing.assert_array_equal(rgb.load_image(0), img)
    np.testing.assert_array_equal(gray.load_image(0), to_gray3(img))


def test_torchscript_export_carries_channels(tmp_path):
    from libreyolo import LibreYOLO, LibreYOLO9

    model = LibreYOLO9._from_scratch(size="t", nb_classes=1, device="cpu")
    model.input_size = 64
    configure_channels(model, 1)
    model.model.eval()
    path = model.export(
        format="torchscript", imgsz=64, output_path=str(tmp_path / "gray.torchscript")
    )
    backend = LibreYOLO(path, device="cpu")
    assert backend.input_channels == 1
    image = np.random.default_rng(0).integers(0, 255, (40, 40, 3), dtype=np.uint8)
    backend.predict(image, conf=0.5)


def test_gray_preprocess_fn_yields_one_channel():
    from libreyolo.utils.input_channels import gray_preprocess_fn

    def preprocess(img, size):
        return img.transpose(2, 0, 1).astype(np.float32) / 255.0, 1.0

    img = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    chw, ratio = gray_preprocess_fn(preprocess)(img, 8)
    assert chw.shape == (1, 8, 8) and ratio == 1.0


def test_ddp_worker_rebuilds_grayscale_before_loading_flat_weights(tmp_path):
    from libreyolo import LibreYOLO9
    from libreyolo.training import ddp_spawn
    from libreyolo.utils.input_channels import pending_model_channels

    model = LibreYOLO9._from_scratch(size="t", nb_classes=1, device="cpu")
    model.input_size = 64
    configure_channels(model, 1)
    kw = ddp_spawn._build_init_kw(model)
    assert kw["_input_channels"] == 1
    flat = tmp_path / "flat.pt"
    torch.save({k: v.cpu() for k, v in model.model.state_dict().items()}, flat)
    init = {k: v for k, v in kw.items() if not k.startswith("_")}
    with pending_model_channels(kw["_input_channels"]):
        worker = LibreYOLO9(str(flat), **init)
    assert worker.input_channels == 1
    torch.testing.assert_close(
        worker.model.backbone.conv0.conv.weight.cpu(),
        model.model.backbone.conv0.conv.weight.cpu(),
    )
