"""Hermetic contracts for the optional WildDet3D adapter and 3D result payload."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import Boxes3D, LibreWildDet3D
from libreyolo.models.wilddet3d import model as adapter
from libreyolo.tasks import normalize_task, suffix_to_task, task_to_suffix
from libreyolo.utils.results import Results

pytestmark = pytest.mark.unit

K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float32)
ROW = [0, 0, 10, 2, 4, 6, 1, 0, 0, 0, 0.72, 0, 0.9, 0.8]


def payload(rows=None):
    return Boxes3D(
        torch.tensor([ROW] if rows is None else rows), (100, 100), torch.tensor(K)
    )


def outputs(ids=(0,)):
    n = len(ids)
    geometry = torch.tensor([ROW[:10]] * n).reshape(n, 10)
    boxes = torch.tensor([[30, 20, 70, 80]] * n).reshape(n, 4)
    return (
        [boxes],
        [geometry],
        [torch.full((n,), 0.72)],
        [torch.full((n,), 0.9)],
        [torch.full((n,), 0.8)],
        [torch.tensor(ids)],
        None,
    )


@pytest.fixture
def model(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not loaded by the unit test")
    return LibreWildDet3D(checkpoint, device="cpu")


def install_fake(model):
    calls = []

    def preprocess(image, intrinsics, depth=None):
        calls.append(("preprocess", image.copy(), intrinsics.copy(), depth))
        data = {
            "images": torch.zeros((1, 3, 8, 8)),
            "intrinsics": torch.tensor(intrinsics),
            "input_hw": (8, 8),
            "original_hw": image.shape[:2],
            "padding": (0, 0, 0, 0),
        }
        if depth is not None:
            data["depth_gt"] = torch.ones((1, 1, 8, 8))
        return data

    def predict(**kwargs):
        calls.append(("predict", kwargs))
        assert not torch.is_grad_enabled()
        return outputs()

    model.device = torch.device("cpu")  # Only the synthetic runtime runs on CPU.
    model._runtime = SimpleNamespace(preprocess=preprocess)
    model._predictor = predict
    return calls


def test_task_roundtrip():
    for alias in ("detect3d", "detection3d", "3d-detection"):
        assert normalize_task(alias) == "detect3d"
    assert suffix_to_task(task_to_suffix("detect3d")) == "detect3d"


def test_geometry_and_quaternion_rotation():
    p = payload()
    np.testing.assert_allclose(p.corners.min(axis=1), [[-2, -3, 9]])
    np.testing.assert_allclose(p.corners.max(axis=1), [[2, 3, 11]])
    row = ROW.copy()
    row[6:10] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]  # 90 degrees about camera z
    rotated = payload([row])
    np.testing.assert_allclose(rotated.corners.min(axis=1), [[-3, -2, 9]], atol=1e-6)
    negated = row.copy()
    negated[6:10] = [-v for v in row[6:10]]
    np.testing.assert_allclose(rotated.corners, payload([negated]).corners)


def test_result_slice_move_json_and_calibration():
    result = LibreWildDet3D._result(
        outputs((0, 1)), (100, 100), K, ["car", "chair"], None
    )
    selected = result[1].numpy()
    assert len(selected) == len(selected.boxes3d) == 1
    assert selected.boxes.cls[0] == selected.boxes3d.cls[0] == 1
    assert selected.boxes3d.intrinsics.shape == (3, 3)
    np.testing.assert_array_equal(selected.boxes3d.intrinsics, K)
    assert isinstance(selected.cpu().to(dtype=torch.float64).boxes3d.data, torch.Tensor)
    assert selected.to(dtype=torch.float64).boxes3d.intrinsics.dtype == torch.float64
    normal = json.loads(selected.to_json())[0]
    normalized = selected.summary(normalize=True)[0]
    assert normal["box3d"] == normalized["box3d"]
    assert normal["box"]["x1"] == 30 and normalized["box"]["x1"] == 0.3
    assert normal["box3d"]["center"] == [0, 0, 10]
    assert normal["box3d"]["quaternion_wxyz"] == [1, 0, 0, 0]
    assert normal["box3d"]["confidence_2d"] == 0.9


def test_empty_result():
    result = LibreWildDet3D._result(outputs(()), (100, 100), K, ["car"], None)
    assert len(result) == 0
    assert result.to_json() == "[]"
    assert result.boxes3d.corners.shape == (0, 8, 3)
    assert result[:0].boxes3d.data.shape == (0, 14)


@pytest.mark.parametrize(
    "column,value",
    [(3, 0), (4, -1), (0, float("nan")), (10, -0.1), (11, 0.5), (11, -1)],
)
def test_invalid_payload(column, value):
    row = ROW.copy()
    row[column] = value
    with pytest.raises(ValueError):
        payload([row])


def test_zero_quaternion():
    row = ROW.copy()
    row[6:10] = [0] * 4
    with pytest.raises(ValueError, match="quaternions"):
        payload([row])


@pytest.mark.parametrize("k", [np.eye(2), np.zeros((3, 3)), np.full((3, 3), np.nan)])
def test_bad_calibration(k):
    with pytest.raises(ValueError, match="intrinsics"):
        Boxes3D(torch.tensor([ROW]), (100, 100), k)


def test_projection_and_near_clipping():
    from libreyolo.utils.drawing import draw_boxes3d

    image = Image.new("RGB", (100, 100), "black")
    drawn = draw_boxes3d(image, payload())
    assert np.asarray(drawn).any()
    assert not np.asarray(image).any()
    row = ROW.copy()
    row[2] = -10
    assert not np.asarray(draw_boxes3d(image, payload([row]))).any()
    row[2] = 1
    assert draw_boxes3d(image, payload([row])).size == image.size
    with pytest.raises(ValueError, match="intrinsics"):
        draw_boxes3d(image, Boxes3D(torch.tensor([ROW])))


def test_load_deferred_and_explicit_device(model, monkeypatch):
    model.device = torch.device("cuda")
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    sentinel = lambda **kwargs: None

    def build(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            build_model=build, preprocess=lambda *a, **k: None
        ),
    )
    assert model._predictor is None
    model._load()
    model._load()
    assert len(calls) == 1
    assert calls[0]["skip_pretrained"] is True
    assert calls[0]["use_predicted_intrinsics"] is False
    assert calls[0]["score_threshold"] == model.DEFAULT_CONF
    assert calls[0]["score_3d_threshold"] == model.DEFAULT_CONF3D
    assert calls[0]["iou_threshold"] == model.DEFAULT_IOU
    assert calls[0]["device"] == "cuda"


def test_missing_runtime_help(model, monkeypatch):
    model.device = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(adapter.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="PYTHONPATH"):
        model._load()


def test_text_original_image_and_save(model, tmp_path):
    calls = install_fake(model)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[..., 0] = 255
    target = tmp_path / "result.png"
    result = model(
        image,
        text=["car"],
        intrinsics=K,
        color_format="rgb",
        save=True,
        output_path=target,
    )
    assert isinstance(result, Results)
    assert calls[0][1][0, 0].tolist() == [255, 0, 0]
    assert calls[1][1]["input_texts"] == ["car"]
    assert calls[1][1]["original_hw"] == [(100, 100)]
    assert target.is_file()
    assert not np.array_equal(np.asarray(Image.open(target)), image)


def test_prompt_groups_forwarded_without_coordinate_changes(model):
    calls = install_fake(model)
    image = Image.new("RGB", (100, 100))
    model(image, intrinsics=K, bboxes=[[3, 4, 25, 30]], prompt_mode="visual")
    assert calls[-1][1]["input_boxes"] == [[3, 4, 25, 30]]
    assert calls[-1][1]["prompt_text"] == "visual"
    model(image, intrinsics=K, points=[[3, 4], [25, 30]], labels=[1, 0])
    assert calls[-1][1]["input_points"] == [[[3, 4, 1], [25, 30, 0]]]


def test_depth_forwarding(model):
    calls = install_fake(model)
    model.use_depth = True
    depth = np.full((100, 100), 2, dtype=np.float32)
    model(Image.new("RGB", (100, 100)), intrinsics=K, text="car", depth=depth)
    np.testing.assert_array_equal(calls[0][3], depth)
    assert "depth_gt" in calls[1][1]


def test_list_and_stream_return_contract(model, tmp_path):
    calls = install_fake(model)
    image = Image.new("RGB", (100, 100))
    model.set_classes(["car"])
    streamed = model([image, image], intrinsics=K, stream=True)
    assert calls == []
    assert len(list(streamed)) == 2
    assert isinstance(model([image], intrinsics=K), list)
    image.save(tmp_path / "a.png")
    assert len(model(tmp_path, intrinsics=K)) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"text": []},
        {"text": [""]},
        {"text": "car", "bboxes": [[1, 1, 2, 2]]},
        {"bboxes": [[1, 1, 0, 0]]},
        {"points": []},
        {"points": [[1, 2]], "labels": [2]},
        {"points": [[1, 2]], "labels": [0]},
        {"text": "car", "depth": np.ones((3, 3))},
        {"text": "car", "labels": [1]},
        {"text": "car", "prompt_mode": "visual"},
    ],
)
def test_invalid_call_rejected_before_loading(model, kwargs):
    with pytest.raises((ValueError, TypeError)):
        model(Image.new("RGB", (100, 100)), intrinsics=K, **kwargs)
    assert model._predictor is None


def test_misaligned_upstream_rejected():
    wrong = list(outputs())
    wrong[1] = []
    with pytest.raises(ValueError, match="different lengths"):
        LibreWildDet3D._result(wrong, (100, 100), K, ["car"], None)
    with pytest.raises(ValueError, match="outside"):
        LibreWildDet3D._result(outputs((5,)), (100, 100), K, ["car"], None)


@pytest.mark.parametrize("method", ["train", "val", "export", "track"])
def test_unsupported_surfaces(model, method):
    with pytest.raises(NotImplementedError):
        getattr(model, method)()


def test_list_payload_and_alignment_guards():
    from libreyolo.utils.results import Boxes

    p = Boxes3D([ROW], (100, 100), K.tolist())
    assert p.xyz.shape == (1, 3)
    assert p.to(dtype=torch.float64).intrinsics.dtype == torch.float64
    with pytest.raises(ValueError, match="row-aligned"):
        Results(None, (100, 100), boxes3d=p)
    result = LibreWildDet3D._result(outputs(), (100, 100), K, ["car"], None)
    with pytest.raises(ValueError, match="row-aligned"):
        result.update(boxes=Boxes(torch.empty((0, 4)), torch.empty(0), torch.empty(0)))
    assert len(result.boxes) == 1  # A rejected update leaves the result intact.
    with pytest.raises(ValueError, match="canvas"):
        Results(result.boxes, (20, 30), boxes3d=p)


def test_numeric_cuda_device_and_no_download_route(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    for device in (0, "0", "cuda:0"):
        assert str(LibreWildDet3D(checkpoint, device=device).device) == "cuda:0"
    assert LibreWildDet3D.get_download_url("unknown.pt") is None
    url = LibreWildDet3D.get_download_url(adapter.WEIGHT_FILE)
    assert adapter.HF_REVISION in url
    assert url.endswith("/wilddet3d_alldata_all_prompt_v1.0.pt")


def test_default_checkpoint_uses_mirror(tmp_path, monkeypatch):
    checkpoint = tmp_path / "mirrored.pt"
    checkpoint.touch()
    calls = []

    def download(cls):
        calls.append(True)
        return checkpoint

    monkeypatch.setattr(LibreWildDet3D, "_download_checkpoint", classmethod(download))
    model = LibreWildDet3D(device="cpu")
    assert model.model_path == checkpoint
    assert calls == [True]


def test_mirror_hash_check(tmp_path, monkeypatch):
    checkpoint = tmp_path / "mirrored.pt"
    checkpoint.write_bytes(b"known checkpoint bytes")
    monkeypatch.setattr(
        adapter,
        "WEIGHT_SHA256",
        "a05545a7ab49d03ef298599bff5bcb657521bd52ef4d71e4b2c2330020492dfc",
    )
    LibreWildDet3D._verify_mirrored_checkpoint(checkpoint)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        LibreWildDet3D._verify_mirrored_checkpoint(checkpoint)


def test_mirror_download_is_revision_pinned(tmp_path, monkeypatch):
    checkpoint = tmp_path / adapter.WEIGHT_FILE
    checkpoint.write_bytes(b"known checkpoint bytes")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(checkpoint)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=download),
    )
    monkeypatch.setattr(
        adapter,
        "WEIGHT_SHA256",
        "a05545a7ab49d03ef298599bff5bcb657521bd52ef4d71e4b2c2330020492dfc",
    )
    assert LibreWildDet3D._download_checkpoint() == checkpoint.resolve()
    assert calls == [
        {
            "repo_id": "LibreYOLO/LibreWildDet3D",
            "filename": adapter.WEIGHT_FILE,
            "revision": adapter.HF_REVISION,
        }
    ]


def test_auto_device_on_mac_uses_cpu(model, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert str(LibreWildDet3D(model.model_path).device) == "cpu"


def test_explicit_mps_explains_upstream_limitation(model):
    with pytest.raises(ValueError, match="mixes CPU and MPS tensors"):
        LibreWildDet3D(model.model_path, device="mps")


def test_explicit_cpu_never_requires_cuda(model, monkeypatch):
    from libreyolo.models.wilddet3d import runtime

    def reject_cuda_probe():
        raise AssertionError("CPU execution must not probe CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", reject_cuda_probe)
    calls = []

    def create_worker(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(runtime, "RuntimeWorker", create_worker)
    model._load()
    assert calls[0]["config"]["device"] == "cpu"


def test_closed_worker_is_replaced_after_failure(model, monkeypatch):
    from libreyolo.models.wilddet3d import runtime

    workers = []

    class Worker:
        def __init__(self, **kwargs):
            self.closed = False
            workers.append(self)

        def predict(self, *args, **kwargs):
            if len(workers) == 1:
                self.closed = True
                raise RuntimeError("worker exited")
            return outputs()

        def close(self):
            self.closed = True

    monkeypatch.setattr(runtime, "RuntimeWorker", Worker)
    image = Image.new("RGB", (100, 100))
    with pytest.raises(RuntimeError, match="worker exited"):
        model(image, intrinsics=K, text=["car"])
    assert model._backend is None
    result = model(image, intrinsics=K, text=["car"])
    assert len(workers) == 2
    assert len(result) == 1


def test_combined_ranking_score_is_not_clamped(model):
    row = ROW.copy()
    row[10] = 1.2393523
    p = payload([row])
    assert float(p.conf[0]) == pytest.approx(1.2393523)
    assert LibreWildDet3D(model.model_path, device="cpu", conf=1.2).conf == 1.2


@pytest.mark.parametrize("argument", ["runtime_path", "runtime_python"])
def test_missing_runtime_location_fails_at_construction(model, tmp_path, argument):
    with pytest.raises(FileNotFoundError, match="not found"):
        LibreWildDet3D(
            model.model_path,
            device="cpu",
            **{argument: tmp_path / "missing"},
        )


def test_runtime_python_preserves_virtualenv_symlink(model, tmp_path):
    interpreter = tmp_path / "python"
    interpreter.symlink_to(Path(sys.executable))
    configured = LibreWildDet3D(
        model.model_path,
        device="cpu",
        runtime_python=interpreter,
    )
    assert configured._runtime_python == interpreter.absolute()
    assert configured._runtime_python.is_symlink()


def test_corners_match_permissive_upstream_golden():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/wilddet3d_corners.json").read_text()
    )
    boxes = torch.tensor(fixture["boxes"], dtype=torch.float32)
    data = torch.cat((boxes, torch.zeros((len(boxes), 4))), dim=1)
    np.testing.assert_allclose(Boxes3D(data).corners, fixture["corners"], atol=2e-6)
