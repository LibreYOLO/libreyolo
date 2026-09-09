"""Unit contract for the optional 3D-MOOD adapter."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import Libre3DMOOD
from libreyolo.models.mood3d import model as adapter

pytestmark = pytest.mark.unit

K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
CUBOIDS = np.array(
    [
        [0, 0, 3, 2, 1, 1, 1, 0, 0, 0],
        [1, 0, 4, 1, 2, 1, 0.7071, 0, 0.7071, 0],
    ],
    dtype=np.float32,
)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "upstream.pt"
    path.write_bytes(b"checkpoint")
    return path


@pytest.fixture
def model(checkpoint):
    return Libre3DMOOD(checkpoint, size="t", device="cpu")


def outputs(shape=(480, 640)):
    return (
        np.array([[10, 20, 100, 120], [30, 40, 130, 150]], dtype=np.float32),
        CUBOIDS.copy(),
        np.array([0.8, 0.4], dtype=np.float32),
        np.array([0, 1], dtype=np.float32),
        np.ones(shape, dtype=np.float32) * 2,
        None,
    )


def test_result_maps_geometry_score_and_depth():
    result = Libre3DMOOD._result(
        outputs(), (480, 640), K, ["chair", "table"], "image.jpg"
    )
    assert len(result) == len(result.boxes3d) == 2
    torch.testing.assert_close(result.boxes.xyxy, torch.tensor(outputs()[0]))
    torch.testing.assert_close(result.boxes3d.data[:, :10], torch.tensor(CUBOIDS))
    torch.testing.assert_close(result.boxes3d.conf, torch.tensor([0.8, 0.4]))
    torch.testing.assert_close(result.boxes3d.conf2d, result.boxes3d.conf)
    torch.testing.assert_close(result.boxes3d.conf3d, torch.ones(2))
    assert result.names == {0: "chair", 1: "table"}
    assert result.depth_map.data.shape == (480, 640)
    assert result.depth_map.mean == 2.0
    assert result.boxes3d.corners.shape == (2, 8, 3)


def test_predict_single_many_stream_and_worker_reuse(model, monkeypatch):
    workers = []

    class Worker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []
            self.closed = False
            workers.append(self)

        def predict(self, image, intrinsics, prompt, depth=None):
            self.calls.append((image, intrinsics, prompt, depth))
            result = outputs(image.shape[:2])
            if len(prompt["text"]) == 1:
                return tuple(value[:1] if hasattr(value, "shape") and value.shape != image.shape[:2] else value for value in result)
            return result

        def close(self):
            self.closed = True

    from libreyolo.models.mood3d import runtime

    monkeypatch.setattr(runtime, "RuntimeWorker", Worker)
    image = Image.new("RGB", (640, 480))
    result = model(image, intrinsics=K, text=["chair", "table"])
    assert len(result) == 2
    assert workers[0].kwargs["config"]["size"] == "t"
    assert workers[0].calls[0][2] == {"text": ["chair", "table"]}
    assert model([image, image], intrinsics=K, text="chair")[0].names == {0: "chair"}
    assert len(workers) == 1
    stream = model(image, intrinsics=K, text=["chair", "table"], stream=True)
    assert iter(stream) is stream
    assert len(next(stream)) == 2
    model.close()
    assert workers[0].closed


def test_set_classes_and_save(model, monkeypatch, tmp_path):
    class Worker:
        def __init__(self, **kwargs):
            pass

        def predict(self, image, intrinsics, prompt, depth=None):
            return outputs(image.shape[:2])

        def close(self):
            pass

    from libreyolo.models.mood3d import runtime

    monkeypatch.setattr(runtime, "RuntimeWorker", Worker)
    destination = tmp_path / "cuboids.png"
    model.set_classes(["chair", "table"])
    result = model(
        Image.new("RGB", (640, 480)),
        intrinsics=K,
        save=True,
        output_path=destination,
    )
    assert destination.is_file()
    assert result.names == {0: "chair", 1: "table"}


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"size": "x"}, "size"),
        ({"conf": -0.1}, "conf"),
        ({"iou": 1.1}, "iou"),
        ({"max_det": 0}, "max_det"),
        ({"max_det": 1.5}, "max_det"),
        ({"device": "xpu"}, "supports"),
    ],
)
def test_constructor_validation(checkpoint, kwargs, match):
    with pytest.raises(ValueError, match=match):
        Libre3DMOOD(checkpoint, **({"device": "cpu"} | kwargs))


def test_auto_prefers_mps_without_cuda(checkpoint, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert Libre3DMOOD(checkpoint).device.type == "mps"


def test_size_is_inferred_and_mismatches_raise(tmp_path):
    checkpoint = tmp_path / "gdino3d_swin-b_120e_omni3d_834c97.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert Libre3DMOOD(checkpoint, device="cpu").size == "b"
    with pytest.raises(ValueError, match="not 't'"):
        Libre3DMOOD(checkpoint, size="t", device="cpu")


def test_prompt_and_calibration_validation(model):
    image = Image.new("RGB", (20, 10))
    with pytest.raises(ValueError, match="text"):
        model(image, intrinsics=K, text=[])
    with pytest.raises(ValueError, match="intrinsics"):
        model(image, intrinsics=np.eye(2), text=["chair"])
    with pytest.raises(ValueError, match="output_path"):
        model(image, intrinsics=K, text=["chair"], output_path="x.png")


def test_missing_runtime_has_install_hint(model, monkeypatch):
    from libreyolo.models.mood3d import runtime

    class MissingRuntime:
        def __init__(self, **kwargs):
            raise RuntimeError("3D-MOOD runtime: No module named 'opendet3d'")

    monkeypatch.setattr(runtime, "RuntimeWorker", MissingRuntime)
    with pytest.raises(ImportError, match="3D-MOOD#installation"):
        model(Image.new("RGB", (20, 10)), intrinsics=K, text=["chair"])


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda values: values.__setitem__(0, np.full((1, 3), 1)), "geometry"),
        (lambda values: values.__setitem__(2, np.array([2, 0.4])), "scores"),
        (lambda values: values.__setitem__(3, np.array([0, 3])), "vocabulary"),
        (lambda values: values.__setitem__(4, np.ones((3, 3))), "canvas"),
    ],
)
def test_malformed_runtime_output_is_rejected(mutator, match):
    values = list(outputs())
    mutator(values)
    with pytest.raises(ValueError, match=match):
        Libre3DMOOD._result(values, (480, 640), K, ["chair", "table"], None)


def test_checkpoint_urls_and_digest(checkpoint, monkeypatch):
    assert "/resolve/f1e163b249b43a8bb7b8b4ef4f446828270e35ef/" in (
        Libre3DMOOD.get_download_url("Libre3DMOODt.pt")
    )
    assert Libre3DMOOD.get_download_url("unknown.pt") is None
    digest = __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setitem(adapter.WEIGHTS["t"], "sha256", digest)
    Libre3DMOOD._verify_checkpoint(checkpoint, "t")
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        Libre3DMOOD._verify_checkpoint(checkpoint, "t")


def test_download_uses_family_repo(tmp_path, monkeypatch):
    checkpoint = tmp_path / "Libre3DMOODt.pt"
    checkpoint.write_bytes(b"weight")
    digest = __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setitem(adapter.WEIGHTS["t"], "sha256", digest)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return checkpoint

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert Libre3DMOOD._download_checkpoint("t") == checkpoint.resolve()
    assert calls == [
        {
            "repo_id": "LibreYOLO/Libre3DMOODt",
            "filename": "Libre3DMOODt.pt",
            "revision": "f1e163b249b43a8bb7b8b4ef4f446828270e35ef",
        }
    ]


@pytest.mark.parametrize("method", ["train", "val", "export", "track"])
def test_unsupported_workflows_raise(model, method):
    with pytest.raises(NotImplementedError):
        getattr(model, method)()


def test_provenance_constants_and_files():
    assert len(adapter.UPSTREAM_REVISION) == 40
    assert len(adapter.UPSTREAM_HF_REVISION) == 40
    assert adapter.UPSTREAM_REPO == "cvg/3D-MOOD"
    assert Path(adapter.__file__).with_name("NOTICE").is_file()


def test_runtime_uses_shared_worker_boundary():
    from libreyolo.models.mood3d.runtime import RuntimeWorker

    assert RuntimeWorker.__mro__[1].__module__ == "libreyolo.models.runtime_worker"
