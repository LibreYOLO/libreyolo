"""DetAny3D geometry, prompt routing and worker lifecycle without external data."""

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.models.detany3d import LibreDetAny3D
from libreyolo.models.detany3d.worker import _portable_attention

pytestmark = pytest.mark.unit


def test_inventory_does_not_require_runtime_in_parent_interpreter(monkeypatch):
    from libreyolo.models import inventory

    original = inventory.importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "detect_anything":
            return None
        return original(name, *args, **kwargs)

    monkeypatch.setattr(inventory.importlib.util, "find_spec", find_spec)
    metadata = inventory.collect_model_inventory()["detany3d"]
    assert metadata["available"] is True
    assert metadata["cli_command"] == "detany3d"


def arrays():
    return {
        "boxes": np.array([[8, 7, 4, 2]], np.float32),
        "centers": np.array([[0, 0, 10]], np.float32),
        "dimensions": np.array([[2, 4, 6]], np.float32),
        "rotations": np.eye(3, dtype=np.float32)[None],
        "scores": np.array([0.7], np.float32),
        "intrinsics": np.array([[100, 0, 8], [0, 100, 7], [0, 0, 1]], np.float32),
        "view_to_original": np.array([[2, 0, 10], [0, 3, 20], [0, 0, 1]], np.float32),
    }


def test_camera_and_local_axis_conversion():
    output = LibreDetAny3D._result(arrays(), ["car"], (100, 200), None)
    np.testing.assert_allclose(output.boxes.xyxy, [[22, 38, 30, 44]])
    np.testing.assert_allclose(
        output.boxes3d.intrinsics, [[200, 0, 26], [0, 300, 41], [0, 0, 1]]
    )
    np.testing.assert_allclose(output.boxes3d.dimensions, [[2, 6, 4]])
    np.testing.assert_allclose(output.boxes3d.corners[0, 0], [1, 2, 13], atol=1e-6)
    assert not output.boxes.is_track
    assert output.names == {0: "car"}
    assert output.boxes3d.conf3d[0] == 1
    assert output.boxes3d.conf[0] == pytest.approx(0.7)


def test_axis_change_is_applied_in_local_frame():
    packet = arrays()
    packet["rotations"][0] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    output = LibreDetAny3D._result(packet, ["car"], (100, 200), None)
    np.testing.assert_allclose(output.boxes3d.corners[0, 0], [-2, 1, 13], atol=1e-6)
    selected = output._select([0])
    np.testing.assert_array_equal(
        selected.boxes3d.intrinsics, output.boxes3d.intrinsics
    )


@pytest.mark.parametrize("rank", [3, 4])
def test_portable_attention_uniform_and_masked(rank):
    q = torch.zeros(1, 2, 1, 2)
    k = torch.zeros(1, 3, 1, 2)
    v = torch.tensor([[[[1.0, 3.0]], [[7.0, 11.0]], [[16.0, 20.0]]]])
    if rank == 3:
        q, k, v = [x.squeeze(2) for x in (q, k, v)]
    result = _portable_attention(q, k, v)
    expected = torch.tensor([8.0, 34 / 3]).expand_as(result)
    torch.testing.assert_close(result, expected)
    bias = torch.tensor([[[[float("-inf"), 0, float("-inf")]]]])
    result = _portable_attention(q, k, v, attn_bias=bias)
    torch.testing.assert_close(result, torch.tensor([7.0, 11.0]).expand_as(result))


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    import libreyolo.models.detany3d.model as module

    instances = []
    calls = []
    replies = ["person", "car"]
    failures = []

    class FakeWorker:
        device = "cpu"

        def __init__(self, **kwargs):
            self.closed = False
            instances.append(self)

        def predict(self, image, prompt):
            calls.append(prompt)
            if failures:
                raise failures.pop()
            return arrays(), [replies.pop(0) if replies else "car"]

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "RuntimeWorker", FakeWorker)
    (tmp_path / "model.pth").touch()
    (tmp_path / "wrap_model.py").touch()
    model = LibreDetAny3D(tmp_path / "model.pth", runtime_path=tmp_path)
    yield model, instances, calls, failures
    model.close()


def test_vocabulary_ids_remain_stable(fake_model):
    model, instances, calls, _ = fake_model
    model.set_classes(["car", "person"])
    image = Image.new("RGB", (100, 100))
    first = model(image)
    second = model(image)
    assert first.boxes.cls.tolist() == [1]
    assert second.boxes.cls.tolist() == [0]
    assert len(instances) == 1
    assert calls[0]["text"] == ["car", "person"]
    assert model.names == {0: "car", 1: "person"}


def test_prompt_shapes_multi_input_save_and_close(fake_model, tmp_path):
    model, instances, calls, _ = fake_model
    image = Image.new("RGB", (100, 100))
    model(
        image, points=[[10, 20], [11, 21]], save=True, output_path=tmp_path / "out.png"
    )
    assert calls[-1]["points"] == [[[10.0, 20.0], [11.0, 21.0]]]
    assert (tmp_path / "out.png").is_file()
    results = model([image, image], bboxes=[1, 2, 30, 40], stream=True)
    assert len(list(results)) == 2
    assert calls[-1]["bboxes"] == [[1.0, 2.0, 30.0, 40.0]]
    model.close()
    assert instances[0].closed


def test_failed_prediction_restarts_worker(fake_model):
    model, instances, _, failures = fake_model
    image = Image.new("RGB", (100, 100))
    failures.append(RuntimeError("failed upstream"))
    with pytest.raises(RuntimeError, match="failed upstream"):
        model(image, points=[10, 20])
    assert instances[0].closed
    model(image, points=[10, 20])
    assert len(instances) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"points": []},
        {"bboxes": [1, 2, 1, 4]},
        {"text": [""]},
        {"points": [1, 2], "text": "car"},
        {"points": [1, 2], "bboxes": [1, 2, 3, 4]},
    ],
)
def test_invalid_prompts_fail_before_prediction(fake_model, kwargs):
    model, _, calls, _ = fake_model
    with pytest.raises(ValueError):
        model(Image.new("RGB", (100, 100)), **kwargs)
    assert calls == []


def test_unsupported_workflows_and_no_download(fake_model):
    model, *_ = fake_model
    assert model.get_download_url("anything.pt") is None
    for method in ("train", "val", "export", "track"):
        with pytest.raises(NotImplementedError):
            getattr(model, method)()


def test_worker_bootstrap_preserves_runtime_dependencies(tmp_path):
    """A parent wheel must not prepend its entire site-packages to the worker."""
    import subprocess
    import sys
    from pathlib import Path

    from libreyolo.models.detany3d import worker

    parent = tmp_path / "parent_site"
    runtime = tmp_path / "runtime_site"
    worker_path = parent / "libreyolo/models/detany3d/worker.py"
    worker_path.parent.mkdir(parents=True)
    runtime.mkdir()
    worker_path.write_text(Path(worker.__file__).read_text())
    (parent / "dependency.py").write_text('ORIGIN="parent"\n')
    (runtime / "dependency.py").write_text('ORIGIN="runtime"\n')
    (parent / "libreyolo/__init__.py").write_text(
        "import dependency\nORIGIN=dependency.ORIGIN\n"
    )
    script = """
import importlib.util,sys
sys.path.insert(0,sys.argv[2])
spec=importlib.util.spec_from_file_location('worker_probe',sys.argv[1])
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
worker._bootstrap_package()
import libreyolo
assert libreyolo.ORIGIN=='runtime',libreyolo.ORIGIN
assert sys.argv[3] not in sys.path
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(worker_path), str(runtime), str(parent)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mirror_download_url_is_revision_pinned():
    from libreyolo.models.detany3d import model as adapter

    assert LibreDetAny3D.get_download_url("unknown.pth") is None
    url = LibreDetAny3D.get_download_url(adapter.WEIGHT_FILE)
    assert adapter.HF_REVISION in url
    assert url.endswith(adapter.WEIGHT_FILE)
    notice = LibreDetAny3D.get_download_notice(adapter.WEIGHT_FILE, url)
    assert "non-commercial" in notice and "MIT" in notice


def test_default_checkpoint_uses_mirror(tmp_path, monkeypatch):
    from libreyolo.models.detany3d import model as adapter

    checkpoint = tmp_path / "mirrored.pth"
    checkpoint.touch()
    calls = []

    def download(cls):
        calls.append(True)
        return checkpoint.resolve()

    monkeypatch.setattr(LibreDetAny3D, "_download_checkpoint", classmethod(download))
    assert LibreDetAny3D._resolve_checkpoint(None) == checkpoint.resolve()
    assert (
        LibreDetAny3D._resolve_checkpoint(adapter.WEIGHT_FILE) == checkpoint.resolve()
    )
    assert calls == [True, True]
    with pytest.raises(FileNotFoundError, match="not found"):
        LibreDetAny3D._resolve_checkpoint(tmp_path / "missing.pth")


def test_mirror_hash_check(tmp_path, monkeypatch):
    from libreyolo.models.detany3d import model as adapter

    checkpoint = tmp_path / "mirrored.pth"
    checkpoint.write_bytes(b"known checkpoint bytes")
    monkeypatch.setattr(
        adapter,
        "WEIGHT_SHA256",
        "a05545a7ab49d03ef298599bff5bcb657521bd52ef4d71e4b2c2330020492dfc",
    )
    LibreDetAny3D._verify_mirrored_checkpoint(checkpoint)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        LibreDetAny3D._verify_mirrored_checkpoint(checkpoint)


def test_mirror_download_is_revision_pinned(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from libreyolo.models.detany3d import model as adapter

    checkpoint = tmp_path / "mirrored.pth"
    checkpoint.write_bytes(b"known checkpoint bytes")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(checkpoint)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
    )
    monkeypatch.setattr(
        adapter,
        "WEIGHT_SHA256",
        "a05545a7ab49d03ef298599bff5bcb657521bd52ef4d71e4b2c2330020492dfc",
    )
    assert LibreDetAny3D._download_checkpoint() == checkpoint.resolve()
    assert calls == [
        {
            "repo_id": "LibreYOLO/LibreDetAny3D",
            "filename": adapter.WEIGHT_FILE,
            "revision": adapter.HF_REVISION,
        }
    ]
