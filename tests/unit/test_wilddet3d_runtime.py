"""Hermetic checks of the isolated runtime boundary and its lifecycle."""

import gc
import os
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from libreyolo.models.wilddet3d.runtime import RuntimeWorker, decode_array, encode_array

pytestmark = pytest.mark.unit


@pytest.fixture
def runtime_path(tmp_path):
    package = tmp_path / "wilddet3d"
    package.mkdir()
    (package / "__init__.py").write_text(
        """
import os
import time
import torch

class Predictor:
    def __init__(self):
        self.calls = 0

    def __call__(self, images, intrinsics, input_hw, original_hw, padding,
                 input_texts=None, depth_gt=None, **kwargs):
        if input_texts == ["fail"]:
            raise ValueError("deliberate test failure")
        if input_texts == ["exit"]:
            os._exit(7)
        if input_texts == ["wait"]:
            time.sleep(.2)
        self.calls += 1
        assert not torch.is_grad_enabled()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            value = images[0, 0, 0, 0] * 2
            if depth_gt is not None:
                value = value + depth_gt[0, 0, 0, 0]
        torch.cuda.current_stream().wait_stream(stream)
        h, w = original_hw[0]
        return ([torch.tensor([[0., 0., float(w), float(h)]])],
                [torch.tensor([[value.item(), 0, 10, 2, 2, 2, 1, 0, 0, 0]])],
                [torch.tensor([1.2 + self.calls / 100])],
                [torch.tensor([.9])], [torch.tensor([.8])],
                [torch.tensor([0])], None)

def build_model(checkpoint, device, **kwargs):
    assert device == "cpu"
    print("upstream startup output")
    return Predictor()

def preprocess(image, intrinsics, depth=None):
    data = dict(images=torch.from_numpy(image).permute(2,0,1)[None],
                intrinsics=torch.as_tensor(intrinsics), input_hw=image.shape[:2],
                original_hw=image.shape[:2], padding=(0,0,0,0))
    if depth is not None:
        data["depth_gt"] = torch.as_tensor(depth)[None,None]
    return data
""",
        encoding="utf-8",
    )
    return tmp_path


def worker(runtime_path):
    return RuntimeWorker(
        config={"checkpoint": "local-test.pt", "device": "cpu"},
        runtime_path=runtime_path,
        runtime_python=sys.executable,
    )


def image(value=7):
    return np.full((3, 5, 3), value, dtype=np.uint8)


def test_real_process_transport_and_reuse(runtime_path):
    original_stream = torch.cuda.Stream
    backend = worker(runtime_path)
    process = backend._process
    try:
        first = backend.predict(image(), np.eye(3), {"input_texts": ["object"]})
        second = backend.predict(
            image(9),
            np.eye(3),
            {"input_texts": ["object"]},
            depth=np.full((3, 5), 4, dtype=np.float32),
        )
        assert process.pid != os.getpid()
        assert backend._process.pid == process.pid
        np.testing.assert_array_equal(first[0][0], [[0, 0, 5, 3]])
        assert first[1][0][0, 0] == 14
        assert second[1][0][0, 0] == 22
        assert first[2][0][0] == pytest.approx(1.21)
        assert second[2][0][0] == pytest.approx(1.22)
        assert torch.cuda.Stream is original_stream
    finally:
        backend.close()
    assert process.poll() is not None
    assert torch.cuda.Stream is original_stream


def test_concurrent_calls_are_not_crossed(runtime_path):
    backend = worker(runtime_path)
    try:

        def predict(value):
            result = backend.predict(
                image(value), np.eye(3), {"input_texts": ["object"]}
            )
            return result[1][0][0, 0]

        with ThreadPoolExecutor(2) as pool:
            assert list(pool.map(predict, [13, 29])) == [26, 58]
    finally:
        backend.close()


@pytest.mark.parametrize(
    "prompt,match", [("fail", "deliberate test failure"), ("exit", "exited")]
)
def test_worker_failures_surface(runtime_path, prompt, match):
    backend = worker(runtime_path)
    try:
        with pytest.raises(RuntimeError, match=match):
            backend.predict(image(), np.eye(3), {"input_texts": [prompt]})
    finally:
        backend.close()
    assert backend._process.poll() is not None


def test_timeout_releases_worker(runtime_path):
    backend = worker(runtime_path)
    backend._timeout = 0.01
    with pytest.raises(RuntimeError, match="timed out"):
        backend.predict(image(), np.eye(3), {"input_texts": ["wait"]})
    assert backend._process.poll() is not None


def test_owner_collection_releases_worker(runtime_path):
    backend = worker(runtime_path)
    process = backend._process
    ref = weakref.ref(backend)
    del backend
    gc.collect()
    assert ref() is None
    assert process.poll() is not None


def test_transport_rejects_wrong_dtype_and_size():
    with pytest.raises(TypeError, match="dtype"):
        encode_array(np.array([object()]))
    message = encode_array(np.zeros((2, 3), dtype=np.float32))
    message["shape"] = [2, 4]
    with pytest.raises(ValueError, match="byte count"):
        decode_array(message)
