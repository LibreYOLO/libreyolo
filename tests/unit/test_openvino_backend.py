import pytest

from libreyolo.backends.openvino import OpenVINOBackend

pytestmark = pytest.mark.unit


class _FakeOutput:
    def __init__(self, shape):
        self._shape = shape

    @property
    def shape(self):
        if isinstance(self._shape, Exception):
            raise self._shape
        return self._shape


class _FakeModel:
    def __init__(self, shape):
        self.inputs = [_FakeOutput(shape)]


def test_openvino_backend_reads_static_input_imgsz():
    model = _FakeModel([1, 3, 384, 384])

    assert OpenVINOBackend._read_static_input_imgsz(model) == 384


def test_openvino_backend_ignores_dynamic_input_shape():
    model = _FakeModel(RuntimeError("to_shape was called on a dynamic shape"))

    assert OpenVINOBackend._read_static_input_imgsz(model) is None


@pytest.mark.parametrize(
    "shape",
    [
        [1, 3],
        [1, 3, -1, -1],
        [1, 3, "?", "?"],
    ],
)
def test_openvino_backend_ignores_non_static_input_imgsz(shape):
    model = _FakeModel(shape)

    assert OpenVINOBackend._read_static_input_imgsz(model) is None
