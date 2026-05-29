import numpy as np
import pytest
import torch

from libreyolo.backends.base import BaseBackend


pytestmark = pytest.mark.unit


class _Backend(BaseBackend):
    def __init__(self, task: str = "segment"):
        super().__init__(
            model_path="model.onnx",
            nb_classes=2,
            device="cpu",
            imgsz=560,
            model_family="rfdetr",
            names={0: "fire", 1: "smoke"},
            model_size="n",
            task=task,
            supported_tasks=("detect", "segment"),
            default_task="detect",
        )

    def _run_inference(self, blob: np.ndarray) -> list:
        batch = blob.shape[0]
        return [
            np.zeros((batch, 100, 4), dtype=np.float32),
            np.zeros((batch, 100, 2), dtype=np.float32),
            np.zeros((batch, 100, 35, 35), dtype=np.float32),
        ]


def test_backend_val_uses_exported_model_adapter(monkeypatch):
    captured = {}

    class _Validator:
        def __init__(self, model, config):
            captured["model"] = model
            captured["config"] = config

        def __call__(self):
            return {"metrics/mAP50": 0.5}

    monkeypatch.setattr("libreyolo.validation.SegmentationValidator", _Validator)

    backend = _Backend(task="segment")
    metrics = backend.val(
        data="data.yaml",
        batch=4,
        imgsz=None,
        conf=0.01,
        iou=0.7,
        workers=0,
        device="cpu",
        split="test",
    )

    assert metrics == {"metrics/mAP50": 0.5}
    assert captured["model"] is backend
    assert captured["config"].imgsz == 560
    assert captured["config"].batch_size == 4
    assert captured["config"].conf_thres == 0.01
    assert backend.FAMILY == "rfdetr"
    assert backend.size == "n"


def test_backend_val_rejects_augment():
    with pytest.raises(ValueError, match="Augmented validation"):
        _Backend().val(data="data.yaml", augment=True)


def test_backend_forward_falls_back_for_fixed_batch_exports():
    class _FixedBatchBackend(_Backend):
        def _run_inference(self, blob: np.ndarray) -> list:
            if blob.shape[0] != 1:
                raise RuntimeError("expected batch 1")
            return [
                np.full((1, 2, 4), blob.sum(), dtype=np.float32),
                np.full((1, 2, 2), blob.sum(), dtype=np.float32),
            ]

    outputs = _FixedBatchBackend(task="detect")._forward(torch.ones(3, 3, 4, 4))

    assert [tuple(output.shape) for output in outputs] == [(3, 2, 4), (3, 2, 2)]
    assert outputs[0][:, 0, 0].tolist() == [48.0, 48.0, 48.0]


def test_backend_init_allows_read_only_size_property():
    class _ReadOnlySizeBackend(_Backend):
        @property
        def size(self) -> str:
            return self.model_size or "computed"

        def __init__(self):
            super().__init__(task="detect")

    backend = _ReadOnlySizeBackend()

    assert backend.size == "n"
    assert backend.FAMILY == "rfdetr"


def test_backend_eval_proxy_has_no_to():
    from libreyolo.backends.base import _BackendEvalProxy

    proxy = _BackendEvalProxy()
    assert not hasattr(proxy, "to")
    assert hasattr(proxy, "eval")


def test_set_device_does_not_call_to_on_backend_proxy():
    """_set_device must not raise when model.model is a _BackendEvalProxy."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from libreyolo.backends.base import _BackendEvalProxy
    from libreyolo.models.base.inference import InferenceRunner

    proxy = _BackendEvalProxy()
    fake_model = SimpleNamespace(
        device=torch.device("cpu"),
        model=proxy,
    )

    runner = object.__new__(InferenceRunner)
    runner.model = fake_model

    # Calling _set_device with a different device must not raise AttributeError.
    with patch("torch.cuda.is_available", return_value=True):
        runner._set_device("cuda:0")

    # model.device is updated; proxy is left untouched (no .to() call).
    assert fake_model.device == torch.device("cuda:0")
