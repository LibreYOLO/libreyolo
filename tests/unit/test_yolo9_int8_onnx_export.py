"""YOLO9 ONNX INT8 export smoke tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = [pytest.mark.unit, pytest.mark.onnx, pytest.mark.export_backend]


@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None
    or importlib.util.find_spec("onnxruntime") is None,
    reason="onnx/onnxruntime not installed",
)
def test_yolo9_detect_onnx_int8_export_loads_and_predicts(tmp_path):
    import onnx
    import onnxruntime as ort

    from libreyolo import LibreYOLO, LibreYOLO9

    image_dir = tmp_path / "images" / "train"
    image_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for idx in range(2):
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(image).save(image_dir / f"{idx}.jpg")

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {tmp_path.as_posix()}",
                "train: images/train",
                "val: images/train",
                "nc: 2",
                "names:",
                "  0: object",
                "  1: other",
            ]
        ),
        encoding="utf-8",
    )

    model = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    fp32_path = tmp_path / "LibreYOLO9t.onnx"
    int8_path = tmp_path / "LibreYOLO9t_int8.onnx"

    model.export(
        "onnx",
        output_path=str(fp32_path),
        imgsz=64,
        simplify=False,
        dynamic=False,
    )
    exported_int8 = model.export(
        "onnx",
        output_path=str(int8_path),
        imgsz=64,
        simplify=False,
        dynamic=False,
        int8=True,
        data=str(data_yaml),
    )

    assert exported_int8 == str(int8_path)
    assert int8_path.stat().st_size < fp32_path.stat().st_size

    proto = onnx.load(str(int8_path))
    metadata = {p.key: p.value for p in proto.metadata_props}
    assert metadata["model_family"] == "yolo9"
    assert metadata["task"] == "detect"
    assert metadata["precision"] == "int8"

    input_type = proto.graph.input[0].type.tensor_type.elem_type
    assert input_type == onnx.TensorProto.FLOAT

    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"images": np.zeros((1, 3, 64, 64), dtype=np.float32)})
    assert outs[0].shape == (1, 6, 84)

    loaded = LibreYOLO(str(int8_path), device="cpu")
    result = loaded.predict(np.zeros((64, 64, 3), dtype=np.uint8), conf=0.0, imgsz=64)
    assert result.boxes is not None
