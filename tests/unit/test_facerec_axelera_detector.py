"""Face detector on the Axelera Metis (raw YuNet head maps + host decode) and the face pipeline's auto-batching."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tests.unit.test_axelera_backend import _install_fake_sdk

pytestmark = pytest.mark.unit


def _tiny_detector_onnx(path, size=64):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.RandomState(0)
    nodes, inits, outs = [], [], []
    channels = {"cls": 1, "obj": 1, "bbox": 4, "kps": 10}
    for k, c in channels.items():
        for s in (8, 16, 32):
            name = f"raw_{k}_{s}"
            w = numpy_helper.from_array((rng.randn(c, 3, s, s) * 0.01).astype(np.float32), name=f"w_{name}")
            inits.append(w)
            nodes.append(helper.make_node("Conv", ["input", w.name], [name], kernel_shape=[s, s], strides=[s, s]))
            outs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, c, size // s, size // s]))
    graph = helper.make_graph(nodes, "tiny_yunet", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, size, size])], outs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return {f"{k}_{s}": f"raw_{k}_{s}" for k in channels for s in (8, 16, 32)}


def test_decode_places_a_face_at_its_grid_cell():
    pytest.importorskip("cv2")
    from libreyolo.models.facerec.axelera_detector import decode_yunet_raw

    size = 64
    maps = {}
    for s in (8, 16, 32):
        n = size // s
        maps[f"cls_{s}"] = np.full((1, 1, n, n), -20.0, np.float32)
        maps[f"obj_{s}"] = np.full((1, 1, n, n), -20.0, np.float32)
        maps[f"bbox_{s}"] = np.zeros((1, 4, n, n), np.float32)
        maps[f"kps_{s}"] = np.zeros((1, 10, n, n), np.float32)
    maps["cls_8"][0, 0, 2, 3] = maps["obj_8"][0, 0, 2, 3] = 10.0  # row 2, col 3 at stride 8
    boxes, marks, scores = decode_yunet_raw(maps, size)
    assert len(boxes) == 1 and scores[0] > 0.99
    np.testing.assert_allclose(boxes[0], [3 * 8 - 4, 2 * 8 - 4, 8, 8], atol=1e-5)  # centre (24, 16), w = h = exp(0) * 8
    np.testing.assert_allclose(marks[0][0], [24, 16], atol=1e-5)


def test_axelera_face_detector_exports_and_batches_like_single_calls(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    import libreyolo.models.facerec.axelera_detector as ad
    from tests.unit.test_facerec import _face_dataset

    sdk = _install_fake_sdk(monkeypatch)
    names = _tiny_detector_onnx(tmp_path / "det.onnx")
    monkeypatch.setattr(ad, "_YUNET_RAW_TENSORS", names)
    out = ad.export_face_detector_axelera(tmp_path / "librefacerec-det_axelera_model", detector_onnx=tmp_path / "det.onnx",
                                          data=str(_face_dataset(tmp_path)), canvas=64)
    det = ad.AxeleraFaceDetector(out, score_threshold=0.0)
    assert det.size == 64 and sorted(det.output_names) == sorted(ad.YUNET_RAW_NAMES)
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, size=(48 + 8 * i, 64, 3), dtype=np.uint8) for i in range(3)]
    single = [det(image) for image in images]
    batched = det.detect_batch(images)
    for a, b in zip(single, batched):
        assert len(a) == len(b)
        for fa, fb in zip(a, b):
            assert fa.xyxy == fb.xyxy and fa.score == fb.score
            np.testing.assert_array_equal(fa.landmarks, fb.landmarks)


def test_face_pipeline_batches_lists_eight_at_a_time_on_axelera(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    import libreyolo.models.facerec.axelera_detector as ad
    from libreyolo.models.facerec import LibreFaceEmbedder
    from tests.unit.test_facerec import _build_tiny_face_onnx, _face_dataset

    sdk = _install_fake_sdk(monkeypatch)
    names = _tiny_detector_onnx(tmp_path / "det.onnx")
    monkeypatch.setattr(ad, "_YUNET_RAW_TENSORS", names)
    out = ad.export_face_detector_axelera(tmp_path / "librefacerec-det_axelera_model", detector_onnx=tmp_path / "det.onnx",
                                          data=str(_face_dataset(tmp_path)), canvas=64)
    head = tmp_path / "head.onnx"
    _build_tiny_face_onnx(str(head), dim=8)
    model = LibreFaceEmbedder(str(head), device="cpu", face_detector=str(out))
    det = model.face_detector
    assert isinstance(det, ad.AxeleraFaceDetector) and det.default_predict_batch == 8
    det.score_threshold = 0.0
    loaded = det._model
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    arrays = [rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8) for _ in range(10)]
    images = [Image.fromarray(a) for a in arrays]
    folder = tmp_path / "images"
    folder.mkdir()
    for index, image in enumerate(images):
        image.save(folder / f"{index:02d}.png")
    video = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    for array in arrays:
        writer.write(array)
    writer.release()

    for source in (images, str(folder), video):
        loaded.batches.clear()
        batched = list(model(source, face_conf=0.0))
        assert loaded.batches == [8, 2], source if isinstance(source, str) else "list"
        loaded.batches.clear()
        single = list(model(source, face_conf=0.0, batch=1))
        assert loaded.batches == []  # batch=1 opts out: image by image
        assert len(batched) == len(single) == 10
        for a, b in zip(batched, single):
            assert a.path == b.path and getattr(a, "frame_idx", None) == getattr(b, "frame_idx", None)
            np.testing.assert_array_equal(np.asarray(a.boxes.xyxy), np.asarray(b.boxes.xyxy))
            np.testing.assert_array_equal(np.asarray(a.embeddings.data), np.asarray(b.embeddings.data))
    with pytest.raises(ValueError, match="batch must be >= 1"):
        model(images, batch=0)
    from libreyolo import LibreYOLO

    with pytest.raises(ValueError, match="face_detector="):
        LibreYOLO(str(out))  # the detector is not a recognition head
