"""Prepare resources for the iOS CoreML smoke test.

This exports a raw and embedded-NMS YOLOX nano CoreML package, plus a canonical
416x416 RGB test image that matches LibreYOLO's YOLOX preprocessing.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RESOURCES = Path(__file__).resolve().parent / "Tests" / "CoreMLSmokeTests" / "Resources"


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _raw_top_score(output: np.ndarray) -> float:
    output = np.asarray(output)
    if output.ndim == 3:
        output = output[0]
    objectness = output[:, 4:5]
    class_scores = output[:, 5:]
    return float(np.max(objectness * class_scores))


def _top_value(output: np.ndarray) -> float:
    return float(np.max(np.asarray(output)))


def main() -> None:
    import coremltools as ct

    from libreyolo import SAMPLE_IMAGE, LibreYOLO
    from libreyolo.models.yolox.utils import preprocess_image

    RESOURCES.mkdir(parents=True, exist_ok=True)

    raw_model = RESOURCES / "LibreYOLOXnRaw.mlpackage"
    nms_model = RESOURCES / "LibreYOLOXnNMS.mlpackage"
    image_path = RESOURCES / "parkour_yolox416.png"
    expected_path = RESOURCES / "expected.json"

    for path in (raw_model, nms_model, image_path, expected_path):
        _remove(path)

    model = LibreYOLO(str(ROOT / "weights" / "LibreYOLOXn.pt"))
    imgsz = model._get_input_size()

    model.export(format="coreml", output_path=str(raw_model))
    model.export(format="coreml", output_path=str(nms_model), nms=True)

    tensor, _, _, _ = preprocess_image(SAMPLE_IMAGE, input_size=imgsz)
    chw_bgr = tensor.numpy()[0]
    rgb = np.transpose(chw_bgr, (1, 2, 0))[..., ::-1]
    canonical = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    Image.fromarray(canonical).save(image_path)

    image = Image.open(image_path)
    raw_prediction = ct.models.MLModel(str(raw_model)).predict({"image": image})
    nms_prediction = ct.models.MLModel(str(nms_model)).predict({"image": image})

    raw_output = next(iter(raw_prediction.values()))
    expected = {
        "rawTopScore": _raw_top_score(raw_output),
        "nmsTopScore": _top_value(nms_prediction["confidence"]),
    }
    expected_path.write_text(json.dumps(expected, indent=2) + "\n")
    print(json.dumps(expected, indent=2))


if __name__ == "__main__":
    main()
