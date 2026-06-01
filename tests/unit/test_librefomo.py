import torch
import pytest

from libreyolo import LibreFOMO, Points
from libreyolo.models.base.inference import InferenceRunner
from libreyolo.models.librefomo.utils import decode_points_from_logits, postprocess
from libreyolo.tasks import normalize_task, resolve_task
from libreyolo.utils.results import Results
from libreyolo.validation.point_validator import PointValidator
from libreyolo.utils.download import _detect_family_from_filename


pytestmark = pytest.mark.unit


def test_point_task_normalization():
    assert normalize_task("points") == "point"
    assert resolve_task(default_task="point", supported_tasks=("point",)) == "point"
    assert _detect_family_from_filename("LibreFOMOs.pt") == "librefomo"


def test_librefomo_forward_shapes():
    for size, imgsz, grid in (("s", 96, 12), ("m", 192, 24), ("l", 224, 28)):
        model = LibreFOMO(model_path=None, size=size, nb_classes=1, device="cpu")
        out = model._forward(torch.zeros(1, 3, imgsz, imgsz))
        assert out.shape == (1, 2, grid, grid)


def test_librefomo_decode_and_postprocess_points():
    logits = torch.zeros(1, 2, 4, 4)
    logits[0, 1, 2, 1] = 8.0
    decoded = decode_points_from_logits(logits, conf_threshold=0.5, nms_radius=1)
    assert decoded[0].shape == (1, 4)
    assert decoded[0][0, :3].tolist() == [1.0, 2.0, 1.0]

    det = postprocess(logits, conf_thres=0.5, input_size=32, original_size=(80, 40))
    assert det["num_detections"] == 1
    assert torch.allclose(det["points"][0], torch.tensor([30.0, 25.0]))
    assert det["classes"][0].item() == 0.0


def test_points_results_summary_and_len():
    points = Points(torch.tensor([[10.0, 20.0, 0.0, 0.9]]), orig_shape=(40, 80))
    result = Results(boxes=None, points=points, orig_shape=(40, 80), names={0: "object"})
    assert len(result) == 1
    assert result.points.xyn.tolist() == [[0.125, 0.5]]
    assert result.summary()[0] == {
        "name": "object",
        "class": 0,
        "confidence": 0.9,
        "point": {"x": 10.0, "y": 20.0},
    }


def test_inference_runner_wraps_empty_points():
    model = LibreFOMO(model_path=None, size="s", nb_classes=1, device="cpu")
    runner = InferenceRunner(model)
    result = runner._wrap_results(
        {
            "points": torch.zeros((0, 2)),
            "scores": torch.zeros((0,)),
            "classes": torch.zeros((0,)),
            "num_detections": 0,
        },
        original_size=(80, 40),
        image_path=None,
        classes=None,
    )
    assert result.boxes is None
    assert result.points is not None
    assert len(result) == 0
    assert result.summary() == []


def test_point_validator_metrics_match_centers():
    validator = PointValidator.__new__(PointValidator)
    validator.config = type(
        "Cfg",
        (),
        {
            "imgsz": 32,
            "conf_thres": 0.5,
            "max_det": 300,
            "point_distance_tolerance": 1.5,
            "point_nms_radius": 1,
        },
    )()
    validator.distance_tolerance = 1.5
    validator.nms_radius = 1
    validator.model = LibreFOMO(model_path=None, size="s", nb_classes=1, device="cpu")
    validator._last_metric_shape = (4, 4)
    validator._init_metrics()

    preds = [torch.tensor([[1.0, 2.0, 0.0, 0.99]])]
    targets = torch.zeros(1, 120, 5)
    targets[0, 0] = torch.tensor([4.0, 12.0, 12.0, 20.0, 0.0])

    validator._update_metrics(preds, targets, None)
    metrics = validator._compute_metrics()
    assert metrics["metrics/precision"] == 1.0
    assert metrics["metrics/recall"] == 1.0
    assert metrics["metrics/F1"] == 1.0
    assert metrics["metrics/mean_distance"] == 0.0
