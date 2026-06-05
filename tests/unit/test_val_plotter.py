from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.unit


def test_per_class_recall_averages_iou_thresholds(monkeypatch, tmp_path):
    from libreyolo.validation.val_plotter import ValPlotter

    captured = {}

    class _Spine:
        def set_visible(self, value):
            pass

    class _Bar:
        def get_y(self):
            return 0.0

        def get_height(self):
            return 1.0

    class _Ax:
        spines = {"top": _Spine(), "right": _Spine()}

        def barh(self, _positions, values, **_kwargs):
            captured["values"] = list(values)
            return [_Bar() for _ in values]

        def set_yticks(self, *_args, **_kwargs):
            pass

        def set_yticklabels(self, labels, **_kwargs):
            captured["labels"] = list(labels)

        def set_xlim(self, *_args, **_kwargs):
            pass

        def set_title(self, title, **_kwargs):
            captured["title"] = title

        def set_xlabel(self, label, **_kwargs):
            captured["xlabel"] = label

        def grid(self, *_args, **_kwargs):
            pass

        def invert_yaxis(self):
            pass

        def text(self, *_args, **_kwargs):
            pass

    class _Fig:
        def tight_layout(self):
            pass

        def savefig(self, *_args, **_kwargs):
            pass

    class _Plot:
        def get_cmap(self, _name):
            return lambda value: value

        def subplots(self, **_kwargs):
            return _Fig(), _Ax()

        def close(self, _fig):
            pass

    class _Params:
        catIds = [1, 2]

    class _Eval:
        params = _Params()
        eval = {
            "recall": np.array(
                [
                    [[[0.2]], [[-1.0]]],
                    [[[0.6]], [[-1.0]]],
                    [[[-1.0]], [[-1.0]]],
                ],
                dtype=np.float32,
            )
        }

    monkeypatch.setattr(ValPlotter, "_require_matplotlib", staticmethod(lambda: _Plot()))

    ValPlotter.plot_per_class_recall(
        _Eval(), ["person", "empty"], tmp_path / "recall.png", "Box"
    )

    assert captured["values"] == pytest.approx([0.4])
    assert "Recall@50-95" in captured["title"]
    assert captured["xlabel"] == "Recall@50-95"


def test_per_class_ap_skips_absent_classes(monkeypatch, tmp_path):
    from libreyolo.validation.val_plotter import ValPlotter

    captured = {}

    class _Spine:
        def set_visible(self, value):
            pass

    class _Bar:
        def get_y(self):
            return 0.0

        def get_height(self):
            return 1.0

    class _Ax:
        spines = {"top": _Spine(), "right": _Spine()}

        def barh(self, _positions, values, **_kwargs):
            captured["values"] = list(values)
            return [_Bar() for _ in values]

        def set_yticks(self, *_args, **_kwargs):
            pass

        def set_yticklabels(self, labels, **_kwargs):
            captured["labels"] = list(labels)

        def set_xlim(self, *_args, **_kwargs):
            pass

        def set_title(self, *_args, **_kwargs):
            pass

        def set_xlabel(self, *_args, **_kwargs):
            pass

        def grid(self, *_args, **_kwargs):
            pass

        def invert_yaxis(self):
            pass

        def text(self, *_args, **_kwargs):
            pass

    class _Fig:
        def tight_layout(self):
            pass

        def savefig(self, *_args, **_kwargs):
            pass

    class _Plot:
        def get_cmap(self, _name):
            return lambda value: value

        def subplots(self, **_kwargs):
            return _Fig(), _Ax()

        def close(self, _fig):
            pass

    class _Params:
        catIds = [90, 91]

    class _Coco:
        def loadCats(self, cat_ids):
            names = {90: "toothbrush", 91: "empty"}
            return [{"id": cat_id, "name": names[cat_id]} for cat_id in cat_ids]

    class _Eval:
        params = _Params()
        cocoGt = _Coco()
        eval = {"precision": np.full((2, 2, 2, 1, 1), -1.0, dtype=np.float32)}

    _Eval.eval["precision"][:, :, 0, 0, 0] = np.array(
        [[0.1, 0.3], [0.5, 0.7]], dtype=np.float32
    )

    monkeypatch.setattr(ValPlotter, "_require_matplotlib", staticmethod(lambda: _Plot()))

    ValPlotter.plot_per_class_ap(_Eval(), ["wrong", "empty"], tmp_path / "ap.png")

    assert captured["values"] == pytest.approx([0.4])
    assert captured["labels"] == ["toothbrush"]


def test_val_sample_draws_pose_keypoints(monkeypatch, tmp_path):
    import cv2

    from libreyolo.validation.val_plotter import ValPlotter

    captured_edges = []

    def _fake_draw_keypoints(img, _keypoints, **kwargs):
        captured_edges.append(kwargs["edges"])
        return img

    monkeypatch.setattr("libreyolo.utils.drawing.draw_keypoints", _fake_draw_keypoints)

    img = np.zeros((48, 48, 3), dtype=np.uint8)
    out = tmp_path / "sample.jpg"

    ValPlotter.plot_val_sample(
        img,
        np.zeros((0, 4), dtype=np.float32),
        np.zeros(0, dtype=int),
        np.zeros((0, 4), dtype=np.float32),
        np.zeros(0, dtype=int),
        np.zeros(0, dtype=np.float32),
        None,
        out,
        gt_keypoints=np.array([[[10.0, 10.0, 2.0], [20.0, 20.0, 2.0]]]),
        pred_keypoints=np.array([[[12.0, 12.0, 0.9], [22.0, 22.0, 0.9]]]),
        keypoint_edges=((0, 1),),
    )

    rendered = cv2.imread(str(Path(out)))
    assert rendered is not None
    assert rendered.sum() > 0
    assert captured_edges == [((0, 1),), ((0, 1),)]
