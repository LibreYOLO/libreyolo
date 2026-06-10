"""Unit tests for LibreEnsemble and ExternalDetector with stub members."""

import subprocess
import sys

import pytest
import torch
from PIL import Image

from libreyolo.ensemble import ExternalDetector, LibreEnsemble
from libreyolo.utils.results import Boxes, Results

pytestmark = pytest.mark.unit

COCOISH = {0: "person", 1: "car"}


class StubMember:
    """Canned-detections member that records every call it receives."""

    def __init__(self, names, boxes=(), scores=(), cls=(), task="detect"):
        self.names = names
        self.task = task
        self.boxes = torch.tensor(list(boxes), dtype=torch.float32).reshape(-1, 4)
        self.scores = torch.tensor(list(scores), dtype=torch.float32)
        self.cls = torch.tensor(list(cls), dtype=torch.float32)
        self.calls = []

    def __call__(self, source, **kwargs):
        self.calls.append({"source": source, **kwargs})
        w, h = source.size
        keep = self.scores > kwargs.get("conf", 0.25)
        return Results(
            boxes=Boxes(self.boxes[keep], self.scores[keep], self.cls[keep]),
            orig_shape=(h, w),
            names=self.names,
        )

    predict = __call__


@pytest.fixture
def image():
    return Image.new("RGB", (100, 80), (40, 90, 160))


def _pair_members():
    a = StubMember(COCOISH, boxes=[[0, 0, 10, 10]], scores=[0.9], cls=[0])
    b = StubMember(COCOISH, boxes=[[0, 0, 10, 12]], scores=[0.7], cls=[0])
    return a, b


class TestConstruction:
    def test_requires_two_members(self):
        with pytest.raises(ValueError, match="at least two members"):
            LibreEnsemble([StubMember(COCOISH)])

    def test_rejects_string_members_argument(self):
        with pytest.raises(TypeError, match="sequence"):
            LibreEnsemble("LibreYOLO9s.pt")

    def test_rejects_non_detect_member(self):
        seg = StubMember(COCOISH, task="segment")
        with pytest.raises(ValueError, match="detect members only"):
            LibreEnsemble([StubMember(COCOISH), seg])

    def test_rejects_bare_callable(self):
        with pytest.raises(TypeError, match="ExternalDetector"):
            LibreEnsemble([StubMember(COCOISH), lambda img: ([], [], [])])

    def test_rejects_unknown_fusion(self):
        with pytest.raises(ValueError, match="unknown fusion"):
            LibreEnsemble(list(_pair_members()), fusion="magic")

    def test_rejects_bad_min_votes(self):
        members = list(_pair_members())
        with pytest.raises(ValueError, match="positive int"):
            LibreEnsemble(members, min_votes=0)
        with pytest.raises(ValueError, match="never be met"):
            LibreEnsemble(members, min_votes=3)

    def test_rejects_nms_with_votes(self):
        with pytest.raises(ValueError, match="cannot count votes"):
            LibreEnsemble(list(_pair_members()), fusion="nms", min_votes=2)

    def test_rejects_bad_weights(self):
        members = list(_pair_members())
        with pytest.raises(ValueError, match="2 members"):
            LibreEnsemble(members, weights=[1.0])
        with pytest.raises(ValueError, match="positive"):
            LibreEnsemble(members, weights=[1.0, 0.0])

    def test_repr_mentions_fusion(self):
        ens = LibreEnsemble(list(_pair_members()), fusion="wbf_seeded")
        assert "wbf_seeded" in repr(ens)


class TestNamesUnion:
    def test_identical_names_pass_through(self):
        ens = LibreEnsemble(list(_pair_members()))
        assert ens.names == COCOISH

    def test_union_by_name_with_remap(self, image):
        # Member B calls "person" class 1; both members' person boxes must fuse.
        a = StubMember({0: "person"}, boxes=[[0, 0, 10, 10]], scores=[0.9], cls=[0])
        b = StubMember(
            {0: "car", 1: "person"}, boxes=[[0, 0, 10, 12]], scores=[0.7], cls=[1]
        )
        ens = LibreEnsemble([a, b])
        assert ens.names == {0: "person", 1: "car"}
        result = ens(image)
        assert len(result) == 1
        assert result.boxes.cls.tolist() == [0.0]
        assert result.names[0] == "person"

    def test_partial_overlap_logs_warning(self, caplog):
        a = StubMember({0: "person"})
        b = StubMember({0: "car", 1: "person"})
        with caplog.at_level("WARNING", logger="libreyolo.ensemble.model"):
            LibreEnsemble([a, b])
        assert any("label spaces differ" in r.message for r in caplog.records)

    def test_vote_cap_for_partially_known_class(self, image):
        # "car" is only known to member B; min_votes=2 must keep it while
        # dropping the solo person box that both members could have confirmed.
        a = StubMember({0: "person"}, boxes=[[0, 0, 10, 10]], scores=[0.9], cls=[0])
        b = StubMember(
            {0: "car", 1: "person"}, boxes=[[50, 50, 60, 60]], scores=[0.8], cls=[0]
        )
        ens = LibreEnsemble([a, b], min_votes=2)
        result = ens(image)
        assert len(result) == 1
        assert result.names[int(result.boxes.cls[0])] == "car"


class TestPredict:
    def test_two_member_fusion(self, image):
        a, b = _pair_members()
        result = ens_result = LibreEnsemble([a, b])(image)
        assert isinstance(ens_result, Results)
        assert len(result) == 1
        assert torch.allclose(
            result.boxes.xyxy, torch.tensor([[0.0, 0.0, 10.0, 10.875]]), atol=1e-4
        )
        assert torch.allclose(result.boxes.conf, torch.tensor([0.8]), atol=1e-5)
        assert result.orig_shape == (80, 100)
        assert result.path is None

    def test_members_receive_same_pil_image(self, image):
        a, b = _pair_members()
        LibreEnsemble([a, b])(image)
        assert isinstance(a.calls[0]["source"], Image.Image)
        assert a.calls[0]["source"] is b.calls[0]["source"]

    def test_per_member_conf_and_broadcast_iou(self, image):
        a, b = _pair_members()
        LibreEnsemble([a, b])(image, conf=[0.3, 0.6], iou=0.5)
        assert a.calls[0]["conf"] == 0.3 and b.calls[0]["conf"] == 0.6
        assert a.calls[0]["iou"] == 0.5 and b.calls[0]["iou"] == 0.5

    def test_per_member_imgsz_list_and_shared_tuple(self, image):
        a, b = _pair_members()
        ens = LibreEnsemble([a, b])
        ens(image, imgsz=[640, 320])
        assert a.calls[0]["imgsz"] == 640 and b.calls[0]["imgsz"] == 320
        ens(image, imgsz=(640, 640))
        assert a.calls[1]["imgsz"] == (640, 640) and b.calls[1]["imgsz"] == (640, 640)

    def test_wrong_sequence_length_raises(self, image):
        ens = LibreEnsemble(list(_pair_members()))
        with pytest.raises(ValueError, match="conf has 3 entries"):
            ens(image, conf=[0.1, 0.2, 0.3])

    def test_members_run_generously_ensemble_trims(self, image):
        boxes = [[i * 20.0, 0.0, i * 20.0 + 10.0, 10.0] for i in range(8)]
        scores = [0.9 - 0.05 * i for i in range(8)]
        a = StubMember(COCOISH, boxes=boxes, scores=scores, cls=[0] * 8)
        b = StubMember(COCOISH)
        result = LibreEnsemble([a, b])(image, max_det=3)
        assert a.calls[0]["max_det"] == 300
        assert len(result) == 3
        assert torch.all(result.boxes.conf[:-1] >= result.boxes.conf[1:])

    def test_classes_filter_uses_union_ids(self, image):
        a = StubMember(
            COCOISH, boxes=[[0, 0, 10, 10], [50, 50, 60, 60]],
            scores=[0.9, 0.8], cls=[0, 1],
        )
        b = StubMember(COCOISH)
        result = LibreEnsemble([a, b])(image, classes=[1])
        assert result.boxes.cls.tolist() == [1.0]

    def test_min_votes_consensus(self, image):
        a = StubMember(
            COCOISH, boxes=[[0, 0, 10, 10], [50, 50, 60, 60]],
            scores=[0.9, 0.95], cls=[0, 0],
        )
        b = StubMember(COCOISH, boxes=[[0, 0, 10, 12]], scores=[0.7], cls=[0])
        result = LibreEnsemble([a, b], min_votes=2)(image)
        assert len(result) == 1
        assert torch.allclose(
            result.boxes.xyxy[0, 3], torch.tensor(10.875), atol=1e-4
        )

    def test_speed_keys(self, image):
        result = LibreEnsemble(list(_pair_members()))(image)
        assert set(result.speed) == {"member_0", "member_1", "fusion"}
        assert all(v >= 0 for v in result.speed.values())

    def test_all_members_empty(self, image):
        result = LibreEnsemble([StubMember(COCOISH), StubMember(COCOISH)])(image)
        assert len(result) == 0
        assert result.boxes.xyxy.shape == (0, 4)
        assert result.names == COCOISH

    def test_one_empty_member_halves_solo_score(self, image):
        a = StubMember(COCOISH, boxes=[[0, 0, 10, 10]], scores=[0.9], cls=[0])
        result = LibreEnsemble([a, StubMember(COCOISH)])(image)
        assert torch.allclose(result.boxes.conf, torch.tensor([0.45]), atol=1e-5)

    def test_nonfinite_member_rows_dropped(self, image):
        a = StubMember(
            COCOISH, boxes=[[0, 0, 10, 10], [0, 0, float("nan"), 10]],
            scores=[0.9, 0.8], cls=[0, 0],
        )
        result = LibreEnsemble([a, StubMember(COCOISH)])(image)
        assert len(result) == 1
        assert torch.isfinite(result.boxes.xyxy).all()

    def test_member_class_id_outside_names_raises(self, image):
        a = StubMember(COCOISH, boxes=[[0, 0, 10, 10]], scores=[0.9], cls=[7])
        with pytest.raises(RuntimeError, match="outside its names"):
            LibreEnsemble([a, StubMember(COCOISH)])(image)

    def test_custom_fusion_callable(self, image):
        seen = {}

        def first_only(boxes, scores, labels, model_ids, *, weights, num_models, **kw):
            seen["num_models"] = num_models
            seen["weights"] = weights
            seen["model_ids"] = model_ids
            keep = model_ids == 0
            return boxes[keep], scores[keep], labels[keep]

        a, b = _pair_members()
        result = LibreEnsemble([a, b], fusion=first_only)(image)
        assert seen["num_models"] == 2
        assert torch.allclose(seen["weights"], torch.ones(2))
        assert set(seen["model_ids"].tolist()) == {0, 1}
        assert len(result) == 1
        assert torch.allclose(result.boxes.conf, torch.tensor([0.9]))

    def test_custom_fusion_bad_return_raises(self, image):
        ens = LibreEnsemble(list(_pair_members()), fusion=lambda *a, **k: "nope")
        with pytest.raises(TypeError, match="fusion must return"):
            ens(image)

    def test_video_raises(self):
        ens = LibreEnsemble(list(_pair_members()))
        with pytest.raises(NotImplementedError, match="video"):
            ens("clip.mp4")

    def test_stream_raises(self, image):
        ens = LibreEnsemble(list(_pair_members()))
        with pytest.raises(NotImplementedError):
            ens(image, stream=True)

    def test_unknown_predict_kwarg_raises(self, image):
        ens = LibreEnsemble(list(_pair_members()))
        with pytest.raises(TypeError, match="Unsupported predict option"):
            ens(image, bogus_option=1)

    def test_directory_input(self, tmp_path, image):
        image.save(tmp_path / "one.jpg")
        image.save(tmp_path / "two.jpg")
        results = LibreEnsemble(list(_pair_members()))(str(tmp_path))
        assert isinstance(results, list) and len(results) == 2
        assert all(isinstance(r, Results) for r in results)
        assert results[0].path.endswith("one.jpg")

    def test_save_writes_annotated_image(self, tmp_path, image):
        src = tmp_path / "in.jpg"
        image.save(src)
        out_dir = tmp_path / "out"
        result = LibreEnsemble(list(_pair_members()))(
            str(src), save=True, output_path=str(out_dir)
        )
        from pathlib import Path

        assert Path(result.saved_path).exists()

    def test_val_and_export_raise(self):
        ens = LibreEnsemble(list(_pair_members()))
        with pytest.raises(NotImplementedError):
            ens.val()
        with pytest.raises(NotImplementedError):
            ens.export()


class TestExternalDetector:
    def test_wraps_plain_lists(self, image):
        det = ExternalDetector(
            lambda img: ([[0, 0, 10, 10]], [0.9], [0]), names={0: "person"}
        )
        result = det(image)
        assert len(result) == 1
        assert result.names == {0: "person"}
        assert result.orig_shape == (80, 100)

    def test_applies_conf(self, image):
        det = ExternalDetector(
            lambda img: ([[0, 0, 10, 10]], [0.2], [0]), names={0: "person"}
        )
        assert len(det(image, conf=0.25)) == 0
        assert len(det(image, conf=0.1)) == 1

    def test_rejects_non_callable(self):
        with pytest.raises(TypeError, match="callable"):
            ExternalDetector("not-a-function", names={0: "person"})

    def test_rejects_bad_names(self):
        with pytest.raises(TypeError, match="names"):
            ExternalDetector(lambda img: ([], [], []), names={})

    def test_rejects_wrong_return_arity(self, image):
        det = ExternalDetector(lambda img: ([], []), names={0: "person"})
        with pytest.raises(TypeError, match="boxes, scores, labels"):
            det(image)

    def test_rejects_bad_box_shape(self, image):
        det = ExternalDetector(
            lambda img: ([[0, 0, 10]], [0.9], [0]), names={0: "person"}
        )
        with pytest.raises(ValueError, match=r"\(N, 4\)"):
            det(image)

    def test_rejects_length_mismatch(self, image):
        det = ExternalDetector(
            lambda img: ([[0, 0, 10, 10]], [0.9, 0.8], [0]), names={0: "person"}
        )
        with pytest.raises(ValueError, match="equal"):
            det(image)

    def test_rejects_unknown_class_id(self, image):
        det = ExternalDetector(
            lambda img: ([[0, 0, 10, 10]], [0.9], [4]), names={0: "person"}
        )
        with pytest.raises(ValueError, match="class ids"):
            det(image)

    def test_as_ensemble_member(self, image):
        external = ExternalDetector(
            lambda img: ([[0, 0, 10, 12]], [0.7], [0]), names={0: "person"}
        )
        a = StubMember({0: "person"}, boxes=[[0, 0, 10, 10]], scores=[0.9], cls=[0])
        result = LibreEnsemble([a, external], min_votes=2)(image)
        assert len(result) == 1
        assert torch.allclose(result.boxes.conf, torch.tensor([0.8]), atol=1e-5)


class TestLazyImport:
    def test_package_import_does_not_load_ensemble(self):
        # The 99.9% of sessions that never touch ensembling must not pay for it.
        code = (
            "import sys, libreyolo; "
            "mods = [m for m in sys.modules if m.startswith('libreyolo') "
            "and ('ensemble' in m or '.ops' in m)]; "
            "assert not mods, f'eagerly imported: {mods}'; "
            "print('clean')"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr
        assert "clean" in out.stdout

    def test_lazy_attribute_resolves(self):
        import libreyolo

        assert libreyolo.LibreEnsemble is LibreEnsemble
        assert libreyolo.ExternalDetector is ExternalDetector
        assert "LibreEnsemble" in libreyolo.__all__
