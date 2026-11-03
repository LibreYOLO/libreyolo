"""Offline PaddleOCR-VL checks: layout post-processing, region preparation,
content clean-up, markdown, the OCRRegions extras and the predict contract."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from libreyolo import LibrePaddleOCRVL, LibreVLM
from libreyolo.models.paddleocrvl import blocks, layout
from libreyolo.models.paddleocrvl.markdown import build_markdown
from libreyolo.utils.results import OCRRegions, Results

pytestmark = pytest.mark.unit

LABEL = {name: i for i, name in enumerate(layout.LAYOUT_LABELS)}


def _row(label, score, box, order):
    return [LABEL[label], score, *box, order]


# =============================================================================
# Layout post-processing
# =============================================================================


def test_label_list_matches_release():
    assert len(layout.LAYOUT_LABELS) == 25
    assert layout.LAYOUT_LABELS[5] == "display_formula"
    assert layout.LAYOUT_LABELS[22] == "text"
    assert layout.LAYOUT_THRESHOLDS[LABEL["text"]] == 0.4
    assert layout.LAYOUT_THRESHOLDS[LABEL["table"]] == 0.5


def test_postprocess_applies_per_class_thresholds_and_reading_order():
    rows = np.array(
        [
            _row("text", 0.45, (10, 300, 200, 340), 2),  # text keeps at 0.4
            _row("table", 0.45, (10, 10, 200, 100), 0),  # table needs 0.5
            _row("doc_title", 0.9, (10, 120, 200, 160), 1),
        ]
    )
    boxes = layout.postprocess_layout(rows, (400, 400))
    assert [b["label"] for b in boxes] == ["doc_title", "text"]
    assert boxes[0]["coordinate"] == [10.0, 120.0, 200.0, 160.0]


def test_postprocess_nms_is_class_aware():
    rows = np.array(
        [
            _row("text", 0.9, (10, 10, 100, 50), 0),
            _row("text", 0.8, (12, 10, 100, 50), 1),  # same class, IoU > 0.6
            _row("table", 0.85, (12, 10, 100, 50), 2),  # other class, IoU < 0.98
        ]
    )
    labels = [b["label"] for b in layout.postprocess_layout(rows, (200, 200))]
    assert labels == ["text", "table"]
    no_nms = layout.postprocess_layout(rows, (200, 200), nms=False)
    assert len(no_nms) == 3


def test_postprocess_large_merge_drops_boxes_inside_titles():
    rows = np.array(
        [
            _row("paragraph_title", 0.9, (0, 0, 100, 40), 0),
            _row("text", 0.9, (5, 5, 95, 35), 1),  # inside the title box
            _row("text", 0.9, (0, 60, 100, 100), 2),
        ]
    )
    boxes = layout.postprocess_layout(rows, (200, 200), nms=False)
    assert [b["label"] for b in boxes] == ["paragraph_title", "text"]
    assert boxes[1]["coordinate"][1] == 60.0


def test_postprocess_drops_page_sized_image_and_clips():
    rows = np.array(
        [
            _row("image", 0.9, (-5, -5, 205, 305), 0),
            _row("text", 0.9, (10, 10, 250, 50), 1),
        ]
    )
    boxes = layout.postprocess_layout(rows, (200, 300))
    assert [b["label"] for b in boxes] == ["text"]
    assert boxes[0]["coordinate"] == [10.0, 10.0, 200.0, 50.0]
    assert layout.postprocess_layout(np.zeros((0, 7)), (10, 10)) == []


def test_order_sequence_ranks():
    q = 4
    assert layout.order_sequence(torch.full((1, q, q), 20.0)).tolist() == [[0, 1, 2, 3]]
    assert layout.order_sequence(torch.full((1, q, q), -20.0)).tolist() == [[3, 2, 1, 0]]


def test_raw_layout_boxes_decodes_cxcywh_to_pixels():
    q, c = 3, len(layout.LAYOUT_LABELS)
    logits = torch.full((1, q, c), -10.0)
    logits[0, 1, LABEL["table"]] = 5.0
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.5, 0.25, 1.0, 0.5], [0.1, 0.1, 0.1, 0.1]]])
    rows = layout.raw_layout_boxes(logits, boxes, torch.full((1, q, q), 20.0), (200, 100))
    assert rows.shape == (q, 7)
    top = rows[0]
    assert int(top[0]) == LABEL["table"]
    np.testing.assert_allclose(top[2:6], [0, 0, 200, 50], atol=1e-4)
    assert int(top[6]) == 1


def test_preprocess_layout_shape_and_range():
    rgb = np.full((50, 70, 3), 255, dtype=np.uint8)
    x = layout.preprocess_layout(rgb)
    assert x.shape == (1, 3, 800, 800)
    assert torch.allclose(x, torch.ones_like(x))


def test_filter_overlap_boxes():
    def box(label, xyxy):
        return {"cls_id": LABEL[label], "label": label, "score": 0.9, "coordinate": list(xyxy)}

    kept = layout.filter_overlap_boxes(
        [
            box("reference", (0, 0, 50, 50)),
            box("text", (0, 100, 100, 200)),
            box("text", (10, 110, 90, 190)),  # inside the first text: dropped
            box("image", (0, 300, 100, 400)),
            box("figure_title", (5, 305, 95, 395)),  # mixed with image: both kept
            box("text", (0, 500, 3, 600)),  # thinner than 6 px
            box("inline_formula", (0, 100, 60, 160)),  # > 50 percent overlap
        ]
    )
    assert [b["label"] for b in kept] == ["text", "image", "figure_title"]


# =============================================================================
# Region preparation and clean-up
# =============================================================================


@pytest.mark.parametrize(
    "label,chart,kind",
    [
        ("text", False, "ocr"),
        ("doc_title", False, "ocr"),
        ("table", False, "table"),
        ("display_formula", False, "formula"),
        ("inline_formula", False, "formula"),
        ("formula_number", False, "ocr"),
        ("chart", True, "chart"),
        ("chart", False, "ocr"),
    ],
)
def test_prompt_kind(label, chart, kind):
    assert blocks.prompt_kind(label, chart) == kind


def test_prepare_region_skips_pictures_and_trims_formulas():
    page = np.full((100, 200, 3), 255, dtype=np.uint8)
    page[40:60, 50:150] = 0
    image = {"label": "image", "coordinate": [0, 0, 100, 100]}
    assert blocks.prepare_region(page, image) is None
    assert blocks.prepare_region(page, {"label": "chart", "coordinate": [0, 0, 9, 9]}) is None
    crop, prompt = blocks.prepare_region(
        page, {"label": "display_formula", "coordinate": [0.0, 0.0, 200.7, 100.2]}
    )
    assert prompt == "Formula Recognition:"
    assert crop.shape == (20, 100, 3)
    crop, prompt = blocks.prepare_region(page, {"label": "text", "coordinate": [10.9, 5.5, 60.2, 30.9]})
    assert prompt == "OCR:" and crop.shape == (25, 50, 3)


def test_otsl_to_html_with_spans():
    otsl = "<fcel>A<lcel><fcel>B<nl><fcel>1<fcel>2<fcel>3<nl><ucel><fcel>x<ecel><nl>"
    html = blocks.otsl_to_html(otsl)
    assert html == (
        '<table><tr><td colspan="2">A</td><td>B</td></tr>'
        '<tr><td rowspan="2">1</td><td>2</td><td>3</td></tr>'
        "<tr><td>x</td><td></td></tr></table>"
    )
    assert blocks.otsl_to_html("") == ""
    assert blocks.otsl_to_html("<fcel>a&b<nl>") == "<table><tr><td>a&amp;b</td></tr></table>"


def test_clean_content():
    assert blocks.clean_content("<fcel>a<fcel>b<nl>", "table") == (
        "<table><tr><td>a</td><td>b</td></tr></table>"
    )
    assert blocks.clean_content("area \\(x^2\\) here", "text") == "area  $ x^2 $ here"
    assert blocks.clean_content("\\[E=mc^2\\]", "display_formula") == " $$ E=mc^2 $$ "
    assert blocks.clean_content("\\((1)\\)", "formula_number") == "  (1) "
    assert blocks.clean_content(None, "text") == ""


def test_truncate_repetitive_content():
    assert blocks.truncate_repetitive_content("short", min_count=1) == "short"
    assert blocks.truncate_repetitive_content("ab" * 40, min_count=10) == "ab"
    looped = "Intro text that is long enough. " * 2 + "the same loop! " * 20
    assert blocks.truncate_repetitive_content(looped, min_count=10).startswith("Intro")
    lines = "\n".join(["same line"] * 12)
    assert blocks.truncate_repetitive_content(lines, min_count=10) == "same line"
    assert blocks.truncate_repetitive_content("ab" * 40) == "ab" * 40  # under min_count


# =============================================================================
# Markdown
# =============================================================================


def test_build_markdown():
    md = build_markdown(
        [
            {"label": "header", "text": "Running header"},
            {"label": "doc_title", "text": "A Title"},
            {"label": "paragraph_title", "text": "2.1 Method."},
            {"label": "text", "text": "line one\nline two"},
            {"label": "display_formula", "text": "$$x$$"},
            {"label": "table", "text": "<table></table>"},
            {"label": "footer", "text": "page 3"},
            {"label": "image", "text": ""},
        ]
    )
    assert md == (
        "# A Title\n\n### 2.1 Method\n\nline one\n\nline two\n\n$$x$$\n\n\n\n<table></table>"
    )


# =============================================================================
# OCRRegions extras
# =============================================================================


def _quads(n):
    return torch.arange(n * 8, dtype=torch.float32).reshape(n, 4, 2)


def test_ocr_regions_labels_and_markdown_survive_moves_and_slices():
    regions = OCRRegions(_quads(3), ["a", "b", "c"], None, None, (10, 10), ["text", "table", "text"], "# md")
    assert regions.cpu().labels == ["text", "table", "text"]
    np_regions = regions.numpy()
    assert np_regions.markdown == "# md"
    assert regions[1].labels == ["table"]
    assert regions[[0, 2]].labels == ["text", "text"]
    assert regions[0:2].markdown == "# md"
    result = Results(boxes=None, orig_shape=(10, 10), ocr=OCRRegions(_quads(1), ["x"], labels=["text"], markdown="x"))
    assert result.ocr.labels == ["text"] and result.ocr.markdown == "x"
    assert result.summary()[0]["name"] == "text"


def test_ocr_regions_defaults_and_validation():
    plain = OCRRegions(_quads(2), ["a", "b"])
    assert plain.labels is None and plain.markdown == ""
    assert plain[0].labels is None
    with pytest.raises(ValueError, match="labels"):
        OCRRegions(_quads(2), ["a", "b"], labels=["text"])


# =============================================================================
# Predict contract (fake networks)
# =============================================================================


class _FakeLayout(nn.Module):
    """Emits fixed regions (normalized cxcywh) in query order."""

    def __init__(self, regions):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.regions = regions

    def forward(self, pixel_values):
        assert pixel_values.shape == (1, 3, 800, 800)
        q, c = 6, len(layout.LAYOUT_LABELS)
        logits = torch.full((1, q, c), -10.0)
        boxes = torch.full((1, q, 4), 0.01)
        for i, (label, box) in enumerate(self.regions):
            logits[0, i, LABEL[label]] = 5.0
            boxes[0, i] = torch.tensor(box)
        return SimpleNamespace(logits=logits, pred_boxes=boxes, order_logits=torch.full((1, q, q), 20.0))


class _FakeVL(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


REGIONS = [
    ("doc_title", (0.5, 0.05, 0.8, 0.06)),
    ("image", (0.5, 0.25, 0.8, 0.3)),
    ("text", (0.5, 0.5, 0.8, 0.2)),
    ("table", (0.5, 0.8, 0.8, 0.2)),
]
RESPONSES = {"OCR:": "Hello", "Table Recognition:": "<fcel>1<fcel>2<nl>"}


@pytest.fixture
def harness(monkeypatch, tmp_path):
    state = SimpleNamespace(prompts=[], sizes=[])

    def load_pretrained(self, snapshot_dir):
        self.layout_model = _FakeLayout(REGIONS)
        return _FakeVL(), SimpleNamespace()

    def recognize(self, image, prompt, **_):
        state.prompts.append(prompt)
        state.sizes.append(image.size)
        text = RESPONSES[prompt]
        if prompt == "OCR:" and image.size[1] < 30:
            text = "Title"
        return {"text": text, "confidence": 0.75, "input_ids": [], "token_ids": []}

    monkeypatch.setattr(LibrePaddleOCRVL, "_ensure_weights", lambda self: str(tmp_path))
    monkeypatch.setattr(LibrePaddleOCRVL, "_load_pretrained", load_pretrained)
    monkeypatch.setattr(LibrePaddleOCRVL, "recognize", recognize)
    return state


@pytest.mark.parametrize("alias", ["paddleocr-vl", "paddleocrvl", "paddleocr-vl-0.9b"])
def test_factory(alias, harness):
    model = LibreVLM(alias, device="cpu")
    assert isinstance(model, LibrePaddleOCRVL)
    assert model.size == "0.9b" and model.task == "ocr"
    assert model.names[LABEL["table"]] == "table"


def test_predict_returns_regions_in_reading_order(harness):
    model = LibreVLM("paddleocr-vl", device="cpu")
    result = model.predict(Image.new("RGB", (400, 400), "white"))
    ocr = result.ocr
    assert ocr.labels == ["doc_title", "text", "table"]  # the image region is skipped
    assert ocr.texts == ["Title", "Hello", "<table><tr><td>1</td><td>2</td></tr></table>"]
    np.testing.assert_allclose(np.asarray(ocr.conf), [0.75] * 3)
    assert (np.asarray(ocr.det_conf) > 0.99).all()
    np.testing.assert_allclose(np.asarray(ocr.xyxy)[1], [40, 160, 360, 240], atol=1e-3)
    assert ocr.markdown == (
        "# Title\n\nHello\n\n\n\n<table><tr><td>1</td><td>2</td></tr></table>"
    )
    assert harness.prompts == ["OCR:", "OCR:", "Table Recognition:"]
    assert result.boxes is None


def test_predict_without_layout_reads_the_whole_image(harness):
    model = LibreVLM("paddleocr-vl", device="cpu", use_layout_detection=False, prompt_label="table")
    result = model.predict(Image.new("RGB", (120, 80), "white"))
    assert result.ocr.labels == ["table"]
    assert harness.prompts == ["Table Recognition:"]
    assert harness.sizes == [(120, 80)]
    np.testing.assert_allclose(np.asarray(result.ocr.xyxy)[0], [0, 0, 120, 80])


def test_predict_saves_annotated_image(harness, tmp_path):
    model = LibreVLM("paddleocr-vl", device="cpu")
    model.predict(Image.new("RGB", (400, 400), "white"), save=True, output_path=str(tmp_path / "out"))
    assert any((tmp_path / "out").rglob("*.jpg"))


def test_contract_errors(harness):
    with pytest.raises(ValueError, match="prompt_label"):
        LibreVLM("paddleocr-vl", device="cpu", prompt_label="seal")
    model = LibreVLM("paddleocr-vl", device="cpu")
    with pytest.raises(ValueError, match="no vocabulary"):
        model.set_classes(["cat"])
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="x.yaml")
    with pytest.raises(NotImplementedError):
        model.export()
    with pytest.raises(ValueError):
        model.set_task("detect")


def test_val_runs_the_ocr_validator(harness, tmp_path):
    import json

    root = tmp_path / "ds"
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels").mkdir()
    Image.new("RGB", (400, 400), "white").save(root / "images" / "val" / "page.png")
    regions = [
        {"polygon": [[40, 160], [360, 160], [360, 240], [40, 240]], "text": "Hello"},
        {"polygon": [[40, 0], [360, 0], [360, 50], [40, 50]], "text": "Missing"},
    ]
    (root / "labels" / "val.jsonl").write_text(json.dumps({"image": "page.png", "regions": regions}) + "\n")
    model = LibreVLM("paddleocr-vl", device="cpu")
    metrics = model.val(data=str(root), verbose=False)
    assert metrics["metrics/det_recall"] == pytest.approx(0.5)
    assert metrics["metrics/e2e_recall"] == pytest.approx(0.5)
    assert metrics["metrics/e2e_precision"] == pytest.approx(1 / 3)


def test_vision_position_grid_matches_upstream():
    from libreyolo.models.paddleocrvl.model import _match_upstream_position_grid

    class _Emb(nn.Module):
        interpolation_align_corners = True

    net = nn.Sequential(nn.Linear(1, 1), _Emb(), nn.Sequential(_Emb()))
    assert _match_upstream_position_grid(net) == 2
    assert all(not m.interpolation_align_corners for m in net.modules() if isinstance(m, _Emb))
    assert _match_upstream_position_grid(nn.Linear(1, 1)) == 0
