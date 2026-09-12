"""Offline Molmo2 grammar, generation and public point-contract checks."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreMolmo2, LibreVLM
from libreyolo.models.vlm import molmo2
from libreyolo.models.vlm.parsing import extract_molmo_points

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "text, expected",
    [
        ('<point x="25" y="75" alt="object">object</point>', [[0.25, 0.75]]),
        ("<point y='0' x='100'/>", [[1.0, 0.0]]),
        (
            '<points x2="90" y2="10" x1="20" y1="30">objects</points>',
            [[0.2, 0.3], [0.9, 0.1]],
        ),
        (
            'Counting: <points coords="1 1 250 750 2 010 099">objects</points>.',
            [[0.25, 0.75], [0.01, 0.099]],
        ),
        ('<points coords="1 1 000 1000"/>', [[0.0, 1.0]]),
        (
            '<points coords="1 1 010 010"/> <point x="10" y="10"/>',
            [[0.01, 0.01], [0.1, 0.1]],
        ),
        ('<points x1="10" y1="20" x2="30"/>', [[0.1, 0.2]]),
    ],
)
def test_point_grammars(text, expected):
    items = extract_molmo_points(text, "Requested Class")
    np.testing.assert_allclose([item["point"] for item in items], expected)
    assert all(item["label"] == "Requested Class" for item in items)


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "There are none.",
        "(25, 75)",
        '<point x="25"/>',
        '<point x="NaN" y="10"/>',
        '<point x="inf" y="10"/>',
        '<point x="-1" y="10"/>',
        '<point x="101" y="10"/>',
        '<point x="1" x="2" y="3"/>',
        '<point x="25" y="75">truncated',
        '<point x="25" y="75"',
        '<points coords="1 1 500"/>',
        '<points coords="1 1 1001 500"/>',
        '<points coords="2 1 500 500"/>',
        '<points coords="0.5 1 500 500"/>',
        '<points coords="1 1 500 500;2 2 700 700"/>',
        '<points coords="1 1 500 500\t2 2 700 700"/>',
        '<points coords="1 1 100 200\t2 2 300 400\t3 3 500 600\t4 4 700 800"/>',
        '<tracks coords="1 1 500 500"/>',
    ],
)
def test_malformed_or_other_canvas_outputs_are_empty(text):
    assert extract_molmo_points(text, "object") == []


class _Batch(dict):
    def to(self, device):
        return _Batch({k: v.to(device) for k, v in self.items()})


@pytest.fixture
def loaded(monkeypatch):
    prompts, generations = [], []

    class Processor:
        def apply_chat_template(self, conversation, **kwargs):
            prompts.append(conversation[0]["content"][1]["text"])
            return _Batch(
                input_ids=torch.tensor([[7, 8]]),
                token_type_ids=torch.tensor([[True, False]]),
                pixel_values=torch.ones((1, 3, 2, 2)),
            )

        def batch_decode(self, tokens, **kwargs):
            assert tokens.tolist() == [[42]]  # Prompt stripped before decoding.
            return ['<points coords="1 1 250 750 2 250 750">paraphrase</points>']

    def init_model(self):
        self.processor = Processor()
        self._model_dtype = torch.float32
        model = torch.nn.Linear(1, 1)

        def generate(**kwargs):
            generations.append(kwargs)
            return torch.tensor([[7, 8, 42]])

        model.generate = generate
        return model

    monkeypatch.setattr(LibreMolmo2, "_init_model", init_model)
    return prompts, generations


@pytest.mark.parametrize(
    "alias,size",
    [
        ("molmo2", "4b"),
        ("molmo2-4b", "4b"),
        ("molmo2-8b", "8b"),
        ("molmo2-o-7b", "o-7b"),
    ],
)
def test_factory(alias, size, loaded):
    model = LibreVLM(alias, names=["boat"], device="cpu")
    assert isinstance(model, LibreMolmo2)
    assert model.size == size
    assert model.task == "point"


def test_predict_wraps_original_canvas_and_preserves_class_and_attention(
    loaded, tmp_path
):
    prompts, generations = loaded
    model = LibreMolmo2(names=["Boat", "Person"], device="cpu", max_new_tokens=32)
    result = model.predict(Image.new("RGB", (400, 200)))
    assert result.orig_shape == (200, 400)
    assert result.boxes is None
    np.testing.assert_allclose(result.points.xy.cpu(), [[100, 150], [100, 150]])
    assert result.points.cls.tolist() == [0, 1]
    assert result.points.conf.tolist() == [1.0, 1.0]
    assert prompts == ["Point to the Boat.", "Point to the Person."]
    assert all(g["token_type_ids"].dtype == torch.bool for g in generations)
    assert all(g["max_new_tokens"] == 32 and not g["do_sample"] for g in generations)
    assert all(g["repetition_penalty"] == 1.0 for g in generations)
    # Exercise the point drawing surface without real model weights.
    saved = tmp_path / "points.png"
    model.predict(Image.new("RGB", (400, 200)), save=True, output_path=str(saved))
    assert saved.is_file()
    assert np.asarray(Image.open(saved)).any()
    filtered = model.predict(Image.new("RGB", (400, 200)), classes=[1], max_det=1)
    assert filtered.points.cls.tolist() == [1]


def test_multiple_images_stream_and_empty_filter(loaded):
    model = LibreMolmo2(names=["boat"], device="cpu")
    images = [Image.new("RGB", (80, 40)), Image.new("RGB", (40, 80))]
    results = model.predict(images)
    assert isinstance(results, list) and len(results) == 2
    np.testing.assert_allclose(results[1].points.xy.cpu(), [[10, 60]])
    assert len(list(model.predict(images, stream=True))) == 2
    empty = model._postprocess([], 0.0, 0.5, (400, 200))
    assert empty == {"points": [], "num_detections": 0}
    out = [{"label": "boat", "point": [0.25, 0.75]}]
    assert model._postprocess(out, 1.1, 0.5, (400, 200))["points"] == []


def test_custom_prompt_and_chat_budget(loaded):
    prompts, generations = loaded
    model = LibreMolmo2(names=["boat"], device="cpu", prompt="Locate {label}.")
    model.predict(Image.new("RGB", (80, 40)))
    assert prompts == ["Locate boat."]
    model.chat(Image.new("RGB", (80, 40)), "Describe the scene.", max_new_tokens=17)
    assert generations[-1]["max_new_tokens"] == 17
    with pytest.raises(ValueError, match="label"):
        LibreMolmo2(prompt="ambiguous")


def test_unsupported_tasks_and_operations(loaded):
    with pytest.raises(ValueError, match="point"):
        LibreVLM("molmo2", task="detect")
    model = LibreMolmo2(names=["boat"], device="cpu")
    for method in [model.train, model.val, model.export]:
        with pytest.raises(NotImplementedError):
            method()
    with pytest.raises(NotImplementedError, match="point"):
        next(model.track(Image.new("RGB", (80, 40))))


def test_remote_pins_and_inventory():
    from libreyolo.models.inventory import collect_model_inventory
    from libreyolo.models.registry import MODEL_GROUPS

    assert LibreMolmo2.TRUST_REMOTE_CODE
    assert set(LibreMolmo2.HF_REPOS) == set(LibreMolmo2.HF_REVISIONS)
    assert all(
        len(sha) == 40 and int(sha, 16) for sha in LibreMolmo2.HF_REVISIONS.values()
    )
    assert MODEL_GROUPS["molmo2"] == "s"
    assert collect_model_inventory()["molmo2"]["optional_extra"] == "molmo2"


def test_wrong_transformers_fails_before_download(monkeypatch):
    monkeypatch.setitem(sys.modules, "einops", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(__version__="5.16.1")
    )
    with pytest.raises(ImportError, match=r"libreyolo\[molmo2\]"):
        molmo2._check_dependencies()
