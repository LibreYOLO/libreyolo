"""Base class for the ``LibreGround`` tier: instruction → click point.

Subclasses are Hugging Face snapshot adapters, same as ``LibreVLM``. The
public primitive is ``Results.points`` on the original image canvas. See
``docs/libreground_design.md`` and ``docs/adr/0020-libreground-contract.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import torch

from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.results import Points, Results
from ..vlm.base import LibreVLMModel
from .parsing import build_point_dict, coerce_queries, extract_clicks


class GroundAPIMixin:
    """``set_query`` + per-call ``prompt=`` / ``query=`` on top of ``predict``.

    Host must be a ``BaseModel`` with ``set_classes`` and the usual
    ``InferenceRunner`` ``__call__``.
    """

    def set_classes(self, classes):
        result = super().set_classes(classes)
        self._queries = [self.names[i] for i in range(len(self.names))]
        return result

    def set_query(self, query: str | Sequence[str]) -> "GroundAPIMixin":
        """Set the sticky referring expression(s) to point at.

        One string is the usual case. A list is one click per query, class id
        matching the query index. Returns ``self`` so calls can chain.
        """
        return self.set_classes(coerce_queries(query))

    def __call__(
        self,
        source=None,
        prompt: str | Sequence[str] | None = None,
        query: str | Sequence[str] | None = None,
        **kwargs,
    ):
        requested = prompt if prompt is not None else query
        if requested is None:
            if not getattr(self, "_queries", None) and not getattr(self, "names", None):
                raise ValueError(
                    "LibreGround needs a query. Pass prompt=\"Bluetooth\" "
                    "or call set_query(...) first."
                )
            if not getattr(self, "_queries", None) and self.names:
                self._queries = [self.names[i] for i in range(len(self.names))]
            if len(getattr(self, "_queries", [])) == 0:
                raise ValueError(
                    "LibreGround needs a query. Pass prompt=\"Bluetooth\" "
                    "or call set_query(...) first."
                )
            return super().__call__(source, **kwargs)

        queries = coerce_queries(requested)
        if len(queries) > 1:
            self._reject_multi_query_batch(source)
            return self._predict_each_query(source, queries, **kwargs)

        previous = list(getattr(self, "_queries", []) or [])
        self.set_query(queries[0])
        try:
            return self._one_click(super().__call__(source, **kwargs))
        finally:
            self._restore_queries(previous)

    def _reject_multi_query_batch(self, source) -> None:
        if isinstance(source, (list, tuple)):
            raise ValueError(
                "A list of queries is only supported for a single image. "
                "Pass one prompt, or call predict once per query."
            )
        if isinstance(source, (str, Path)) and Path(source).is_dir():
            raise ValueError(
                "A list of queries is only supported for a single image. "
                "Use set_query(\"one thing\") for a folder."
            )

    def _predict_each_query(self, source, queries: List[str], **kwargs) -> Results:
        previous = list(getattr(self, "_queries", []) or [])
        results: List[Results] = []
        try:
            for item in queries:
                self.set_query(item)
                result = super().__call__(source, **kwargs)
                if isinstance(result, list):
                    raise ValueError(
                        "A list of queries is only supported for a single image."
                    )
                results.append(self._one_click(result))
        finally:
            self._restore_queries(previous)
        return _merge_point_results(results, queries)

    def _restore_queries(self, previous: List[str]) -> None:
        """Put sticky query state back the way the caller left it."""
        if previous:
            self.set_query(previous)
            return
        self._queries = []
        self.names = {}
        if hasattr(self, "nb_classes"):
            self.nb_classes = 0
        if hasattr(self, "_name_to_id"):
            self._name_to_id = {}

    def _one_click(self, result: Results) -> Results:
        """Keep the first point. The contract is one click per query."""
        if result.points is None or len(result.points) <= 1:
            return result
        data = result.points.data[:1]
        return Results(
            boxes=None,
            orig_shape=result.orig_shape,
            path=result.path,
            names=result.names,
            points=Points(data, result.orig_shape),
        )


def _merge_point_results(results: List[Results], queries: List[str]) -> Results:
    rows = []
    for index, result in enumerate(results):
        if result.points is None or len(result.points) == 0:
            continue
        data = result.points.data
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)
        cloned = data.clone().float()
        if cloned.ndim == 1:
            cloned = cloned.unsqueeze(0)
        cloned[:, 2] = float(index)
        rows.append(cloned)
    if rows:
        points = torch.cat(rows, dim=0)
    else:
        points = torch.zeros((0, 4), dtype=torch.float32)
    first = results[0]
    return Results(
        boxes=None,
        orig_shape=first.orig_shape,
        path=first.path,
        names={i: name for i, name in enumerate(queries)},
        points=Points(points, first.orig_shape),
    )


class LibreGroundModel(GroundAPIMixin, LibreVLMModel):
    """Generative / task-token grounder that returns ``Results.points``."""

    SUPPORTED_TASKS: ClassVar[tuple] = ("point",)
    DEFAULT_TASK: ClassVar[str] = "point"
    MAX_NEW_TOKENS: ClassVar[int] = 128
    # unit | milli | pixel | pixel_view — see parsing.scale_point
    COORD_SPACE: ClassVar[str] = "unit"

    def __init__(
        self,
        size: str,
        *,
        names: Optional[list] = None,
        query: str | Sequence[str] | None = None,
        prompt: Optional[str] = None,
        **kwargs,
    ):
        # Constructor ``prompt=`` / ``query=`` is the initial instruction, not
        # a LibreVLM custom detection prompt.
        initial = query if query is not None else names
        if initial is None and prompt is not None:
            initial = prompt
            prompt = None
        if initial is not None:
            initial = coerce_queries(initial)
        super().__init__(size=size, names=initial, prompt=None, **kwargs)
        if initial is None:
            self.names = {}
            self.nb_classes = 0
            self._name_to_id = {}
            self._queries: List[str] = []
        else:
            self._queries = list(initial)
        self._view_size: Tuple[int, int] | None = None

    def _active_query(self) -> str:
        if getattr(self, "_queries", None):
            return self._queries[0]
        if self.names:
            return self.names[0]
        raise ValueError(
            "LibreGround needs a query. Pass prompt=\"Bluetooth\" "
            "or call set_query(...) first."
        )

    def _grounding_prompt(self) -> str:
        if self._custom_prompt:
            return self._custom_prompt
        return self._format_grounding_prompt(self._active_query())

    def _format_grounding_prompt(self, query: str) -> str:
        return (
            f"Point to: {query}. "
            "Output only the click location as [x, y] with coordinates "
            "normalized to [0, 1]."
        )

    def _detection_prompt(self) -> str:
        # InferenceRunner still calls the VLM hook name from subclasses that
        # reuse _preprocess. Grounding families should call _grounding_prompt
        # themselves; this keeps a stray call honest.
        return self._grounding_prompt()

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        self._view_size = img.size
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": self._grounding_prompt()},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        return inputs, img, img.size, 1.0

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        text = self._decode_generated(output)
        items = extract_clicks(text)
        for item in items:
            if not item.get("label"):
                item["label"] = self._active_query()
        return build_point_dict(
            items,
            getattr(self, "_name_to_id", {}) or {self._active_query().lower(): 0},
            original_size,
            coord_space=self.COORD_SPACE,
            view_size=getattr(self, "_view_size", None),
            conf_thres=conf_thres,
            max_det=1 if max_det is None else min(int(max_det), 1),
            classes=kwargs.get("classes"),
            default_score=self._score_detections(items),
        )

    def _decode_generated(self, output: Any) -> str:
        if isinstance(output, str):
            return output
        decoder = self.processor
        return decoder.batch_decode(output, skip_special_tokens=True)[0]

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            f"Training is out of scope for {type(self).__name__}. "
            "Fine-tune the grounder upstream and load the resulting weights."
        )

    def val(self, *args, **kwargs):
        raise NotImplementedError(
            f"Dataset validation is not supported for {type(self).__name__}: "
            "generated clicks carry only a placeholder confidence. "
            "Evaluate on ScreenSpot (click-in-box) outside this API."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} does not export to {format!r}. "
            "Run it through predict() and read Results.points."
        )
