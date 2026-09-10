"""Promptable 3D detection through the optional DetAny3D runtime."""

import typer

from ..command_utils import exit_with_error, help_json_callback
from ..output import OutputHandler


def detany3d_cmd(
    source: str = typer.Option(..., help="Image path or directory"),
    model: str = typer.Option(..., help="Local official full DetAny3D checkpoint"),
    runtime_path: str | None = typer.Option(
        None, help="DetAny3D checkout; defaults to DETANY3D_PATH"
    ),
    runtime_python: str | None = typer.Option(
        None, help="Interpreter for the separate runtime environment"
    ),
    text: str | None = typer.Option(None, help="JSON category list or a text caption"),
    bboxes: str | None = typer.Option(None, help="JSON original-pixel xyxy box list"),
    points: str | None = typer.Option(
        None, help="JSON positive points, optionally grouped by object"
    ),
    conf: float | None = typer.Option(
        None, help="Text detector box-confidence threshold"
    ),
    text_threshold: float | None = typer.Option(
        None, help="Text token-confidence threshold"
    ),
    grounding_checkpoint: str | None = typer.Option(
        None, help="Optional GroundingDINO Swin-B checkpoint path"
    ),
    grounding_config: str | None = typer.Option(
        None, help="Optional GroundingDINO Swin-B config path"
    ),
    device: str = typer.Option("auto", help="auto, cpu, or CUDA device"),
    save: bool = typer.Option(False, help="Save projected cuboids"),
    output_path: str | None = typer.Option(None, help="Single-image output filename"),
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
    help_json: bool = typer.Option(
        False,
        "--help-json",
        is_eager=True,
        callback=help_json_callback,
        help="Dump command schema as JSON",
    ),
) -> None:
    """Detect metric cuboids from text, boxes or points with predicted calibration."""
    import json
    import os
    import sys
    from contextlib import ExitStack, redirect_stderr, redirect_stdout

    out = OutputHandler(json_mode=json_output, quiet=quiet)
    try:
        prompt = {
            key: json.loads(value)
            for key, value in (("bboxes", bboxes), ("points", points))
            if value is not None
        }
        if text is not None:
            prompt["text"] = json.loads(text) if text.lstrip().startswith("[") else text
        thresholds = {
            key: value
            for key, value in (("conf", conf), ("text_threshold", text_threshold))
            if value is not None
        }
        with ExitStack() as stack:
            destination = (
                stack.enter_context(open(os.devnull, "w")) if quiet else sys.stderr
            )
            stack.enter_context(redirect_stdout(destination))
            if quiet:
                stack.enter_context(redirect_stderr(destination))
            from libreyolo import LibreDetAny3D

            with LibreDetAny3D(
                model,
                runtime_path=runtime_path,
                runtime_python=runtime_python,
                grounding_checkpoint=grounding_checkpoint,
                grounding_config=grounding_config,
                device=device,
                **thresholds,
            ) as detector:
                results = detector.predict(
                    source, save=save, output_path=output_path, **prompt
                )
        results = results if isinstance(results, list) else [results]
        out.result(
            {
                "task": "detect3d",
                "model": model,
                "results": [
                    {
                        "path": result.path,
                        "detections": result.summary(),
                        "intrinsics": result.boxes3d.numpy().intrinsics.tolist(),
                    }
                    for result in results
                ],
            }
        )
    except (ValueError, TypeError) as exc:
        exit_with_error(out, "config_type_error", str(exc))
    except (ImportError, RuntimeError) as exc:
        exit_with_error(out, "model_load_failed", str(exc))
    except OSError as exc:
        exit_with_error(out, "io_error", str(exc))
