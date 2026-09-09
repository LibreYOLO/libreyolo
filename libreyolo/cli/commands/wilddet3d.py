"""Promptable 3D inference through the optional WildDet3D runtime."""

import typer

from ..command_utils import exit_with_error, help_json_callback
from ..output import OutputHandler


def wilddet3d_cmd(
    source: str = typer.Option(..., help="Image path or directory"),
    model: str = typer.Option(
        "wilddet3d_alldata_all_prompt_v1.0.pt",
        help="Local checkpoint or the mirrored default filename",
    ),
    intrinsics: str = typer.Option(
        ..., help="Original-image 3x3 calibration .npy file"
    ),
    text: str | None = typer.Option(
        None, help='JSON category list, e.g. ["car","person"]'
    ),
    bboxes: str | None = typer.Option(None, help="JSON original-pixel xyxy box list"),
    points: str | None = typer.Option(None, help="JSON xy points, optionally grouped"),
    labels: str | None = typer.Option(None, help="JSON binary labels matching points"),
    prompt_mode: str = typer.Option("geometric", help="geometric or visual"),
    depth: str | None = typer.Option(
        None, help="Original-resolution depth .npy in metres"
    ),
    device: str = typer.Option("auto", help="auto, cpu, or a CUDA device"),
    runtime_path: str | None = typer.Option(
        None, help="Optional upstream runtime checkout"
    ),
    runtime_python: str | None = typer.Option(
        None, help="Python interpreter for the Mac/CPU runtime"
    ),
    conf: float | None = typer.Option(None, help="Combined ranking-score threshold"),
    conf3d: float | None = typer.Option(None, help="3D confidence threshold"),
    iou: float | None = typer.Option(None, help="2D NMS IoU threshold"),
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
    """Detect 3D objects with text, boxes or points (optional upstream runtime)."""
    import json
    import os
    import sys
    from contextlib import ExitStack, redirect_stderr, redirect_stdout

    import numpy as np

    out = OutputHandler(json_mode=json_output, quiet=quiet)
    try:
        prompts = {
            name: json.loads(value)
            for name, value in (
                ("text", text),
                ("bboxes", bboxes),
                ("points", points),
                ("labels", labels),
            )
            if value is not None
        }
        calibration = np.load(intrinsics, allow_pickle=False)
        depth_array = np.load(depth, allow_pickle=False) if depth else None
        thresholds = {
            key: value
            for key, value in (("conf", conf), ("conf3d", conf3d), ("iou", iou))
            if value is not None
        }
        # Upstream may print during loading; preserve the machine stdout contract.
        with ExitStack() as stack:
            destination = (
                stack.enter_context(open(os.devnull, "w")) if quiet else sys.stderr
            )
            stack.enter_context(redirect_stdout(destination))
            if quiet:
                stack.enter_context(redirect_stderr(destination))
            from libreyolo import LibreWildDet3D

            with LibreWildDet3D(
                model,
                device=device,
                use_depth=depth is not None,
                runtime_path=runtime_path,
                runtime_python=runtime_python,
                **thresholds,
            ) as model_obj:
                results = model_obj.predict(
                    source,
                    intrinsics=calibration,
                    depth=depth_array,
                    prompt_mode=prompt_mode,
                    save=save,
                    output_path=output_path,
                    **prompts,
                )
        results = results if isinstance(results, list) else [results]
        out.result(
            {
                "task": "detect3d",
                "model": model,
                "results": [
                    {
                        "path": r.path,
                        "detections": r.summary(),
                        "intrinsics": r.boxes3d.numpy().intrinsics.tolist(),
                    }
                    for r in results
                ],
            }
        )
    except (ValueError, TypeError) as exc:
        exit_with_error(out, "config_type_error", str(exc))
    except (ImportError, RuntimeError) as exc:
        exit_with_error(out, "model_load_failed", str(exc))
    except OSError as exc:
        exit_with_error(out, "io_error", str(exc))
