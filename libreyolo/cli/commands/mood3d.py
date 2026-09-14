"""Open-set monocular 3D detection through the optional 3D-MOOD runtime."""

import typer

from ..command_utils import exit_with_error, help_json_callback
from ..output import OutputHandler


def mood3d_cmd(
    source: str = typer.Option(..., help="Image path or directory"),
    model: str | None = typer.Option(None, help="Local checkpoint path"),
    size: str | None = typer.Option(
        None, help="Model size: t or b; inferred from official filenames"
    ),
    intrinsics: str = typer.Option(
        ..., help="Original-image 3x3 calibration .npy file"
    ),
    text: str = typer.Option(..., help='JSON category list, e.g. ["car","person"]'),
    device: str = typer.Option("auto", help="auto, cpu, mps, or a CUDA device"),
    runtime_path: str | None = typer.Option(
        None, help="Optional upstream 3D-MOOD checkout"
    ),
    runtime_python: str | None = typer.Option(
        None, help="Python interpreter for the upstream runtime"
    ),
    conf: float | None = typer.Option(None, help="Detection confidence threshold"),
    iou: float | None = typer.Option(None, help="Class-agnostic NMS IoU threshold"),
    max_det: int | None = typer.Option(None, help="Maximum detections per image"),
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
    """Detect open-set 3D objects and predict metric depth with 3D-MOOD."""
    import json
    import os
    import sys
    from contextlib import ExitStack, redirect_stderr, redirect_stdout

    import numpy as np

    out = OutputHandler(json_mode=json_output, quiet=quiet)
    try:
        classes = json.loads(text)
        calibration = np.load(intrinsics, allow_pickle=False)
        settings = {
            key: value
            for key, value in (("conf", conf), ("iou", iou), ("max_det", max_det))
            if value is not None
        }
        with ExitStack() as stack:
            destination = (
                stack.enter_context(open(os.devnull, "w")) if quiet else sys.stderr
            )
            stack.enter_context(redirect_stdout(destination))
            if quiet:
                stack.enter_context(redirect_stderr(destination))
            from libreyolo import Libre3DMOOD

            with Libre3DMOOD(
                model,
                size=size,
                device=device,
                runtime_path=runtime_path,
                runtime_python=runtime_python,
                **settings,
            ) as model_obj:
                resolved_size = model_obj.size
                results = model_obj.predict(
                    source,
                    intrinsics=calibration,
                    text=classes,
                    save=save,
                    output_path=output_path,
                )
        results = results if isinstance(results, list) else [results]
        out.result(
            {
                "task": "detect3d",
                "model": model or f"Libre3DMOOD{resolved_size}.pt",
                "results": [
                    {
                        "path": result.path,
                        "detections": result.summary(),
                        "intrinsics": result.boxes3d.numpy().intrinsics.tolist(),
                        "depth": {
                            "min": result.depth_map.min,
                            "max": result.depth_map.max,
                            "mean": result.depth_map.mean,
                        },
                    }
                    for result in results
                ],
            }
        )
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        exit_with_error(out, "config_type_error", str(exc))
    except (ImportError, RuntimeError) as exc:
        exit_with_error(out, "model_load_failed", str(exc))
    except OSError as exc:
        exit_with_error(out, "io_error", str(exc))
