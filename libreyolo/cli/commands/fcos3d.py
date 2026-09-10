"""Calibrated monocular 3D detection with native FCOS3D."""

import typer

from ..command_utils import exit_with_error, help_json_callback
from ..output import OutputHandler


def fcos3d_cmd(
    source: str = typer.Option(..., help="Image path or directory"),
    model: str = typer.Option(
        ..., help="Local official FCOS3D R101 nuScenes checkpoint"
    ),
    intrinsics: str = typer.Option(
        ..., help="Original-image 3x3 camera calibration .npy file"
    ),
    device: str = typer.Option("auto", help="auto, cpu, or a CUDA device"),
    conf: float | None = typer.Option(None, help="Joint confidence threshold"),
    iou: float | None = typer.Option(
        None, help="Rotated bird's-eye-view NMS threshold"
    ),
    max_det: int = typer.Option(200, help="Maximum number of detections per image"),
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
    """Detect metric 3D boxes with a known camera calibration."""
    import os
    import sys
    from contextlib import ExitStack, redirect_stderr, redirect_stdout

    import numpy as np

    out = OutputHandler(json_mode=json_output, quiet=quiet)
    try:
        calibration = np.load(intrinsics, allow_pickle=False)
        thresholds = {
            key: value
            for key, value in (("conf", conf), ("iou", iou))
            if value is not None
        }
        with ExitStack() as stack:
            destination = (
                stack.enter_context(open(os.devnull, "w")) if quiet else sys.stderr
            )
            stack.enter_context(redirect_stdout(destination))
            if quiet:
                stack.enter_context(redirect_stderr(destination))
            from libreyolo import LibreFCOS3D

            model_obj = LibreFCOS3D(model, device=device)
            results = model_obj.predict(
                source,
                intrinsics=calibration,
                max_det=max_det,
                save=save,
                output_path=output_path,
                **thresholds,
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
