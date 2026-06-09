"""Doctor command: dataset health checks before training."""

from typing import Optional

import typer

from ..command_utils import exit_with_error, help_json_callback
from ..output import OutputHandler


def doctor_cmd(
    data: Optional[str] = typer.Argument(
        None, help="Dataset YAML (YOLO detection format, e.g. coco8.yaml)"
    ),
    data_opt: Optional[str] = typer.Option(
        None, "--data", help="Dataset YAML (alternative to the positional form)"
    ),
    imgsz: int = typer.Option(
        640, help="Training image size used for pixel-based checks (tiny objects)"
    ),
    fast: bool = typer.Option(
        False,
        "--fast",
        help="Skip image decoding (no corruption/duplicate/leakage checks)",
    ),
    skip: str = typer.Option(
        "",
        help="Comma-separated check ids or families to skip (e.g. images,labels.tiny_object)",
    ),
    only: str = typer.Option(
        "", help="Comma-separated check ids or families to run exclusively"
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Warnings also fail the exit code (CI gates)"
    ),
    download: bool = typer.Option(
        False,
        "--download",
        help="Allow URL-based dataset download if missing (never scripts)",
    ),
    # Agent flags
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
    """Check a dataset for problems before training (exit 1 when errors are found)."""
    from libreyolo.doctor import (
        DatasetNotFoundError,
        DoctorError,
        NotADetectionDatasetError,
        UnknownCheckError,
        diagnose,
    )

    out = OutputHandler(json_mode=json_output, quiet=quiet)

    if data and data_opt and data != data_opt:
        exit_with_error(
            out,
            "config_conflict",
            f"Dataset given twice with different values: '{data}' and "
            f"'{data_opt}'. Pass it once.",
        )
    data = data or data_opt
    if not data:
        exit_with_error(
            out,
            "config_required_key",
            "Missing dataset. Usage: libreyolo doctor <data.yaml>",
        )

    out.progress(f"Checking {data}...")
    try:
        report = diagnose(
            data,
            imgsz=imgsz,
            fast=fast,
            skip=[s for s in skip.split(",") if s.strip()],
            only=[s for s in only.split(",") if s.strip()],
            progress=not quiet and not json_output,
            autodownload=download,
        )
    except UnknownCheckError as exc:
        exit_with_error(out, "config_unknown_key", str(exc))
    except NotADetectionDatasetError as exc:
        exit_with_error(
            out,
            "data_invalid",
            str(exc),
            suggestion="Support for other tasks is planned; see docs/libredoctor_design.md.",
        )
    except DatasetNotFoundError as exc:
        exit_with_error(out, "data_not_found", str(exc))
    except DoctorError as exc:
        exit_with_error(out, "data_invalid", str(exc))

    code = report.exit_code(strict=strict)
    data_out = report.to_dict()
    data_out["data"] = data
    data_out["exit_code"] = code
    if not json_output:
        data_out["_human_text"] = report.render_human()
    out.result(data_out)

    if code:
        raise typer.Exit(code=code)
