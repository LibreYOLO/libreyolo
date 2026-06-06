"""UI command: launch the local drag-and-drop inference UI in the browser."""

import errno

import typer

from ..command_utils import exit_with_error
from ..output import OutputHandler


def ui_cmd(
    host: str = typer.Option("127.0.0.1", help="Host/interface to bind"),
    port: int = typer.Option(8000, help="Port to bind (auto-bumps if taken)"),
    device: str = typer.Option("auto", help="Device: 0, cpu, mps, auto"),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Do not auto-open the browser"
    ),
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose stderr output"),
) -> None:
    """Launch a local web UI: drag/drop/paste images, run inference, view results."""
    out = OutputHandler(json_mode=json_output, quiet=quiet)

    from libreyolo.ui.server import serve

    httpd = None
    url = ""
    for candidate in range(port, port + 20):
        try:
            httpd, url = serve(
                host=host,
                port=candidate,
                device=device,
                open_browser=not no_browser,
            )
            break
        except OSError as exc:
            if exc.errno in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", 10048)):
                continue
            exit_with_error(out, "io_error", f"Failed to start UI server: {exc}")

    if httpd is None:
        exit_with_error(
            out,
            "io_error",
            f"No free port found in range {port}-{port + 19}.",
            suggestion="Pass a different --port.",
        )

    out.result(
        {
            "url": url,
            "device": device,
            "_human_text": (
                f"LibreYOLO UI running at {url}\n"
                "Drop images or a folder, pick a model, hit Run inference. "
                "Press Ctrl+C to stop."
            ),
        }
    )

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        out.progress("Shutting down UI...")
    finally:
        httpd.shutdown()
        httpd.server_close()
