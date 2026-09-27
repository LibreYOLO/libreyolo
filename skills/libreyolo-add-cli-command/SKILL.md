---
name: libreyolo-add-cli-command
description: >-
  Add or extend a LibreYOLO CLI command or flag correctly: the Typer +
  KeyValueCommand architecture, both argument grammars (key=value and
  --key value), the --json/--quiet/--help-json contract, mode-aware aliases,
  family-derived defaults, and the unit tests that gate CLI behavior. Use
  when someone asks to add a CLI command, add a flag to
  predict/train/val/export, expose a Python feature on the CLI, or when a
  CLI parsing bug is reported (bool flags, aliases, key=value quirks). API
  naming decisions belong to libreyolo-api-conventions; this is the
  implementation side.
---

# Add or extend a LibreYOLO CLI command

The CLI is Typer with one custom layer: `KeyValueCommand`
(`libreyolo/cli/parsing.py`) rewrites YOLO-style `key=value` tokens into
`--key value` before Click parses, so every command accepts both grammars
with full Typer type-checking, help, and completion intact. Understand that
layer before touching anything; most CLI bugs live at its edges.

## The map

- `libreyolo/cli/__init__.py` `entrypoint()`: registers everything. Special
  commands (`version checks models formats cfg info metadata`) come from
  `commands/special.py`; core modes (`predict train val export ui monitor
  label doctor`) are one module each under `commands/`; `profile` is a
  Typer sub-app (command group) - use that pattern for any new
  multi-subcommand tool.
- `command_utils.py`, `output.py`, `errors.py`: shared option
  handling, machine-readable output, and error rendering. Reuse; don't
  reimplement `--json` printing or error exits per command.
- `aliases.py`: mode-aware translation of user-facing shorthand to internal
  config names (e.g. train `mosaic` to `mosaic_prob`, val `conf` to
  `conf_thres`). The single source of truth for name translation; never
  inline a rename in a command function.
- `config.py`: CLI-side config resolution.

## Adding a flag to an existing command

1. Add it as a typed Typer parameter in the command function, named per the
   ecosystem convention (`libreyolo-api-conventions`; check the standard's
   documented name first).
2. If users know it by a different name than the internal config field, add
   the mapping to the right `*_ALIASES` dict in `aliases.py`.
3. **Bool flags need a decision**: negatable (`--flag/--no-flag`) or
   one-way (`--json`, `--quiet`)? `key=false` on a one-way flag must drop
   the flag, not emit a nonexistent `--no-<flag>`; the `negatable` set in
   `rewrite_known_bool_flags` encodes this and getting it wrong is a known
   bug class (see the comments in `cli/__init__.py` around the logging
   flags). Register new bool flags accordingly.
4. Mirror it in the Python API in the same PR (CLI and Python land
   together), and thread it through, verifying the value actually reaches
   the code that uses it; an accepted-and-ignored flag is a blocking review
   finding.
5. Defaults come from the loaded model/family where they are family-shaped
   (imgsz, thresholds): read them off the model, don't hardcode
   (REVIEW.md: "CLI defaults are family-derived").

## Adding a new command

1. New module `libreyolo/cli/commands/<name>.py` exposing `<name>_cmd`
   (or a `<name>_app` Typer sub-app for command groups).
2. Register in `entrypoint()` with `cls=KeyValueCommand` (or
   `app.add_typer` for groups). Without `KeyValueCommand` the command
   silently loses `key=value` support, which users will file as a bug.
3. Support the output contract: `--json` (machine-readable to stdout,
   nothing else on stdout), `--quiet`, and correct exit codes via the
   shared error helpers. Map an exception raised by a model call with
   `model_call_error_code()`, so a `ValueError` exits 2 as a config error
   rather than 1 as `io_error`. Agents and scripts consume these; they are part of
   why the CLI is self-describing.
4. Keep imports inside the command function or module lazy where heavy
   (torch etc.): `libreyolo --help` must stay fast, and the entrypoint's
   deferred imports exist for that reason.
5. Docs: the command shows up in `libreyolo --help` automatically; add it
   to the website docs command reference when user-facing.

## Testing (the part that actually gates)

CLI tests are unit tests (`tests/unit/cli/` for parsing/aliases/output/
errors, `tests/unit/cli/commands/test_<cmd>.py` per command) and run in the
PR gate, so they must be hermetic: mock the model layer, use `tmp_path`,
never download. Cover, at minimum:

- both grammars for the new surface (`key=value` and `--key value`),
- bool-flag behavior including `key=false` on one-way flags,
- alias translation if you touched `aliases.py`,
- `--json` output shape (assert keys, not prose),
- the failure path (bad input produces the intended error, not a
  traceback).

`--help-json` output comes from the declared Typer parameters, so it stays
correct if you declared the flag properly and wrong if you hand-parsed
argv; that is the test for whether you wired it right.

## Traps

- Hand-parsing `sys.argv` in a command. Everything goes through Typer
  parameters + `KeyValueCommand`, or grammar support fractures.
- Renaming inside the command function instead of `aliases.py`.
- Printing progress to stdout (breaks `--json` consumers); human chatter
  goes to stderr / logging, respecting `--quiet`.
- A new command that imports torch at module top (slows every `libreyolo`
  invocation, including `--help`).
- Adding a `track` / task-prefixed verb without checking
  `_strip_task_prefix` in `cli/__init__.py` for how task-prefixed argv is
  normalized.

## Related

- `skills/libreyolo-api-conventions/`: what to name it and whether it
  should exist on both surfaces (yes).
- `skills/libreyolo-run-unit-tests/`: keeping the new tests PR-gate-safe.
- `tests/unit/cli/`: the exemplars to copy.
