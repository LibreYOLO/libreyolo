"""Findings and report for LibreDoctor.

A check emits :class:`Finding` objects; the runner collects them into a
:class:`Report` that renders for humans (``render_human``) or agents
(``to_dict`` → JSON).
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class Severity(str, Enum):
    """How bad a finding is.

    ERROR: training will crash or silently learn garbage.
    WARNING: likely hurts results; the user should look.
    INFO: a statistic worth knowing; no action implied.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


_SEVERITY_ORDER = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
_SEVERITY_MARK = {Severity.ERROR: "E", Severity.WARNING: "W", Severity.INFO: "i"}


@dataclass
class Finding:
    """One issue (or statistic) discovered in the dataset."""

    check_id: str
    severity: Severity
    message: str
    split: Optional[str] = None
    paths: list[Path] = field(default_factory=list)  # capped examples
    count: Optional[int] = None  # total occurrences (may exceed len(paths))
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "check_id": self.check_id,
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.split is not None:
            data["split"] = self.split
        if self.count is not None:
            data["count"] = self.count
        if self.paths:
            data["paths"] = [str(p) for p in self.paths]
        if self.details:
            data["details"] = self.details
        return data


@dataclass
class Report:
    """The outcome of a doctor run."""

    findings: list[Finding]
    stats: dict[str, Any]
    skipped_checks: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def infos(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.INFO]

    def exit_code(self, strict: bool = False) -> int:
        """0 when healthy, 1 when errors (or, with strict, warnings) exist."""
        if self.errors:
            return 1
        if strict and self.warnings:
            return 1
        return 0

    def summary(self) -> dict[str, int]:
        return {
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "infos": len(self.infos),
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation (stable contract for agents and CI)."""
        return {
            "summary": self.summary(),
            "stats": self.stats,
            "findings": [f.to_dict() for f in sorted_findings(self.findings)],
            "skipped_checks": list(self.skipped_checks),
        }

    def render_human(self, max_paths: int = 3) -> str:
        """Plain-text report (ASCII-safe for Windows consoles)."""
        lines: list[str] = []
        dataset = self.stats.get("yaml") or self.stats.get("root") or "dataset"
        lines.append(f"LibreDoctor report for {dataset}")

        splits = self.stats.get("splits", {})
        for name, s in splits.items():
            lines.append(
                f"  {name}: {s.get('images', 0)} images, "
                f"{s.get('instances', 0)} instances, "
                f"{s.get('background', 0)} background"
            )
        nc = self.stats.get("nc")
        if nc is not None:
            lines.append(f"  classes: {nc}")
        lines.append("")

        for severity, title in (
            (Severity.ERROR, "ERRORS"),
            (Severity.WARNING, "WARNINGS"),
            (Severity.INFO, "INFO"),
        ):
            group = [f for f in self.findings if f.severity is severity]
            if not group:
                continue
            lines.append(f"{title} ({len(group)})")
            for f in group:
                where = f" [{f.split}]" if f.split else ""
                mark = _SEVERITY_MARK[severity]
                lines.append(f"  {mark} {f.check_id}{where}: {f.message}")
                shown = f.paths[:max_paths]
                for p in shown:
                    lines.append(f"      {p}")
                hidden = (f.count or len(f.paths)) - len(shown)
                if f.paths and hidden > 0:
                    lines.append(f"      ... and {hidden} more")
            lines.append("")

        if self.skipped_checks:
            lines.append(f"skipped checks: {', '.join(self.skipped_checks)}")

        s = self.summary()
        if s["errors"] or s["warnings"]:
            lines.append(
                f"{s['errors']} error(s), {s['warnings']} warning(s), "
                f"{s['infos']} info(s)."
            )
        else:
            lines.append("No problems found.")
        return "\n".join(lines)


def sorted_findings(findings: list[Finding]) -> list[Finding]:
    """Errors first, then warnings, then infos; stable by check id."""
    return sorted(
        findings,
        key=lambda f: (_SEVERITY_ORDER[f.severity], f.check_id, f.split or ""),
    )
