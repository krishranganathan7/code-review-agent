"""`grep(pattern, path_glob=None)` — content search via ripgrep, as a subprocess.

ripgrep is the only backend. If `rg` is absent the tool fails with a clear error
rather than falling back to a slower or less correct Python scan: a silent
downgrade would change what the agent finds without telling it, and "no matches"
from a weaker searcher is indistinguishable from "no matches" from ripgrep.

**Subprocess safety.** The pattern and the glob are model output. Both reach
ripgrep as elements of an argument *list* — never a shell string, so there is no
shell to inject into. The pattern is passed after `--regexp` and the search path
is a fixed `.` after `--`, so a pattern or glob beginning with `-` is a pattern,
not a flag, and no argument can introduce a second search root. `cwd` is the
sandbox root, and every path ripgrep reports is re-checked through the sandbox
before it is returned.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Sequence

from ..types import ToolResult, ToolSpec
from .base import SandboxedTool, ToolInputError
from .sandbox import Sandbox, SandboxViolation

__all__ = ["Grep", "Match", "MAX_MATCHES", "MAX_LINE_CHARS", "TIMEOUT_SECONDS"]

MAX_MATCHES = 200
"""Matches returned in one call before truncation."""

MAX_LINE_CHARS = 500
"""A single matched line is clipped to this many characters."""

TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class Match:
    """One matching line, located."""

    file: str
    line_number: int
    line: str


class Grep(SandboxedTool):
    """Search file contents under the repository root."""

    name = "grep"
    spec = ToolSpec(
        name="grep",
        description=(
            "Search the contents of files in the repository for a regular "
            "expression. Returns matching lines with their file and line number. "
            "Optionally restrict the search to files matching a glob."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression to search for.",
                },
                "path_glob": {
                    "type": "string",
                    "description": (
                        "Optional glob restricting which files are searched, "
                        "e.g. '*.py' or 'src/**/*.ts'."
                    ),
                },
            },
            "required": ["pattern"],
        },
    )

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        rg_command: Sequence[str] | None = None,
        max_matches: int = MAX_MATCHES,
        timeout: float = TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(sandbox)
        if rg_command is None:
            found = shutil.which("rg")
            self.rg_command: list[str] | None = [found] if found else None
        else:
            self.rg_command = list(rg_command)
        self.max_matches = max_matches
        self.timeout = timeout

    @property
    def available(self) -> bool:
        """Whether a ripgrep executable was found."""
        return self.rg_command is not None

    def _run(self, **kwargs: Any) -> ToolResult:
        self.only(kwargs, {"pattern", "path_glob"})
        pattern = self.required_str(kwargs, "pattern")
        path_glob = self.optional_str(kwargs, "path_glob")
        if not pattern:
            raise ToolInputError("'pattern' must not be empty")

        if self.rg_command is None:
            return self.failure(
                "ripgrep (rg) was not found on PATH; the grep tool requires it. "
                "Install ripgrep, or use glob and read_file to inspect files "
                "directly.",
                unavailable=True,
            )

        argv = [*self.rg_command, "--json", "--regexp", pattern]
        if path_glob:
            argv += ["--glob", path_glob]
        argv += ["--", "."]

        try:
            completed = subprocess.run(  # noqa: S603 - argument list, never a shell
                argv,
                cwd=self.sandbox.root,
                capture_output=True,
                shell=False,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            return self.failure(
                f"ripgrep executable {self.rg_command[0]!r} could not be run",
                unavailable=True,
            )
        except subprocess.TimeoutExpired:
            return self.failure(
                f"grep timed out after {self.timeout:g}s; narrow the pattern "
                "or restrict it with path_glob",
                timed_out=True,
            )

        # ripgrep: 0 = matches, 1 = no matches (not an error), 2 = real failure.
        if completed.returncode not in (0, 1):
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            return self.failure(
                f"grep failed: {detail or f'ripgrep exited {completed.returncode}'}"
            )

        matches, parse_failures = self._parse(completed.stdout)
        truncated = len(matches) > self.max_matches
        if truncated:
            matches = matches[: self.max_matches]

        return self.success(
            _render(matches, pattern, path_glob, truncated),
            pattern=pattern,
            path_glob=path_glob,
            match_count=len(matches),
            file_count=len({match.file for match in matches}),
            truncated=truncated,
            unparsed_lines=parse_failures,
            matches=[
                {"file": m.file, "line_number": m.line_number, "line": m.line}
                for m in matches
            ],
        )

    def _parse(self, stdout: bytes) -> tuple[list[Match], int]:
        """Read ripgrep's newline-delimited JSON into `Match` objects.

        Non-match event types (`begin`, `end`, `summary`) are skipped. Every path
        is re-verified through the sandbox: ripgrep is told to search `.` and does
        not follow symlinks by default, but the containment guarantee is the
        sandbox's to make, not a subprocess's.
        """
        matches: list[Match] = []
        parse_failures = 0

        for raw in stdout.decode("utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                parse_failures += 1
                continue
            if not isinstance(event, dict) or event.get("type") != "match":
                continue

            data = event.get("data")
            if not isinstance(data, dict):
                parse_failures += 1
                continue

            path_text = _text_of(data.get("path"))
            line_number = data.get("line_number")
            if path_text is None or not isinstance(line_number, int):
                # A non-UTF-8 path or a missing line number: not citable evidence.
                parse_failures += 1
                continue

            try:
                resolved = self.sandbox.resolve(path_text)
            except SandboxViolation:
                parse_failures += 1
                continue

            text = _text_of(data.get("lines")) or ""
            matches.append(
                Match(
                    file=self.sandbox.relative(resolved),
                    line_number=line_number,
                    line=_clip(text.rstrip("\r\n")),
                )
            )

        return matches, parse_failures


def _text_of(field: Any) -> str | None:
    """ripgrep encodes strings as `{"text": ...}`, or `{"bytes": ...}` if not UTF-8."""
    if isinstance(field, dict):
        text = field.get("text")
        if isinstance(text, str):
            return text
    return None


def _clip(line: str) -> str:
    if len(line) <= MAX_LINE_CHARS:
        return line
    return line[:MAX_LINE_CHARS] + f"... [+{len(line) - MAX_LINE_CHARS} chars]"


def _render(
    matches: list[Match], pattern: str, path_glob: str | None, truncated: bool
) -> str:
    if not matches:
        scope = f" in files matching {path_glob}" if path_glob else ""
        return f"No matches for {pattern!r}{scope}."

    lines = [f"{m.file}:{m.line_number}: {m.line}" for m in matches]
    if truncated:
        lines.append(
            f"... truncated: showing the first {len(matches)} matches. "
            "Narrow the pattern or pass path_glob to see the rest."
        )
    return "\n".join(lines)
