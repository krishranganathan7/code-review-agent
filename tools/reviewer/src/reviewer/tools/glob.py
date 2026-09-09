"""`glob(pattern)` — find files by name pattern, scoped to the repository root.

The pattern is model output. `pathlib` will honour `..` segments and absolute
patterns on newer Pythons, so the pattern is rejected up front if it tries to
address anything outside the root — and, independently, every path the walk
yields is re-checked through the sandbox. Neither check relies on the other.
"""

from __future__ import annotations

from typing import Any

from ..types import ToolResult, ToolSpec
from .base import SandboxedTool, ToolInputError
from .sandbox import Sandbox, SandboxViolation

__all__ = ["Glob", "MAX_RESULTS"]

MAX_RESULTS = 500
"""Paths returned in one call before truncation."""


class Glob(SandboxedTool):
    """List repository files whose path matches a glob pattern."""

    name = "glob"
    spec = ToolSpec(
        name="glob",
        description=(
            "Find files in the repository by path pattern. Supports '*', '?' and "
            "'**' for recursive matching, e.g. '**/*.py' or 'src/**/test_*.ts'. "
            "Returns repository-relative paths."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Glob pattern, relative to the repository root, "
                        "e.g. '**/*.py'."
                    ),
                }
            },
            "required": ["pattern"],
        },
    )

    def __init__(
        self, sandbox: Sandbox, *, max_results: int = MAX_RESULTS
    ) -> None:
        super().__init__(sandbox)
        self.max_results = max_results

    def _run(self, **kwargs: Any) -> ToolResult:
        self.only(kwargs, {"pattern"})
        pattern = self.required_str(kwargs, "pattern")
        if not pattern:
            raise ToolInputError("'pattern' must not be empty")

        normalized = pattern.replace("\\", "/").strip()
        if normalized.startswith("/") or ":" in normalized.split("/")[0]:
            raise ToolInputError(
                f"pattern {pattern!r} must be relative to the repository root"
            )
        if ".." in normalized.split("/"):
            raise ToolInputError(
                f"pattern {pattern!r} may not contain '..'; "
                "glob is scoped to the repository root"
            )

        try:
            # Sorted by the posix string, not by Path: Path comparison is
            # case-insensitive on Windows, which would make the order of a
            # review's evidence depend on the OS it ran on.
            candidates = sorted(
                self.sandbox.root.glob(normalized), key=lambda p: p.as_posix()
            )
        except (ValueError, NotImplementedError, IndexError) as exc:
            # pathlib rejects some malformed patterns outright.
            raise ToolInputError(f"invalid glob pattern {pattern!r}: {exc}") from exc

        found: list[str] = []
        directories = 0
        for candidate in candidates:
            try:
                resolved = self.sandbox.resolve(str(candidate))
            except SandboxViolation:
                # A match reached through a link that leaves the tree.
                continue
            if resolved.is_dir():
                directories += 1
                continue
            found.append(self.sandbox.relative(resolved))
            if len(found) > self.max_results:
                break

        truncated = len(found) > self.max_results
        if truncated:
            found = found[: self.max_results]

        return self.success(
            _render(found, pattern, truncated),
            pattern=pattern,
            match_count=len(found),
            directories_skipped=directories,
            truncated=truncated,
            paths=list(found),
        )


def _render(paths: list[str], pattern: str, truncated: bool) -> str:
    if not paths:
        return f"No files match {pattern!r}."
    lines = list(paths)
    if truncated:
        lines.append(
            f"... truncated: showing the first {len(paths)} paths. "
            "Use a narrower pattern to see the rest."
        )
    return "\n".join(lines)
