"""`list_dir(path)` — list a directory's entries, scoped to the repository root.

Directories are marked with a trailing `/` so the agent can navigate without a
second call to find out what each entry is. Entries whose real location is
outside the root are omitted rather than shown-and-refused: offering a path the
agent is not allowed to open would only waste a turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..types import ToolResult, ToolSpec
from .base import SandboxedTool
from .sandbox import Sandbox, SandboxViolation

__all__ = ["ListDir", "Entry", "MAX_ENTRIES"]

MAX_ENTRIES = 1000
"""Entries returned in one call before truncation."""


@dataclass(frozen=True)
class Entry:
    """One directory entry."""

    name: str
    is_dir: bool

    @property
    def display(self) -> str:
        return f"{self.name}/" if self.is_dir else self.name


class ListDir(SandboxedTool):
    """List the contents of a directory."""

    name = "list_dir"
    spec = ToolSpec(
        name="list_dir",
        description=(
            "List the entries of a directory in the repository. Directories are "
            "shown with a trailing '/'. Omit path, or pass '.', for the "
            "repository root."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Repository-relative directory path. Defaults to the "
                        "repository root."
                    ),
                }
            },
            "required": [],
        },
    )

    def __init__(self, sandbox: Sandbox, *, max_entries: int = MAX_ENTRIES) -> None:
        super().__init__(sandbox)
        self.max_entries = max_entries

    def _run(self, **kwargs: Any) -> ToolResult:
        self.only(kwargs, {"path"})
        requested = self.optional_str(kwargs, "path")

        target = self.sandbox.resolve(requested)
        relative = self.sandbox.relative(target)

        if not target.exists():
            return self.failure(f"no such directory: {relative}", path=relative)
        if not target.is_dir():
            return self.failure(
                f"{relative} is a file, not a directory; use read_file",
                path=relative,
            )

        entries: list[Entry] = []
        skipped = 0
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            try:
                resolved = self.sandbox.resolve(str(child))
            except SandboxViolation:
                # A link or junction whose real target is outside the tree.
                skipped += 1
                continue
            entries.append(Entry(name=child.name, is_dir=resolved.is_dir()))

        truncated = len(entries) > self.max_entries
        if truncated:
            entries = entries[: self.max_entries]

        return self.success(
            _render(entries, relative, truncated),
            path=relative,
            entry_count=len(entries),
            dir_count=sum(1 for entry in entries if entry.is_dir),
            outside_root_skipped=skipped,
            truncated=truncated,
            entries=[
                {"name": entry.name, "is_dir": entry.is_dir} for entry in entries
            ],
        )


def _render(entries: list[Entry], relative: str, truncated: bool) -> str:
    if not entries:
        return f"{relative} is empty."
    lines = [entry.display for entry in entries]
    if truncated:
        lines.append(
            f"... truncated: showing the first {len(entries)} entries of {relative}."
        )
    return "\n".join(lines)
