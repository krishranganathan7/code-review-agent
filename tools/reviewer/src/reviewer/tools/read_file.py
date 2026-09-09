"""`read_file(path, start=None, end=None)` — read a file or a line slice.

Output is line-numbered. Every finding this system publishes needs a file and a
line range as evidence (CLAUDE.md §2.4), so the agent must be able to cite a line
it has actually seen rather than counting from the top of a blob.

Caps are always applied. A file the agent cannot read in one call is not an
error — the result says how much was withheld and how to ask for the rest.
"""

from __future__ import annotations

from typing import Any

from ..types import ToolResult, ToolSpec
from .base import SandboxedTool, ToolInputError
from .sandbox import Sandbox

__all__ = ["ReadFile", "MAX_LINES", "MAX_BYTES", "BINARY_SNIFF_BYTES"]

MAX_LINES = 2000
"""Lines returned in one call before truncation."""

MAX_BYTES = 256 * 1024
"""Bytes read from disk before truncation. Bounds work on a huge single-line file."""

BINARY_SNIFF_BYTES = 8192
"""How much of the head is examined to decide whether a file is text."""


class ReadFile(SandboxedTool):
    """Read a text file, or a 1-indexed inclusive line range of one."""

    name = "read_file"
    spec = ToolSpec(
        name="read_file",
        description=(
            "Read a text file from the repository, or a line range of it. "
            "Returns line-numbered content. Paths are relative to the repository "
            "root. Use start and end to read a specific range of a large file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repository-relative path to the file.",
                },
                "start": {
                    "type": "integer",
                    "description": "First line to return, 1-indexed inclusive.",
                    "minimum": 1,
                },
                "end": {
                    "type": "integer",
                    "description": "Last line to return, 1-indexed inclusive.",
                    "minimum": 1,
                },
            },
            "required": ["path"],
        },
    )

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        max_lines: int = MAX_LINES,
        max_bytes: int = MAX_BYTES,
    ) -> None:
        super().__init__(sandbox)
        self.max_lines = max_lines
        self.max_bytes = max_bytes

    def _run(self, **kwargs: Any) -> ToolResult:
        self.only(kwargs, {"path", "start", "end"})
        requested = self.required_str(kwargs, "path")
        start = self.optional_int(kwargs, "start", minimum=1)
        end = self.optional_int(kwargs, "end", minimum=1)
        if start is not None and end is not None and end < start:
            raise ToolInputError(f"end ({end}) is before start ({start})")

        target = self.sandbox.resolve(requested)
        relative = self.sandbox.relative(target)

        if not target.exists():
            return self.failure(f"no such file: {relative}", path=relative)
        if target.is_dir():
            return self.failure(
                f"{relative} is a directory, not a file; use list_dir",
                path=relative,
            )
        if not target.is_file():
            return self.failure(
                f"{relative} is not a regular file", path=relative
            )

        size = target.stat().st_size
        # Read one byte past the cap so an exactly-at-cap file is not called truncated.
        with target.open("rb") as handle:
            head = handle.read(self.max_bytes + 1)
        bytes_truncated = len(head) > self.max_bytes
        if bytes_truncated:
            head = head[: self.max_bytes]

        if _looks_binary(head):
            return self.failure(
                f"{relative} appears to be a binary file ({size} bytes); "
                "read_file only returns text",
                path=relative,
                binary=True,
                size_bytes=size,
            )

        text = head.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if bytes_truncated and len(lines) > 1:
            # The final line was cut mid-way by the byte cap; drop the fragment.
            # Unless it is the only line there is — on a file that is one huge
            # line, the clipped fragment is the entire result.
            lines.pop()

        total_available = len(lines)
        first = start if start is not None else 1

        if total_available == 0 and first == 1:
            # An empty file is empty, not an error.
            return self.success(
                "",
                path=relative,
                start_line=1,
                end_line=0,
                line_count=0,
                total_lines=0,
                size_bytes=size,
                truncated=False,
            )

        if first > total_available:
            return self.failure(
                f"{relative}: start line {first} is past the end of the file "
                f"({total_available} lines available)",
                path=relative,
                total_lines=total_available,
            )

        last = end if end is not None else total_available
        last = min(last, total_available)

        selected = lines[first - 1 : last]
        lines_truncated = len(selected) > self.max_lines
        if lines_truncated:
            selected = selected[: self.max_lines]
            last = first + self.max_lines - 1

        truncated = lines_truncated or bytes_truncated
        body = _number(selected, first)
        if truncated:
            body += "\n" + _marker(relative, last, bytes_truncated, size)

        return self.success(
            body,
            path=relative,
            start_line=first,
            end_line=last,
            line_count=len(selected),
            total_lines=total_available,
            size_bytes=size,
            truncated=truncated,
        )


def _looks_binary(head: bytes) -> bool:
    """A NUL byte, or undecodable text, means this is not something to read."""
    sample = head[:BINARY_SNIFF_BYTES]
    if b"\x00" in sample:
        return True
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A multi-byte character clipped by the sniff boundary is not binary.
        return exc.reason != "unexpected end of data"
    return False


def _number(lines: list[str], first: int) -> str:
    width = max(len(str(first + len(lines) - 1)), 4) if lines else 4
    return "\n".join(
        f"{number:>{width}}\t{line}"
        for number, line in enumerate(lines, start=first)
    )


def _marker(relative: str, last: int, by_bytes: bool, size: int) -> str:
    reason = (
        f"file exceeds the {MAX_BYTES // 1024} KiB read cap ({size} bytes)"
        if by_bytes
        else "line cap reached"
    )
    return (
        f"... truncated: {reason}. Showing {relative} through line {last}; "
        f"call read_file again with start={last + 1} for more."
    )
