"""Unified-diff parsing, in one place.

Four separate parsers used to live in `risk.py`, `verification.py` and the
Python adapter, and all four carried the same defect: inside a hunk they treated
any line starting with ``+++`` or ``---`` as a file header and skipped it
*without advancing the new-side line counter*.

That is wrong twice over. A patch line is one marker character followed by the
line's own text, so adding the line ``++count;`` produces ``+++count;`` — real
added content, common in any language with an increment operator, and in any
diff-of-a-diff. Skipping it loses the line, and failing to advance the counter
shifts the reported number of every added line after it in the hunk.

The header lines the old check was reaching for (``--- a/f``, ``+++ b/f``) only
ever appear *before* a hunk, so the correct rule is: a ``diff --git`` line ends
the current hunk, and everything between ``@@`` markers is content.

Language-neutral by construction — this is diff syntax, not source syntax, so it
belongs in the core rather than behind a `LanguageAdapter` (CLAUDE.md §2.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Literal

__all__ = ["HUNK_HEADER", "PatchLine", "LineKind", "iter_patch_lines", "added_lines"]

HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
"""A hunk header, capturing the new-side start line (and length, if given)."""

_FILE_BOUNDARY = "diff --git "
"""Starts a new file within a multi-file diff, which ends the current hunk."""

LineKind = Literal["add", "remove", "context"]


@dataclass(frozen=True)
class PatchLine:
    """One content line of a hunk, with where it sits on the new side.

    ``new_line`` is the line's own number for an addition or a context line. For
    a removal there is no new-side line, so it carries the position the removal
    sits *at* — the number the following line will take.
    """

    kind: LineKind
    new_line: int
    text: str

    @property
    def added(self) -> bool:
        return self.kind == "add"


def iter_patch_lines(patch: str | None) -> Iterator[PatchLine]:
    """Walk a unified diff, yielding each content line with its new-side number.

    Handles multi-file diffs: a ``diff --git`` line closes the open hunk, so the
    ``--- a/f`` / ``+++ b/f`` headers that follow are outside any hunk and are
    skipped as headers rather than mistaken for content.
    """
    if not patch:
        return

    new_line = 0
    in_hunk = False

    for raw in patch.splitlines():
        if raw.startswith(_FILE_BOUNDARY):
            in_hunk = False
            continue

        header = HUNK_HEADER.match(raw)
        if header:
            new_line = int(header.group(1))
            in_hunk = True
            continue

        if not in_hunk:
            continue

        if raw.startswith("\\"):
            # "\ No newline at end of file" — an annotation, not a line.
            continue

        if raw.startswith("+"):
            yield PatchLine(kind="add", new_line=new_line, text=raw[1:])
            new_line += 1
        elif raw.startswith("-"):
            # A removal occupies no new-side line; report the position it sits at.
            yield PatchLine(kind="remove", new_line=new_line, text=raw[1:])
        else:
            # A context line. Some producers strip the leading space on a blank
            # context line, so an empty string is context, not a malformed line.
            yield PatchLine(
                kind="context", new_line=new_line, text=raw[1:] if raw else ""
            )
            new_line += 1


def added_lines(patch: str | None) -> list[tuple[int, str]]:
    """Added lines of a patch with their new-side numbers."""
    return [(line.new_line, line.text) for line in iter_patch_lines(patch) if line.added]
