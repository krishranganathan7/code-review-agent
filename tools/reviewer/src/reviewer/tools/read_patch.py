"""`read_patch(path=None)` — the change under review, fetched rather than handed over.

Six live runs published good findings while making **zero** tool calls, and the
cause was this: the task prompt inlined every patch, so the agent had nothing
left to fetch. Given a complete review assignment the natural reply is the
review. The same system prompt with a terse task carrying only the diff produced
three tool calls on the first turn; with the full task, none.

Anthropic's own PR reviewer (`anthropics/claude-code-action`) does not inline the
diff either. It inlines the PR metadata and a changed-file list —
`- path (modified) +2/-2 SHA: ...` — and tells the agent to run
`git diff origin/<base>...HEAD` itself, with `Glob`, `Grep`, `LS` and `Read`
allowed. This tool is that arrangement, kept inside our own sandboxed tool set:
the agent asks for the diff, one file at a time.

The patches come from stage 1, not from `git`, so this works identically for a
GitHub-ingested review and a local checkout, and the diff the agent reads is
byte-for-byte the diff the pipeline analyzed. Returned content is repository
content and reaches the model through the same untrusted-data delimiting as any
other tool result (§2.5).
"""

from __future__ import annotations

from typing import Any, Mapping

from ..types import ToolResult, ToolSpec
from .base import SandboxedTool
from .sandbox import Sandbox

__all__ = ["ReadPatch", "MAX_PATCH_CHARS"]

MAX_PATCH_CHARS = 60_000
"""Characters of one patch returned before truncation.

A generated lockfile or a vendored blob can carry a patch far larger than the
rest of the change put together. The cap keeps one such file from consuming the
whole token budget on its own.
"""


class ReadPatch(SandboxedTool):
    """List the files this pull request changed, or read one file's diff."""

    name = "read_patch"
    spec = ToolSpec(
        name="read_patch",
        description=(
            "The diff under review. Call with no arguments to list every file "
            "this pull request changed, with its status and line counts. Call "
            "with a path to read that file's unified diff. This is the change "
            "itself — read it before judging the code. To see a changed file "
            "in full, including the parts the diff does not show, use "
            "read_file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Repository-relative path of a changed file. Omit to "
                        "list all of them."
                    ),
                },
            },
        },
    )

    def __init__(
        self,
        sandbox: Sandbox,
        patches: Mapping[str, str],
        *,
        statuses: Mapping[str, str] | None = None,
        counts: Mapping[str, tuple[int, int]] | None = None,
        max_chars: int = MAX_PATCH_CHARS,
    ) -> None:
        super().__init__(sandbox)
        self.patches = dict(patches)
        self.statuses = dict(statuses or {})
        self.counts = dict(counts or {})
        self.max_chars = max_chars

    def _run(self, **kwargs: Any) -> ToolResult:
        self.only(kwargs, {"path"})
        requested = self.optional_str(kwargs, "path")

        if requested is None or not requested.strip():
            return self._listing()

        # Normalized through the sandbox, so `./src/x.ts`, `src\x.ts` and an
        # absolute in-root path all find the same entry, and a path outside the
        # root is refused by the same seam as every other tool (§2.7).
        relative = self.sandbox.relative(self.sandbox.resolve(requested))

        patch = self.patches.get(relative)
        if patch is None:
            return self.failure(
                f"{relative} is not changed by this pull request. "
                f"Changed files: {', '.join(sorted(self.patches)) or '(none)'}",
                path=relative,
            )
        if not patch.strip():
            # A rename with no edits, or a binary file: GitHub sends no patch.
            return self.success(
                f"{relative} ({self.statuses.get(relative, 'modified')}) has no "
                "textual diff — it was renamed, is binary, or is too large for "
                "the host to render. Use read_file to see its contents.",
                path=relative,
                empty=True,
            )

        body = patch
        truncated = len(body) > self.max_chars
        if truncated:
            body = body[: self.max_chars]
            withheld = len(patch) - self.max_chars
            body += (
                f"\n... [{withheld} characters withheld: this patch exceeds the "
                f"{self.max_chars}-character cap. Use read_file on the file "
                "itself to see any part not shown here.]"
            )

        return self.success(
            body,
            path=relative,
            status=self.statuses.get(relative, "modified"),
            truncated=truncated,
        )

    def _listing(self) -> ToolResult:
        """Every changed file, in the shape the reference implementation uses."""
        if not self.patches:
            return self.success(
                "This pull request changes no files with a readable diff.",
                files=0,
            )
        lines = []
        for path in sorted(self.patches):
            status = self.statuses.get(path, "modified")
            added, removed = self.counts.get(path, (0, 0))
            lines.append(f"- {path} ({status}) +{added}/-{removed}")
        return self.success(
            "\n".join(lines) + "\n\nCall read_patch with a path to read one diff.",
            files=len(lines),
        )
