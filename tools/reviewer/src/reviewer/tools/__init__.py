"""Stage 4 substrate: the read-only agentic-search tools (CLAUDE.md §4).

`read_file`, `grep`, `glob`, `list_dir`, `git_history` — each scoped to the
checkout root through the single `Sandbox` chokepoint and protected against path
escape. They are pure deterministic Python: no model is involved in running one,
only in choosing to call it.

`read_patch` joins them when a pull request's patches are supplied. It serves the
diff under review, which used to be inlined into the task prompt — and, being
inlined, left the agent nothing to fetch and so no reason to search at all.
See :mod:`reviewer.tools.read_patch`.

Build the set the agent is offered with `build_toolset(repo_root)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .base import SandboxedTool, Tool, ToolInputError
from .git_history import Commit, GitHistory
from .glob import Glob
from .grep import Grep, Match
from .list_dir import Entry, ListDir
from .read_file import ReadFile
from .read_patch import ReadPatch
from .sandbox import Sandbox, SandboxViolation

__all__ = [
    "Commit",
    "Entry",
    "GitHistory",
    "Glob",
    "Grep",
    "ListDir",
    "Match",
    "ReadFile",
    "ReadPatch",
    "Sandbox",
    "SandboxViolation",
    "SandboxedTool",
    "Tool",
    "ToolInputError",
    "build_toolset",
]


def build_toolset(
    repo_root: str | Path | Sandbox,
    *,
    patches: Mapping[str, str] | None = None,
    statuses: Mapping[str, str] | None = None,
    counts: Mapping[str, tuple[int, int]] | None = None,
) -> list[Tool]:
    """The tools, all sharing one sandbox, in the order they are offered.

    Ordering is stable so the tool list a provider sees does not change between
    runs — a varying tool list would invalidate prompt caching for no reason.

    `patches` adds `read_patch`, which serves the diff under review. Omit it and
    the tool is left out rather than offered with nothing to return: a tool that
    can only fail is worse than no tool.
    """
    sandbox = repo_root if isinstance(repo_root, Sandbox) else Sandbox(repo_root)
    tools: list[Tool] = [
        ReadFile(sandbox),
        Grep(sandbox),
        Glob(sandbox),
        ListDir(sandbox),
        GitHistory(sandbox),
    ]
    if patches is not None:
        tools.append(
            ReadPatch(sandbox, patches, statuses=statuses, counts=counts)
        )
    return tools
