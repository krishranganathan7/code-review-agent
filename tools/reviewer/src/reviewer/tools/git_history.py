"""`git_history(path, limit=20)` — recent commits touching a path, via `git log`.

Same subprocess discipline as `grep`: an argument list, never a shell, with the
path passed after `--` so a filename can never be read as a revision or a flag.
`git` runs with `cwd` set to the sandbox root and the path is resolved through
the sandbox first.

Two conditions are ordinary rather than exceptional and are reported as such: a
checkout that is not a git repository, and a path git knows nothing about (new,
untracked, or ignored). Both are things the agent should be told plainly — the
first means history is unavailable everywhere, the second is a genuine signal
about the file.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Sequence

from ..types import ToolResult, ToolSpec
from .base import SandboxedTool
from .sandbox import Sandbox

__all__ = ["GitHistory", "Commit", "DEFAULT_LIMIT", "MAX_LIMIT", "TIMEOUT_SECONDS"]

DEFAULT_LIMIT = 20
MAX_LIMIT = 100
"""Ceiling on `limit`, whatever the model asks for."""

TIMEOUT_SECONDS = 30

_SEPARATOR = "\x1f"
"""ASCII unit separator: cannot occur in a commit subject, unlike any punctuation."""

_FORMAT = _SEPARATOR.join(["%H", "%an", "%aI", "%s"])


@dataclass(frozen=True)
class Commit:
    """One commit touching the requested path."""

    sha: str
    author: str
    date: str
    subject: str

    @property
    def short_sha(self) -> str:
        return self.sha[:12]


class GitHistory(SandboxedTool):
    """Recent commits that touched a file or directory."""

    name = "git_history"
    spec = ToolSpec(
        name="git_history",
        description=(
            "List recent commits that touched a file or directory, most recent "
            "first. Useful for seeing how and why code changed, and who has been "
            "working on it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repository-relative file or directory path.",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Maximum commits to return (default {DEFAULT_LIMIT}, "
                        f"capped at {MAX_LIMIT})."
                    ),
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
        git_command: Sequence[str] = ("git",),
        timeout: float = TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(sandbox)
        self.git_command = list(git_command)
        self.timeout = timeout

    def _run(self, **kwargs: Any) -> ToolResult:
        self.only(kwargs, {"path", "limit"})
        requested = self.required_str(kwargs, "path")
        limit = self.optional_int(kwargs, "limit", minimum=1) or DEFAULT_LIMIT
        limit = min(limit, MAX_LIMIT)

        target = self.sandbox.resolve(requested)
        relative = self.sandbox.relative(target)

        if not target.exists():
            return self.failure(f"no such path: {relative}", path=relative)

        argv = [
            *self.git_command,
            "--no-pager",
            "log",
            f"--max-count={limit}",
            f"--format={_FORMAT}",
            "--",
            relative,
        ]

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
                f"git executable {self.git_command[0]!r} was not found on PATH; "
                "history is unavailable",
                unavailable=True,
                path=relative,
            )
        except subprocess.TimeoutExpired:
            return self.failure(
                f"git_history timed out after {self.timeout:g}s",
                timed_out=True,
                path=relative,
            )

        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            if _not_a_repository(detail):
                return self.failure(
                    "this checkout is not a git repository, so commit history "
                    "is unavailable",
                    unavailable=True,
                    not_a_repository=True,
                    path=relative,
                )
            return self.failure(
                f"git_history failed: {detail or f'git exited {completed.returncode}'}",
                path=relative,
            )

        commits = _parse(completed.stdout.decode("utf-8", errors="replace"))

        return self.success(
            _render(commits, relative),
            path=relative,
            commit_count=len(commits),
            limit=limit,
            truncated=len(commits) == limit,
            commits=[
                {
                    "sha": commit.sha,
                    "author": commit.author,
                    "date": commit.date,
                    "subject": commit.subject,
                }
                for commit in commits
            ],
        )


def _not_a_repository(stderr: str) -> bool:
    lowered = stderr.lower()
    return "not a git repository" in lowered or "detected dubious ownership" in lowered


def _parse(stdout: str) -> list[Commit]:
    commits: list[Commit] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split(_SEPARATOR)
        if len(fields) != 4:
            continue
        sha, author, date, subject = fields
        commits.append(
            Commit(sha=sha, author=author, date=date, subject=subject)
        )
    return commits


def _render(commits: list[Commit], relative: str) -> str:
    if not commits:
        return (
            f"No commits found for {relative}. The path may be newly added, "
            "untracked, or ignored."
        )
    return "\n".join(
        f"{commit.short_sha}  {commit.date}  {commit.author}  {commit.subject}"
        for commit in commits
    )
