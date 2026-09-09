"""The review workspace: file content as of `head_sha`, read through the sandbox.

Phase 4 left symbol attribution working from a reconstruction of the patch,
because nothing supplied real file content. This module supplies it.

**The seam.** :class:`SourceProvider` is one method — `read(path) -> str | None`,
`None` meaning "not available at head", which covers a deletion, a binary file,
an oversized file and a refused path alike. The change analyzer depends on that
and nothing more, so a review can run against a real checkout, an in-memory
fixture, or nothing at all with no change to stage 2.

**The workspace.** :class:`ReviewWorkspace` owns one directory holding the head
revision for one review. It is created per review under the system temp
directory (or `REVIEWER_WORKSPACE_ROOT`), and removed when the review ends —
nothing is shared between reviews, and nothing survives one. This directory is
deliberately the *same* root the Phase 2 tools are anchored to: it is the shared
workspace the Phase 5 review agents will explore, so what an agent can read and
what stage 2 parsed are the same bytes by construction.

**Every read goes through the Phase 2 `Sandbox`** (CLAUDE.md §2.7). There is no
path handling in this module: `Sandbox.resolve` is the only thing that turns a
path into a path, so the checkout inherits the escape guarantee already proven
for the tools — including the tar extraction that fills it.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import TracebackType
from typing import Iterator, Mapping, Protocol, runtime_checkable

from ..tools.read_file import BINARY_SNIFF_BYTES, MAX_BYTES
from ..tools.sandbox import Sandbox, SandboxViolation
from .ingest import PullRequest

__all__ = [
    "SourceProvider",
    "InMemorySource",
    "CheckoutSource",
    "Materializer",
    "GitArchiveMaterializer",
    "GitCloneMaterializer",
    "ReviewWorkspace",
    "WorkspaceError",
    "open_workspace",
    "WORKSPACE_ROOT_ENV_VAR",
]

logger = logging.getLogger("reviewer.analysis.source")

WORKSPACE_ROOT_ENV_VAR = "REVIEWER_WORKSPACE_ROOT"
"""Optional override for where review workspaces are created."""

WORKSPACE_PREFIX = "reviewer-"


class WorkspaceError(RuntimeError):
    """A review workspace could not be created or populated."""


# --------------------------------------------------------------------------
# The seam
# --------------------------------------------------------------------------


@runtime_checkable
class SourceProvider(Protocol):
    """File content as of the revision under review.

    ``read`` returns ``None`` rather than raising for every "cannot give you
    this" case — absent at head, binary, too large, refused by the sandbox. The
    caller's fallback is the same in all of them, and a provider that raised
    would make stage 2 responsible for distinguishing cases it cannot act on
    differently.
    """

    def read(self, path: str) -> str | None: ...


@dataclass(frozen=True)
class InMemorySource:
    """A provider backed by a dict. For tests: no git, no filesystem, no network."""

    files: Mapping[str, str]

    def read(self, path: str) -> str | None:
        return self.files.get(path)


class CheckoutSource:
    """A provider backed by a materialized checkout, read through the sandbox.

    The byte and binary limits are deliberately the *same* ones
    :class:`~reviewer.tools.read_file.ReadFile` applies. A file stage 2 refuses
    to parse is then exactly a file an agent would be refused too, so the
    analyzer never claims precision the agent cannot corroborate.
    """

    def __init__(self, sandbox: Sandbox, *, max_bytes: int = MAX_BYTES) -> None:
        self.sandbox = sandbox
        self.max_bytes = max_bytes

    def __repr__(self) -> str:
        return f"CheckoutSource({str(self.sandbox.root)!r})"

    def read(self, path: str) -> str | None:
        try:
            target = self.sandbox.resolve(path)
        except SandboxViolation as violation:
            logger.warning(
                "source_refused path=%s reason=%s",
                path,
                violation.reason,
                extra={
                    "event": "source_refused",
                    "path": path,
                    "reason": violation.reason,
                },
            )
            return None

        try:
            if not target.is_file():
                return None
            if target.stat().st_size > self.max_bytes:
                logger.info(
                    "source_too_large path=%s bytes=%d",
                    path,
                    target.stat().st_size,
                    extra={"event": "source_too_large", "path": path},
                )
                return None
            data = target.read_bytes()
        except OSError as exc:
            logger.warning(
                "source_unreadable path=%s error=%s",
                path,
                exc,
                extra={"event": "source_unreadable", "path": path},
            )
            return None

        if b"\x00" in data[:BINARY_SNIFF_BYTES]:
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            # Undecodable is indistinguishable from binary for parsing purposes.
            return None


# --------------------------------------------------------------------------
# Materializing the head revision
# --------------------------------------------------------------------------


@runtime_checkable
class Materializer(Protocol):
    """Fills an empty directory with the head revision of a pull request."""

    def materialize(self, pull_request: PullRequest, destination: Path) -> None: ...


class GitArchiveMaterializer:
    """Extracts `head_sha` out of a local clone, with no working-tree side effects.

    `git archive` streams a tar of one commit; the extraction happens here in
    Python with every member path validated through a :class:`Sandbox` anchored
    at the destination. That means a hostile archive entry (`../../etc/passwd`,
    an absolute path, a symlink pointing out of the tree) is refused by the same
    resolver the tools use rather than by ad-hoc checking — and it leaves no git
    metadata behind, so cleanup is an ordinary directory removal.

    A worktree or a plain `git checkout` would both be shorter and worse: the
    first leaves a `.git` file pointing at the parent repository inside the tree
    an agent can read, the second mutates the source repository's index.
    """

    def __init__(
        self,
        repo: str | Path,
        *,
        git: str = "git",
        timeout: float = 120.0,
    ) -> None:
        self.repo = Path(repo)
        self.git = git
        self.timeout = timeout

    def materialize(self, pull_request: PullRequest, destination: Path) -> None:
        if not pull_request.head_sha:
            raise WorkspaceError(
                f"{pull_request.repo}#{pull_request.number} has no head sha to check out"
            )

        archive = self._archive(pull_request.head_sha)
        sandbox = Sandbox(destination)
        written = 0

        with tarfile.open(fileobj=BytesIO(archive), mode="r|") as tar:
            for member in tar:
                if not (member.isfile() or member.isdir()):
                    # Symlinks, devices and hardlinks are not review material and
                    # are the classic archive-escape vectors. Skip them.
                    continue
                try:
                    target = sandbox.resolve(member.name)
                except SandboxViolation as violation:
                    logger.warning(
                        "workspace_member_refused name=%s reason=%s",
                        member.name,
                        violation.reason,
                        extra={
                            "event": "workspace_member_refused",
                            # not "name": that key is reserved on a LogRecord.
                            "member": member.name,
                            "reason": violation.reason,
                        },
                    )
                    continue

                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                extracted = tar.extractfile(member)
                if extracted is None:  # pragma: no cover - defensive
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(extracted.read())
                written += 1

        logger.info(
            "workspace_materialized repo=%s pr=%d sha=%s files=%d",
            pull_request.repo,
            pull_request.number,
            pull_request.head_sha[:12],
            written,
            extra={
                "event": "workspace_materialized",
                "repo": pull_request.repo,
                "pr": pull_request.number,
                "files": written,
            },
        )

    def _archive(self, sha: str) -> bytes:
        try:
            completed = subprocess.run(  # noqa: S603 - argument list, never a shell
                [self.git, "archive", "--format=tar", sha],
                cwd=str(self.repo),
                capture_output=True,
                timeout=self.timeout,
                shell=False,
                check=False,
            )
        except FileNotFoundError as exc:
            raise WorkspaceError(
                f"{self.git!r} is not installed or not on PATH; "
                "a head checkout cannot be materialized"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise WorkspaceError(
                f"git archive timed out after {self.timeout:g}s for {sha}"
            ) from exc

        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise WorkspaceError(
                f"git archive failed for {sha} in {self.repo}: {detail or 'unknown error'}"
            )
        return completed.stdout


class GitCloneMaterializer:
    """Clones the repository, then delegates to :class:`GitArchiveMaterializer`.

    The path for a review driven purely from the GitHub API, where no local clone
    exists. It reaches the network, so it is exercised only by opt-in tests; the
    extraction it delegates to is the fully covered one above.

    The clone is blobless and headless — the objects for one commit are fetched,
    not the project's history — and lands in a sibling directory of the
    workspace that is removed with it.
    """

    def __init__(
        self,
        clone_url: str,
        *,
        token: str | None = None,
        git: str = "git",
        timeout: float = 600.0,
    ) -> None:
        self.clone_url = clone_url
        self.token = token
        self.git = git
        self.timeout = timeout

    def materialize(self, pull_request: PullRequest, destination: Path) -> None:
        mirror = destination.parent / f"{destination.name}.git"
        mirror.mkdir(parents=True, exist_ok=True)
        self._run(["clone", "--no-checkout", "--filter=blob:none", self._url(), "."], mirror)
        self._run(["fetch", "--depth=1", "origin", pull_request.head_sha], mirror)
        GitArchiveMaterializer(mirror, git=self.git, timeout=self.timeout).materialize(
            pull_request, destination
        )

    def _url(self) -> str:
        if not self.token:
            return self.clone_url
        if self.clone_url.startswith("https://"):
            return self.clone_url.replace("https://", f"https://x-access-token:{self.token}@", 1)
        return self.clone_url

    def _run(self, args: list[str], cwd: Path) -> None:
        try:
            completed = subprocess.run(  # noqa: S603 - argument list, never a shell
                [self.git, *args],
                cwd=str(cwd),
                capture_output=True,
                timeout=self.timeout,
                shell=False,
                check=False,
            )
        except FileNotFoundError as exc:
            raise WorkspaceError(f"{self.git!r} is not installed or not on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise WorkspaceError(
                f"git {args[0]} timed out after {self.timeout:g}s"
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            # The URL may carry a token; never let it reach a log or a message.
            raise WorkspaceError(f"git {args[0]} failed: {_redact(detail)}")


def _redact(text: str) -> str:
    """Strip any embedded credential out of a git error message."""
    import re

    return re.sub(r"://[^@/\s]+@", "://***@", text)


# --------------------------------------------------------------------------
# The workspace
# --------------------------------------------------------------------------


class ReviewWorkspace:
    """One directory holding the head revision for one review.

    Lifecycle, deliberately explicit:

    * **created** per review by :func:`open_workspace`, as a fresh temp directory
      under the system temp root (or `REVIEWER_WORKSPACE_ROOT`);
    * **shared** for the length of that review — stage 2 parses out of it and the
      Phase 5 agents search the same root, so they cannot disagree about what the
      code says;
    * **removed** by :meth:`close`, or on leaving the context manager, including
      when materialization fails half-way. Nothing is reused between reviews and
      nothing survives one.

    Read-only is enforced by the things that read it — the Phase 2 tools never
    write — rather than by filesystem permissions, which would break cleanup on
    Windows for no gain.
    """

    def __init__(self, root: Path, *, owned: bool = True, label: str = "") -> None:
        self.sandbox = Sandbox(root)
        self.root = self.sandbox.root
        self.label = label
        self._owned = owned
        self._closed = False

    def __repr__(self) -> str:
        return f"ReviewWorkspace({str(self.root)!r}, label={self.label!r})"

    def source(self, *, max_bytes: int = MAX_BYTES) -> CheckoutSource:
        """A :class:`SourceProvider` reading this workspace through its sandbox."""
        return CheckoutSource(self.sandbox, max_bytes=max_bytes)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Remove the workspace. Idempotent, and never raises."""
        if self._closed:
            return
        self._closed = True
        if not self._owned:
            return
        _force_rmtree(self.root)
        for sibling in (self.root.parent / f"{self.root.name}.git",):
            if sibling.exists():
                _force_rmtree(sibling)
        logger.info(
            "workspace_closed root=%s label=%s",
            self.root,
            self.label,
            extra={"event": "workspace_closed", "label": self.label},
        )

    def __enter__(self) -> ReviewWorkspace:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def open_workspace(
    pull_request: PullRequest,
    materializer: Materializer | None = None,
    *,
    parent: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> ReviewWorkspace:
    """Create a workspace for one review and populate it with the head revision.

    With no ``materializer`` the workspace is created empty — a valid state: the
    source provider then answers ``None`` for everything and stage 2 falls back
    to the patch, exactly as it did before this phase.

    If materialization fails the directory is removed before the error
    propagates, so a failed review leaves nothing behind.
    """
    source_env = os.environ if env is None else env
    if parent is None:
        parent = source_env.get(WORKSPACE_ROOT_ENV_VAR) or None
    if parent is not None:
        Path(parent).mkdir(parents=True, exist_ok=True)

    label = f"{pull_request.repo}#{pull_request.number}"
    slug = label.replace("/", "-").replace("#", "-")
    root = Path(tempfile.mkdtemp(prefix=f"{WORKSPACE_PREFIX}{slug}-", dir=parent))

    logger.info(
        "workspace_opened root=%s label=%s sha=%s",
        root,
        label,
        (pull_request.head_sha or "")[:12],
        extra={"event": "workspace_opened", "label": label},
    )

    workspace = ReviewWorkspace(root, label=label)
    if materializer is None:
        return workspace

    try:
        materializer.materialize(pull_request, workspace.root)
    except BaseException:
        workspace.close()
        raise
    return workspace


def workspaces(
    pull_request: PullRequest, materializer: Materializer | None = None
) -> Iterator[ReviewWorkspace]:
    """Generator form, for callers that prefer `yield from` over `with`."""
    workspace = open_workspace(pull_request, materializer)
    try:
        yield workspace
    finally:
        workspace.close()


def _force_rmtree(root: Path) -> None:
    """Remove a tree, clearing read-only bits if the first attempt is blocked.

    Git writes read-only objects on some platforms, and Windows refuses to unlink
    a read-only file. Chmod-and-retry rather than leaving a temp directory behind.
    """
    if not root.exists():
        return
    try:
        shutil.rmtree(root)
        return
    except OSError:
        pass

    for path in sorted(root.rglob("*"), reverse=True):
        try:
            path.chmod(stat.S_IWRITE | stat.S_IREAD)
        except OSError:  # pragma: no cover - best effort
            pass
    try:
        shutil.rmtree(root)
    except OSError as exc:  # pragma: no cover - a locked file elsewhere
        logger.warning(
            "workspace_cleanup_failed root=%s error=%s",
            root,
            exc,
            extra={"event": "workspace_cleanup_failed", "root": str(root)},
        )
