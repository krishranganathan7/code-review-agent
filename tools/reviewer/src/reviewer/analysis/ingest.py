"""Stage 1 — fetch the PR: diff, metadata, base/head, repo config.

This is the only place in the pipeline that talks to GitHub. Everything
downstream consumes the normalized :class:`PullRequest` and never reaches the
network again.

Two seams keep that honest:

* :class:`GitHubTransport` is a narrow protocol — one authenticated ``GET``. The
  real implementation uses the standard library; tests script a fake. Swapping in
  an HTTP client later touches nothing but the transport.
* every failure becomes a typed :class:`IngestError` subclass naming the cause.
  A revoked token, a bad PR number and a rate limit are three different
  problems with three different fixes, and the caller is told which it has —
  never a raw traceback, never a bare ``KeyError`` on a JSON field.

Credentials come from the environment (``GITHUB_TOKEN``), never from source.
"""

from __future__ import annotations

import json
import logging
import os
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

__all__ = [
    "ChangedFile",
    "RepoConfig",
    "PullRequest",
    "IngestError",
    "PullRequestNotFound",
    "RepositoryNotFound",
    "CredentialsRejected",
    "InsufficientPermissions",
    "RateLimited",
    "GitHubUnavailable",
    "MalformedResponse",
    "GitHubResponse",
    "GitHubTransport",
    "UrllibTransport",
    "GitHubIngest",
    "fetch_pull_request",
    "TOKEN_ENV_VAR",
    "CONFIG_PATH",
    "FILE_STATUSES",
]

logger = logging.getLogger("reviewer.analysis.ingest")

TOKEN_ENV_VAR = "GITHUB_TOKEN"
"""Where the credential comes from. Never read into source, never logged."""

CONFIG_PATH = ".reviewer.toml"
"""Optional per-repository configuration, read from the head commit."""

DEFAULT_API_ROOT = "https://api.github.com"
MAX_FILE_PAGES = 30
"""GitHub caps `/files` at 3000 entries; stop there rather than paging forever."""

FILE_STATUSES = (
    "added",
    "modified",
    "removed",
    "renamed",
    "copied",
    "changed",
    "unchanged",
)


# --------------------------------------------------------------------------
# Typed failures
# --------------------------------------------------------------------------


class IngestError(RuntimeError):
    """Base class for every ingest failure. Always names the cause."""


class RepositoryNotFound(IngestError):
    """The repository does not exist, or the token cannot see it."""


class PullRequestNotFound(IngestError):
    """The pull request number does not exist in that repository."""


class CredentialsRejected(IngestError):
    """The token was rejected (missing, malformed, expired or revoked)."""


class InsufficientPermissions(IngestError):
    """The token authenticated but lacks the scope this read requires."""


class RateLimited(IngestError):
    """GitHub is rate-limiting the token.

    ``reset_at`` is the epoch second the limit resets, when GitHub said so.
    """

    def __init__(self, message: str, reset_at: int | None = None) -> None:
        super().__init__(message)
        self.reset_at = reset_at


class GitHubUnavailable(IngestError):
    """GitHub could not be reached, or answered with a server error."""


class MalformedResponse(IngestError):
    """GitHub answered successfully but the payload was not what we expect."""


# --------------------------------------------------------------------------
# The normalized model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangedFile:
    """One file the pull request touches.

    ``patch`` is the per-file unified diff. GitHub omits it for binary files and
    for very large diffs, so it may legitimately be empty — ``has_patch`` says
    which, and the change analyzer degrades rather than guessing.
    """

    path: str
    status: str
    additions: int = 0
    deletions: int = 0
    patch: str = ""
    previous_path: str | None = None

    @property
    def has_patch(self) -> bool:
        return bool(self.patch)

    @property
    def changed_lines(self) -> int:
        return self.additions + self.deletions

    @property
    def is_rename(self) -> bool:
        return self.status in {"renamed", "copied"} or self.previous_path is not None

    @property
    def directory(self) -> str:
        """POSIX-style parent directory, or ``""`` for a root-level file."""
        head, _, _ = self.path.rpartition("/")
        return head


@dataclass(frozen=True)
class RepoConfig:
    """Repository-level facts and opt-in configuration.

    ``settings`` is the parsed `.reviewer.toml` if the repo has one, otherwise
    empty. It is held as raw data here: ingest does not interpret it, and the
    stages that own each section read their own keys (the risk engine reads
    ``[risk]``, the policy engine will read ``[policy]``). That keeps stage 1
    free of downstream policy.
    """

    repo: str
    default_branch: str = "main"
    private: bool = False
    settings: dict[str, Any] = field(default_factory=dict)

    def section(self, name: str) -> dict[str, Any]:
        """One top-level config table, or ``{}`` when absent or malformed."""
        value = self.settings.get(name)
        return dict(value) if isinstance(value, dict) else {}


@dataclass(frozen=True)
class PullRequest:
    """Everything the rest of the pipeline needs about one pull request.

    Nothing downstream touches GitHub again: if a stage needs a fact about the
    PR, it belongs here.
    """

    repo: str
    number: int
    title: str
    description: str
    author: str
    base_ref: str
    head_ref: str
    base_sha: str
    head_sha: str
    files: list[ChangedFile] = field(default_factory=list)
    diff: str = ""
    config: RepoConfig | None = None
    draft: bool = False
    files_truncated: bool = False

    @property
    def additions(self) -> int:
        return sum(file.additions for file in self.files)

    @property
    def deletions(self) -> int:
        return sum(file.deletions for file in self.files)

    @property
    def changed_lines(self) -> int:
        return self.additions + self.deletions

    @property
    def paths(self) -> list[str]:
        return [file.path for file in self.files]

    def file(self, path: str) -> ChangedFile | None:
        for changed in self.files:
            if changed.path == path:
                return changed
        return None

    def settings(self, section: str) -> dict[str, Any]:
        """A repo-config section, empty when the repo has no configuration."""
        return self.config.section(section) if self.config else {}


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GitHubResponse:
    """A raw HTTP answer, in the little of it that ingest needs."""

    status: int
    body: str
    headers: Mapping[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        try:
            return json.loads(self.body)
        except json.JSONDecodeError as exc:
            raise MalformedResponse(
                f"GitHub returned a {self.status} that is not valid JSON: {exc}"
            ) from exc

    def header(self, name: str) -> str | None:
        """Case-insensitive header lookup."""
        lowered = name.lower()
        for key, value in self.headers.items():
            if key.lower() == lowered:
                return value
        return None


@runtime_checkable
class GitHubTransport(Protocol):
    """One authenticated GET against the GitHub REST API.

    Deliberately narrow: ingest only reads. A transport that could write would
    be a capability this pipeline has no use for (§2.7's spirit — least
    privilege — applied to the network edge).
    """

    def get(self, path: str, *, accept: str | None = None) -> GitHubResponse: ...


class UrllibTransport:
    """The real transport, on the standard library.

    Only GETs are ever issued, so this needs no HTTP client dependency. The
    protocol above is the seam if that ever changes.
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        api_root: str = DEFAULT_API_ROOT,
        timeout: float = 30.0,
        env: Mapping[str, str] | None = None,
    ) -> None:
        source = os.environ if env is None else env
        self.api_root = api_root.rstrip("/")
        self.timeout = timeout
        self._token = token if token is not None else source.get(TOKEN_ENV_VAR)

    def get(self, path: str, *, accept: str | None = None) -> GitHubResponse:
        url = path if path.startswith("http") else f"{self.api_root}{path}"
        request = urllib.request.Request(url, method="GET")
        request.add_header("Accept", accept or "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        request.add_header("User-Agent", "reviewer-agent")
        if self._token:
            request.add_header("Authorization", f"Bearer {self._token}")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                return GitHubResponse(
                    status=response.status,
                    body=body,
                    headers=dict(response.headers.items()),
                )
        except urllib.error.HTTPError as exc:
            # An HTTP error is a real answer: return it so the caller can map the
            # status onto a typed error rather than guessing from an exception.
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            return GitHubResponse(
                status=exc.code, body=body, headers=dict(exc.headers.items())
            )
        except urllib.error.URLError as exc:
            raise GitHubUnavailable(f"could not reach GitHub at {url}: {exc.reason}") from exc
        except OSError as exc:  # pragma: no cover - platform-dependent
            raise GitHubUnavailable(f"could not reach GitHub at {url}: {exc}") from exc


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------


class GitHubIngest:
    """Stage 1. Turns a repo plus a PR number into a :class:`PullRequest`."""

    def __init__(self, transport: GitHubTransport) -> None:
        self.transport = transport

    def fetch(self, repo: str, number: int) -> PullRequest:
        """Fetch and normalize one pull request.

        Raises an :class:`IngestError` subclass — never anything else — when the
        PR cannot be read.
        """
        _validate_target(repo, number)
        logger.info(
            "ingest_start repo=%s pr=%d",
            repo,
            number,
            extra={"event": "ingest_start", "repo": repo, "pr": number},
        )

        metadata = self._pull_request_metadata(repo, number)
        files, truncated = self._changed_files(repo, number)
        diff = self._diff(repo, number)
        head_sha = _string(metadata, ("head", "sha"))
        config = self._repo_config(repo, head_sha)

        pull_request = PullRequest(
            repo=repo,
            number=number,
            title=_string(metadata, ("title",)),
            description=_string(metadata, ("body",)),
            author=_string(metadata, ("user", "login")),
            base_ref=_string(metadata, ("base", "ref")),
            head_ref=_string(metadata, ("head", "ref")),
            base_sha=_string(metadata, ("base", "sha")),
            head_sha=head_sha,
            files=files,
            diff=diff,
            config=config,
            draft=bool(metadata.get("draft", False)),
            files_truncated=truncated,
        )

        logger.info(
            "ingest_done repo=%s pr=%d files=%d additions=%d deletions=%d",
            repo,
            number,
            len(files),
            pull_request.additions,
            pull_request.deletions,
            extra={
                "event": "ingest_done",
                "repo": repo,
                "pr": number,
                "files": len(files),
                "files_truncated": truncated,
                "has_diff": bool(diff),
            },
        )
        return pull_request

    # -- individual reads -------------------------------------------------

    def _pull_request_metadata(self, repo: str, number: int) -> dict[str, Any]:
        response = self.transport.get(f"/repos/{repo}/pulls/{number}")
        self._raise_for_status(response, repo=repo, number=number)
        payload = response.json()
        if not isinstance(payload, dict):
            raise MalformedResponse(
                f"expected an object for {repo}#{number}, got {type(payload).__name__}"
            )
        return payload

    def _changed_files(self, repo: str, number: int) -> tuple[list[ChangedFile], bool]:
        files: list[ChangedFile] = []
        page = 1
        while page <= MAX_FILE_PAGES:
            response = self.transport.get(
                f"/repos/{repo}/pulls/{number}/files?per_page=100&page={page}"
            )
            self._raise_for_status(response, repo=repo, number=number)
            payload = response.json()
            if not isinstance(payload, list):
                raise MalformedResponse(
                    f"expected a list of files for {repo}#{number}, "
                    f"got {type(payload).__name__}"
                )
            files.extend(_changed_file(entry) for entry in payload if isinstance(entry, dict))
            if len(payload) < 100:
                return files, False
            page += 1
        return files, True

    def _diff(self, repo: str, number: int) -> str:
        response = self.transport.get(
            f"/repos/{repo}/pulls/{number}", accept="application/vnd.github.v3.diff"
        )
        if response.status == 406:
            # GitHub refuses to render a diff beyond ~20k lines. The per-file
            # patches still arrived, so this is a degradation, not a failure.
            logger.warning(
                "ingest_diff_too_large repo=%s pr=%d",
                repo,
                number,
                extra={"event": "ingest_diff_too_large", "repo": repo, "pr": number},
            )
            return ""
        self._raise_for_status(response, repo=repo, number=number)
        return response.body

    def _repo_config(self, repo: str, ref: str) -> RepoConfig:
        response = self.transport.get(f"/repos/{repo}")
        self._raise_for_status(response, repo=repo, number=None)
        payload = response.json()
        if not isinstance(payload, dict):
            raise MalformedResponse(f"expected an object for repo {repo}")

        return RepoConfig(
            repo=repo,
            default_branch=str(payload.get("default_branch") or "main"),
            private=bool(payload.get("private", False)),
            settings=self._settings(repo, ref),
        )

    def _settings(self, repo: str, ref: str) -> dict[str, Any]:
        """Read `.reviewer.toml` from the head commit. Absent is normal."""
        path = f"/repos/{repo}/contents/{CONFIG_PATH}"
        if ref:
            path = f"{path}?ref={ref}"
        response = self.transport.get(path, accept="application/vnd.github.raw")

        if response.status == 404:
            return {}
        self._raise_for_status(response, repo=repo, number=None)

        try:
            return tomllib.loads(response.body)
        except tomllib.TOMLDecodeError as exc:
            # Malformed configuration must not take the review down: fall back to
            # defaults and say so loudly.
            logger.warning(
                "ingest_bad_config repo=%s error=%s",
                repo,
                exc,
                extra={"event": "ingest_bad_config", "repo": repo, "error": str(exc)},
            )
            return {}

    # -- status mapping ---------------------------------------------------

    @staticmethod
    def _raise_for_status(
        response: GitHubResponse, *, repo: str, number: int | None
    ) -> None:
        """Map an HTTP status onto a typed error naming the cause."""
        if 200 <= response.status < 300:
            return

        target = f"{repo}#{number}" if number is not None else repo
        detail = _api_message(response)

        if response.status == 401:
            raise CredentialsRejected(
                f"GitHub rejected the credential reading {target}: {detail}. "
                f"Check {TOKEN_ENV_VAR} — it is missing, malformed, expired or revoked."
            )

        if response.status in (403, 429):
            remaining = response.header("x-ratelimit-remaining")
            looks_limited = (
                response.status == 429
                or remaining == "0"
                or "rate limit" in detail.lower()
                or "secondary rate" in detail.lower()
            )
            if looks_limited:
                raise RateLimited(
                    f"GitHub is rate-limiting this token reading {target}: {detail}.",
                    reset_at=_int_header(response, "x-ratelimit-reset"),
                )
            raise InsufficientPermissions(
                f"the credential lacks permission to read {target}: {detail}. "
                "A pull-request review needs read access to the repository's "
                "contents and pull requests."
            )

        if response.status == 404:
            if number is None:
                raise RepositoryNotFound(
                    f"repository {repo!r} does not exist, or this credential "
                    f"cannot see it: {detail}."
                )
            raise PullRequestNotFound(
                f"pull request {target} does not exist: {detail}."
            )

        if response.status >= 500:
            raise GitHubUnavailable(
                f"GitHub returned {response.status} reading {target}: {detail}."
            )

        raise IngestError(
            f"GitHub returned {response.status} reading {target}: {detail}."
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _validate_target(repo: str, number: int) -> None:
    if not isinstance(number, int) or isinstance(number, bool) or number < 1:
        raise PullRequestNotFound(
            f"pull request number must be a positive integer, got {number!r}."
        )
    parts = repo.split("/")
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise RepositoryNotFound(
            f"repository must be given as 'owner/name', got {repo!r}."
        )


def _api_message(response: GitHubResponse) -> str:
    """GitHub's own explanation, when it gave one."""
    try:
        payload = json.loads(response.body)
    except (json.JSONDecodeError, TypeError):
        return response.body.strip()[:200] or f"HTTP {response.status}"
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, str) and message:
            return message
    return f"HTTP {response.status}"


def _int_header(response: GitHubResponse, name: str) -> int | None:
    raw = response.header(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _string(payload: Mapping[str, Any], path: Sequence[str]) -> str:
    """Read a nested string, tolerating nulls and missing keys.

    GitHub sends ``"body": null`` for an empty description and can omit a user
    on a deleted account. A missing field is empty, not a crash.
    """
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return ""
        current = current.get(key)
    return current if isinstance(current, str) else ""


def _changed_file(entry: Mapping[str, Any]) -> ChangedFile:
    status = str(entry.get("status") or "modified")
    return ChangedFile(
        path=str(entry.get("filename") or ""),
        status=status if status in FILE_STATUSES else "modified",
        additions=_non_negative(entry.get("additions")),
        deletions=_non_negative(entry.get("deletions")),
        patch=str(entry.get("patch") or ""),
        previous_path=(
            str(entry["previous_filename"])
            if isinstance(entry.get("previous_filename"), str)
            else None
        ),
    )


def _non_negative(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(number, 0)


def fetch_pull_request(
    repo: str,
    number: int,
    *,
    token: str | None = None,
    api_root: str = DEFAULT_API_ROOT,
    env: Mapping[str, str] | None = None,
) -> PullRequest:
    """Convenience entry point: build the real transport and fetch."""
    transport = UrllibTransport(token, api_root=api_root, env=env)
    return GitHubIngest(transport).fetch(repo, number)
