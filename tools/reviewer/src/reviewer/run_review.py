"""The entry point: a repository and a PR number in, a verdict out.

This is what a CI job or a CLI actually calls, and it is the last connective
piece — everything it does is sequencing stages that already exist:

```
ingest  →  materialize the head workspace  →  pipeline  →  publish  →  close
```

It lives in its own module rather than in `pipeline.py` because it owns two
concerns the orchestrator deliberately does not: **credentials** and
**workspace lifetime**. `ReviewPipeline` takes a provider, tools and a source
and reviews what it is given; this decides where those come from and guarantees
the workspace is removed afterwards.

**Cleanup is guaranteed.** The workspace is opened in a `with` block, so it is
removed on success, on a typed failure, on an unexpected exception, and on
`KeyboardInterrupt`. A review that dies half-way leaves nothing on disk.

**Failures are typed.** Every stage already raises a specific error naming the
cause — `PullRequestNotFound`, `CredentialsRejected`, `RateLimited`,
`WorkspaceError`, `ProviderError`, `PublishError`. Those propagate unchanged
because they are already the clean answer. Anything unexpected is wrapped in
:class:`ReviewError` so a caller can catch one type, and :func:`main` turns each
into a one-line message and an exit code — never a traceback.

**Least privilege.** See :data:`REQUIRED_SCOPES`. Read-only review needs no
write access at all; posting needs exactly one write scope, and the entry point
refuses to pretend it can publish without a credential for it.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .agents.selection import available_agents
from .analysis.ingest import (
    TOKEN_ENV_VAR,
    GitHubIngest,
    IngestError,
    PullRequest,
    UrllibTransport,
)
from .analysis.source import (
    GitArchiveMaterializer,
    GitCloneMaterializer,
    Materializer,
    ReviewWorkspace,
    WorkspaceError,
    open_workspace,
)
from .findings.judge import Judge
from .languages import default_adapters
from .pipeline import ReviewOutcome, ReviewPipeline
from .policy.engine import GateDecision
from .providers.base import LLMProvider, ProviderError
from .providers.factory import build_provider
from .publish.github import (
    GitHubPublisher,
    PublishError,
    ReviewTransport,
    UrllibReviewTransport,
)
from .tools import build_toolset
from .tools.sandbox import SandboxViolation
from .trace import ReviewReport, Stopwatch, build_report, review_context

__all__ = [
    "ReviewError",
    "ReviewSettings",
    "ReviewRun",
    "run_review",
    "main",
    "REQUIRED_SCOPES",
    "REVIEW_FAILURES",
    "EXIT_CODES",
]

logger = logging.getLogger("reviewer.run")


class ReviewError(RuntimeError):
    """An unexpected failure during a review, reported instead of a traceback."""


REVIEW_FAILURES: tuple[type[Exception], ...] = (
    IngestError,
    WorkspaceError,
    ProviderError,
    PublishError,
    SandboxViolation,
    ReviewError,
)
"""Every failure a caller should expect. One `except` clause covers a review."""


REQUIRED_SCOPES: dict[str, str] = {
    "contents:read": (
        "read the repository at head, to materialize the review workspace"
    ),
    "pull_requests:read": (
        "read the pull request: its metadata, changed files and diff"
    ),
    "pull_requests:write": (
        "post the review and its inline comments. NOT required for a dry run "
        "(--no-publish), which is the default"
    ),
}
"""The complete set of GitHub permissions the reviewer uses, and nothing more.

A fine-grained token needs `Contents: Read` and `Pull requests: Read` to review,
plus `Pull requests: Read and write` only to post. On a classic token that is
`repo:status` + `public_repo` at most — and for a private repository, `repo`,
which is broader than the reviewer needs and worth avoiding where fine-grained
tokens are available.

Deliberately absent: no `workflow`, no `admin`, no `write:packages`, no
`delete_repo`, no org or member scopes, no ability to merge, approve, dismiss a
review, or change branch protection. The gate's verdict is *reported*; a human or
a branch-protection rule acts on it, so the reviewer never needs the authority to
merge — and therefore should never hold it.

Bot-triggered events get no implicit trust: a review runs the same way whoever
opened the pull request, and no author, label or comment can change the gate
(see `policy/engine.py`, which reads severities and policy only).
"""


EXIT_CODES: dict[str, int] = {
    "pass": 0,
    "warn": 0,
    "block": 1,
    "error": 2,
}
"""Process exit codes. A blocking review fails the job; a warning does not."""


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewSettings:
    """Everything the entry point needs that is not the PR itself."""

    model: str | None = None
    """A `"<provider>/<model>"` spec. `None` reads `REVIEWER_MODEL`."""
    judge_model: str | None = None
    """A separate spec for the judge. `None` reuses `model`.

    Exists because the judge is one tool-free call with a fixed contract, so it
    is the cheapest stage to move onto a different provider — running it on
    `claude-code/sonnet` while the agents stay on an API key, or the reverse."""
    clone: str | None = None
    """A local clone to materialize head from. Faster than cloning."""
    clone_url: str | None = None
    """Where to clone from when no local clone is given. Requires network."""
    api_root: str = "https://api.github.com"
    publish: bool = False
    """Off by default: a review that cannot post needs no write scope."""
    judge: bool = True
    max_workers: int = 4
    finding_cap: int | None = 50
    workspace_root: str | None = None
    review_id: str | None = None

    def token(self, env: Mapping[str, str] | None = None) -> str | None:
        source = os.environ if env is None else env
        return source.get(TOKEN_ENV_VAR)


@dataclass(frozen=True)
class ReviewRun:
    """The result of one end-to-end review."""

    outcome: ReviewOutcome
    report: ReviewReport
    posted: bool = False
    workspace_root: str = ""
    """Where the workspace was. It no longer exists by the time you read this."""

    @property
    def decision(self) -> GateDecision:
        return self.outcome.decision

    @property
    def verdict(self) -> str:
        return self.outcome.decision.outcome

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.verdict, EXIT_CODES["error"])


# --------------------------------------------------------------------------
# The entry point
# --------------------------------------------------------------------------


def run_review(
    repo: str,
    number: int,
    settings: ReviewSettings | None = None,
    *,
    provider: LLMProvider | None = None,
    ingest: GitHubIngest | None = None,
    materializer: Materializer | None = None,
    publish_transport: ReviewTransport | None = None,
    pull_request: PullRequest | None = None,
    env: Mapping[str, str] | None = None,
    **pipeline_kwargs: Any,
) -> ReviewRun:
    """Review one pull request, end to end.

    Every collaborator can be injected, which is how the default test path runs
    with fixtures and a scripted provider and no network. Left unset, each is
    built from the environment.

    Raises one of :data:`REVIEW_FAILURES`. Never leaks a workspace.
    """
    config = settings or ReviewSettings()
    watch = Stopwatch()

    with review_context(config.review_id) as review_id:
        logger.info(
            "run_start review=%s repo=%s pr=%d publish=%s",
            review_id,
            repo,
            number,
            config.publish,
            extra={
                "event": "run_start",
                "repo": repo,
                "pr": number,
                "publish": config.publish,
            },
        )

        # Stage 1 — ingest. Typed errors propagate unchanged.
        with watch.stage("ingest"):
            request = pull_request or _ingest(repo, number, config, ingest, env)

        provider = provider or _provider(config)
        judge_provider = (
            build_provider(config.judge_model)
            if config.judge_model and provider is not None
            else provider
        )
        publisher = _publisher(config, publish_transport, env)
        chosen = materializer if materializer is not None else _materializer(config)

        # The workspace is opened inside the guard and removed by the `with`,
        # whatever happens — success, typed failure, an unexpected crash during
        # materialization itself, or an interrupt. `open_workspace` removes a
        # half-populated directory before re-raising, so nothing is left behind
        # on either path.
        root = ""
        try:
            with watch.stage("workspace"):
                workspace = open_workspace(
                    request, chosen, parent=config.workspace_root, env=env
                )
            root = str(workspace.root)
            with workspace:
                run = _review(
                    request,
                    workspace,
                    provider,
                    judge_provider,
                    publisher,
                    config,
                    watch,
                    review_id,
                    pipeline_kwargs,
                )
        except REVIEW_FAILURES:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise ReviewError(
                f"review of {repo}#{number} failed unexpectedly: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        logger.info(
            "run_done review=%s repo=%s pr=%d outcome=%s findings=%d",
            review_id,
            repo,
            number,
            run.verdict,
            len(run.outcome.findings),
            extra={
                "event": "run_done",
                "repo": repo,
                "pr": number,
                "outcome": run.verdict,
                "findings": len(run.outcome.findings),
            },
        )
        return ReviewRun(
            outcome=run.outcome,
            report=run.report,
            posted=run.posted,
            workspace_root=root,
        )


def _review(
    request: PullRequest,
    workspace: ReviewWorkspace,
    provider: LLMProvider,
    judge_provider: LLMProvider,
    publisher: GitHubPublisher,
    config: ReviewSettings,
    watch: Stopwatch,
    review_id: str,
    pipeline_kwargs: Mapping[str, Any],
) -> ReviewRun:
    """The part that runs inside the open workspace."""
    source = workspace.source()
    # The patches make `read_patch` available, which is how the agent obtains
    # the diff: it is no longer inlined into the task prompt.
    tools = build_toolset(
        workspace.sandbox,
        patches={f.path: f.patch or "" for f in request.files},
        statuses={f.path: f.status for f in request.files},
        counts={f.path: (f.additions, f.deletions) for f in request.files},
    )

    pipeline = ReviewPipeline(
        provider,
        tools=tools,
        source=source,
        adapters=default_adapters(),
        agents=available_agents(),
        judge=Judge(judge_provider, source=source) if config.judge else None,
        publisher=publisher,
        finding_cap=config.finding_cap,
        max_workers=config.max_workers,
        **dict(pipeline_kwargs),
    )

    with watch.stage("pipeline"):
        outcome = pipeline.run(request, publish=config.publish)

    with watch.stage("trace"):
        report = build_report(outcome, review_id=review_id, timings=watch.timings)

    return ReviewRun(outcome=outcome, report=report, posted=config.publish)


# --------------------------------------------------------------------------
# Building the collaborators
# --------------------------------------------------------------------------


def _ingest(
    repo: str,
    number: int,
    config: ReviewSettings,
    ingest: GitHubIngest | None,
    env: Mapping[str, str] | None,
) -> PullRequest:
    reader = ingest or GitHubIngest(
        UrllibTransport(api_root=config.api_root, env=env)
    )
    return reader.fetch(repo, number)


def _provider(config: ReviewSettings) -> LLMProvider:
    return build_provider(config.model)


def _materializer(config: ReviewSettings) -> Materializer | None:
    """How to fill the workspace, from what the settings offer.

    With neither a local clone nor a URL, the workspace is left empty: the review
    still runs, symbol attribution falls back to the patch, and the agents' tools
    find nothing. Useful for a smoke test, not for a real review — so it is
    logged as the degradation it is.
    """
    if config.clone:
        return GitArchiveMaterializer(config.clone)
    if config.clone_url:
        return GitCloneMaterializer(config.clone_url, token=config.token())
    logger.warning(
        "run_no_workspace_source reason=no clone or clone_url configured",
        extra={"event": "run_no_workspace_source"},
    )
    return None


def _publisher(
    config: ReviewSettings,
    transport: ReviewTransport | None,
    env: Mapping[str, str] | None,
) -> GitHubPublisher:
    """A publisher, with a transport only if we are actually posting.

    Least privilege in practice: a dry run never constructs a write-capable
    transport, so it cannot post even by mistake. Asking to publish without a
    credential fails now, with the scope named, rather than at the last step of a
    review that has already been paid for.
    """
    if not config.publish:
        return GitHubPublisher()
    if transport is not None:
        return GitHubPublisher(transport)
    if not config.token(env):
        raise PublishError(
            f"publishing needs {TOKEN_ENV_VAR} with the "
            "'pull_requests:write' permission; set it, or run without --publish"
        )
    return GitHubPublisher(
        UrllibReviewTransport(api_root=config.api_root, env=env)
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reviewer",
        description="Review a pull request and report evidence-backed findings.",
    )
    # Optional at the parser level so `--scopes` can be asked on its own; the
    # entry point requires them for an actual review. Marking them required here
    # made `--scopes` -- documented as "print the permissions and exit" --
    # impossible to run without naming a pull request it was never going to read.
    parser.add_argument("repo", nargs="?", help="owner/name")
    parser.add_argument("number", nargs="?", type=int, help="pull request number")
    parser.add_argument("--model", help='provider spec, e.g. "anthropic/claude-opus-5"')
    parser.add_argument(
        "--judge-model",
        help="separate provider spec for the judge; defaults to --model",
    )
    parser.add_argument("--clone", help="path to a local clone to check head out of")
    parser.add_argument("--clone-url", help="clone from this URL instead (network)")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="post the review (needs pull_requests:write). Off by default.",
    )
    parser.add_argument(
        "--no-judge", action="store_true", help="skip the independent judge"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--cap", type=int, default=50, help="maximum findings to publish"
    )
    parser.add_argument(
        "--trace", metavar="PATH", help="write the review trace to this file as JSON"
    )
    parser.add_argument(
        "--scopes",
        action="store_true",
        help="print the GitHub permissions the reviewer needs, and exit",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a review from the command line. Returns a process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)

    if args.scopes:
        print("GitHub permissions the reviewer needs:\n")
        for scope, why in REQUIRED_SCOPES.items():
            print(f"  {scope:24} {why}")
        print(f"\nCredential is read from {TOKEN_ENV_VAR}. Nothing else is used.")
        return 0

    if args.repo is None or args.number is None:
        # argparse cannot enforce this: the positionals are optional so that
        # `--scopes` can be asked on its own. Fail here, with usage, rather than
        # letting `None` reach ingest and surface as "not a positive integer".
        parser.error("repo and number are required (unless --scopes is given)")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s [%(review_id)s] %(message)s",
    )
    from .trace import install_review_id_logging

    install_review_id_logging()

    settings = ReviewSettings(
        model=args.model,
        judge_model=args.judge_model,
        clone=args.clone,
        clone_url=args.clone_url,
        publish=args.publish,
        judge=not args.no_judge,
        max_workers=args.workers,
        finding_cap=args.cap,
    )

    try:
        run = run_review(args.repo, args.number, settings)
    except REVIEW_FAILURES as exc:
        # A named cause and a non-zero exit. Never a traceback.
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_CODES["error"]

    print(run.report.why())
    print()
    print(run.outcome.rendered.summary if run.outcome.rendered else "")

    if args.trace:
        with open(args.trace, "w", encoding="utf-8") as handle:
            handle.write(run.report.to_json())
        print(f"\ntrace written to {args.trace}")

    return run.exit_code


if __name__ == "__main__":  # pragma: no cover - process entry
    raise SystemExit(main())
