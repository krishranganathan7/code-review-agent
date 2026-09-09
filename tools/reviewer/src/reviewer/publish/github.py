"""Stage 10 — post evidence-backed findings to GitHub.

The artifact is the **findings**, not a score (CLAUDE.md §9). Every published
finding carries where it is, how bad it is, what supports it, whether a
deterministic check corroborated it, and how confident the review is. A reader
who disagrees can check the evidence; a number offers nothing to check.

**Untrusted content is escaped here, at the boundary.** A finding's `message`
and `evidence` were written by a model, from repository content it read. By the
time they are rendered into Markdown they are being handed to a system that
interprets Markdown and HTML, so this module neutralizes both: fences cannot be
broken out of, HTML cannot be injected, and `@mentions` cannot be turned into
notifications. That is done on the way *out*, once, rather than trusted to have
been done upstream.

Posting goes through a narrow :class:`ReviewTransport`, so rendering is testable
with no network and the real HTTP path is one small class.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ..findings.finding import Finding
from ..policy.engine import GateDecision

__all__ = [
    "InlineComment",
    "RenderedReview",
    "ReviewRenderer",
    "ReviewTransport",
    "RecordingTransport",
    "UrllibReviewTransport",
    "GitHubPublisher",
    "PublishError",
    "escape_untrusted",
    "OUTCOME_HEADLINE",
    "MAX_FIELD_CHARS",
]

logger = logging.getLogger("reviewer.publish")

MAX_FIELD_CHARS = 4000
"""Cap on any single rendered field. A review comment is not a place for a novel."""

OUTCOME_HEADLINE = {
    "block": "Changes requested — blocking findings",
    "warn": "Reviewed with findings",
    "pass": "No blocking findings",
}

VERIFICATION_NOTE = {
    "VERIFIED": "verified by a deterministic check",
    "UNVERIFIED": "not deterministically verifiable — rests on the reviewer's reasoning",
    "REFUTED": "refuted",
}

TOKEN_ENV_VAR = "GITHUB_TOKEN"


class PublishError(RuntimeError):
    """The review could not be posted."""


# --------------------------------------------------------------------------
# Escaping untrusted content
# --------------------------------------------------------------------------

_HTML = re.compile(r"[<>]")
_MENTION = re.compile(r"(?<![\w`])([@#])(?=[A-Za-z0-9_-])")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def escape_untrusted(text: str, *, limit: int = MAX_FIELD_CHARS) -> str:
    """Neutralize model- and repository-authored text for Markdown output.

    Four things are defused, in order:

    * **backticks** — collapsed to a lookalike, so no fence or inline span can be
      closed early and no raw HTML block can be opened after it;
    * **angle brackets** — entity-encoded, so no HTML tag survives;
    * **``@`` and ``#`` prefixes** — zero-width-joined, so a mention or issue
      reference cannot notify a person or cross-link a ticket from inside a
      finding;
    * **control characters** — removed.

    The text is then capped. Escaping is idempotent in the sense that matters:
    running it twice cannot produce active markup.
    """
    cleaned = _CONTROL.sub("", text or "")
    # A lookalike rather than a backslash escape: backslash-escaping backticks
    # is unreliable inside fenced blocks, where escapes are not processed.
    cleaned = cleaned.replace("`", "ˋ")
    cleaned = _HTML.sub(lambda match: "&lt;" if match.group() == "<" else "&gt;", cleaned)
    cleaned = _MENTION.sub(lambda match: f"{match.group(1)}​", cleaned)

    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned


# --------------------------------------------------------------------------
# The rendered review
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InlineComment:
    """One finding, positioned on a line of the diff."""

    path: str
    line: int
    body: str
    severity: str
    start_line: int | None = None

    def payload(self) -> dict[str, Any]:
        """The GitHub review-comment shape."""
        comment: dict[str, Any] = {
            "path": self.path,
            "line": self.line,
            "side": "RIGHT",
            "body": self.body,
        }
        if self.start_line is not None and self.start_line < self.line:
            comment["start_line"] = self.start_line
            comment["start_side"] = "RIGHT"
        return comment


@dataclass(frozen=True)
class RenderedReview:
    """Everything to be posted, rendered and escaped, ready to send."""

    outcome: str
    summary: str
    body: str
    comments: list[InlineComment] = field(default_factory=list)
    event: str = "COMMENT"
    """GitHub review event. Always COMMENT — see `GitHubPublisher`."""

    def payload(self) -> dict[str, Any]:
        return {
            "body": self.body,
            "event": self.event,
            "comments": [comment.payload() for comment in self.comments],
        }


class ReviewRenderer:
    """Turns a gate decision into evidence-backed review output."""

    def __init__(self, *, max_inline_comments: int = 25) -> None:
        self.max_inline_comments = max_inline_comments

    def render(
        self,
        decision: GateDecision,
        *,
        repo: str = "",
        number: int = 0,
        notes: Sequence[str] = (),
    ) -> RenderedReview:
        findings = decision.findings
        summary = self._summary(decision)
        body = self._body(decision, summary, notes)
        comments = [
            self._comment(finding)
            for finding in findings[: self.max_inline_comments]
        ]

        logger.info(
            "publish_render repo=%s pr=%d outcome=%s findings=%d comments=%d",
            repo,
            number,
            decision.outcome,
            len(findings),
            len(comments),
            extra={
                "event": "publish_render",
                "repo": repo,
                "pr": number,
                "outcome": decision.outcome,
                "findings": len(findings),
                "comments": len(comments),
            },
        )
        return RenderedReview(
            outcome=decision.outcome,
            summary=summary,
            body=body,
            comments=comments,
        )

    # -- pieces -----------------------------------------------------------

    def _summary(self, decision: GateDecision) -> str:
        counts = decision.counts()
        headline = OUTCOME_HEADLINE.get(decision.outcome, decision.outcome)
        if not decision.findings:
            return f"{headline}. No findings."
        breakdown = ", ".join(
            f"{counts[severity]} {severity}"
            for severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
            if counts.get(severity)
        )
        verified = sum(1 for f in decision.findings if f.verification == "VERIFIED")
        return (
            f"{headline}. {len(decision.findings)} finding(s): {breakdown}. "
            f"{verified} corroborated by a deterministic check."
        )

    def _body(
        self, decision: GateDecision, summary: str, notes: Sequence[str]
    ) -> str:
        lines = [
            f"## {OUTCOME_HEADLINE.get(decision.outcome, decision.outcome)}",
            "",
            summary,
            "",
            f"**Decision: {decision.outcome.upper()}**",
            "",
        ]

        lines.append("Why:")
        for reason in decision.reasons or ["no findings"]:
            lines.append(f"- {escape_untrusted(reason, limit=400)}")

        if decision.waived:
            lines += ["", "Waived by repository policy:"]
            for waiver in decision.waived:
                lines.append(f"- {escape_untrusted(waiver.describe(), limit=400)}")

        if decision.findings:
            lines += ["", "### Findings", ""]
            for index, finding in enumerate(decision.findings, start=1):
                lines += self._finding_block(index, finding)

        if notes:
            lines += ["", "### Review notes", ""]
            for note in notes:
                lines.append(f"- {escape_untrusted(note, limit=400)}")

        lines += [
            "",
            "---",
            "",
            "Every finding above cites the evidence behind it. The decision is "
            "made by a deterministic policy engine from finding severities and "
            "this repository's policy — not by a model.",
        ]
        return "\n".join(lines)

    def _finding_block(self, index: int, finding: Finding) -> list[str]:
        location = f"{finding.file}:{finding.line_start}"
        if finding.line_end > finding.line_start:
            location += f"-{finding.line_end}"
        blocking = " · **blocks merge**" if finding.blocking else ""
        return [
            f"#### {index}. {finding.severity} · {finding.category} · "
            f"`{_escape_path(location)}`{blocking}",
            "",
            escape_untrusted(finding.message),
            "",
            f"> **Evidence** ({escape_untrusted(finding.source, limit=120)}): "
            f"{escape_untrusted(finding.evidence)}",
            "",
            f"_Confidence {finding.confidence}/100 · "
            f"{VERIFICATION_NOTE.get(finding.verification, finding.verification)}._",
            "",
        ]

    def _comment(self, finding: Finding) -> InlineComment:
        blocking = " — **blocks merge**" if finding.blocking else ""
        body = "\n".join(
            [
                f"**{finding.severity} · {finding.category}**{blocking}",
                "",
                escape_untrusted(finding.message),
                "",
                f"> **Evidence** ({escape_untrusted(finding.source, limit=120)}): "
                f"{escape_untrusted(finding.evidence, limit=1500)}",
                "",
                f"_Confidence {finding.confidence}/100 · "
                f"{VERIFICATION_NOTE.get(finding.verification, finding.verification)}._",
            ]
        )
        return InlineComment(
            path=finding.file,
            line=finding.line_end,
            start_line=finding.line_start,
            body=body,
            severity=finding.severity,
        )


def _escape_path(path: str) -> str:
    """A path is repository-authored too, and it lands inside a code span.

    The backtick is what matters — it would close the span and let whatever
    follows out into live markup. Angle brackets are defused as well, so the
    path is inert even if some later renderer treats it as text rather than code.
    """
    return (
        path.replace("`", "ˋ")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


@runtime_checkable
class ReviewTransport(Protocol):
    """One authenticated POST. The publisher needs nothing else."""

    def post(self, path: str, payload: Mapping[str, Any]) -> int: ...


@dataclass
class RecordingTransport:
    """Records what would be posted. For tests, and for a dry run."""

    posts: list[tuple[str, Mapping[str, Any]]] = field(default_factory=list)
    status: int = 200
    raises: Exception | None = None

    def post(self, path: str, payload: Mapping[str, Any]) -> int:
        if self.raises is not None:
            raise self.raises
        self.posts.append((path, payload))
        return self.status

    @property
    def last(self) -> Mapping[str, Any]:
        assert self.posts, "nothing was posted"
        return self.posts[-1][1]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuses redirects instead of replaying the credential at a new host.

    `urllib`'s default redirect handler copies every request header onto the
    redirected request except ``content-length`` and ``content-type`` — the
    ``Authorization`` header included — so a 3xx pointing off-host would hand
    the GitHub token to whoever answered. The API does not redirect POSTs in
    normal operation, so refusing is both safe and informative.
    """

    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        raise PublishError(
            f"GitHub redirected the review POST to {newurl!r}; refusing to "
            "resend the credential to a redirect target"
        )


class UrllibReviewTransport:
    """The real transport: one POST, on the standard library."""

    def __init__(
        self,
        token: str | None = None,
        *,
        api_root: str = "https://api.github.com",
        timeout: float = 30.0,
        env: Mapping[str, str] | None = None,
    ) -> None:
        source = os.environ if env is None else env
        self.api_root = api_root.rstrip("/")
        self.timeout = timeout
        self._token = token if token is not None else source.get(TOKEN_ENV_VAR)
        self._host = urllib.parse.urlsplit(self.api_root).hostname or ""

    def _opener(self) -> urllib.request.OpenerDirector:
        """An opener that will not follow a redirect. See :class:`_NoRedirect`."""
        return urllib.request.build_opener(_NoRedirect)

    def _resolve(self, path: str) -> str:
        """Turn a caller's path into a URL, or refuse it.

        This request carries a `Bearer` credential, so where it goes is a
        security decision and not a formatting one. Two rules:

        * **https only.** The scheme test used to be ``path.startswith("http")``,
          which accepts ``http://`` — the token in clear text — and also matches
          a path that merely begins with those four letters.
        * **the configured host only.** An absolute URL is accepted only if it
          resolves to the same host as ``api_root``, so a repo name or a caller
          -supplied path cannot redirect the credential somewhere else.
        """
        if "://" in path:
            parts = urllib.parse.urlsplit(path)
            if parts.scheme != "https":
                raise PublishError(
                    f"refusing to send a credential over {parts.scheme!r}; "
                    "the GitHub API must be reached over https"
                )
            if (parts.hostname or "") != self._host:
                raise PublishError(
                    f"refusing to post to {parts.hostname!r}: the configured API "
                    f"host is {self._host!r}"
                )
            return path

        if not path.startswith("/"):
            raise PublishError(f"expected an absolute API path, got {path!r}")
        return f"{self.api_root}{path}"

    def post(self, path: str, payload: Mapping[str, Any]) -> int:
        url = self._resolve(path)
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("Content-Type", "application/json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        request.add_header("User-Agent", "reviewer-agent")
        if self._token:
            request.add_header("Authorization", f"Bearer {self._token}")

        try:
            with self._opener().open(request, timeout=self.timeout) as response:
                return int(response.status)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300] if exc.fp else ""
            raise PublishError(
                f"GitHub returned {exc.code} posting the review: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise PublishError(f"could not reach GitHub: {exc.reason}") from exc


# --------------------------------------------------------------------------
# The publisher
# --------------------------------------------------------------------------


class GitHubPublisher:
    """Stage 10. Renders a decision and posts it as a pull-request review.

    The review is always posted as ``COMMENT``, never ``REQUEST_CHANGES``: the
    merge decision belongs to the deterministic gate and to branch protection
    reading it, not to a bot's review state. The decision is stated plainly in
    the body — but the bot does not cast the vote.
    """

    def __init__(
        self,
        transport: ReviewTransport | None = None,
        renderer: ReviewRenderer | None = None,
    ) -> None:
        self.transport = transport
        self.renderer = renderer or ReviewRenderer()

    def render(
        self,
        decision: GateDecision,
        *,
        repo: str = "",
        number: int = 0,
        notes: Sequence[str] = (),
    ) -> RenderedReview:
        return self.renderer.render(decision, repo=repo, number=number, notes=notes)

    def publish(
        self,
        decision: GateDecision,
        *,
        repo: str,
        number: int,
        commit_sha: str = "",
        notes: Sequence[str] = (),
    ) -> RenderedReview:
        """Render and post. Raises :class:`PublishError` if posting fails."""
        rendered = self.render(decision, repo=repo, number=number, notes=notes)
        if self.transport is None:
            logger.info(
                "publish_skipped repo=%s pr=%d reason=no transport configured",
                repo,
                number,
                extra={"event": "publish_skipped", "repo": repo, "pr": number},
            )
            return rendered

        payload = dict(rendered.payload())
        if commit_sha:
            payload["commit_id"] = commit_sha

        status = self.transport.post(f"/repos/{repo}/pulls/{number}/reviews", payload)
        logger.info(
            "publish_posted repo=%s pr=%d status=%s comments=%d",
            repo,
            number,
            status,
            len(rendered.comments),
            extra={
                "event": "publish_posted",
                "repo": repo,
                "pr": number,
                "status": status,
            },
        )
        return rendered
