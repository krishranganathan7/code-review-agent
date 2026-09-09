"""Stage 11 — the trace: what a review did, and why it decided what it decided.

One `ReviewReport` per review, assembled from what the stages already recorded.
It answers two questions from the trace alone:

* **"why did this PR get this verdict?"** — the risk level and the signals that
  set it, which agents were selected and which were skipped and why, every
  finding that survived with its severity, confidence and verification, every
  finding that was dropped and at which stage, and the gate's own reasons.
* **"what did the agent look at?"** — per agent, per group: the tools it called,
  the arguments it called them with, whether each call succeeded, how many
  characters came back, how many model round-trips it took, and why it stopped.

**References and counts, never payloads.** No file contents, no tool output, no
message transcripts, and no credentials. A tool call is recorded as
``read_file(path="src/pr.py") ok 812 chars`` — enough to retrace the search,
nothing that leaks the repository into a log aggregator. `AgentResult.transcript`
is deliberately *not* serialized; it stays in memory for the caller that wants
it.

A `review_id` is threaded through the logging of every stage by
:func:`review_context`, so two concurrent reviews can be told apart in a shared
log stream — the gap Phase 6 flagged.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

__all__ = [
    "ReviewReport",
    "StageTiming",
    "ToolCallRecord",
    "AgentRecord",
    "FindingRecord",
    "DropRecord",
    "build_report",
    "review_context",
    "current_review_id",
    "ReviewIdFilter",
    "install_review_id_logging",
    "redact",
    "SECRET_SHAPES",
]

logger = logging.getLogger("reviewer.trace")

_REVIEW_ID: ContextVar[str] = ContextVar("reviewer_review_id", default="-")

_FACTORY_INSTALLED = False
"""Set once the record factory is in place; installing it twice would nest wrappers."""


# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------


def current_review_id() -> str:
    """The review this thread is working on, or ``"-"`` outside a review."""
    return _REVIEW_ID.get()


@contextmanager
def review_context(review_id: str | None = None) -> Iterator[str]:
    """Tag everything logged inside this block with a review id.

    A `ContextVar`, so it follows the orchestrator's thread pool: an agent
    running concurrently inherits the id of the review that submitted it.
    """
    assigned = review_id or f"rv_{uuid.uuid4().hex[:12]}"
    token = _REVIEW_ID.set(assigned)
    try:
        yield assigned
    finally:
        _REVIEW_ID.reset(token)


class ReviewIdFilter(logging.Filter):
    """Adds ``review_id`` to every record, so a formatter can print it.

    Note for callers: once :func:`install_review_id_logging` has run,
    ``review_id`` is a reserved record attribute. Passing it again in a
    ``logging`` ``extra=`` dict raises `KeyError: Attempt to overwrite
    'review_id' in LogRecord` — the correlation id is supplied automatically and
    should never be passed by hand.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.review_id = current_review_id()
        return True


def install_review_id_logging(logger_name: str = "reviewer") -> ReviewIdFilter:
    """Put ``review_id`` on every log record, so a formatter can rely on it.

    A filter attached to a logger only sees records logged *on that logger* —
    records from child loggers propagate to ancestor **handlers** without ever
    passing through ancestor **filters**. Attaching to `reviewer` therefore left
    every `reviewer.analysis.*` record without the attribute, and a formatter
    referencing it raised `ValueError: Formatting field not found in record`.

    A record factory runs for every record created anywhere, which is what the
    guarantee actually requires. The filter is still attached and returned, so a
    caller can add it to a specific handler if it wants one.
    """
    review_filter = ReviewIdFilter()
    logging.getLogger(logger_name).addFilter(review_filter)

    global _FACTORY_INSTALLED
    if not _FACTORY_INSTALLED:
        previous = logging.getLogRecordFactory()

        def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = previous(*args, **kwargs)
            if not hasattr(record, "review_id"):
                record.review_id = current_review_id()
            return record

        logging.setLogRecordFactory(factory)
        _FACTORY_INSTALLED = True
    return review_filter


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------

SECRET_SHAPES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\b[sr]k_live_[A-Za-z0-9]{16,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
"""Credential shapes scrubbed from anything that reaches the trace.

The trace should never see one — nothing routes a credential into it — so this
is the belt to the design's braces. A finding's evidence is model-authored text
that quoted a file, and a file can contain anything.
"""

REDACTED = "[redacted]"


def redact(text: str, *, limit: int | None = None) -> str:
    """Scrub credential-shaped strings, and optionally cap the length."""
    cleaned = text
    for shape in SECRET_SHAPES:
        cleaned = shape.sub(REDACTED, cleaned)
    if limit is not None and len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StageTiming:
    """How long one stage took."""

    stage: str
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "seconds": round(self.seconds, 4)}


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool call: what was asked for, and how much came back. Not what."""

    iteration: int
    tool: str
    arguments: dict[str, Any]
    ok: bool
    error: str | None = None
    content_chars: int = 0
    unknown_tool: bool = False

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "iteration": self.iteration,
            "tool": self.tool,
            "arguments": {
                key: redact(str(value), limit=200)
                for key, value in sorted(self.arguments.items())
            },
            "ok": self.ok,
            "content_chars": self.content_chars,
        }
        if self.error:
            record["error"] = redact(self.error, limit=300)
        if self.unknown_tool:
            record["unknown_tool"] = True
        return record


@dataclass(frozen=True)
class AgentRecord:
    """One agent's run over one change group."""

    agent: str
    group: str
    stop_reason: str
    iterations: int = 0
    tool_calls: int = 0
    tokens_used: int = 0
    elapsed_seconds: float = 0.0
    provider_calls: int = 0
    findings: int = 0
    dropped: list[str] = field(default_factory=list)
    tools: list[ToolCallRecord] = field(default_factory=list)
    context_elided_exchanges: int = 0
    context_clipped_results: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "group": self.group,
            "stop_reason": self.stop_reason,
            "iterations": self.iterations,
            "provider_calls": self.provider_calls,
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "findings": self.findings,
            "dropped": [redact(reason, limit=300) for reason in self.dropped],
            "context": {
                "elided_exchanges": self.context_elided_exchanges,
                "clipped_results": self.context_clipped_results,
            },
            "tools": [call.to_dict() for call in self.tools],
        }


@dataclass(frozen=True)
class FindingRecord:
    """A published finding, as the trace records it."""

    severity: str
    category: str
    file: str
    line_start: int
    line_end: int
    confidence: int
    verification: str
    blocking: bool
    source: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "location": f"{self.file}:{self.line_start}-{self.line_end}",
            "confidence": self.confidence,
            "verification": self.verification,
            "blocking": self.blocking,
            "source": self.source,
            "message": redact(self.message, limit=400),
        }


@dataclass(frozen=True)
class DropRecord:
    """A finding that did not survive, and the stage that removed it."""

    stage: str
    reason: str
    location: str = ""

    def to_dict(self) -> dict[str, Any]:
        record = {"stage": self.stage, "reason": redact(self.reason, limit=400)}
        if self.location:
            record["location"] = self.location
        return record


@dataclass(frozen=True)
class ReviewReport:
    """The whole trace of one review."""

    review_id: str
    repo: str
    number: int
    head_sha: str = ""
    outcome: str = "pass"
    risk_level: str = "LOW"
    risk_signals: list[str] = field(default_factory=list)
    risk_why: str = ""
    files_changed: int = 0
    groups: list[dict[str, Any]] = field(default_factory=list)
    agents_selected: list[str] = field(default_factory=list)
    agents_skipped: dict[str, str] = field(default_factory=dict)
    selection_why: str = ""
    attribution: dict[str, int] = field(default_factory=dict)
    agent_runs: list[AgentRecord] = field(default_factory=list)
    aggregation: dict[str, int] = field(default_factory=dict)
    verification: dict[str, int] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)
    gate: dict[str, Any] = field(default_factory=dict)
    published: dict[str, Any] = field(default_factory=dict)
    findings: list[FindingRecord] = field(default_factory=list)
    dropped: list[DropRecord] = field(default_factory=list)
    timings: list[StageTiming] = field(default_factory=list)
    tokens_used: int = 0
    tool_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "repo": self.repo,
            "pr": self.number,
            "head_sha": self.head_sha[:12],
            "outcome": self.outcome,
            "risk": {
                "level": self.risk_level,
                "signals": self.risk_signals,
                "why": redact(self.risk_why, limit=1200),
            },
            "change": {
                "files": self.files_changed,
                "groups": self.groups,
                "attribution": self.attribution,
            },
            "selection": {
                "selected": self.agents_selected,
                "skipped": self.agents_skipped,
                "why": self.selection_why,
            },
            "agent_runs": [record.to_dict() for record in self.agent_runs],
            "aggregation": self.aggregation,
            "verification": self.verification,
            "judge": self.judge,
            "gate": self.gate,
            "published": self.published,
            "findings": [record.to_dict() for record in self.findings],
            "dropped": [record.to_dict() for record in self.dropped],
            "totals": {
                "tokens_used": self.tokens_used,
                "tool_calls": self.tool_calls,
            },
            "timings": [timing.to_dict() for timing in self.timings],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def why(self) -> str:
        """The verdict in one paragraph, from the trace alone."""
        gate_reasons = self.gate.get("reasons") or []
        return (
            f"{self.repo}#{self.number} [{self.review_id}]: {self.outcome.upper()}. "
            f"Risk {self.risk_level}"
            + (f" ({', '.join(self.risk_signals)})" if self.risk_signals else "")
            + f"; agents {', '.join(self.agents_selected) or 'none'}; "
            f"{len(self.findings)} finding(s) published, "
            f"{len(self.dropped)} dropped. "
            + ("Gate: " + "; ".join(str(reason) for reason in gate_reasons))
        )

    def tools_used(self) -> list[str]:
        """Every tool called anywhere in the review, in name order."""
        return sorted({call.tool for run in self.agent_runs for call in run.tools})

    def files_read(self) -> list[str]:
        """Every path any agent asked a tool for. The search trail."""
        paths: set[str] = set()
        for run in self.agent_runs:
            for call in run.tools:
                value = call.arguments.get("path")
                if isinstance(value, str) and value:
                    paths.add(value)
        return sorted(paths)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_report(
    outcome: Any,
    *,
    review_id: str | None = None,
    timings: Sequence[StageTiming] = (),
) -> ReviewReport:
    """Assemble the trace from a `ReviewOutcome`.

    Takes the outcome structurally rather than by import, so `trace` stays a leaf
    module that the pipeline can use without a cycle.
    """
    pull_request = outcome.pull_request
    analysis = outcome.analysis
    assessment = outcome.assessment
    decision = outcome.decision

    agent_runs = [_agent_record(review) for review in outcome.reviews]

    dropped: list[DropRecord] = []
    for review in outcome.reviews:
        for item in review.dropped:
            dropped.append(DropRecord(stage=f"agent:{review.agent}", reason=item.reason))
    for refuted in outcome.verification.refuted:
        dropped.append(
            DropRecord(
                stage="verification",
                reason=refuted.reason,
                location=f"{refuted.finding.file}:{refuted.finding.line_start}",
            )
        )
    for rejected in outcome.judgement.rejected:
        dropped.append(
            DropRecord(
                stage="judge",
                reason=rejected.reason,
                location=f"{rejected.finding.file}:{rejected.finding.line_start}",
            )
        )
    if outcome.aggregation.capped:
        dropped.append(
            DropRecord(
                stage="aggregation",
                reason=(
                    f"{outcome.aggregation.capped} finding(s) below the publication "
                    "cap, after ranking by severity and confidence"
                ),
            )
        )

    report = ReviewReport(
        review_id=review_id or current_review_id(),
        repo=pull_request.repo,
        number=pull_request.number,
        head_sha=pull_request.head_sha,
        outcome=decision.outcome,
        risk_level=assessment.level,
        risk_signals=list(assessment.signal_ids),
        risk_why=assessment.why,
        files_changed=analysis.scope.files_touched,
        groups=[
            {
                "key": group.group_key,
                "label": group.label,
                "files": len(group.paths),
                "agents": group.agents_run,
                "findings": group.findings,
                "dropped": group.dropped,
            }
            for group in outcome.trace.groups
        ],
        agents_selected=list(outcome.trace.agents_selected),
        agents_skipped=dict(outcome.trace.agents_skipped),
        selection_why=outcome.trace.selection_why,
        attribution=analysis.tier_counts(),
        agent_runs=agent_runs,
        aggregation={
            "considered": outcome.aggregation.considered,
            "merged": outcome.aggregation.merged,
            "agreed": outcome.aggregation.agreed,
            "capped": outcome.aggregation.capped,
            "published": outcome.aggregation.published,
        },
        verification=outcome.verification.counts(),
        judge={
            **outcome.judgement.counts(),
            "skipped": outcome.judgement.skipped,
            "downgraded_detail": [
                redact(item, limit=300) for item in outcome.judgement.downgraded
            ],
        },
        gate={
            "outcome": decision.outcome,
            "blocking": len(decision.blocking),
            "waived": [redact(item.describe(), limit=300) for item in decision.waived],
            "reasons": [redact(reason, limit=400) for reason in decision.reasons],
            "block_at": decision.policy.block_at,
        },
        published={
            "posted": bool(getattr(outcome, "posted", False)),
            "comments": (
                len(outcome.rendered.comments) if outcome.rendered is not None else 0
            ),
            "summary": (
                redact(outcome.rendered.summary, limit=600)
                if outcome.rendered is not None
                else ""
            ),
        },
        findings=[_finding_record(finding) for finding in decision.findings],
        dropped=dropped,
        timings=list(timings),
        tokens_used=sum(run.tokens_used for run in agent_runs),
        tool_calls=sum(run.tool_calls for run in agent_runs),
    )

    logger.info(
        "review_trace review=%s repo=%s pr=%d outcome=%s findings=%d dropped=%d "
        "tokens=%d tool_calls=%d",
        report.review_id,
        report.repo,
        report.number,
        report.outcome,
        len(report.findings),
        len(report.dropped),
        report.tokens_used,
        report.tool_calls,
        extra={
            "event": "review_trace",
            "repo": report.repo,
            "pr": report.number,
            "outcome": report.outcome,
            "findings": len(report.findings),
            "dropped": len(report.dropped),
            "tokens_used": report.tokens_used,
            "tool_calls": report.tool_calls,
        },
    )
    return report


def _agent_record(review: Any) -> AgentRecord:
    result = review.result
    if result is None:
        return AgentRecord(
            agent=review.agent,
            group=review.group_key,
            stop_reason=review.stop_reason,
            findings=len(review.findings),
            dropped=[item.reason for item in review.dropped],
        )
    return AgentRecord(
        agent=review.agent,
        group=review.group_key,
        stop_reason=result.stop_reason,
        iterations=result.iterations,
        tool_calls=result.tool_calls,
        tokens_used=result.tokens_used,
        elapsed_seconds=result.elapsed_seconds,
        provider_calls=len(result.provider_calls),
        findings=len(review.findings),
        dropped=[item.reason for item in review.dropped],
        tools=[
            ToolCallRecord(
                iteration=call.iteration,
                tool=call.tool,
                arguments=dict(call.arguments),
                ok=call.ok,
                error=call.error,
                content_chars=call.content_chars,
                unknown_tool=call.unknown_tool,
            )
            for call in result.tool_trace
        ],
        context_elided_exchanges=max(
            (record.elided_exchanges for record in result.provider_calls), default=0
        ),
        context_clipped_results=sum(
            record.clipped_results for record in result.provider_calls
        ),
    )


def _finding_record(finding: Any) -> FindingRecord:
    return FindingRecord(
        severity=finding.severity,
        category=finding.category,
        file=finding.file,
        line_start=finding.line_start,
        line_end=finding.line_end,
        confidence=finding.confidence,
        verification=finding.verification,
        blocking=finding.blocking,
        source=finding.source,
        message=finding.message,
    )


class Stopwatch:
    """Times stages for the trace. Not a metric system; just enough to explain."""

    def __init__(self) -> None:
        self.timings: list[StageTiming] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.monotonic()
        try:
            yield
        finally:
            self.timings.append(
                StageTiming(stage=name, seconds=time.monotonic() - started)
            )

    def record(self, name: str, seconds: float) -> None:
        self.timings.append(StageTiming(stage=name, seconds=seconds))
