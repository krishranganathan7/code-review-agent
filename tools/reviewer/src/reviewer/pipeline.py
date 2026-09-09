"""The conductor: a pull request in, a published verdict out.

This is not a pipeline stage — it is the thing that runs them, in the order
CLAUDE.md §3 lays down:

```
change analysis  →  risk  →  agent selection  →  agents (concurrent)
      →  aggregate  →  verify  →  judge  →  gate  →  publish
```

It lives as a root-level module rather than a package because it owns no stage
of its own; every box above belongs to `analysis/`, `agents/`, `findings/`,
`policy/` or `publish/`, and this file only sequences them. (CLAUDE.md §7's
layout is indicative and lists the stage packages; this addition is noted in
PROGRESS.md.)

**Concurrency with deterministic output.** Agents reviewing independent change
groups have no reason to wait for each other, so they run on a thread pool.
Determinism is then a separate guarantee, obtained by sorting rather than by
scheduling: results are collected as they finish and **sorted by
(group key, agent name) before anything is combined**. Two runs over the same
pull request produce byte-identical findings whatever order the agents complete
in — a property tested by running the same review at one worker and at four.

The deterministic stages contain no model call. Only the agents and the judge
use the provider, and neither can reach the gate.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

from .agent import LoopControls
from .agents.base import AgentContext, AgentReview, ReviewAgent
from .agents.selection import AgentPlan, AgentSelection, plan_review
from .analysis.change_analyzer import ChangeAnalysis, ChangeAnalyzer, ChangeGroup
from .analysis.ingest import PullRequest
from .analysis.risk import RiskAssessment, RiskEngine
from .findings.aggregator import AggregationResult, Aggregator
from .findings.finding import Finding
from .findings.judge import Judge, JudgeResult
from .findings.verification import VerificationResult, Verifier
from .languages.base import LanguageAdapter
from .policy.engine import GateDecision, PolicyEngine
from .providers.base import LLMProvider
from .publish.github import GitHubPublisher, RenderedReview
from .tools.base import Tool

__all__ = [
    "ReviewPipeline",
    "ReviewOutcome",
    "ReviewTrace",
    "GroupRun",
    "review_pull_request",
    "ProviderFor",
]

logger = logging.getLogger("reviewer.pipeline")

ProviderFor = Callable[[str, str], LLMProvider]
"""Given an agent name and a group key, the provider that agent should use.

Exists because a scripted test provider is stateful and not thread-safe: each
concurrent agent needs its own. Production passes a single shared provider and
never sets this.
"""


@dataclass(frozen=True)
class GroupRun:
    """What happened for one change group."""

    group_key: str
    label: str
    paths: list[str] = field(default_factory=list)
    agents_run: list[str] = field(default_factory=list)
    agents_skipped: dict[str, str] = field(default_factory=dict)
    findings: int = 0
    dropped: int = 0


@dataclass(frozen=True)
class ReviewTrace:
    """The record of one review: what ran, what was skipped, and why.

    Kept as a first-class result rather than only in the logs, because "why did
    the security agent not look at this?" is a question about a specific review.
    """

    level: str = "LOW"
    signals: list[str] = field(default_factory=list)
    groups: list[GroupRun] = field(default_factory=list)
    agents_selected: list[str] = field(default_factory=list)
    agents_skipped: dict[str, str] = field(default_factory=dict)
    selection_why: str = ""
    workers: int = 1

    @property
    def group_keys(self) -> list[str]:
        return [group.group_key for group in self.groups]

    def summary(self) -> str:
        return (
            f"{len(self.groups)} change group(s), risk {self.level}"
            + (f" (signals: {', '.join(self.signals)})" if self.signals else "")
            + f"; agents: {', '.join(self.agents_selected) or 'none'}"
        )


@dataclass(frozen=True)
class ReviewOutcome:
    """Everything one review produced, stage by stage."""

    pull_request: PullRequest
    analysis: ChangeAnalysis
    assessment: RiskAssessment
    plan: AgentPlan
    reviews: list[AgentReview] = field(default_factory=list)
    aggregation: AggregationResult = field(default_factory=AggregationResult)
    verification: VerificationResult = field(default_factory=VerificationResult)
    judgement: JudgeResult = field(default_factory=JudgeResult)
    decision: GateDecision = field(default_factory=GateDecision)  # type: ignore[arg-type]
    rendered: RenderedReview | None = None
    trace: ReviewTrace = field(default_factory=ReviewTrace)

    @property
    def findings(self) -> list[Finding]:
        """The published findings, as the gate decided them."""
        return self.decision.findings

    @property
    def outcome(self) -> str:
        return self.decision.outcome

    @property
    def blocked(self) -> bool:
        return self.decision.blocked

    def notes(self) -> list[str]:
        """Honest caveats about this review's completeness, for publication."""
        notes: list[str] = []

        failed = sorted(
            {review.agent for review in self.reviews if review.failed}
        )
        if failed:
            notes.append(
                "The following reviewer(s) failed and contributed nothing: "
                + "; ".join(
                    f"{review.agent} ({review.error})"
                    for review in self.reviews
                    if review.failed
                )
                + ". This review is incomplete."
            )

        limited = [review.agent for review in self.reviews if review.limited]
        if limited:
            notes.append(
                "Search budget ran out for "
                + ", ".join(sorted(set(limited)))
                + "; those reviews are less complete and their findings carry "
                "reduced confidence."
            )

        dropped = sum(len(review.dropped) for review in self.reviews)
        if dropped:
            notes.append(
                f"{dropped} candidate finding(s) were discarded for lacking "
                "evidence or citing code that is not there."
            )
        if self.verification.refuted:
            notes.append(
                f"{len(self.verification.refuted)} finding(s) were refuted by a "
                "deterministic check and dropped."
            )
        if self.judgement.rejected:
            notes.append(
                f"{len(self.judgement.rejected)} finding(s) were rejected by an "
                "independent judge and dropped."
            )
        if self.judgement.skipped:
            notes.append(f"The independent judge did not run: {self.judgement.skipped}")
        if self.aggregation.capped:
            notes.append(
                f"{self.aggregation.capped} lower-ranked finding(s) were omitted "
                "for length; findings are ranked by severity and confidence, so "
                "nothing more severe was dropped."
            )
        if self.analysis.degraded_paths:
            notes.append(
                "Symbols could not be resolved in "
                + ", ".join(sorted(self.analysis.degraded_paths)[:5])
                + "; those files were reviewed as whole files."
            )
        return notes


class ReviewPipeline:
    """Runs a pull request through every stage and publishes the verdict."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        tools: Sequence[Tool] | None = None,
        source: Any | None = None,
        adapters: Sequence[LanguageAdapter] | None = None,
        agents: Sequence[ReviewAgent] | None = None,
        judge: Judge | None = None,
        publisher: GitHubPublisher | None = None,
        finding_cap: int | None = 50,
        max_workers: int = 4,
        provider_for: ProviderFor | None = None,
        controls: LoopControls | None = None,
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.tools = list(tools or [])
        self.source = source
        self.adapters = list(adapters) if adapters is not None else None
        self.agents = list(agents) if agents is not None else None
        self.judge = judge
        self.publisher = publisher or GitHubPublisher()
        self.finding_cap = finding_cap
        self.max_workers = max(1, max_workers)
        self.provider_for = provider_for
        self.controls = controls

    # -- the run ----------------------------------------------------------

    def run(self, pull_request: PullRequest, *, publish: bool = False) -> ReviewOutcome:
        # Stages 2-3: deterministic.
        analysis = ChangeAnalyzer(self.adapters, source=self.source).analyze(
            pull_request
        )
        assessment = RiskEngine.for_analysis(analysis).assess(analysis)

        # Stage 5 selection, then the agents themselves.
        plan = plan_review(assessment, self.agents, controls=self.controls)
        reviews, groups = self._run_agents(pull_request, analysis, assessment, plan)

        trace = ReviewTrace(
            level=assessment.level,
            signals=list(assessment.signal_ids),
            groups=groups,
            agents_selected=plan.names,
            agents_skipped=dict(plan.skipped),
            selection_why=plan.why(),
            workers=self.max_workers,
        )

        # Stage 6: one ranked, deduplicated list.
        aggregation = Aggregator(cap=self.finding_cap).combine(
            finding for review in reviews for finding in review.findings
        )

        # Stage 7: deterministic corroboration.
        verification = Verifier(pull_request, self.source).check(aggregation.findings)

        # Stage 8: the independent judge, on a fresh context.
        judgement = (
            self.judge.review(verification.findings)
            if self.judge is not None
            else JudgeResult(
                findings=list(verification.findings), skipped="no judge configured"
            )
        )

        # Stage 9: the deterministic gate. Nothing above decides this.
        # An explicit `settings` mapping overrides the repository's own
        # `.reviewer.toml`; without one the repository's policy stands.
        policy_settings = (
            self.settings.get("policy")
            if self.settings is not None
            else pull_request.settings("policy")
        )
        engine = PolicyEngine.from_settings(
            policy_settings if isinstance(policy_settings, Mapping) else None
        )
        decision = engine.decide(judgement.findings)

        outcome = ReviewOutcome(
            pull_request=pull_request,
            analysis=analysis,
            assessment=assessment,
            plan=plan,
            reviews=reviews,
            aggregation=aggregation,
            verification=verification,
            judgement=judgement,
            decision=decision,
            trace=trace,
        )

        logger.info(
            "review repo=%s pr=%d outcome=%s findings=%d groups=%d agents=%s",
            pull_request.repo,
            pull_request.number,
            decision.outcome,
            len(decision.findings),
            len(groups),
            ",".join(plan.names),
            extra={
                "event": "review",
                "repo": pull_request.repo,
                "pr": pull_request.number,
                "outcome": decision.outcome,
                "findings": len(decision.findings),
                "risk": assessment.level,
                "groups": len(groups),
                "agents": plan.names,
            },
        )

        # Stage 10.
        rendered = (
            self.publisher.publish(
                decision,
                repo=pull_request.repo,
                number=pull_request.number,
                commit_sha=pull_request.head_sha,
                notes=outcome.notes(),
            )
            if publish
            else self.publisher.render(
                decision,
                repo=pull_request.repo,
                number=pull_request.number,
                notes=outcome.notes(),
            )
        )

        return replace(outcome, rendered=rendered)

    # -- stage 5 ----------------------------------------------------------

    def _run_agents(
        self,
        pull_request: PullRequest,
        analysis: ChangeAnalysis,
        assessment: RiskAssessment,
        plan: AgentPlan,
    ) -> tuple[list[AgentReview], list[GroupRun]]:
        """Run every selected agent over every change group.

        Concurrent, then sorted. The sort is what makes the result
        deterministic — not the scheduling, which is deliberately not relied on.
        """
        jobs: list[tuple[ChangeGroup, AgentSelection]] = [
            (group, selection)
            for group in analysis.groups
            for selection in plan.selections
        ]
        if not jobs:
            return [], [_group_run(group, plan, []) for group in analysis.groups]

        def execute(job: tuple[ChangeGroup, AgentSelection]) -> AgentReview:
            group, selection = job
            context = AgentContext(
                pull_request=pull_request,
                group=group,
                analysis=analysis,
                assessment=assessment,
                source=self.source,
            )
            provider = (
                self.provider_for(selection.agent.name, group.key)
                if self.provider_for is not None
                else self.provider
            )
            try:
                return selection.agent.review(
                    context, provider, self.tools, selection.controls
                )
            except Exception as exc:
                # One agent failing — a rate limit, a provider outage, a bug in
                # a custom agent — must not discard every finding the others
                # already produced. The failure becomes an empty review that
                # carries its reason, and `notes()` reports it.
                logger.exception(
                    "agent_failed agent=%s group=%s error=%s",
                    selection.agent.name,
                    group.key,
                    exc,
                    extra={
                        "event": "agent_failed",
                        "agent": selection.agent.name,
                        "group": group.key,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                return AgentReview(
                    agent=selection.agent.name,
                    group_key=group.key,
                    error=f"{type(exc).__name__}: {exc}",
                )

        if self.max_workers == 1 or len(jobs) == 1:
            collected = [execute(job) for job in jobs]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                collected = list(pool.map(execute, jobs))

        # Deterministic aggregation order, whatever order they finished in.
        reviews = sorted(collected, key=lambda review: (review.group_key, review.agent))

        by_group: dict[str, list[AgentReview]] = {}
        for review in reviews:
            by_group.setdefault(review.group_key, []).append(review)

        groups = [
            _group_run(group, plan, by_group.get(group.key, []))
            for group in analysis.groups
        ]
        return reviews, groups


def _group_run(
    group: ChangeGroup, plan: AgentPlan, reviews: Sequence[AgentReview]
) -> GroupRun:
    return GroupRun(
        group_key=group.key,
        label=group.label,
        paths=list(group.paths),
        agents_run=sorted(review.agent for review in reviews),
        agents_skipped=dict(plan.skipped),
        findings=sum(len(review.findings) for review in reviews),
        dropped=sum(len(review.dropped) for review in reviews),
    )


def review_pull_request(
    pull_request: PullRequest,
    provider: LLMProvider,
    *,
    tools: Sequence[Tool] | None = None,
    source: Any | None = None,
    judge: Judge | None = None,
    publisher: GitHubPublisher | None = None,
    publish: bool = False,
    max_workers: int = 4,
    provider_for: ProviderFor | None = None,
    settings: Mapping[str, Any] | None = None,
) -> ReviewOutcome:
    """Convenience entry point over :class:`ReviewPipeline`.

    ``settings`` overrides the repository's own `.reviewer.toml` for this run.
    """
    return ReviewPipeline(
        provider,
        tools=tools,
        source=source,
        judge=judge,
        publisher=publisher,
        max_workers=max_workers,
        provider_for=provider_for,
        settings=settings,
    ).run(pull_request, publish=publish)
