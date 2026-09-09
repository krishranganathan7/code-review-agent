"""Stage 6 — normalize, dedupe, and rank by (severity, confidence) before any cap.

Three rules, in this order, and the order is the point.

**Dedupe on span, not on words.** Two agents that notice the same problem will
describe it differently — a missing null check on an auth path is a `bug` to one
reviewer and a `security` flaw to the other. Matching on message text would
publish both; matching on ``(file, overlapping line range)`` publishes one, with
*both* perspectives kept in the merged evidence.

**Agreement raises confidence.** A span two independent agents flagged, working
from different prompts, is more likely to be real than one either found alone.
That is the corroboration signal Phase 5 produced and did not use.

**Rank before cap** (CLAUDE.md §2.8). Whenever findings are limited, they are
sorted by ``(severity, confidence)`` *first* and truncated *after*. This is a
corrective for a known prior-tool bug where a HIGH finding could be discarded
because low-severity noise happened to arrive first. A regression test holds the
line: a HIGH arriving last still survives a cap that drops LOWs.

Deterministic throughout — no model is consulted here. The output order is fully
determined by the findings themselves, so the same set aggregates identically
however the agents that produced it were scheduled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

from .finding import Finding, severity_rank

__all__ = [
    "AggregationResult",
    "Aggregator",
    "aggregate",
    "rank",
    "AGREEMENT_BONUS",
    "MAX_AGREEMENT_BONUS",
    "MAX_MERGE_SPAN",
    "MAX_SPAN_RATIO",
    "MERGE_ADJACENCY",
]

logger = logging.getLogger("reviewer.findings.aggregator")

AGREEMENT_BONUS = 15
"""Confidence added for each *additional* agent that flagged the same span."""

MAX_AGREEMENT_BONUS = 30
"""Ceiling on the agreement boost, so a crowd cannot manufacture certainty."""

MAX_MERGE_SPAN = 40
"""Widest range, in lines, that may be merged with another finding at all.

A finding citing more lines than this is making a *file-level* claim, and must
not absorb the specific findings inside it. Span overlap is transitive -- A-B
overlap and B-C overlap put A and C together however unrelated they are -- so
one wide range collapses everything it touches.
"""

MERGE_ADJACENCY = 1
"""Lines of gap tolerated between two *single-line* findings from different agents.

A one-line citation is a pointer at a statement, and statements often span two
lines — so two reviewers pointing a line apart are usually describing the same
one. On a live review the bug agent reported a unit mismatch at line 28 and the
architecture agent reported the same mismatch at line 29, and both were
published.

Deliberately the narrowest rule that fixes it. Ranges of more than one line are
not merged across a gap at all: two adjacent three-line blocks are far more
likely to be two distinct problems than one described twice. And it requires
*different* sources, because one agent reporting twice a line apart has chosen
to report two things.
"""

MAX_SPAN_RATIO = 3.0
"""How different in length two ranges may be and still be the same finding.

An absolute cap alone is not enough: on a live review a finding citing lines
8-32 of a 32-line file swallowed eight others, and 25 lines is under any
sensible absolute threshold. What gives it away is the *disparity* -- a 3-line
claim and a 25-line claim are not two descriptions of one problem, they are a
specific finding and a broad one. Comparable ranges still merge, which is what
dedupe is for.
"""

SOURCE_SEPARATOR = "+"


@dataclass(frozen=True)
class AggregationResult:
    """One ranked, deduplicated finding list, and what it took to get there."""

    findings: list[Finding] = field(default_factory=list)
    considered: int = 0
    """Findings received from all agents, before merging."""
    merged: int = 0
    """Duplicates folded into another finding."""
    agreed: int = 0
    """Surviving findings that more than one agent independently flagged."""
    capped: int = 0
    """Findings dropped by the cap — always the lowest-ranked ones."""

    @property
    def published(self) -> int:
        return len(self.findings)

    def by_severity(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        return counts

    def sources_of(self, finding: Finding) -> list[str]:
        """The agents behind a finding, recovered from its merged source field."""
        return finding.source.split(SOURCE_SEPARATOR)


class Aggregator:
    """Stage 6. Deterministic: same findings in, same ranked list out."""

    def __init__(
        self,
        *,
        cap: int | None = None,
        agreement_bonus: int = AGREEMENT_BONUS,
        max_agreement_bonus: int = MAX_AGREEMENT_BONUS,
    ) -> None:
        if cap is not None and cap < 1:
            raise ValueError("cap must be at least 1 when set")
        self.cap = cap
        self.agreement_bonus = agreement_bonus
        self.max_agreement_bonus = max_agreement_bonus

    def combine(self, findings: Iterable[Finding]) -> AggregationResult:
        """Dedupe, boost agreement, rank, then cap."""
        received = list(findings)
        clusters = _cluster(received)

        merged_findings: list[Finding] = []
        merged_count = 0
        agreed = 0

        for cluster in clusters:
            finding, agents = self._merge(cluster)
            merged_findings.append(finding)
            merged_count += len(cluster) - 1
            if agents > 1:
                agreed += 1

        ranked = rank(merged_findings)
        capped = 0
        if self.cap is not None and len(ranked) > self.cap:
            capped = len(ranked) - self.cap
            # Ranked first, truncated second — never the other way round.
            ranked = ranked[: self.cap]

        logger.info(
            "aggregate considered=%d published=%d merged=%d agreed=%d capped=%d",
            len(received),
            len(ranked),
            merged_count,
            agreed,
            capped,
            extra={
                "event": "aggregate",
                "considered": len(received),
                "published": len(ranked),
                "merged": merged_count,
                "agreed": agreed,
                "capped": capped,
            },
        )

        return AggregationResult(
            findings=ranked,
            considered=len(received),
            merged=merged_count,
            agreed=agreed,
            capped=capped,
        )

    def _merge(self, cluster: Sequence[Finding]) -> tuple[Finding, int]:
        """Fold a cluster of overlapping findings into one, and count the agents."""
        if len(cluster) == 1:
            return cluster[0], 1

        # The most severe, then most confident, finding leads: its category and
        # message describe the merged result.
        ordered = rank(list(cluster))
        lead = ordered[0]

        agents: list[str] = []
        for finding in ordered:
            for source in finding.source.split(SOURCE_SEPARATOR):
                if source and source not in agents:
                    agents.append(source)

        boost = min(
            self.agreement_bonus * (len(agents) - 1), self.max_agreement_bonus
        )
        confidence = min(100, max(item.confidence for item in ordered) + boost)

        return (
            replace(
                lead,
                confidence=confidence,
                line_start=min(item.line_start for item in ordered),
                line_end=max(item.line_end for item in ordered),
                evidence=_merge_evidence(ordered),
                source=SOURCE_SEPARATOR.join(agents),
            ),
            len(agents),
        )


def rank(findings: list[Finding]) -> list[Finding]:
    """Sort by severity, then confidence — the order any cap must respect.

    The trailing keys are there only to make ties deterministic, so two runs
    over the same findings publish them in the same order.
    """
    return sorted(
        findings,
        key=lambda finding: (
            -severity_rank(finding.severity),
            -finding.confidence,
            finding.file,
            finding.line_start,
            finding.line_end,
            finding.category,
            finding.source,
            finding.message,
        ),
    )


def aggregate(
    findings: Iterable[Finding], *, cap: int | None = None
) -> AggregationResult:
    """Convenience entry point over :class:`Aggregator`."""
    return Aggregator(cap=cap).combine(findings)


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------


def _cluster(findings: Sequence[Finding]) -> list[list[Finding]]:
    """Group findings that cover overlapping lines of the same file.

    Deterministic: files are visited in sorted order and findings within a file
    in line order, so the clustering does not depend on the order the agents
    happened to finish in.
    """
    by_file: dict[str, list[Finding]] = {}
    for finding in findings:
        by_file.setdefault(finding.file, []).append(finding)

    clusters: list[list[Finding]] = []
    for path in sorted(by_file):
        ordered = sorted(
            by_file[path],
            key=lambda finding: (finding.line_start, finding.line_end),
        )
        current: list[Finding] = []
        current_start = -1
        current_end = -1
        for finding in ordered:
            if _is_wide(finding):
                # A file-level claim: publish it on its own rather than let it
                # swallow the specific findings it happens to span.
                if current:
                    clusters.append(current)
                    current = []
                    current_start = -1
                    current_end = -1
                clusters.append([finding])
                continue
            overlaps = finding.line_start <= current_end
            if current and (
                (overlaps and _comparable(current_start, current_end, finding))
                or _restates(current, current_start, current_end, finding)
            ):
                current.append(finding)
                current_end = max(current_end, finding.line_end)
                continue
            if current:
                clusters.append(current)
            current = [finding]
            current_start = finding.line_start
            current_end = finding.line_end
        if current:
            clusters.append(current)
    return clusters


def _restates(
    cluster: Sequence[Finding],
    cluster_start: int,
    cluster_end: int,
    finding: Finding,
) -> bool:
    """Whether a finding is another agent's citation of the same single statement.

    See :data:`MERGE_ADJACENCY`. All four conditions must hold: the cluster is a
    single line, the candidate is a single line, they are within the adjacency
    gap, and they come from different agents.
    """
    if cluster_start != cluster_end or finding.line_start != finding.line_end:
        return False
    if finding.line_start - cluster_end > MERGE_ADJACENCY:
        return False
    seen = {
        source
        for item in cluster
        for source in item.source.split(SOURCE_SEPARATOR)
    }
    return not seen & set(finding.source.split(SOURCE_SEPARATOR))


def _comparable(cluster_start: int, cluster_end: int, finding: Finding) -> bool:
    """Whether a finding's range is close enough in size to join the cluster."""
    cluster_span = max(cluster_end - cluster_start + 1, 1)
    finding_span = max(finding.line_end - finding.line_start + 1, 1)
    larger, smaller = max(cluster_span, finding_span), min(cluster_span, finding_span)
    return larger / smaller <= MAX_SPAN_RATIO


def _is_wide(finding: Finding) -> bool:
    """Whether a finding's range is too broad to be about one specific thing."""
    return finding.line_end - finding.line_start + 1 > MAX_MERGE_SPAN


def _merge_evidence(ordered: Sequence[Finding]) -> str:
    """Keep every agent's perspective, attributed, in the merged evidence.

    Losing the second agent's reasoning would throw away the very thing that
    makes the merged finding more credible than either half.
    """
    seen: set[tuple[str, str]] = set()
    parts: list[str] = []
    for finding in ordered:
        key = (finding.source, finding.evidence.strip())
        if key in seen or not finding.evidence.strip():
            continue
        seen.add(key)
        parts.append(f"[{finding.source}] {finding.evidence.strip()}")

    if len(parts) > 1:
        header = (
            f"Independently flagged by {len(parts)} reviewers "
            f"({', '.join(dict.fromkeys(f.source for f in ordered))}):"
        )
        return "\n".join([header, *parts])
    return parts[0] if parts else ""
