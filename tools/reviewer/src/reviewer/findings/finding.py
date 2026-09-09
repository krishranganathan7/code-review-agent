"""The canonical :class:`Finding` schema — one shape used everywhere.

Every stage that produces, transforms, ranks, verifies, or publishes a review
result uses this dataclass. There is no second finding shape (CLAUDE.md §5).

Phase 0: the dataclass and the documented value sets. No validation and no
behavior — enforcement of the vocabularies belongs to the stages that build and
verify findings in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "Category",
    "CATEGORIES",
    "Severity",
    "SEVERITIES",
    "SEVERITY_ORDER",
    "Verification",
    "VERIFICATIONS",
    "Finding",
    "severity_rank",
]

Category = Literal[
    "bug",
    "security",
    "architecture",
    "performance",
    "test",
    "docs",
    "dependency",
    "standards",
]

CATEGORIES: tuple[Category, ...] = (
    "bug",
    "security",
    "architecture",
    "performance",
    "test",
    "docs",
    "dependency",
    "standards",
)

Severity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

SEVERITIES: tuple[Severity, ...] = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

SEVERITY_ORDER: dict[Severity, int] = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}
"""Ascending rank. Used by the aggregator to rank *before* capping (CLAUDE.md §2.8)."""

Verification = Literal["UNVERIFIED", "VERIFIED", "REFUTED"]

VERIFICATIONS: tuple[Verification, ...] = ("UNVERIFIED", "VERIFIED", "REFUTED")


def severity_rank(severity: str) -> int:
    """Ascending rank for a severity given as a plain `str`.

    `Finding.severity` is typed `str` per CLAUDE.md §5, while
    :data:`SEVERITY_ORDER` is keyed by the `Severity` literal. This is the one
    place that bridges the two, so callers ranking findings do not each need a
    cast. An unrecognized severity sorts below LOW rather than raising — an
    out-of-vocabulary value should be treated as least important, not fatal.
    """
    for level in SEVERITIES:
        if severity == level:
            return SEVERITY_ORDER[level]
    return -1


@dataclass
class Finding:
    """One evidence-backed review result.

    Field types are ``str``/``int`` per CLAUDE.md §5. The documented vocabularies
    live alongside as :data:`CATEGORIES`, :data:`SEVERITIES` and
    :data:`VERIFICATIONS`.

    Attributes:
        category: one of :data:`CATEGORIES`.
        severity: one of :data:`SEVERITIES`.
        confidence: 0-100.
        file: repository-relative path.
        line_start: first line of the range (1-indexed).
        line_end: last line of the range (inclusive).
        message: what is wrong and why, in plain language.
        evidence: concrete support — cited code, a relationship, a test, tool output.
        source: which agent or check produced it.
        blocking: whether this finding, by policy, can block a merge.
        verification: one of :data:`VERIFICATIONS`.
    """

    category: str
    severity: str
    confidence: int
    file: str
    line_start: int
    line_end: int
    message: str
    evidence: str
    source: str
    blocking: bool
    verification: str
