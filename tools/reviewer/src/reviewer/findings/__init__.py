"""Stages 6-8: the canonical `Finding`, aggregation, verification, judge.

The `Finding` schema is stable from Phase 0. Stages 6 and 7 are deterministic —
no model is consulted. Stage 8, the judge, is the one model call, and it runs on
a context built only from the findings themselves.
"""

from __future__ import annotations

from .aggregator import (
    AGREEMENT_BONUS,
    MAX_AGREEMENT_BONUS,
    AggregationResult,
    Aggregator,
    aggregate,
    rank,
)
from .finding import (
    CATEGORIES,
    SEVERITIES,
    SEVERITY_ORDER,
    VERIFICATIONS,
    Category,
    Finding,
    Severity,
    Verification,
    severity_rank,
)
from .judge import Judge, JudgeResult, JudgeVerdict, RejectedFinding
from .verification import RefutedFinding, VerificationResult, Verifier, verify

__all__ = [
    "AGREEMENT_BONUS",
    "CATEGORIES",
    "MAX_AGREEMENT_BONUS",
    "SEVERITIES",
    "SEVERITY_ORDER",
    "VERIFICATIONS",
    "AggregationResult",
    "Aggregator",
    "Category",
    "Finding",
    "Judge",
    "JudgeResult",
    "JudgeVerdict",
    "RefutedFinding",
    "RejectedFinding",
    "Severity",
    "Verification",
    "VerificationResult",
    "Verifier",
    "aggregate",
    "rank",
    "severity_rank",
    "verify",
]
