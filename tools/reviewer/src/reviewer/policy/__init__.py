"""Stage 9: the deterministic policy engine that owns the merge decision.

No model output can flip the gate by itself (CLAUDE.md §2.6). `decide()` reads
finding severities, categories and paths, plus this repository's policy — and
nothing else. Waivers are explicit, narrow and audited; a CRITICAL security
finding cannot be waived at all.
"""

from __future__ import annotations

from .engine import (
    OUTCOMES,
    UNWAIVABLE,
    AppliedWaiver,
    GateDecision,
    GatePolicy,
    Outcome,
    PolicyEngine,
    Waiver,
    decide,
)

__all__ = [
    "OUTCOMES",
    "UNWAIVABLE",
    "AppliedWaiver",
    "GateDecision",
    "GatePolicy",
    "Outcome",
    "PolicyEngine",
    "Waiver",
    "decide",
]
