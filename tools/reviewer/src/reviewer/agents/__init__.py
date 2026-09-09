"""Stage 5: specialized review agents (bug, security, architecture).

Each is an agentic-search task: the Phase 3 loop, a concern-specific system
prompt, the Phase 2 tools over the Phase 4.5 head workspace, and a bounded
context — its change group, never the whole repository (CLAUDE.md §2.3).

Which agents run is decided by the risk *signals* (`selection.plan_review`); how
deep they may search is decided by the risk *level*. Every finding they emit is
validated against the canonical `Finding` schema and dropped if it cannot be
supported by evidence (§2.4).

Nothing here is provider- or language-specific: agents reach the model only
through `LLMProvider`, and the model reaches code only through `Tool`.
"""

from __future__ import annotations

from .architecture import ArchitectureAgent
from .base import (
    CORROBORATION_BONUS,
    DEFAULT_CONFIDENCE,
    LIMITED_PENALTY,
    OUTPUT_CONTRACT,
    AgentContext,
    AgentReview,
    DroppedFinding,
    ReviewAgent,
)
from .bug import BugAgent
from .security import SecurityAgent
from .selection import (
    ALWAYS_RUN,
    ARCHITECTURE_SIGNALS,
    SECURITY_SIGNALS,
    SIGNAL_MAP,
    AgentPlan,
    AgentSelection,
    available_agents,
    plan_review,
    select_agents,
)

__all__ = [
    "ALWAYS_RUN",
    "ARCHITECTURE_SIGNALS",
    "CORROBORATION_BONUS",
    "DEFAULT_CONFIDENCE",
    "LIMITED_PENALTY",
    "OUTPUT_CONTRACT",
    "SECURITY_SIGNALS",
    "SIGNAL_MAP",
    "AgentContext",
    "AgentPlan",
    "AgentReview",
    "AgentSelection",
    "ArchitectureAgent",
    "BugAgent",
    "DroppedFinding",
    "ReviewAgent",
    "SecurityAgent",
    "available_agents",
    "plan_review",
    "select_agents",
]
