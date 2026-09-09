"""Stage 4: the agentic tool-use loop and its budget controls (CLAUDE.md §4).

`max_tool_calls`, token budget, timeout; on any limit the loop stops and the
model is forced to conclude with what it has. Context growth is bounded by a
configurable policy — the view sent to the provider is capped while the full
transcript is preserved for tracing.

Nothing here is provider- or language-specific: the loop drives `LLMProvider` and
`Tool` through provider-neutral message shapes.
"""

from __future__ import annotations

from .context import BoundedView, ContextPolicy, bound_transcript, untrusted
from .controls import (
    LIMIT_STOPS,
    Clock,
    LoopControls,
    LoopStop,
    ManualClock,
    MonotonicClock,
)
from .loop import (
    CONCLUDE_INSTRUCTION,
    AgentLoop,
    AgentResult,
    ProviderCallRecord,
    ToolInvocation,
    run,
)

__all__ = [
    "CONCLUDE_INSTRUCTION",
    "LIMIT_STOPS",
    "AgentLoop",
    "AgentResult",
    "BoundedView",
    "Clock",
    "ContextPolicy",
    "LoopControls",
    "LoopStop",
    "ManualClock",
    "MonotonicClock",
    "ProviderCallRecord",
    "ToolInvocation",
    "bound_transcript",
    "run",
    "untrusted",
]
