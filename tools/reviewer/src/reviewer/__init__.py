"""reviewer — an LLM-provider-agnostic, codebase-agnostic PR review agent.

The pull request is the trigger, not the whole context: the agent explores the
repository on demand with read-only tools, reasons about the change with
specialized review agents, and produces verified, evidence-backed findings.
Deterministic controls remain authoritative — no model output can, on its own,
approve or block a merge (CLAUDE.md §1, §2.6).

Phase 1 exposes the stable core shapes plus the provider seam.
"""

from __future__ import annotations

from .findings.finding import (
    CATEGORIES,
    SEVERITIES,
    SEVERITY_ORDER,
    VERIFICATIONS,
    Category,
    Finding,
    Severity,
    Verification,
)
from .languages.base import LanguageAdapter
from .providers import LLMProvider, ProviderConfig, ProviderError, build_provider
from .tools.base import Tool
from .types import (
    ROLES,
    STOP_REASONS,
    ContentPart,
    LLMResponse,
    Message,
    Role,
    StopReason,
    Symbol,
    TextPart,
    ToolCall,
    ToolCallPart,
    ToolResult,
    ToolResultPart,
    ToolSpec,
    assistant_turn,
    text_of,
    tool_result_turn,
)

__all__ = [
    "CATEGORIES",
    "SEVERITIES",
    "SEVERITY_ORDER",
    "VERIFICATIONS",
    "ROLES",
    "STOP_REASONS",
    "Category",
    "ContentPart",
    "Finding",
    "LLMProvider",
    "LLMResponse",
    "LanguageAdapter",
    "Message",
    "ProviderConfig",
    "ProviderError",
    "Role",
    "Severity",
    "StopReason",
    "Symbol",
    "TextPart",
    "Tool",
    "ToolCall",
    "ToolCallPart",
    "ToolResult",
    "ToolResultPart",
    "ToolSpec",
    "Verification",
    "__version__",
    "assistant_turn",
    "build_provider",
    "text_of",
    "tool_result_turn",
]

__version__ = "0.0.0"
