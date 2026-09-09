"""Provider-neutral, language-neutral core types.

These are the shared value objects that cross package boundaries: messages and
tool-calling shapes used by :mod:`reviewer.providers`, the tool result shape used
by :mod:`reviewer.tools`, and the symbol shape produced by
:mod:`reviewer.languages`.

Hard constraint (CLAUDE.md §2.1/§2.2): nothing here may reference a specific LLM
provider or a specific programming language. Provider adapters translate these
shapes to and from their native formats; nothing outside a provider adapter sees
a native format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

__all__ = [
    "Role",
    "ROLES",
    "StopReason",
    "STOP_REASONS",
    "TextPart",
    "ToolCallPart",
    "ToolResultPart",
    "ContentPart",
    "Message",
    "ToolSpec",
    "ToolCall",
    "LLMResponse",
    "ToolResult",
    "Symbol",
    "as_parts",
    "text_of",
    "assistant_turn",
    "tool_result_turn",
]

Role = Literal["system", "user", "assistant", "tool"]
"""Provider-neutral message roles."""

ROLES: tuple[Role, ...] = ("system", "user", "assistant", "tool")

StopReason = Literal[
    "end_turn",
    "tool_use",
    "max_tokens",
    "stop_sequence",
    "refusal",
    "other",
]
"""Provider-neutral reason a generation stopped.

Each adapter maps its provider's native vocabulary onto this set, so the agentic
loop branches on one vocabulary regardless of who served the request.
"""

STOP_REASONS: tuple[StopReason, ...] = (
    "end_turn",
    "tool_use",
    "max_tokens",
    "stop_sequence",
    "refusal",
    "other",
)


# --------------------------------------------------------------------------
# Message content
#
# `Message.content` is `str | list[ContentPart]`. The plain string is the common
# path and stays the common path — a part list is only needed when a turn carries
# something a string cannot express: an assistant turn that requested tools, or a
# turn returning several tool results at once.
# --------------------------------------------------------------------------


@dataclass
class TextPart:
    """Plain text within a message."""

    text: str
    type: Literal["text"] = "text"


@dataclass
class ToolCallPart:
    """A tool call the assistant made, replayed back into conversation history.

    Both providers require the assistant turn that requested a tool to be present
    in history before the matching result is sent, so the loop must be able to
    express "the assistant asked for this" as a :class:`Message`.
    """

    call: ToolCall
    type: Literal["tool_call"] = "tool_call"


@dataclass
class ToolResultPart:
    """The result of one tool call, bound to the call that requested it."""

    tool_call_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


ContentPart = TextPart | ToolCallPart | ToolResultPart
"""Tagged union of message content parts, discriminated by ``.type``."""


@dataclass
class Message:
    """One turn in a conversation, in provider-neutral form.

    ``content`` is either a plain string (the common case) or a list of
    :data:`ContentPart`. ``tool_call_id`` is a convenience for the single-result
    case: a ``tool`` role message with string content and a ``tool_call_id`` is
    equivalent to one :class:`ToolResultPart`.
    """

    role: Role
    content: str | list[ContentPart]
    tool_call_id: str | None = None


@dataclass
class ToolSpec:
    """Provider-neutral declaration of a tool the model may call.

    ``parameters`` is a JSON Schema object. A provider adapter converts this into
    whatever tool/function schema its API expects.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """A model's request to invoke a tool, normalized across providers."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """A normalized model response: text plus any requested tool calls."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class ToolResult:
    """The outcome of running a tool.

    Tool output is untrusted *data* (CLAUDE.md §2.5). ``content`` is never treated
    as instructions to the model.
    """

    content: str
    ok: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Symbol:
    """A named code entity a :class:`~reviewer.languages.base.LanguageAdapter`
    extracted from a change."""

    name: str
    kind: str
    file: str
    line_start: int
    line_end: int


# --------------------------------------------------------------------------
# Helpers — provider-neutral, used by adapters and (from Phase 3) by the loop.
# --------------------------------------------------------------------------


def as_parts(message: Message) -> list[ContentPart]:
    """Normalize a message's content to a part list.

    A ``tool`` role message with string content becomes a single
    :class:`ToolResultPart`; any other string content becomes one
    :class:`TextPart`.
    """
    if isinstance(message.content, str):
        if message.role == "tool" and message.tool_call_id is not None:
            return [
                ToolResultPart(
                    tool_call_id=message.tool_call_id, content=message.content
                )
            ]
        return [TextPart(text=message.content)]
    return list(message.content)


def text_of(message: Message) -> str:
    """All text in a message, concatenated. Non-text parts are ignored."""
    return "".join(p.text for p in as_parts(message) if isinstance(p, TextPart))


def assistant_turn(response: LLMResponse) -> Message:
    """Build the assistant history entry for a model response.

    The loop appends this before sending tool results back, so the provider sees
    its own tool request replayed in the position it requires.
    """
    if not response.tool_calls:
        return Message(role="assistant", content=response.text)
    parts: list[ContentPart] = []
    if response.text:
        parts.append(TextPart(text=response.text))
    parts.extend(ToolCallPart(call=call) for call in response.tool_calls)
    return Message(role="assistant", content=parts)


def tool_result_turn(results: Iterable[tuple[str, ToolResult]]) -> Message:
    """Build one ``tool`` turn carrying the results of one or more tool calls.

    Takes ``(tool_call_id, result)`` pairs. Adapters decide how this lands
    natively — one batched message or one message per result.
    """
    parts: list[ContentPart] = [
        ToolResultPart(
            tool_call_id=call_id,
            content=result.content if result.ok else (result.error or result.content),
            is_error=not result.ok,
        )
        for call_id, result in results
    ]
    return Message(role="tool", content=parts)
