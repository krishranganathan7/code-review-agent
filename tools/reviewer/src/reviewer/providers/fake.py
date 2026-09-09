"""``FakeProvider`` — a scripted :class:`~reviewer.providers.base.LLMProvider`.

No network, no SDK, no model. You hand it an ordered script of turns and it
returns them one per :meth:`FakeProvider.complete` call, recording what it was
asked. This is what Phase 3's agentic loop and Phase 5's review agents are tested
against.

Scripting is deliberately terse — a bare string is a text turn::

    provider = FakeProvider([
        tool_turn("grep", pattern="def handle_pr"),
        tool_turn(
            ("read_file", {"path": "src/pr.py"}),
            ("grep", {"pattern": "handle_pr"}),
        ),
        "Found it: src/pr.py:42 does not check the base ref.",
    ])

Tool-call ids are assigned deterministically (``call_1``, ``call_2``, ...) across
the whole script, so a test can bind results back without capturing ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..types import LLMResponse, Message, ToolCall, ToolSpec
from .base import ProviderError, log_call, log_response

__all__ = [
    "FakeProvider",
    "RecordedCall",
    "ScriptExhausted",
    "Turn",
    "text_turn",
    "tool_turn",
]

PROVIDER = "fake"


class ScriptExhausted(ProviderError):
    """The loop asked for more turns than the script provides.

    Raised rather than repeating the last turn, so a runaway loop fails loudly
    in tests instead of spinning.
    """


@dataclass
class Turn:
    """One scripted model turn."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class RecordedCall:
    """What the provider was asked for on one :meth:`FakeProvider.complete` call."""

    messages: list[Message]
    tools: list[ToolSpec]
    max_tokens: int


ScriptItem = str | Turn | LLMResponse
"""What a script entry may be: a bare string, a :class:`Turn`, or a full response."""


def text_turn(text: str, *, stop_reason: str | None = None) -> Turn:
    """A plain text turn. Equivalent to putting the bare string in the script."""
    return Turn(text=text, stop_reason=stop_reason or "end_turn")


def tool_turn(
    *calls: str | tuple[str, dict[str, Any]],
    text: str = "",
    **arguments: Any,
) -> Turn:
    """A turn in which the model requests one or more tools.

    Two forms::

        tool_turn("grep", pattern="def foo")              # one call, kwargs args
        tool_turn(("grep", {"pattern": "x"}),             # several calls at once
                  ("read_file", {"path": "a.py"}))
    """
    if not calls:
        raise ValueError("tool_turn requires at least one tool call")
    if arguments and any(not isinstance(call, str) for call in calls):
        raise ValueError(
            "tool_turn: pass arguments as kwargs for the single-call form, or "
            "inline per call for the multi-call form — not both"
        )

    pending: list[tuple[str, dict[str, Any]]] = []
    for call in calls:
        if isinstance(call, str):
            pending.append((call, dict(arguments)))
        else:
            name, call_arguments = call
            pending.append((name, dict(call_arguments)))

    return Turn(
        text=text,
        # ids are filled in by FakeProvider so they are unique across the script
        tool_calls=[ToolCall(id="", name=name, arguments=args) for name, args in pending],
        stop_reason="tool_use",
    )


class FakeProvider:
    """Replays a scripted sequence of turns, recording every request."""

    native_tools = True
    """Scripted tool calls arrive as structured turns, like a native SDK's."""

    def __init__(self, script: Sequence[ScriptItem] | None = None) -> None:
        self.calls: list[RecordedCall] = []
        self._responses: list[LLMResponse] = []
        self._next_call_id = 1
        for item in script or []:
            self._responses.append(self._as_response(item))

    # -- LLMProvider -------------------------------------------------------

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        log_call(PROVIDER, "scripted", messages, tools)
        self.calls.append(
            RecordedCall(
                messages=list(messages),
                tools=list(tools or []),
                max_tokens=max_tokens,
            )
        )
        if not self._responses:
            raise ScriptExhausted(
                f"FakeProvider script exhausted after {len(self.calls) - 1} turn(s)"
            )
        response = self._responses.pop(0)
        log_response(PROVIDER, "scripted", response)
        return response

    # -- introspection for tests ------------------------------------------

    @property
    def call_count(self) -> int:
        """How many times the loop called the model."""
        return len(self.calls)

    @property
    def remaining(self) -> int:
        """Scripted turns not yet consumed."""
        return len(self._responses)

    def last_call(self) -> RecordedCall:
        """The most recent request. Raises if the provider was never called."""
        if not self.calls:
            raise ProviderError("FakeProvider has not been called")
        return self.calls[-1]

    # -- script normalization ---------------------------------------------

    def _as_response(self, item: ScriptItem) -> LLMResponse:
        if isinstance(item, LLMResponse):
            return item
        if isinstance(item, str):
            item = text_turn(item)
        return LLMResponse(
            text=item.text,
            tool_calls=[self._with_id(call) for call in item.tool_calls],
            stop_reason=item.stop_reason
            or ("tool_use" if item.tool_calls else "end_turn"),
            usage=dict(item.usage),
        )

    def _with_id(self, call: ToolCall) -> ToolCall:
        if call.id:
            return ToolCall(id=call.id, name=call.name, arguments=dict(call.arguments))
        call_id = f"call_{self._next_call_id}"
        self._next_call_id += 1
        return ToolCall(id=call_id, name=call.name, arguments=dict(call.arguments))
