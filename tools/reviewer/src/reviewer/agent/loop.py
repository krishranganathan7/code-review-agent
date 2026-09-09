"""Stage 4: the agentic tool-use loop (CLAUDE.md §3, §4).

The cycle is small and the discipline around it is the point:

1. call the provider with the bounded conversation and the available tool specs;
2. no tool calls in the response means the model is done — stop and return it;
3. otherwise execute every requested call, append the results as one ``tool``
   turn, and go round again;
4. check the controls on every iteration; when one trips, make **one** final call
   that instructs the model to conclude with what it has, and return.

Three things are deliberately never fatal, because the model has to be able to
recover from them by calling something else:

* a tool returning ``ok=False`` — the error is fed back as the result;
* a tool the loop does not have — a clear "unknown tool" error is fed back;
* a tool raising despite Phase 2's contract — caught and fed back.

Nothing provider-specific and nothing language-specific appears here (§2.1/§2.2).
The loop talks to `LLMProvider` and `Tool`, and every message shape it builds
comes from :mod:`reviewer.types`, so the same code drives either provider.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..providers.base import LLMProvider
from ..tools.base import Tool
from ..types import (
    LLMResponse,
    Message,
    ToolCall,
    ToolResult,
    ToolSpec,
    assistant_turn,
    tool_result_turn,
)
from .context import BoundedView, ContextPolicy, bound_transcript
from .controls import LIMIT_STOPS, Clock, LoopControls, LoopStop, MonotonicClock

__all__ = [
    "AgentLoop",
    "AgentResult",
    "ToolInvocation",
    "ProviderCallRecord",
    "run",
    "CONCLUDE_INSTRUCTION",
    "SEARCH_FIRST_INSTRUCTION",
]

logger = logging.getLogger("reviewer.agent")

CONCLUDE_INSTRUCTION = (
    "Stop searching now: {reason}. Do not request any more tools. "
    "Answer the original task with what you have already found, and say plainly "
    "which parts you could not verify."
)
"""The loop's own words, never mixed with tool output — a control channel."""

SEARCH_FIRST_INSTRUCTION = (
    "You have answered without calling a single tool, so nothing you said is "
    "grounded in this repository: you have not opened the change, and you have "
    "not read any of the code it depends on. That answer is a guess.\n\n"
    "Do not answer again yet. Use the tools first — you have: {tools}. Read the "
    "change itself, then the definitions, callers and tests around it. Once you "
    "have actually looked, give your answer in the format you were asked for."
)
"""Sent once when a run concludes on its first turn having searched nothing.

A control-channel message like `CONCLUDE_INSTRUCTION`: the loop's own words, on
the `user` role, never routed through `untrusted()`. See
`LoopControls.require_tool_use` for why this is enforced rather than prompted.
"""


@dataclass
class ToolInvocation:
    """One tool call and what came back. The unit of the tool trace."""

    iteration: int
    call_id: str
    tool: str
    arguments: dict[str, Any]
    ok: bool
    error: str | None = None
    content_chars: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    unknown_tool: bool = False


@dataclass
class ProviderCallRecord:
    """One model round-trip: what was sent, what bounding did, what came back."""

    iteration: int
    messages_sent: int
    tools_offered: int
    stop_reason: str | None
    tool_calls_requested: int
    usage: dict[str, int] = field(default_factory=dict)
    elided_exchanges: int = 0
    elided_messages: int = 0
    clipped_results: int = 0
    forced_conclusion: bool = False
    tokens: int = 0
    """Tokens this round-trip cost, reported or estimated."""
    usage_estimated: bool = False
    """Whether `tokens` is an estimate because the provider reported no usage."""


@dataclass
class AgentResult:
    """Everything one agentic-search run produced.

    ``transcript`` is the **full** history, including content that was elided or
    clipped from what was actually sent — bounding shapes the view, never the
    record. Later stages read `final_text` for the answer and the traces for the
    evidence trail behind it.
    """

    final_text: str
    stop_reason: LoopStop
    transcript: list[Message] = field(default_factory=list)
    tool_trace: list[ToolInvocation] = field(default_factory=list)
    provider_calls: list[ProviderCallRecord] = field(default_factory=list)
    iterations: int = 0
    tool_calls: int = 0
    tokens_used: int = 0
    elapsed_seconds: float = 0.0
    forced_conclusion: bool = False
    search_nudges: int = 0
    """How many times the run was pushed back for answering without searching."""

    @property
    def searched(self) -> bool:
        """Whether this run called any tool at all."""
        return self.tool_calls > 0

    @property
    def concluded(self) -> bool:
        """Whether the model finished on its own rather than being cut off."""
        return self.stop_reason == "concluded"

    @property
    def limited(self) -> bool:
        """Whether a control tripped."""
        return self.stop_reason in LIMIT_STOPS

    @property
    def failed_tool_calls(self) -> list[ToolInvocation]:
        return [call for call in self.tool_trace if not call.ok]

    def tools_used(self) -> list[str]:
        """Distinct tool names invoked, in first-use order."""
        seen: dict[str, None] = {}
        for call in self.tool_trace:
            seen.setdefault(call.tool, None)
        return list(seen)


class AgentLoop:
    """Runs one agentic-search task to a conclusion or a bounded stop."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: Sequence[Tool] | None = None,
        controls: LoopControls | None = None,
        *,
        context_policy: ContextPolicy | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.provider = provider
        self.tools = list(tools or [])
        self.controls = controls or LoopControls()
        self.context_policy = context_policy or ContextPolicy()
        self.clock = clock or MonotonicClock()
        self._by_name = {tool.name: tool for tool in self.tools}
        self._specs = [tool.spec for tool in self.tools]

    # -- the loop ----------------------------------------------------------

    def run(self, task: str, system_prompt: str) -> AgentResult:
        transcript: list[Message] = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=task),
        ]
        trace: list[ToolInvocation] = []
        calls: list[ProviderCallRecord] = []

        started = self.clock.now()
        iterations = 0
        tool_calls = 0
        tokens_used = 0
        nudged = False

        while True:
            tripped = self.controls.tripped(
                iterations=iterations,
                tool_calls=tool_calls,
                tokens_used=tokens_used,
                elapsed=self.clock.now() - started,
            )
            if tripped is not None:
                return self._conclude_under_limit(
                    tripped,
                    transcript=transcript,
                    trace=trace,
                    calls=calls,
                    iterations=iterations,
                    tool_calls=tool_calls,
                    tokens_used=tokens_used,
                    started=started,
                    nudged=nudged,
                )

            iterations += 1
            response, record = self._call_provider(
                transcript, iterations, offer_tools=True
            )
            calls.append(record)
            tokens_used += record.tokens
            transcript.append(assistant_turn(response))

            if not response.tool_calls:
                if self._should_nudge(nudged, tool_calls):
                    nudged = True
                    logger.warning(
                        "loop_no_search iteration=%d tools_offered=%d",
                        iterations,
                        len(self._specs),
                        extra={
                            "event": "loop_no_search",
                            "iteration": iterations,
                            "tools_offered": len(self._specs),
                        },
                    )
                    transcript.append(
                        Message(
                            role="user",
                            content=SEARCH_FIRST_INSTRUCTION.format(
                                tools=", ".join(sorted(self._by_name))
                            ),
                        )
                    )
                    continue

                return AgentResult(
                    final_text=response.text,
                    stop_reason="concluded",
                    transcript=transcript,
                    tool_trace=trace,
                    provider_calls=calls,
                    iterations=iterations,
                    tool_calls=tool_calls,
                    tokens_used=tokens_used,
                    elapsed_seconds=self.clock.now() - started,
                    search_nudges=1 if nudged else 0,
                )

            results: list[tuple[str, ToolResult]] = []
            for call in response.tool_calls:
                invocation, result = self._invoke(call, iterations)
                trace.append(invocation)
                results.append((call.id, result))
                tool_calls += 1

            # One turn carrying every result, whatever shape the provider wants
            # it in — the Phase 1 adapters batch or fan out as required.
            transcript.append(tool_result_turn(results))

    # -- the forced-conclusion path ---------------------------------------

    def _should_nudge(self, already_nudged: bool, tool_calls: int) -> bool:
        """Whether to push back on an answer given without searching.

        Once per run, only when this run has called nothing at all, and only
        when there were tools to call.
        """
        return (
            self.controls.require_tool_use
            and not already_nudged
            and tool_calls == 0
            and bool(self._specs)
        )

    def _conclude_under_limit(
        self,
        stop: LoopStop,
        *,
        transcript: list[Message],
        trace: list[ToolInvocation],
        calls: list[ProviderCallRecord],
        iterations: int,
        tool_calls: int,
        tokens_used: int,
        started: float,
        nudged: bool = False,
    ) -> AgentResult:
        """A tripped limit ends in a bounded answer, never a hang or a half-state.

        The instruction is the loop's own message, and the call offers **no
        tools** — that is what makes the conclusion forced rather than merely
        requested. Any tool call the model returns anyway is ignored, not
        executed.
        """
        logger.info(
            "loop_limit stop=%s iterations=%d tool_calls=%d tokens=%d",
            stop,
            iterations,
            tool_calls,
            tokens_used,
            extra={
                "event": "loop_limit",
                "stop_reason": stop,
                "iterations": iterations,
                "tool_calls": tool_calls,
                "tokens_used": tokens_used,
            },
        )

        transcript.append(
            Message(
                role="user",
                content=CONCLUDE_INSTRUCTION.format(
                    reason=self.controls.describe(stop)
                ),
            )
        )

        response, record = self._call_provider(
            transcript, iterations + 1, offer_tools=False, forced=True
        )
        calls.append(record)
        tokens_used += record.tokens
        transcript.append(assistant_turn(response))

        return AgentResult(
            final_text=response.text,
            stop_reason=stop,
            transcript=transcript,
            tool_trace=trace,
            provider_calls=calls,
            iterations=iterations,
            tool_calls=tool_calls,
            tokens_used=tokens_used,
            elapsed_seconds=self.clock.now() - started,
            forced_conclusion=True,
            search_nudges=1 if nudged else 0,
        )

    # -- provider and tool plumbing ---------------------------------------

    def _call_provider(
        self,
        transcript: list[Message],
        iteration: int,
        *,
        offer_tools: bool,
        forced: bool = False,
    ) -> tuple[LLMResponse, ProviderCallRecord]:
        view: BoundedView = bound_transcript(transcript, self.context_policy)
        specs: list[ToolSpec] | None = self._specs if offer_tools and self._specs else None

        logger.info(
            "loop_provider_call iteration=%d messages=%d tools=%d bounded=%s",
            iteration,
            len(view.messages),
            len(specs or []),
            view.bounded,
            extra={
                "event": "loop_provider_call",
                "iteration": iteration,
                "messages_sent": len(view.messages),
                "tools_offered": len(specs or []),
                "elided_exchanges": view.elided_exchanges,
                "clipped_results": view.clipped_results,
                "forced_conclusion": forced,
            },
        )

        response = self.provider.complete(
            view.messages, tools=specs, max_tokens=self.controls.max_tokens_per_call
        )

        reported = bool(
            {"total_tokens", "input_tokens", "output_tokens"} & set(response.usage)
        )
        if not reported:
            logger.warning(
                "loop_usage_estimated iteration=%d",
                iteration,
                extra={"event": "loop_usage_estimated", "iteration": iteration},
            )

        return response, ProviderCallRecord(
            iteration=iteration,
            messages_sent=len(view.messages),
            tools_offered=len(specs or []),
            stop_reason=response.stop_reason,
            tool_calls_requested=len(response.tool_calls),
            usage=dict(response.usage),
            elided_exchanges=view.elided_exchanges,
            elided_messages=view.elided_messages,
            clipped_results=view.clipped_results,
            forced_conclusion=forced,
            tokens=_tokens_of(response, view.messages),
            usage_estimated=not reported,
        )

    def _invoke(self, call: ToolCall, iteration: int) -> tuple[ToolInvocation, ToolResult]:
        """Run one requested tool call. Never raises."""
        tool = self._by_name.get(call.name)

        if tool is None:
            result = ToolResult(
                content="",
                ok=False,
                error=(
                    f"unknown tool {call.name!r}. Available tools: "
                    f"{', '.join(sorted(self._by_name)) or 'none'}."
                ),
                metadata={"unknown_tool": True},
            )
            invocation = ToolInvocation(
                iteration=iteration,
                call_id=call.id,
                tool=call.name,
                arguments=dict(call.arguments),
                ok=False,
                error=result.error,
                unknown_tool=True,
            )
            logger.warning(
                "loop_unknown_tool iteration=%d tool=%s",
                iteration,
                call.name,
                extra={
                    "event": "loop_unknown_tool",
                    "iteration": iteration,
                    "tool": call.name,
                },
            )
            return invocation, result

        try:
            result = tool.run(**call.arguments)
        except Exception as exc:  # pragma: no cover - Phase 2 tools do not raise
            # A tool breaking its contract must not end the run: report it as a
            # failed result so the model can try something else.
            result = ToolResult(
                content="",
                ok=False,
                error=f"tool {call.name!r} raised {type(exc).__name__}: {exc}",
                metadata={"raised": True},
            )

        return (
            ToolInvocation(
                iteration=iteration,
                call_id=call.id,
                tool=call.name,
                arguments=dict(call.arguments),
                ok=result.ok,
                error=result.error,
                content_chars=len(result.content),
                metadata=dict(result.metadata),
            ),
            result,
        )


CHARS_PER_TOKEN = 4
"""Rough characters-per-token, for providers that report no usage.

Deliberately crude. It exists so `token_budget` is an enforceable control rather
than an inert one — a provider that reports nothing (the CLI adapter, a stub, a
provider whose usage block is missing on some responses) used to leave the
budget permanently at zero, so it never tripped however long the run went. An
approximate bound that fires is worth more than an exact one that cannot.
"""


def _tokens_of(response: LLMResponse, sent: Sequence[Message] | None = None) -> int:
    """Tokens a response cost, from whichever shared key the provider filled in.

    Falls back to a character-count estimate when the provider reports no usage,
    so the budget still converges. Estimated calls are marked by the caller.
    """
    usage = response.usage
    if "total_tokens" in usage:
        return int(usage["total_tokens"])
    if "input_tokens" in usage or "output_tokens" in usage:
        return int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
    return _estimate_tokens(response, sent or [])


def _estimate_tokens(response: LLMResponse, sent: Sequence[Message]) -> int:
    """A characters/4 estimate of one round-trip, prompt included."""
    characters = len(response.text)
    for call in response.tool_calls:
        characters += len(call.name) + sum(
            len(str(value)) for value in call.arguments.values()
        )
    for message in sent:
        if isinstance(message.content, str):
            characters += len(message.content)
        else:
            for part in message.content:
                characters += len(getattr(part, "text", "") or "") + len(
                    getattr(part, "content", "") or ""
                )
    return max(1, characters // CHARS_PER_TOKEN)


def run(
    task: str,
    system_prompt: str,
    provider: LLMProvider,
    tools: Sequence[Tool] | None = None,
    controls: LoopControls | None = None,
    *,
    context_policy: ContextPolicy | None = None,
    clock: Clock | None = None,
) -> AgentResult:
    """Run one agentic-search task. Convenience wrapper over :class:`AgentLoop`."""
    return AgentLoop(
        provider,
        tools,
        controls,
        context_policy=context_policy,
        clock=clock,
    ).run(task, system_prompt)
