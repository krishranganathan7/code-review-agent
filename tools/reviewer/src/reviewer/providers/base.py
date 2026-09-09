"""The :class:`LLMProvider` interface — the only way the system calls a model.

Hard constraint (CLAUDE.md §2.1): no provider SDK is imported anywhere except
inside an adapter in this package. Swapping providers must require zero changes
outside :mod:`reviewer.providers`.

The adapter's job is translation only, in both directions:

* our :class:`~reviewer.types.ToolSpec` list into the provider's native tool
  schema, and our :class:`~reviewer.types.Message` list into its native message
  shape (including how *that* provider wants a tool result returned);
* the provider's native response back into a normalized
  :class:`~reviewer.types.LLMResponse` — text, :class:`~reviewer.types.ToolCall`
  list, a stop reason drawn from :data:`~reviewer.types.STOP_REASONS`, and usage
  under shared keys.

We own the agentic loop (:mod:`reviewer.agent`), not the provider. There is no
shared third-party model-routing layer; each provider gets its own adapter
written against that provider's official SDK.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from ..types import LLMResponse, Message, StopReason, ToolSpec

__all__ = [
    "LLMProvider",
    "ProviderError",
    "log_call",
    "log_response",
    "normalize_stop_reason",
]

logger = logging.getLogger("reviewer.providers")


class ProviderError(RuntimeError):
    """A provider-layer failure, raised in provider-neutral form.

    Adapters translate their SDK's exceptions and malformed-response conditions
    into this type, so nothing outside :mod:`reviewer.providers` ever catches a
    provider-specific exception class.
    """


@runtime_checkable
class LLMProvider(Protocol):
    """A normalized, provider-independent chat-completion call."""

    native_tools: bool
    """Whether this provider's tool-calling is native, or prompted.

    Native means the model emits a tool call as a first-class action the SDK
    returns as structured output. Prompted means the adapter describes the tools
    in the prompt and asks the model to name them in a reply envelope, which the
    adapter then parses.

    The difference is not cosmetic, and stage 5 has to know about it. A model
    with native tool-calling will search when the task requires it; a model
    being asked to search *through a prompt* competes with every other
    instruction it was given, and loses. Measured: with the diff withheld and
    six tools offered, a prompted provider made zero tool calls on a HIGH-risk
    auth PR and returned zero findings, while the identical task with the diff
    inlined returned nine verified findings.

    So this is a capability of the provider, normalized like any other, not a
    fact about which provider is in use — nothing outside `providers/` may ask
    that (§2.1, §2.2). What callers do with it is their business; agents use it
    to decide whether the change must be handed over or can be fetched.
    """

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send ``messages`` (and optionally callable ``tools``) to the model.

        Returns a normalized :class:`~reviewer.types.LLMResponse`: text plus any
        tool calls the model requested. Raises :class:`ProviderError`, never a
        provider-specific exception type, across this boundary.
        """
        ...


def log_call(
    provider: str, model: str, messages: list[Message], tools: list[ToolSpec] | None
) -> None:
    """Structured log of an outbound model call.

    Message *content* is deliberately not logged — it carries untrusted
    repository data (CLAUDE.md §2.5) and can be large. Phase 7 hardens tracing.
    """
    logger.info(
        "llm_call provider=%s model=%s messages=%d tools=%d",
        provider,
        model,
        len(messages),
        len(tools or []),
        extra={
            "event": "llm_call",
            "provider": provider,
            "model": model,
            "message_count": len(messages),
            "tool_count": len(tools or []),
        },
    )


def log_response(provider: str, model: str, response: LLMResponse) -> None:
    """Structured log of a normalized model response."""
    logger.info(
        "llm_response provider=%s model=%s stop=%s tool_calls=%d",
        provider,
        model,
        response.stop_reason,
        len(response.tool_calls),
        extra={
            "event": "llm_response",
            "provider": provider,
            "model": model,
            "stop_reason": response.stop_reason,
            "tool_call_count": len(response.tool_calls),
            "tool_names": [call.name for call in response.tool_calls],
            "usage": response.usage,
        },
    )


def normalize_stop_reason(native: str | None, mapping: dict[str, StopReason]) -> str | None:
    """Map a provider's native stop reason onto :data:`~reviewer.types.STOP_REASONS`.

    An unrecognized value becomes ``"other"`` rather than leaking a native token
    into the loop. ``None`` stays ``None``.
    """
    if native is None:
        return None
    return mapping.get(native, "other")
