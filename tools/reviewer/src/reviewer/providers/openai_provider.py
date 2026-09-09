"""OpenAI adapter for :class:`~reviewer.providers.base.LLMProvider`.

Written against the official ``openai`` SDK (Chat Completions). This module and
its sibling adapters are the only places in the codebase permitted to import a
provider SDK (CLAUDE.md §2.1).

Native shapes this adapter owns, and hides from everything else:

* tools are ``{"type": "function", "function": {...}}`` with a ``parameters``
  JSON Schema;
* the system prompt is an ordinary message with ``role: "system"``;
* a tool result is its **own message** with ``role: "tool"`` and a
  ``tool_call_id`` — so a turn carrying several results **fans out** into several
  messages, the opposite of Anthropic's batching;
* an assistant turn that called tools replays as a ``tool_calls`` array whose
  arguments are a **JSON string**, not an object.
"""

from __future__ import annotations

import json
from typing import Any

import openai

from ..types import (
    ContentPart,
    LLMResponse,
    Message,
    StopReason,
    TextPart,
    ToolCall,
    ToolCallPart,
    ToolResultPart,
    ToolSpec,
    as_parts,
)
from .base import ProviderError, log_call, log_response, normalize_stop_reason

__all__ = ["OpenAIProvider"]

PROVIDER = "openai"

STOP_REASONS: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class OpenAIProvider:
    """Calls OpenAI models through the ``openai`` SDK.

    ``client`` exists for tests: pass a stand-in and no SDK client is
    constructed, so nothing touches the network. In production leave it unset —
    with no ``api_key`` the SDK reads ``OPENAI_API_KEY`` itself.
    """

    native_tools = True
    """The SDK carries tool schemas and returns tool_calls directly."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        if client is not None:
            self._client: Any = client
            return
        try:
            self._client = (
                openai.OpenAI(api_key=api_key)
                if api_key is not None
                else openai.OpenAI()
            )
        except openai.OpenAIError as exc:
            raise ProviderError(
                f"could not construct the openai client ({exc}); "
                "set OPENAI_API_KEY or pass api_key"
            ) from exc

    # -- LLMProvider -------------------------------------------------------

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        log_call(PROVIDER, self.model, messages, tools)

        request: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "messages": self.to_native_messages(messages),
        }
        if tools:
            request["tools"] = [self.to_native_tool(tool) for tool in tools]

        try:
            raw = self._client.chat.completions.create(**request)
        except openai.OpenAIError as exc:
            raise ProviderError(f"openai call failed: {exc}") from exc

        response = self.from_native_response(raw)
        log_response(PROVIDER, self.model, response)
        return response

    # -- outbound: ours -> native -----------------------------------------

    @staticmethod
    def to_native_tool(tool: ToolSpec) -> dict[str, Any]:
        """A :class:`~reviewer.types.ToolSpec` as an OpenAI function tool."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
                or {"type": "object", "properties": {}},
            },
        }

    @classmethod
    def to_native_messages(cls, messages: list[Message]) -> list[dict[str, Any]]:
        """Our messages as OpenAI chat messages.

        One input turn may produce several output messages: OpenAI binds each
        tool result to its own ``role: "tool"`` message, so a turn holding N
        results fans out into N messages.
        """
        native: list[dict[str, Any]] = []

        for message in messages:
            parts = as_parts(message)
            results = [p for p in parts if isinstance(p, ToolResultPart)]
            if results:
                if len(results) != len(parts):
                    raise ProviderError(
                        "openai: a tool-result turn may not carry other content"
                    )
                native.extend(cls._tool_result_message(p) for p in results)
                continue

            native.append(cls._standard_message(message.role, parts))

        return native

    @staticmethod
    def _tool_result_message(part: ToolResultPart) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": part.tool_call_id,
            "content": part.content,
        }

    @staticmethod
    def _standard_message(role: str, parts: list[ContentPart]) -> dict[str, Any]:
        text = "".join(p.text for p in parts if isinstance(p, TextPart))
        calls = [p.call for p in parts if isinstance(p, ToolCallPart)]

        unsupported = [
            p for p in parts if not isinstance(p, (TextPart, ToolCallPart))
        ]
        if unsupported:
            raise ProviderError(
                f"openai: unsupported content part {unsupported[0].type!r}"
            )

        message: dict[str, Any] = {"role": role, "content": text}
        if calls:
            # An assistant turn that only called tools sends null content.
            message["content"] = text or None
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in calls
            ]
        return message

    # -- inbound: native -> ours ------------------------------------------

    @staticmethod
    def from_native_response(raw: Any) -> LLMResponse:
        """An OpenAI ``ChatCompletion`` as a normalized :class:`LLMResponse`."""
        choices = getattr(raw, "choices", None)
        if not choices:
            raise ProviderError("openai: response carried no choices")
        choice = choices[0]
        native_message = choice.message

        tool_calls = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=_decode_arguments(call.id, call.function.arguments),
            )
            for call in (native_message.tool_calls or [])
        ]

        return LLMResponse(
            text=native_message.content or "",
            tool_calls=tool_calls,
            stop_reason=normalize_stop_reason(
                getattr(choice, "finish_reason", None), STOP_REASONS
            ),
            usage=_usage(raw),
        )


def _decode_arguments(call_id: str, arguments: str | None) -> dict[str, Any]:
    """OpenAI serializes tool arguments as a JSON string; we hand back an object."""
    if not arguments:
        return {}
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            f"openai: tool call {call_id!r} arguments are not valid JSON: {exc}"
        ) from exc
    if not isinstance(decoded, dict):
        raise ProviderError(
            f"openai: tool call {call_id!r} arguments are not a JSON object"
        )
    return decoded


def _usage(raw: Any) -> dict[str, int]:
    usage = getattr(raw, "usage", None)
    if usage is None:
        return {}
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = getattr(usage, "total_tokens", None)
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": int(total) if total is not None else prompt + completion,
    }
