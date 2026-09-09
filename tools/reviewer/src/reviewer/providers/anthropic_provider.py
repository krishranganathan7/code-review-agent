"""Anthropic adapter for :class:`~reviewer.providers.base.LLMProvider`.

Written against the official ``anthropic`` SDK. This module and its sibling
adapters are the only places in the codebase permitted to import a provider SDK
(CLAUDE.md §2.1).

Native shapes this adapter owns, and hides from everything else:

* tools are ``{"name", "description", "input_schema"}``;
* the system prompt is a **top-level parameter**, not a message;
* a tool result is a ``tool_result`` **content block inside a user message**, and
  several results are batched into one such message;
* an assistant turn that called tools replays as ``tool_use`` content blocks.
"""

from __future__ import annotations

from typing import Any, ClassVar

import anthropic

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

__all__ = ["AnthropicProvider"]

PROVIDER = "anthropic"

STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "refusal": "refusal",
}


class AnthropicProvider:
    """Calls Claude models through the ``anthropic`` SDK.

    ``client`` exists for tests: pass a stand-in and no SDK client is
    constructed, so nothing touches the network. In production leave it unset —
    with no ``api_key`` the SDK resolves credentials itself (``ANTHROPIC_API_KEY``
    or a configured profile).
    """

    native_tools = True
    """The SDK carries tool schemas and returns tool_use blocks directly."""

    provider_name: ClassVar[str] = PROVIDER
    """The name this adapter logs under.

    A class attribute rather than the module constant because Bedrock serves
    this same Messages API and reuses every translation below; only the client
    and the name differ. See `bedrock_provider.BedrockProvider`.
    """

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
                anthropic.Anthropic(api_key=api_key)
                if api_key is not None
                else anthropic.Anthropic()
            )
        except anthropic.AnthropicError as exc:
            raise ProviderError(
                f"could not construct the anthropic client ({exc}); "
                "set ANTHROPIC_API_KEY or pass api_key"
            ) from exc

    # -- LLMProvider -------------------------------------------------------

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        log_call(self.provider_name, self.model, messages, tools)
        system, native_messages = self.to_native_messages(messages)

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": native_messages,
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [self.to_native_tool(tool) for tool in tools]

        try:
            raw = self._client.messages.create(**request)
        except anthropic.AnthropicError as exc:
            raise ProviderError(f"anthropic call failed: {exc}") from exc

        response = self.from_native_response(raw)
        log_response(self.provider_name, self.model, response)
        return response

    # -- outbound: ours -> native -----------------------------------------

    @staticmethod
    def to_native_tool(tool: ToolSpec) -> dict[str, Any]:
        """A :class:`~reviewer.types.ToolSpec` as an Anthropic tool definition."""
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters or {"type": "object", "properties": {}},
        }

    @classmethod
    def to_native_messages(
        cls, messages: list[Message]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Split our messages into Anthropic's ``(system, messages)``.

        System turns are lifted out into the top-level system parameter. Tool
        results become ``tool_result`` blocks in a **user** message — Anthropic
        has no tool role — and a turn carrying several results stays a single
        message, which is how Anthropic wants parallel results returned.
        """
        system_chunks: list[str] = []
        native: list[dict[str, Any]] = []

        for message in messages:
            if message.role == "system":
                system_chunks.append(_plain_text(message))
                continue

            parts = as_parts(message)
            results = [p for p in parts if isinstance(p, ToolResultPart)]
            if results:
                if len(results) != len(parts):
                    raise ProviderError(
                        "anthropic: a tool-result turn may not carry other content"
                    )
                native.append(
                    {
                        "role": "user",
                        "content": [cls._tool_result_block(p) for p in results],
                    }
                )
                continue

            native.append(
                {
                    "role": message.role,
                    "content": [cls._content_block(p) for p in parts],
                }
            )

        return "\n\n".join(chunk for chunk in system_chunks if chunk), native

    @staticmethod
    def _tool_result_block(part: ToolResultPart) -> dict[str, Any]:
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": part.tool_call_id,
            "content": part.content,
        }
        if part.is_error:
            block["is_error"] = True
        return block

    @staticmethod
    def _content_block(part: ContentPart) -> dict[str, Any]:
        if isinstance(part, TextPart):
            return {"type": "text", "text": part.text}
        if isinstance(part, ToolCallPart):
            return {
                "type": "tool_use",
                "id": part.call.id,
                "name": part.call.name,
                "input": part.call.arguments,
            }
        raise ProviderError(f"anthropic: unsupported content part {part.type!r}")

    # -- inbound: native -> ours ------------------------------------------

    @staticmethod
    def from_native_response(raw: Any) -> LLMResponse:
        """An Anthropic ``Message`` as a normalized :class:`LLMResponse`."""
        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in raw.content:
            kind = getattr(block, "type", None)
            if kind == "text":
                text_chunks.append(block.text)
            elif kind == "tool_use":
                arguments = block.input
                if not isinstance(arguments, dict):
                    raise ProviderError(
                        f"anthropic: tool_use {block.id!r} input is not an object"
                    )
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(arguments))
                )
            # thinking / other block types carry no loop-facing content.

        return LLMResponse(
            text="".join(text_chunks),
            tool_calls=tool_calls,
            stop_reason=normalize_stop_reason(
                getattr(raw, "stop_reason", None), STOP_REASONS
            ),
            usage=_usage(raw),
        )


def _plain_text(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(p.text for p in message.content if isinstance(p, TextPart))


def _usage(raw: Any) -> dict[str, int]:
    usage = getattr(raw, "usage", None)
    if usage is None:
        return {}
    prompt = int(getattr(usage, "input_tokens", 0) or 0)
    completion = int(getattr(usage, "output_tokens", 0) or 0)
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
    }
