"""Stage: model access. `LLMProvider` and its adapters (CLAUDE.md §6).

The only package permitted to import a provider SDK. One adapter per provider,
each written against that provider's official SDK — there is no shared
third-party model-routing layer. Adding a provider means adding a module here and
one entry in `factory.REGISTRY`; nothing outside this package changes.

Callers should name a provider by config string and use `build_provider`, rather
than importing a concrete adapter.
"""

from __future__ import annotations

from .anthropic_provider import AnthropicProvider
from .base import LLMProvider, ProviderError
from .bedrock_provider import BedrockProvider
from .claude_code_provider import ClaudeCodeProvider
from .config import API_KEY_ENV_VARS, DEFAULT_SPEC, MODEL_ENV_VAR, ProviderConfig
from .factory import REGISTRY, available_providers, build_provider
from .fake import FakeProvider, RecordedCall, ScriptExhausted, Turn, text_turn, tool_turn
from .openai_provider import OpenAIProvider

__all__ = [
    "API_KEY_ENV_VARS",
    "DEFAULT_SPEC",
    "MODEL_ENV_VAR",
    "REGISTRY",
    "AnthropicProvider",
    "BedrockProvider",
    "ClaudeCodeProvider",
    "FakeProvider",
    "LLMProvider",
    "OpenAIProvider",
    "ProviderConfig",
    "ProviderError",
    "RecordedCall",
    "ScriptExhausted",
    "Turn",
    "available_providers",
    "build_provider",
    "text_turn",
    "tool_turn",
]
