"""Build an :class:`~reviewer.providers.base.LLMProvider` from a config string.

This is the seam that makes provider choice a configuration decision rather than
a code decision. Callers outside :mod:`reviewer.providers` name a provider with a
string and receive something satisfying the interface — they never import an
adapter, so they never reach a provider SDK.
"""

from __future__ import annotations

from typing import Callable, Mapping

from .anthropic_provider import AnthropicProvider
from .base import LLMProvider, ProviderError
from .bedrock_provider import BedrockProvider
from .claude_code_provider import ClaudeCodeProvider
from .config import ProviderConfig
from .fake import FakeProvider
from .openai_provider import OpenAIProvider

__all__ = ["build_provider", "REGISTRY", "available_providers"]


def _anthropic(config: ProviderConfig) -> LLMProvider:
    return AnthropicProvider(config.model, api_key=config.api_key)


def _openai(config: ProviderConfig) -> LLMProvider:
    return OpenAIProvider(config.model, api_key=config.api_key)


def _bedrock(config: ProviderConfig) -> LLMProvider:
    # No api_key: Bedrock authenticates with AWS credentials from the ordinary
    # chain (AWS_PROFILE / AWS_REGION / SSO / instance role).
    return BedrockProvider(config.model)


def _claude_code(config: ProviderConfig) -> LLMProvider:
    # No api_key: the CLI resolves the machine's own Claude credentials.
    return ClaudeCodeProvider(config.model)


def _fake(config: ProviderConfig) -> LLMProvider:
    # An unscripted fake; tests that need turns construct FakeProvider directly.
    return FakeProvider()


REGISTRY: dict[str, Callable[[ProviderConfig], LLMProvider]] = {
    "anthropic": _anthropic,
    "bedrock": _bedrock,
    "openai": _openai,
    "claude-code": _claude_code,
    "fake": _fake,
}
"""Adding a provider means adding an adapter module and one entry here."""


def available_providers() -> list[str]:
    """Provider names this build can construct."""
    return sorted(REGISTRY)


def build_provider(
    source: str | ProviderConfig | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> LLMProvider:
    """Construct the provider named by ``source``.

    ``source`` may be a spec string (``"openai/gpt-4o"``), a
    :class:`~reviewer.providers.config.ProviderConfig`, or ``None`` to read the
    environment.
    """
    if source is None:
        config = ProviderConfig.from_env(env)
    elif isinstance(source, str):
        config = ProviderConfig.parse(source)
    else:
        config = source

    try:
        build = REGISTRY[config.provider]
    except KeyError:
        raise ProviderError(
            f"unknown provider {config.provider!r}; "
            f"available: {', '.join(available_providers())}"
        ) from None
    return build(config)
