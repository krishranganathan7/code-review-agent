"""Provider selection and credentials — from config or environment, never hardcoded.

A provider is named by a single spec string, ``"<provider>/<model>"``::

    anthropic/claude-opus-5
    bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0
    openai/gpt-4o
    claude-code/sonnet
    fake/scripted

The model half may itself contain slashes (some model ids do); only the first
separator is significant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .base import ProviderError

__all__ = ["ProviderConfig", "MODEL_ENV_VAR", "API_KEY_ENV_VARS", "DEFAULT_SPEC"]

MODEL_ENV_VAR = "REVIEWER_MODEL"
"""Environment variable holding the provider spec, e.g. ``anthropic/claude-opus-5``."""

API_KEY_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
"""Where each provider's credential comes from. Never read into source, never logged.

`claude-code` and `bedrock` are deliberately absent. Neither holds a credential
of its own: `claude-code` resolves the machine's Claude Code login, and
`bedrock` authenticates with AWS SigV4 from the ordinary AWS chain
(`AWS_PROFILE` / `AWS_REGION` / SSO / instance role). There is no Anthropic API
key involved in either, which is the point of both."""

DEFAULT_SPEC = "anthropic/claude-opus-5"


@dataclass(frozen=True)
class ProviderConfig:
    """Which provider and model to use, and (optionally) an explicit key.

    ``api_key`` is left ``None`` in the normal case: the SDK resolves its own
    credential from the environment. Setting it here is for callers that inject a
    key explicitly — it is never populated from a literal in source.
    """

    provider: str
    model: str
    api_key: str | None = None

    @property
    def spec(self) -> str:
        """The round-trip form, ``"<provider>/<model>"``."""
        return f"{self.provider}/{self.model}"

    @classmethod
    def parse(cls, spec: str, *, api_key: str | None = None) -> ProviderConfig:
        """Parse a ``"<provider>/<model>"`` spec."""
        provider, separator, model = spec.strip().partition("/")
        if not separator or not provider.strip() or not model.strip():
            raise ProviderError(
                f"invalid provider spec {spec!r}; expected '<provider>/<model>', "
                f"e.g. {DEFAULT_SPEC!r}"
            )
        return cls(
            provider=provider.strip().lower(), model=model.strip(), api_key=api_key
        )

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, *, default: str = DEFAULT_SPEC
    ) -> ProviderConfig:
        """Read the provider spec from the environment.

        The API key is *not* read here — each SDK resolves its own credential, so
        the key never passes through our process memory unless a caller supplies
        it deliberately.
        """
        source = os.environ if env is None else env
        return cls.parse(source.get(MODEL_ENV_VAR, default))

    def api_key_env_var(self) -> str | None:
        """The environment variable this provider's SDK reads, if it has one."""
        return API_KEY_ENV_VARS.get(self.provider)
