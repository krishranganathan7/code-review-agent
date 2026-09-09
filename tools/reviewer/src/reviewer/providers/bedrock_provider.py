"""AWS Bedrock adapter for :class:`~reviewer.providers.base.LLMProvider`.

Bedrock serves Anthropic models through **the same Messages API** the direct
Anthropic SDK speaks. The official `anthropic` package ships an
`AnthropicBedrock` client that differs from `Anthropic` in exactly two ways:
how it authenticates (AWS SigV4 through the boto3 credential chain, not an
`ANTHROPIC_API_KEY`) and how models are named (a Bedrock model id or inference
profile, not an Anthropic model name).

So this adapter subclasses :class:`~reviewer.providers.anthropic_provider.AnthropicProvider`
and overrides only the client construction. Every translation — tool schemas,
the top-level system prompt, batched `tool_result` blocks, `tool_use` replay,
stop reasons, usage — is inherited unchanged. That is not laziness: the wire
format really is the same one, and a second hand-written copy could only drift
from it. A test asserts the two adapters normalize an identical interaction
identically.

**Why this provider exists.** `native_tools` is `True` here, and that is the
point of it. Tool calling is a first-class API action, so
`ReviewAgent.resolve_inlining` withholds the diff and the agent must call
`read_patch` to see the change at all — the structural forcing function that the
prompted `claude-code` path cannot provide, where a live run made zero tool
calls in every configuration tried.

**Credentials.** Never read into source and never passed through config. The
client resolves them from the ordinary AWS chain — `AWS_PROFILE`, `AWS_REGION`,
environment keys, SSO, instance role — exactly as the AWS CLI would. `region`
and `profile` may be passed explicitly by a caller that has them; both default
to the chain.

**Dependency.** Requires the `bedrock` extra (`pip install -e ".[bedrock]"`),
which pulls `boto3`. Without it the SDK raises at client construction, and this
adapter converts that into a `ProviderError` naming the fix rather than letting
an `ImportError` escape the provider boundary (§6).
"""

from __future__ import annotations

from typing import Any, ClassVar

import anthropic

from .anthropic_provider import AnthropicProvider
from .base import ProviderError

__all__ = ["BedrockProvider", "PROVIDER", "DEFAULT_REGION"]

PROVIDER = "bedrock"

DEFAULT_REGION = "us-east-1"
"""Region used when neither the caller nor the environment names one.

Not a preference so much as a fact about availability: Anthropic inference
profiles are not offered in every region, and a Bedrock call to a region without
one fails with a bare "model identifier is invalid" that says nothing about the
region being the problem. Defaulting to a region that has them turns a confusing
failure into a working call, and an explicit `region` or `AWS_REGION` still wins.
"""


class BedrockProvider(AnthropicProvider):
    """Calls Claude models on AWS Bedrock through the ``anthropic`` SDK.

    ``client`` exists for tests: pass a stand-in and no SDK client is
    constructed, so nothing touches AWS or the network.

    Model ids are Bedrock's, not Anthropic's — usually a cross-region inference
    profile such as ``us.anthropic.claude-sonnet-4-5-20250929-v1:0``.
    """

    provider_name: ClassVar[str] = PROVIDER

    def __init__(
        self,
        model: str,
        *,
        region: str | None = None,
        profile: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.region = region
        self.profile = profile
        if client is not None:
            self._client: Any = client
            return
        try:
            self._client = anthropic.AnthropicBedrock(
                aws_region=region or DEFAULT_REGION,
                aws_profile=profile,
            )
        except anthropic.AnthropicError as exc:
            raise ProviderError(
                f"could not construct the bedrock client ({exc}); "
                "check AWS_PROFILE / AWS_REGION and that credentials are valid"
            ) from exc
        except ImportError as exc:
            # boto3 ships with the SDK's `bedrock` extra, which is optional here
            # so the core install stays light. Name the fix rather than leaking
            # an ImportError across the provider boundary.
            raise ProviderError(
                f"the bedrock provider needs boto3 ({exc}); "
                'install it with: pip install -e ".[bedrock]"'
            ) from exc
