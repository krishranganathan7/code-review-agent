"""Claude Code adapter for :class:`~reviewer.providers.base.LLMProvider`.

Drives the `claude` CLI in print mode as a subprocess. Its reason to exist is
**authentication**: Claude Code reads the machine's OAuth/keychain credentials,
so a developer with a Claude subscription can run a review locally without an
API key. The `anthropic` and `openai` adapters remain the path for anything
automated.

This is still a provider adapter and nothing more (CLAUDE.md §2.1/§6): it
translates our `Message`/`ToolSpec` shapes into a CLI invocation and translates
the JSON back into a normalized `LLMResponse`. **We keep the agentic loop** — the
CLI is invoked once per loop iteration with the bounded transcript, its own
built-in tools are switched off with ``--tools ""``, and every control in
:mod:`reviewer.agent.controls` still applies.

Two mechanics are worth knowing, both established by probing the CLI:

* **Tool-calling is prompted, not native.** The CLI offers no way to declare
  *our* tool schemas, so the adapter renders them into the system prompt and
  requires a fixed reply envelope. That envelope is enforced by ``--json-schema``
  and comes back **already parsed** in ``structured_output`` — so this is
  schema-validated structured output, not text we parse hopefully.
* **``--json-schema`` empties ``result``.** When a schema is in play the answer
  is in ``structured_output``; otherwise it is in ``result``. The adapter reads
  whichever applies.

There is no SDK import here — the whole adapter is `subprocess` — so the Phase 0
provider-isolation guard has nothing to catch.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from typing import Any, Mapping, Sequence

from ..types import (
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

__all__ = [
    "ClaudeCodeProvider",
    "DEFAULT_COMMAND",
    "DEFAULT_MODEL",
    "ENVELOPE_SCHEMA",
    "TOOL_PROTOCOL",
]

logger = logging.getLogger("reviewer.providers.claude_code")

PROVIDER = "claude-code"

DEFAULT_COMMAND: tuple[str, ...] = ("claude",)
DEFAULT_MODEL = "sonnet"
DEFAULT_TIMEOUT = 900.0

MAX_SYSTEM_PROMPT_ARGV = 16_000
"""Above this, the system prompt moves into stdin rather than the command line.

Windows caps a command line at ~32k characters. Our system prompts are a few
thousand, so this is a guard rather than a routine path.
"""

SPAWN_FAILURES = frozenset({3221225794, 3221225495, 3221226356})
"""Windows exit codes that mean the process never started, not that it failed.

`0xC0000142` STATUS_DLL_INIT_FAILED, `0xC0000017` STATUS_NO_MEMORY and
`0xC0000374` heap corruption all show up when a machine is under pressure from
several concurrent `claude` launches. One was observed killing the judge
mid-review: the CLI exited 3221225794 having produced no output at all. A
process that never ran has done no work and cost nothing, so retrying it once is
safe — unlike retrying a call that may have already spent tokens.
"""

SPAWN_RETRIES = 1

STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "refusal": "refusal",
}


# --------------------------------------------------------------------------
# The prompted tool-calling protocol
# --------------------------------------------------------------------------

ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "Your reply. Empty if you are only calling tools.",
        },
        "tool_calls": {
            "type": "array",
            "description": "Tools to run now. Empty when you are finished.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
        },
    },
    "required": ["text", "tool_calls"],
}
"""The reply envelope, enforced by the CLI rather than hoped for.

`tool_calls: []` is how the model says it is done — one unambiguous signal,
rather than the absence of something.
"""

TOOL_PROTOCOL = """
## Tools

The tools listed below are real and they work. They are the only ones you have.

You are running inside another program, and this is the part that is easy to get
wrong: your usual tools are switched off on purpose. You have no Read, no Grep,
no Bash, no MCP server, and no code index. If some part of you believes
otherwise, it is wrong about this session. Do not conclude that you cannot
search — you can, through the mechanism below, and it is the only mechanism.

You do not run these tools. You name them, and the program running you executes
them and hands you the results on your next turn.

Reply with a JSON object of exactly this shape, and nothing else:

  {{"text": "...", "tool_calls": [{{"name": "<tool>", "arguments": {{...}}}}]}}

* To call tools, put them in `tool_calls`. You may ask for several at once when
  they do not depend on each other. Leave `text` empty, or use it to say briefly
  what you are looking for.
* When you have finished and want to give your final answer, return
  `"tool_calls": []` and put the whole answer in `text`.
* Never invent a tool. Never put anything outside this object.
* Never write out a tool call as prose or as markup inside `text`. Nothing in
  `text` is ever executed, so a call written there is silently lost and you will
  be left guessing at its result.
* Never state what a file contains unless its contents reached you in a tool
  result. If you have not read it, say so, or ask for it.

Available tools:

{catalogue}
"""


def _catalogue(tools: Sequence[ToolSpec]) -> str:
    """Render tool specs as a readable catalogue with their JSON schemas."""
    entries: list[str] = []
    for tool in tools:
        schema = json.dumps(
            tool.parameters or {"type": "object", "properties": {}},
            indent=2,
            sort_keys=True,
        )
        entries.append(f"### {tool.name}\n{tool.description}\n\nArguments:\n{schema}")
    return "\n\n".join(entries)


# --------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------


class ClaudeCodeProvider:
    """Calls Claude models through the `claude` CLI, on subscription auth.

    ``command`` exists for tests: point it at a stand-in executable and nothing
    launches the real CLI or reaches the network — the same shape Phase 2 used
    for ripgrep.
    """

    native_tools = False
    """Tool-calling here is prompted, not native, and it shows.

    The CLI has no tool-calling API of its own, so `TOOL_PROTOCOL` describes the
    tools in the system prompt and `--json-schema` enforces a reply envelope the
    adapter parses. The model complies when calling a tool is the obvious next
    move, and otherwise does not: given a full review assignment it answers the
    assignment. Callers that need the model to search must not rely on it
    choosing to.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        command: Sequence[str] = DEFAULT_COMMAND,
        timeout: float = DEFAULT_TIMEOUT,
        max_budget_usd: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.command = list(command)
        self.timeout = timeout
        self.max_budget_usd = max_budget_usd
        self.env = dict(env) if env is not None else None
        self._counter = 0
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"ClaudeCodeProvider(model={self.model!r}, command={self.command!r})"

    # -- LLMProvider -------------------------------------------------------

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        log_call(PROVIDER, self.model, messages, tools)

        system, prompt = self.to_native_prompt(messages, tools)
        argv = self.to_native_argv(bool(tools))

        if system:
            if len(system) <= MAX_SYSTEM_PROMPT_ARGV:
                argv += ["--system-prompt", system]
            else:
                # Too long for a command line; fold it into stdin, clearly
                # marked so it still reads as instruction rather than content.
                prompt = f"# Instructions\n\n{system}\n\n---\n\n{prompt}"

        raw = self._invoke(argv, prompt)
        response = self.from_native_response(raw, expecting_tools=bool(tools))
        log_response(PROVIDER, self.model, response)
        return response

    # -- outbound: ours -> native -----------------------------------------

    def to_native_argv(self, with_tools: bool) -> list[str]:
        """The command line, without the system prompt."""
        argv = [
            *self.command,
            "-p",
            "--output-format",
            "json",
            # Claude Code's own tools are switched off: the agent searches
            # through *our* sandboxed tools, or not at all (§2.7).
            "--tools",
            "",
            # No user, project or local settings, which is how the CLI
            # discovers CLAUDE.md files. Without this the developer's own
            # CLAUDE.md is loaded into every review: a live run was observed
            # reaching for CodeGraph and declining to search because a
            # personal memory file told it to, and repository content is
            # supposed to be the only context (§2.3, §2.5). It also makes a
            # review reproducible rather than a function of whose machine ran
            # it.
            "--setting-sources",
            "",
            "--model",
            self.model,
        ]
        if with_tools:
            argv += ["--json-schema", json.dumps(ENVELOPE_SCHEMA)]
        if self.max_budget_usd is not None:
            argv += ["--max-budget-usd", str(self.max_budget_usd)]
        return argv

    def to_native_prompt(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None
    ) -> tuple[str, str]:
        """Render our transcript into ``(system prompt, stdin prompt)``.

        The CLI takes one prompt, so the conversation is serialized with role
        headings. Tool results arrive already wrapped in their untrusted region
        (Phase 7), and that wrapping is preserved verbatim.
        """
        system_chunks: list[str] = []
        turns: list[str] = []

        for message in messages:
            if message.role == "system":
                system_chunks.append(_plain_text(message))
                continue

            parts = as_parts(message)
            text = "".join(p.text for p in parts if isinstance(p, TextPart))
            calls = [p.call for p in parts if isinstance(p, ToolCallPart)]
            results = [p for p in parts if isinstance(p, ToolResultPart)]

            if results:
                rendered = ["## Tool results"]
                for result in results:
                    status = " (failed)" if result.is_error else ""
                    rendered.append(f"\n### Result of {result.tool_call_id}{status}")
                    rendered.append(result.content)
                turns.append("\n".join(rendered))
                continue

            heading = "## Assistant" if message.role == "assistant" else "## User"
            block = [heading]
            if text:
                block.append(text)
            for call in calls:
                block.append(
                    f"Requested tool {call.id}: {call.name}"
                    f"({json.dumps(call.arguments, sort_keys=True)})"
                )
            turns.append("\n".join(block))

        if tools:
            system_chunks.append(
                TOOL_PROTOCOL.format(catalogue=_catalogue(list(tools)))
            )

        return "\n\n".join(c for c in system_chunks if c), "\n\n".join(turns)

    # -- inbound: native -> ours ------------------------------------------

    def from_native_response(
        self, raw: Mapping[str, Any], *, expecting_tools: bool = False
    ) -> LLMResponse:
        """A CLI result object as a normalized :class:`LLMResponse`."""
        if raw.get("is_error") or raw.get("subtype") not in (None, "success"):
            raise ProviderError(
                "claude CLI reported a failure "
                f"({raw.get('subtype') or 'error'}): "
                f"{raw.get('result') or raw.get('api_error_status') or 'no detail'}"
            )

        text = ""
        tool_calls: list[ToolCall] = []
        structured = raw.get("structured_output")

        if expecting_tools and isinstance(structured, Mapping):
            text = str(structured.get("text") or "")
            tool_calls = self._tool_calls(structured.get("tool_calls"))
        elif isinstance(structured, Mapping):
            # A schema was in play but we did not ask for tools: hand the whole
            # object back as text, so a caller's own parser sees what it expects.
            text = json.dumps(structured)
        else:
            text = str(raw.get("result") or "")

        stop = (
            "tool_use"
            if tool_calls
            else normalize_stop_reason(raw.get("stop_reason"), STOP_REASONS)
        )

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop,
            usage=_usage(raw),
        )

    def _tool_calls(self, entries: Any) -> list[ToolCall]:
        """Turn the envelope's `tool_calls` into ours, assigning ids."""
        if not isinstance(entries, list):
            return []
        calls: list[ToolCall] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                continue
            arguments = entry.get("arguments")
            calls.append(
                ToolCall(
                    id=self._next_id(),
                    name=name,
                    arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
                )
            )
        return calls

    def _next_id(self) -> str:
        """Unique within this provider, so ids do not repeat across turns."""
        with self._lock:
            self._counter += 1
            return f"call_{self._counter}"

    # -- the subprocess ---------------------------------------------------

    def _invoke(self, argv: Sequence[str], prompt: str) -> Mapping[str, Any]:
        """Run the CLI once, retrying only a failure to start the process."""
        for attempt in range(SPAWN_RETRIES + 1):
            try:
                return self._invoke_once(argv, prompt)
            except _SpawnFailure as failure:
                if attempt == SPAWN_RETRIES:
                    raise ProviderError(str(failure)) from failure
                logger.warning(
                    "claude_spawn_retry exit=%d attempt=%d",
                    failure.returncode,
                    attempt + 1,
                    extra={
                        "event": "claude_spawn_retry",
                        "exit_code": failure.returncode,
                        "attempt": attempt + 1,
                    },
                )
        raise ProviderError("claude CLI could not be started")  # pragma: no cover

    def _invoke_once(self, argv: Sequence[str], prompt: str) -> Mapping[str, Any]:
        try:
            completed = subprocess.run(  # noqa: S603 - argument list, never a shell
                list(argv),
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=self.timeout,
                shell=False,
                check=False,
                env=self.env,
            )
        except FileNotFoundError as exc:
            raise ProviderError(
                f"{argv[0]!r} is not installed or not on PATH. The claude-code "
                "provider needs the Claude Code CLI; install it, or use an "
                "API-key provider such as anthropic/claude-opus-5."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(
                f"claude CLI timed out after {self.timeout:g}s"
            ) from exc
        except OSError as exc:  # pragma: no cover - platform-dependent
            raise ProviderError(f"could not run the claude CLI: {exc}") from exc

        stdout = completed.stdout.decode("utf-8", errors="replace")
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()

        if completed.returncode != 0:
            detail = (
                f"claude CLI exited {completed.returncode}: "
                f"{stderr[:400] or stdout[:400] or 'no output'}"
            )
            if completed.returncode in SPAWN_FAILURES and not stdout.strip():
                # The process never started; nothing was spent, so it is safe to
                # try again rather than fail the whole review.
                raise _SpawnFailure(detail, completed.returncode)
            raise ProviderError(detail)

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"claude CLI did not return JSON ({exc}): {stdout[:300]!r}"
            ) from exc

        if not isinstance(payload, Mapping):
            raise ProviderError(
                f"claude CLI returned {type(payload).__name__}, expected an object"
            )
        return payload


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


class _SpawnFailure(Exception):
    """Internal: the CLI process never started. Retryable, unlike a real error."""

    def __init__(self, message: str, returncode: int) -> None:
        super().__init__(message)
        self.returncode = returncode


def _plain_text(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(p.text for p in message.content if isinstance(p, TextPart))


def _usage(raw: Mapping[str, Any]) -> dict[str, int]:
    """Normalized usage, plus the CLI's own cost figure in whole micro-dollars.

    Cost is genuinely useful — it is the one provider that reports it — and
    `LLMResponse.usage` is `dict[str, int]`, so it is carried as micro-USD
    rather than a float.
    """
    usage = raw.get("usage")
    counts: dict[str, int] = {}
    if isinstance(usage, Mapping):
        prompt = _int(usage.get("input_tokens"))
        completion = _int(usage.get("output_tokens"))
        cached = _int(usage.get("cache_read_input_tokens"))
        created = _int(usage.get("cache_creation_input_tokens"))
        counts = {
            "input_tokens": prompt,
            "output_tokens": completion,
            # Cache tokens are reported separately, NOT folded into the total:
            # `LoopControls.token_budget` compares this number across providers,
            # and the API adapters count only input+output. Including Claude
            # Code's cache traffic here would trip the budget several times
            # earlier for the same work.
            "total_tokens": prompt + completion,
        }
        if cached:
            counts["cache_read_input_tokens"] = cached
        if created:
            counts["cache_creation_input_tokens"] = created

    cost = raw.get("total_cost_usd")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        counts["cost_micro_usd"] = int(round(float(cost) * 1_000_000))
    return counts


def _int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
