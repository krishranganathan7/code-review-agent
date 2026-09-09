"""The `Tool` interface for the read-only agentic-search tool set, and its base.

Hard constraints (CLAUDE.md §2.3/§2.7): the agent gathers context only by calling
these tools in a loop — it is never handed the whole repository. Every tool is
read-only, never reaches the network, and is scoped to the checkout root via
:mod:`reviewer.tools.sandbox`; a path that escapes the root is refused.

The LLM only *chooses* when to call a tool; the tool itself is plain Python. Tool
arguments are model output and tool results are repository content — both are
untrusted **data**. A tool therefore never interprets what it reads, and never
builds a subprocess through a shell.

`SandboxedTool` funnels every failure into a `ToolResult(ok=False)`. The agent has
to be able to *read* an error and correct itself; an exception escaping into the
loop would end the turn instead. Exceptions stay reserved for genuine programming
errors.

The interface is deliberately open so a future graph-query tool could be added
without touching the loop — but no such tool is built in this design (§9).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Iterable, Protocol, runtime_checkable

from ..types import ToolResult, ToolSpec
from .sandbox import Sandbox, SandboxViolation

__all__ = [
    "Tool",
    "SandboxedTool",
    "ToolInputError",
    "log_tool_call",
    "log_tool_result",
]

logger = logging.getLogger("reviewer.tools")


@runtime_checkable
class Tool(Protocol):
    """One read-only, sandboxed capability the agent may invoke."""

    name: str
    """The name the model calls, e.g. ``"read_file"``."""

    spec: ToolSpec
    """Provider-neutral schema describing this tool's arguments."""

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool. Returns a :class:`~reviewer.types.ToolResult`.

        Failures — including a refused out-of-root path — are reported as a
        result with ``ok=False``, not raised.
        """
        ...


class ToolInputError(Exception):
    """The model called a tool with arguments the tool cannot use.

    Caught by :meth:`SandboxedTool.run` and returned as a failed result, so the
    model sees what was wrong and can call again.
    """


class SandboxedTool(ABC):
    """Base for the five search tools: sandbox held, failures funnelled, calls logged.

    Subclasses implement :meth:`_run` and may assume that anything it raises —
    bad arguments, a refused path, an OS error — becomes a failed result rather
    than an exception in the loop.
    """

    name: str
    spec: ToolSpec

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    def run(self, **kwargs: Any) -> ToolResult:
        log_tool_call(self.name, kwargs)
        try:
            result = self._run(**kwargs)
        except SandboxViolation as exc:
            result = self.failure(str(exc), refused=True, requested=exc.requested)
        except ToolInputError as exc:
            result = self.failure(f"{self.name}: {exc}", invalid_arguments=True)
        except OSError as exc:
            result = self.failure(f"{self.name}: {_os_error_text(exc)}")
        log_tool_result(self.name, result)
        return result

    @abstractmethod
    def _run(self, **kwargs: Any) -> ToolResult:
        """The tool's actual work. May raise; `run` converts."""

    # -- result construction ----------------------------------------------

    def success(self, content: str, **metadata: Any) -> ToolResult:
        metadata.setdefault("truncated", False)
        metadata["tool"] = self.name
        return ToolResult(content=content, ok=True, metadata=metadata)

    def failure(self, error: str, **metadata: Any) -> ToolResult:
        metadata["tool"] = self.name
        return ToolResult(content="", ok=False, error=error, metadata=metadata)

    # -- argument handling -------------------------------------------------
    #
    # Arguments come from the model, so every one is validated. A wrong type is
    # reported, never coerced silently into something that would search the
    # wrong thing.

    @staticmethod
    def only(kwargs: dict[str, Any], allowed: Iterable[str]) -> None:
        """Reject arguments this tool does not accept."""
        permitted = set(allowed)
        unexpected = sorted(set(kwargs) - permitted)
        if unexpected:
            raise ToolInputError(
                f"unexpected argument(s) {', '.join(unexpected)}; "
                f"accepts {', '.join(sorted(permitted))}"
            )

    @staticmethod
    def required_str(kwargs: dict[str, Any], key: str) -> str:
        if key not in kwargs or kwargs[key] is None:
            raise ToolInputError(f"missing required argument {key!r}")
        return SandboxedTool._as_str(kwargs[key], key)

    @staticmethod
    def optional_str(kwargs: dict[str, Any], key: str) -> str | None:
        value = kwargs.get(key)
        if value is None:
            return None
        return SandboxedTool._as_str(value, key)

    @staticmethod
    def optional_int(
        kwargs: dict[str, Any], key: str, *, minimum: int | None = None
    ) -> int | None:
        value = kwargs.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            # Models often send numbers as strings; accept a clean integer string.
            if isinstance(value, str) and value.strip().lstrip("-").isdigit():
                value = int(value.strip())
            else:
                raise ToolInputError(f"{key!r} must be an integer, got {value!r}")
        if minimum is not None and value < minimum:
            raise ToolInputError(f"{key!r} must be >= {minimum}, got {value}")
        return value

    @staticmethod
    def _as_str(value: Any, key: str) -> str:
        if not isinstance(value, str):
            raise ToolInputError(f"{key!r} must be a string, got {type(value).__name__}")
        return value


def _os_error_text(exc: OSError) -> str:
    """An OS error without the absolute path of the machine in it."""
    return exc.strerror or str(exc)


def log_tool_call(name: str, arguments: dict[str, Any]) -> None:
    """Structured log of a tool invocation.

    Argument *values* are logged: they are short, model-authored, and needed to
    reproduce a run. Tool *output* never is — see :func:`log_tool_result`.
    """
    logger.info(
        "tool_call tool=%s args=%s",
        name,
        sorted(arguments),
        extra={"event": "tool_call", "tool": name, "arguments": arguments},
    )


def log_tool_result(name: str, result: ToolResult) -> None:
    """Structured log of a tool outcome.

    Deliberately records size and metadata but **not** content: tool output is
    untrusted repository data (CLAUDE.md §2.5) and can be large.
    """
    logger.info(
        "tool_result tool=%s ok=%s bytes=%d truncated=%s",
        name,
        result.ok,
        len(result.content),
        result.metadata.get("truncated"),
        extra={
            "event": "tool_result",
            "tool": name,
            "ok": result.ok,
            "error": result.error,
            "content_bytes": len(result.content),
            "metadata": result.metadata,
        },
    )
