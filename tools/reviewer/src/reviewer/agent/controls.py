"""Loop controls: `max_tool_calls`, token budget, timeout — and the clock.

CLAUDE.md §4: every control is checked on every iteration, and on any limit the
loop stops and forces the model to conclude with what it has. A limit is a
*bounded stop*, never a hang and never a half-state.

Time is read through a :class:`Clock` so the timeout is testable without waiting.
Production uses :class:`MonotonicClock`; tests advance a :class:`ManualClock` by
hand.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

__all__ = [
    "LoopControls",
    "LoopStop",
    "LIMIT_STOPS",
    "Clock",
    "MonotonicClock",
    "ManualClock",
]

LoopStop = Literal[
    "concluded",
    "max_tool_calls",
    "token_budget",
    "timeout",
    "max_iterations",
]
"""Why the loop stopped. ``"concluded"`` is the model finishing on its own."""

LIMIT_STOPS: tuple[LoopStop, ...] = (
    "max_tool_calls",
    "token_budget",
    "timeout",
    "max_iterations",
)
"""Every stop reason that means a control tripped rather than the model finishing."""


@runtime_checkable
class Clock(Protocol):
    """A monotonic source of seconds. Injected so timeouts need no real waiting."""

    def now(self) -> float: ...


class MonotonicClock:
    """Wall-clock elapsed time, immune to system clock adjustments."""

    def now(self) -> float:
        return time.monotonic()


class ManualClock:
    """A clock that only moves when a test moves it.

    Shipped rather than kept in the test tree: the loop's timeout is a real
    control, and anything driving the loop — Phase 5's review agents included —
    needs a way to exercise it deterministically.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> float:
        """Move time forward and return the new reading."""
        self._now += seconds
        return self._now


@dataclass(frozen=True)
class LoopControls:
    """Bounds on one agentic-search run.

    Attributes:
        max_tool_calls: total tool invocations allowed across the run. An unknown
            tool still counts — it consumed a turn.
        token_budget: cumulative tokens (as reported by the provider) allowed
            before the run must conclude. ``None`` disables the check.
        timeout_seconds: wall-clock seconds allowed. ``None`` disables the check.
        max_iterations: hard ceiling on provider round-trips. This is runaway
            protection, not a tuning knob — a loop that hits it has misbehaved.
        max_tokens_per_call: per-response cap passed to the provider.
        require_tool_use: whether a run that answers on its first turn without
            calling a single tool is pushed back **once** before its answer is
            accepted. See :data:`~reviewer.agent.loop.SEARCH_FIRST_INSTRUCTION`.

            Off on a bare `LoopControls` so a caller driving the loop directly
            gets exactly the turns it scripted. It is switched **on** for every
            real review by `DEPTH_BY_LEVEL`, which is where an agent's depth
            comes from — because the failure it corrects is silent and total:
            handed a task that reads like a complete assignment, a model answers
            the assignment rather than searching, and returns a confident review
            of code it never opened. Nothing downstream can tell that apart from
            a review that searched and found nothing. The push-back costs one
            extra round-trip only in the case where the agent did no work.

            It fires at most once per run, and only when tools were actually
            offered — a loop with no tools has nothing to push back towards.
    """

    max_tool_calls: int = 40
    token_budget: int | None = None
    timeout_seconds: float | None = None
    max_iterations: int = 100
    max_tokens_per_call: int = 4096
    require_tool_use: bool = False

    def __post_init__(self) -> None:
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.token_budget is not None and self.token_budget < 1:
            raise ValueError("token_budget must be at least 1 when set")
        if self.timeout_seconds is not None and self.timeout_seconds < 0:
            raise ValueError("timeout_seconds must not be negative")
        if self.max_tokens_per_call < 1:
            raise ValueError("max_tokens_per_call must be at least 1")

    def tripped(
        self,
        *,
        iterations: int,
        tool_calls: int,
        tokens_used: int,
        elapsed: float,
    ) -> LoopStop | None:
        """The first limit that has been reached, or ``None`` to keep going.

        Order is deliberate: the cheapest-to-explain limit wins, and the hard
        iteration ceiling is checked last because reaching it means one of the
        others should have fired first.
        """
        if tool_calls >= self.max_tool_calls:
            return "max_tool_calls"
        if self.token_budget is not None and tokens_used >= self.token_budget:
            return "token_budget"
        if self.timeout_seconds is not None and elapsed >= self.timeout_seconds:
            return "timeout"
        if iterations >= self.max_iterations:
            return "max_iterations"
        return None

    def describe(self, stop: LoopStop) -> str:
        """A plain-language reason, for the message that forces a conclusion.

        Branches rather than a lookup table: an unset limit must not be formatted
        just because a different one tripped.
        """
        if stop == "concluded":
            return "the task is complete"
        if stop == "max_tool_calls":
            return f"the tool-call budget of {self.max_tool_calls} has been used up"
        if stop == "token_budget":
            return f"the token budget of {self.token_budget} has been used up"
        if stop == "timeout":
            seconds = "" if self.timeout_seconds is None else f"{self.timeout_seconds:g}s"
            return f"the time limit of {seconds} has been reached"
        return (
            f"the hard ceiling of {self.max_iterations} model round-trips "
            "has been reached"
        )
