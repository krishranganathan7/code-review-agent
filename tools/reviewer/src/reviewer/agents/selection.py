"""Which agents run, and how much they may spend.

Selection is **signal-driven, not level-driven**. The risk level says how deep to
search; the individual signals say what to search *for*. A CRITICAL change that
is critical because it rewrites a migration does not need a security reviewer,
and a HIGH change that touches auth does — level alone cannot tell those apart.

Depth is the other half: the level maps to a
:class:`~reviewer.agent.controls.LoopControls` (Phase 4's `DEPTH_BY_LEVEL`), and
every selected agent gets it. Risk buys search budget; signals decide who spends
it.

The map is deliberately small and inspectable — the selection a review made is
part of what a reviewer audits, so :class:`AgentPlan` records why each agent was
chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..agent import LoopControls
from ..analysis.risk import RiskAssessment
from .architecture import ARCHITECTURE_SIGNALS, ArchitectureAgent
from .base import ReviewAgent
from .bug import BugAgent
from .security import SECURITY_SIGNALS, SecurityAgent

__all__ = [
    "AgentSelection",
    "AgentPlan",
    "select_agents",
    "plan_review",
    "SECURITY_SIGNALS",
    "ARCHITECTURE_SIGNALS",
    "ALWAYS_RUN",
    "available_agents",
]

ALWAYS_RUN: tuple[str, ...] = ("bug-agent",)
"""Names of the shipped agents that run on every change, for documentation.

Authority lives on the agent (`ReviewAgent.always_run`); this is a readable
summary of the shipped roster, not the thing selection consults.
"""


@dataclass(frozen=True)
class AgentSelection:
    """One agent chosen to run, with its budget and the reason it was chosen."""

    agent: ReviewAgent
    controls: LoopControls
    reason: str
    signals: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.agent.name


@dataclass(frozen=True)
class AgentPlan:
    """The full selection for one review, and the case for it."""

    selections: list[AgentSelection] = field(default_factory=list)
    level: str = "LOW"
    skipped: dict[str, str] = field(default_factory=dict)
    """Agents not selected, and why — as auditable as the ones that were."""

    @property
    def names(self) -> list[str]:
        return [selection.name for selection in self.selections]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.selections)

    def __len__(self) -> int:
        return len(self.selections)

    def selection(self, name: str) -> AgentSelection | None:
        for item in self.selections:
            if item.name == name:
                return item
        return None

    def why(self) -> str:
        """One line per agent, chosen or not. The audit trail of the selection."""
        lines = [
            f"{item.name}: {item.reason}"
            + (f" [{', '.join(item.signals)}]" if item.signals else "")
            for item in self.selections
        ]
        lines += [f"{name}: skipped — {reason}" for name, reason in sorted(self.skipped.items())]
        return "; ".join(lines)


def available_agents() -> list[ReviewAgent]:
    """Every agent this build ships. Adding one means adding it here."""
    return [BugAgent(), SecurityAgent(), ArchitectureAgent()]


def _triggers(agent: ReviewAgent) -> frozenset[str]:
    """The risk signals that summon this agent, as the agent declares them."""
    return frozenset(getattr(agent, "triggers", frozenset()))


def _always_run(agent: ReviewAgent) -> bool:
    """Whether this agent runs regardless of signal, as the agent declares it."""
    return bool(getattr(agent, "always_run", False))


def plan_review(
    assessment: RiskAssessment,
    agents: Sequence[ReviewAgent] | None = None,
    *,
    controls: LoopControls | None = None,
) -> AgentPlan:
    """Decide which agents run for this assessment, and with what budget.

    Each agent declares its own `always_run` flag and `triggers` set, so a
    caller-supplied roster is selected on the same terms as the shipped one.

    ``controls`` overrides the depth for every agent; by default each gets the
    depth the risk level bought, after any repository override.
    """
    roster = list(agents) if agents is not None else available_agents()
    depth = controls if controls is not None else assessment.controls()
    fired = set(assessment.signal_ids)

    selections: list[AgentSelection] = []
    skipped: dict[str, str] = {}

    for agent in roster:
        if _always_run(agent):
            selections.append(
                AgentSelection(
                    agent=agent,
                    controls=depth,
                    reason="runs on every change; correctness has no signal of its own",
                )
            )
            continue

        triggers = _triggers(agent)
        matched = tuple(sorted(fired & triggers))
        if matched:
            selections.append(
                AgentSelection(
                    agent=agent,
                    controls=depth,
                    reason="summoned by risk signal",
                    signals=matched,
                )
            )
        elif not triggers:
            # Neither always-run nor triggered by anything: this agent can never
            # be selected. That is a configuration mistake, not a quiet skip.
            skipped[agent.name] = (
                "declares neither always_run nor any triggers, so it can never "
                "be selected; set one on the agent class"
            )
        else:
            skipped[agent.name] = (
                "no matching risk signal fired "
                f"(needs one of: {', '.join(sorted(triggers))})"
            )

    return AgentPlan(selections=selections, level=assessment.level, skipped=skipped)


def select_agents(
    assessment: RiskAssessment, agents: Sequence[ReviewAgent] | None = None
) -> list[AgentSelection]:
    """The selected agents alone, for callers that do not need the audit trail."""
    return plan_review(assessment, agents).selections


def depth_for(assessment: RiskAssessment) -> LoopControls:
    """The search budget every selected agent gets for this assessment."""
    return assessment.controls()


SIGNAL_MAP: Mapping[str, frozenset[str]] = {
    agent.name: _triggers(agent)
    for agent in available_agents()
    if _triggers(agent)
}
"""The selection map, derived from the agents themselves.

Built from `available_agents()` rather than written out again, so it cannot
drift from the triggers selection actually consults — the previous hand-written
copy did, and editing it changed nothing.
"""
