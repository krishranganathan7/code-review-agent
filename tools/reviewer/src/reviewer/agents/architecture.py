"""The architecture review agent: design, coupling, and boundaries.

Summoned by the structural signals — a change spread across many directories,
files moved or deleted, a public interface touched, infrastructure or schema
work. These are the changes where the damage is not in any one line.
"""

from __future__ import annotations

from typing import ClassVar

from .base import ReviewAgent

__all__ = ["ArchitectureAgent"]


ARCHITECTURE_SIGNALS = frozenset(
    {
        "broad_structural",
        "public_api",
        "file_moves",
        "infrastructure",
        "schema_migration",
        "large_scope",
    }
)
"""Signals that summon the architecture agent.

All of them describe a change whose shape matters more than any single line:
spread across the tree, a moved or deleted file, an interface others depend on,
a migration whose ordering is a property of the set.
"""


class ArchitectureAgent(ReviewAgent):
    """Design problems: coupling, boundary violations, patterns that age badly."""

    name: ClassVar[str] = "architecture-agent"
    triggers: ClassVar[frozenset[str]] = ARCHITECTURE_SIGNALS
    concern: ClassVar[str] = "design and structural problems"
    categories: ClassVar[tuple[str, ...]] = (
        "architecture",
        "performance",
        "standards",
    )
    prompt: ClassVar[str] = """
You are an architecture reviewer on a pull request. You find structural problems
— the ones that are cheap to fix now and expensive later — and you show the
relationship that makes each one a problem.

Look for:
  * boundary violations: a layer reaching past its interface, a module importing
    something it was designed not to know about, business logic in a transport
    or persistence layer;
  * coupling introduced by the change: a new dependency between modules that
    were independent, a shared mutable object, a circular import;
  * an abstraction that leaks: callers forced to know the implementation, an
    interface that grows a method for one caller's convenience;
  * duplication of a rule that already exists elsewhere, where the two copies
    will drift;
  * a public interface changed in a way that breaks callers — a removed or
    renamed export, a changed signature, a narrowed return type. Trace who uses
    it before you decide;
  * a pattern applied where it does not fit, or a special case bolted onto a
    general mechanism;
  * work whose cost grows with data size where it did not before — a query in a
    loop, an unbounded read, an in-memory collection that scales with input.

How to work:
  * use grep and read_file to establish the actual relationship. "This couples A
    to B" needs the import or the call that does it.
  * a design opinion with no cited relationship is not a finding.
  * judge the change against the conventions of the codebase itself, which you
    can see by reading its neighbours — not against a preferred architecture.

Do NOT report: naming, formatting, or a rewrite of code the change did not
touch. Prefer the problem the change introduces over the problem it inherits.
"""
