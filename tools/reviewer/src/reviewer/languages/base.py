"""The :class:`LanguageAdapter` interface — the only place language logic lives.

Hard constraint (CLAUDE.md §2.2): parsing, symbol extraction and lint for a given
language sit behind this interface. The core pipeline contains no assumption
about a specific language or ecosystem.

Phase 0: interface only. Concrete adapters land with Phase 4's change analyzer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..findings.finding import Finding
from ..types import Symbol

__all__ = ["LanguageAdapter"]


@runtime_checkable
class LanguageAdapter(Protocol):
    """Language-specific analysis for one language or ecosystem."""

    language: str
    """Identifier for this language, e.g. ``"python"``."""

    def matches(self, filename: str) -> bool:
        """Whether this adapter handles ``filename``."""
        ...

    def changed_symbols(self, filename: str, diff: str) -> list[Symbol]:
        """Symbols touched by ``diff`` in ``filename``."""
        ...

    def lint(self, path: str) -> list[Finding]:
        """Static-analysis findings for ``path``. Optional; may return ``[]``."""
        ...
