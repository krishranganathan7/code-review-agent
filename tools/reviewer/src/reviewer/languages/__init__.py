"""`LanguageAdapter` and its implementations (CLAUDE.md §2.2).

The only package permitted to contain language-specific logic — `ast`, and any
future parser, is imported here and nowhere else. A guard test enforces that.
"""

from __future__ import annotations

from .base import LanguageAdapter
from .python_adapter import PythonAdapter

__all__ = ["LanguageAdapter", "PythonAdapter", "default_adapters"]


def default_adapters() -> list[LanguageAdapter]:
    """Every adapter this build ships, in match order.

    Adding a language means adding a module here and one entry in this list;
    nothing in the core pipeline changes.
    """
    return [PythonAdapter()]
