"""`LanguageAdapter` for Python: symbol extraction and lint.

CLAUDE.md §2.2: this module and its siblings are the only places allowed to
contain language-specific logic, and `ast` is imported nowhere else. The core
pipeline sees a `LanguageAdapter` and never learns which language it got.

**Mapping changed lines to symbols.** The interface hands over a filename and a
unified diff, so the adapter has to recover enough of the *new* file to parse it.
It tries three tiers, in order, and degrades rather than failing:

1. **Full source** — when the caller can supply the head version of the file
   (via the optional ``source`` argument), the AST is exact and so are the line
   ranges. The change analyzer passes it whenever a checkout is available.
2. **Reconstructed source** — otherwise the new-side lines are rebuilt from the
   patch at their true line numbers, with gaps padded so numbering still lines
   up. A whole-file patch (any added file, and most small ones) parses cleanly
   and gives the same answer as tier 1.
3. **File level** — a partial patch usually will not parse: a hunk from the
   middle of a function is a dangling indented block. Rather than guess, the
   adapter returns one ``kind="file"`` symbol spanning the changed lines. Callers
   see less detail, never a crash and never a wrong line range.

A syntax error in the head version itself lands in tier 3 too — a PR that breaks
the parser is exactly a PR worth reviewing, so it must not take the run down.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from ..findings.finding import Finding
from ..analysis.patch import iter_patch_lines
from ..types import Symbol

__all__ = [
    "PythonAdapter",
    "EXTENSIONS",
    "FILE_KIND",
    "changed_line_numbers",
    "reconstruct_new_side",
]

EXTENSIONS = (".py", ".pyi")
"""Extensions this adapter claims. `.pyx`/`.pyd` are deliberately excluded."""

FILE_KIND = "file"
"""Kind used for the file-level degradation, distinct from any real symbol kind."""


_FUNCTION_KINDS = {"function", "async function", "method", "async method"}


@dataclass(frozen=True)
class _Scope:
    """One symbol found in the AST, with the lines it spans."""

    name: str
    kind: str
    line_start: int
    line_end: int
    depth: int


class PythonAdapter:
    """Python implementation of `LanguageAdapter`."""

    language = "python"

    def matches(self, filename: str) -> bool:
        """Whether this adapter handles ``filename``."""
        return filename.lower().endswith(EXTENSIONS)

    def changed_symbols(
        self, filename: str, diff: str, *, source: str | None = None
    ) -> list[Symbol]:
        """Symbols containing the lines this diff changes.

        ``source`` is an optional extra: the head version of the file, when the
        caller has it. Supplying it makes the result exact; omitting it keeps the
        locked interface signature working, at the cost of falling back to a
        reconstruction. Either way the return type is the same.
        """
        lines = changed_line_numbers(diff)
        if not lines:
            return []

        scopes = self._scopes(source if source is not None else reconstruct_new_side(diff))
        if scopes is None:
            return [_file_symbol(filename, lines)]

        symbols = _attribute(filename, lines, scopes)
        return symbols or [_file_symbol(filename, lines)]

    def lint(self, path: str) -> list[Finding]:
        """Static-analysis findings for ``path``.

        Intentionally empty: the shape is wired so stage 7's verification can
        call it uniformly, but no lint engine is integrated yet. Returning `[]`
        is a valid answer under the interface, and an empty list can never
        produce an evidence-free finding.
        """
        return []

    # -- parsing ----------------------------------------------------------

    @staticmethod
    def _scopes(source: str) -> list[_Scope] | None:
        """Every function/class in ``source``, or ``None`` if it will not parse."""
        if not source.strip():
            return None
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError, RecursionError):
            # ValueError covers source containing null bytes; RecursionError a
            # pathologically nested file. All three mean "cannot parse".
            return None
        return _walk(tree)


def _walk(tree: ast.Module) -> list[_Scope]:
    """Collect scopes depth-first, recording nesting so the innermost wins."""
    scopes: list[_Scope] = []

    def visit(node: ast.AST, container: str | None, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                name = _qualify(container, child.name)
                scopes.append(
                    _Scope(name, "class", child.lineno, _end_of(child), depth)
                )
                visit(child, name, depth + 1)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = _qualify(container, child.name)
                scopes.append(
                    _Scope(
                        name,
                        _function_kind(child, inside_class=container is not None),
                        _decorated_start(child),
                        _end_of(child),
                        depth,
                    )
                )
                visit(child, name, depth + 1)
            else:
                # Not a scope itself, but may contain one (a `def` under `if
                # TYPE_CHECKING`, inside a `try`, or in a `with` block).
                visit(child, container, depth)

    visit(tree, None, 0)
    return scopes


def _qualify(container: str | None, name: str) -> str:
    """``ClassName.method`` — the qualified name is what a reviewer needs."""
    return f"{container}.{name}" if container else name


def _function_kind(
    node: ast.FunctionDef | ast.AsyncFunctionDef, *, inside_class: bool
) -> str:
    base = "method" if inside_class else "function"
    return f"async {base}" if isinstance(node, ast.AsyncFunctionDef) else base


def _decorated_start(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Include decorators: changing `@property` changes the method."""
    if node.decorator_list:
        return min(decorator.lineno for decorator in node.decorator_list)
    return node.lineno


def _end_of(node: ast.AST) -> int:
    end = getattr(node, "end_lineno", None)
    if isinstance(end, int):
        return end
    return int(getattr(node, "lineno", 1))


def _attribute(
    filename: str, lines: set[int], scopes: list[_Scope]
) -> list[Symbol]:
    """Map each changed line to the innermost scope containing it.

    A change inside a method is attributed to the method, not to its class — the
    class only surfaces when the change is in the class body itself (a field, a
    docstring) or when it spans the whole class.
    """
    matched: dict[tuple[str, str], _Scope] = {}

    for line in lines:
        innermost: _Scope | None = None
        for scope in scopes:
            if scope.line_start <= line <= scope.line_end:
                if innermost is None or scope.depth > innermost.depth:
                    innermost = scope
        if innermost is not None:
            matched.setdefault((innermost.name, innermost.kind), innermost)

    return [
        Symbol(
            name=scope.name,
            kind=scope.kind,
            file=filename,
            line_start=scope.line_start,
            line_end=scope.line_end,
        )
        for scope in sorted(matched.values(), key=lambda item: item.line_start)
    ]


def _file_symbol(filename: str, lines: set[int]) -> Symbol:
    """The tier-3 degradation: the file itself, spanning what changed."""
    return Symbol(
        name=filename,
        kind=FILE_KIND,
        file=filename,
        line_start=min(lines),
        line_end=max(lines),
    )


# --------------------------------------------------------------------------
# Patch reading — new-side line numbers and reconstruction
# --------------------------------------------------------------------------


def changed_line_numbers(diff: str) -> set[int]:
    """New-side line numbers this patch adds or modifies.

    Deletions have no new-side line of their own; they are represented by the
    line they sit next to, so a pure deletion still attributes to the symbol it
    was removed from.
    """
    changed: set[int] = set()
    for line in iter_patch_lines(diff):
        if line.kind == "add":
            changed.add(line.new_line)
        elif line.kind == "remove":
            # Attribute the removal to the surrounding new-side position.
            changed.add(max(line.new_line, 1))
    return changed


def reconstruct_new_side(diff: str) -> str:
    """Rebuild the new version of the file from a patch, preserving line numbers.

    Context and added lines are placed at their true new-side positions; unknown
    stretches become blank lines. For a whole-file patch this reproduces the file
    exactly, which is why tier 2 works at all. For a partial patch the result
    usually will not parse — that is the signal to degrade, not a bug.
    """
    known: dict[int, str] = {}
    for line in iter_patch_lines(diff):
        if line.kind in ("add", "context"):
            known[line.new_line] = line.text

    if not known:
        return ""
    return "\n".join(known.get(number, "") for number in range(1, max(known) + 1))
