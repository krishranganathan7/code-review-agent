"""Stage 2 — diff to changed symbols and coherent change groups.

Two jobs, both deterministic:

**Symbol extraction.** Each changed file is routed to the first
:class:`~reviewer.languages.base.LanguageAdapter` that claims it, and the adapter
maps the diff's changed lines onto the symbols containing them. A file no adapter
claims — or one whose patch GitHub omitted — degrades to a file-level record with
the reason recorded. The pipeline must handle any language, adapter or not
(CLAUDE.md §2.2), so "no adapter" is a normal outcome rather than an error.

**Grouping.** The specialized agents in stage 5 review a *coherent unit*, not a
token window. Grouping is by relationship — a module with its tests, the
dependency manifests, the migrations, then package cohesion — and every group
records *why* its members belong together, so a reviewer can audit the split.

No language-specific logic appears here: this module knows only the
`LanguageAdapter` interface.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Literal, Sequence

from ..languages.base import LanguageAdapter
from ..types import Symbol
from .ingest import ChangedFile, PullRequest
from .source import SourceProvider

__all__ = [
    "AttributionTier",
    "ATTRIBUTION_TIERS",
    "SourceLookup",
    "FileChange",
    "ChangeGroup",
    "ChangeScope",
    "ChangeAnalysis",
    "ChangeAnalyzer",
    "analyze",
    "MANIFEST_NAMES",
    "MANIFEST_SUFFIXES",
    "MIGRATION_MARKERS",
    "TEST_MARKERS",
]

logger = logging.getLogger("reviewer.analysis.change")

AttributionTier = Literal["exact", "reconstructed", "file"]
"""How a file's symbols were resolved — the tier ladder, made observable.

* ``exact`` — parsed from the real head revision (tier 1).
* ``reconstructed`` — parsed from the patch rebuilt at its true line numbers
  (tier 2). Correct, but only as complete as the patch.
* ``file`` — neither parsed; one file-level symbol spanning the changed lines
  (tier 3). The safety net, never a guessed range.
"""

ATTRIBUTION_TIERS: tuple[AttributionTier, ...] = ("exact", "reconstructed", "file")


SourceLookup = Callable[[str], str | None]
"""Returns the head version of a repo-relative path, or ``None`` if unavailable.

Optional. When supplied — typically backed by the Phase 2 sandbox over a local
checkout — adapters get exact source and exact line numbers instead of working
from a reconstruction of the patch."""

MANIFEST_NAMES = frozenset(
    {
        "requirements.txt",
        "requirements-dev.txt",
        "constraints.txt",
        "pyproject.toml",
        "poetry.lock",
        "pipfile",
        "pipfile.lock",
        "setup.py",
        "setup.cfg",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "go.mod",
        "go.sum",
        "cargo.toml",
        "cargo.lock",
        "gemfile",
        "gemfile.lock",
        "composer.json",
        "composer.lock",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "gradle.lockfile",
        "packages.config",
    }
)
"""Dependency manifests and lockfiles, matched on basename (case-insensitive)."""

MANIFEST_SUFFIXES = (".csproj", ".fsproj", ".vbproj")

MIGRATION_MARKERS = ("migration", "migrations", "alembic", "schema", "flyway", "liquibase")
"""Path segments that mark schema-evolution work."""

TEST_MARKERS = ("test", "tests", "spec", "specs", "__tests__")
"""Path segments that mark a test tree."""


# --------------------------------------------------------------------------
# The change model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FileChange:
    """One changed file, understood as far as the available adapters allow."""

    file: ChangedFile
    language: str | None = None
    symbols: list[Symbol] = field(default_factory=list)
    degraded: bool = False
    reason: str | None = None
    attribution: AttributionTier = "file"
    """Which tier resolved this file's symbols. See :data:`AttributionTier`."""
    source_available: bool = False
    """Whether head content was obtained for this file at all."""

    @property
    def path(self) -> str:
        return self.file.path

    @property
    def exact(self) -> bool:
        """Whether symbols came from the real head revision."""
        return self.attribution == "exact"

    @property
    def symbol_names(self) -> list[str]:
        return [symbol.name for symbol in self.symbols]

    @property
    def understood(self) -> bool:
        """Whether real symbols were recovered, as opposed to a file-level record."""
        return bool(self.symbols) and not self.degraded


@dataclass(frozen=True)
class ChangeGroup:
    """A set of changed files that belong together, and why.

    One group is one review unit in stage 5. ``reason`` is part of the output
    rather than an implementation detail: a grouping a reviewer cannot audit is a
    grouping nobody can trust.
    """

    key: str
    label: str
    reason: str
    paths: list[str] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.paths)


@dataclass(frozen=True)
class ChangeScope:
    """How big and how spread out the change is. Inputs to the risk engine."""

    files_touched: int = 0
    symbols_touched: int = 0
    additions: int = 0
    deletions: int = 0
    directories_touched: int = 0
    languages: list[str] = field(default_factory=list)
    unsupported_files: int = 0
    renames: int = 0
    deletions_of_files: int = 0
    groups: int = 0
    exact_files: int = 0
    """Files whose symbols came from the real head revision (tier 1)."""
    reconstructed_files: int = 0
    """Files resolved from the patch alone (tier 2)."""
    file_level_files: int = 0
    """Files that fell back to a file-level symbol (tier 3)."""

    @property
    def changed_lines(self) -> int:
        return self.additions + self.deletions

    @property
    def spread(self) -> float:
        """Directories per file, in ``0.0..1.0``.

        Near 1.0 means every file sits in its own directory — a change smeared
        across the tree. Near 0.0 means one tight location.
        """
        if not self.files_touched:
            return 0.0
        return self.directories_touched / self.files_touched


@dataclass(frozen=True)
class ChangeAnalysis:
    """The structural understanding stage 5 builds on."""

    pull_request: PullRequest
    file_changes: list[FileChange] = field(default_factory=list)
    groups: list[ChangeGroup] = field(default_factory=list)
    scope: ChangeScope = field(default_factory=ChangeScope)

    @property
    def symbols(self) -> list[Symbol]:
        return [symbol for change in self.file_changes for symbol in change.symbols]

    @property
    def paths(self) -> list[str]:
        return [change.path for change in self.file_changes]

    def change_for(self, path: str) -> FileChange | None:
        for change in self.file_changes:
            if change.path == path:
                return change
        return None

    def group_for(self, path: str) -> ChangeGroup | None:
        for group in self.groups:
            if path in group.paths:
                return group
        return None

    @property
    def exact_paths(self) -> list[str]:
        """Files whose symbols came from the real head revision."""
        return [change.path for change in self.file_changes if change.exact]

    def tier_counts(self) -> dict[str, int]:
        """How many files resolved at each tier. The accuracy of one analysis."""
        counts: dict[str, int] = {tier: 0 for tier in ATTRIBUTION_TIERS}
        for change in self.file_changes:
            counts[change.attribution] += 1
        return counts

    @property
    def degraded_paths(self) -> list[str]:
        """Files understood only at file level — reduced confidence downstream."""
        return [change.path for change in self.file_changes if change.degraded]


# --------------------------------------------------------------------------
# The analyzer
# --------------------------------------------------------------------------


class ChangeAnalyzer:
    """Stage 2. Deterministic: same PR in, same analysis out."""

    def __init__(
        self,
        adapters: Sequence[LanguageAdapter] | None = None,
        *,
        source: SourceProvider | None = None,
        source_lookup: SourceLookup | None = None,
    ) -> None:
        """
        ``source`` is the preferred way to supply head content: a
        :class:`~reviewer.analysis.source.SourceProvider`, normally a
        `ReviewWorkspace.source()`. ``source_lookup`` is the same thing as a bare
        callable, kept because it is the lower-level form and some callers
        already have one.

        With neither, attribution works from the patch alone — the pre-Phase-4.5
        behaviour, still fully supported.
        """
        if adapters is None:
            from ..languages import default_adapters

            adapters = default_adapters()
        self.adapters = list(adapters)
        self.source = source
        self.source_lookup = source_lookup
        if source is not None:
            self._read_source: SourceLookup | None = source.read
        else:
            self._read_source = source_lookup

    def analyze(self, pull_request: PullRequest) -> ChangeAnalysis:
        changes = [self._analyze_file(changed) for changed in pull_request.files]
        groups = _group(changes)
        scope = _scope(changes, groups)

        logger.info(
            "change_analysis repo=%s pr=%d files=%d symbols=%d groups=%d",
            pull_request.repo,
            pull_request.number,
            scope.files_touched,
            scope.symbols_touched,
            scope.groups,
            extra={
                "event": "change_analysis",
                "repo": pull_request.repo,
                "pr": pull_request.number,
                "files": scope.files_touched,
                "symbols": scope.symbols_touched,
                "groups": scope.groups,
                "unsupported_files": scope.unsupported_files,
                "languages": scope.languages,
                "exact_files": scope.exact_files,
                "reconstructed_files": scope.reconstructed_files,
                "file_level_files": scope.file_level_files,
            },
        )
        return ChangeAnalysis(
            pull_request=pull_request,
            file_changes=changes,
            groups=groups,
            scope=scope,
        )

    def _analyze_file(self, changed: ChangedFile) -> FileChange:
        adapter = self._adapter_for(changed.path)

        if adapter is None:
            return FileChange(
                file=changed,
                language=None,
                symbols=[_whole_file(changed)],
                degraded=True,
                reason="no language adapter claims this file",
                attribution="file",
            )

        if not changed.has_patch:
            return FileChange(
                file=changed,
                language=adapter.language,
                symbols=[_whole_file(changed)],
                degraded=True,
                reason="no patch available (binary file, or a diff GitHub truncated)",
                attribution="file",
            )

        if changed.status == "removed":
            # A deleted file has no head revision to parse — the workspace
            # correctly does not contain it. The reviewable fact is that it is
            # gone, and by how much; naming the symbols that went with it would
            # mean parsing the patch's *old* side, which the interface's
            # new-side contract does not cover.
            return FileChange(
                file=changed,
                language=adapter.language,
                symbols=[_whole_file(changed)],
                degraded=True,
                reason="file deleted; no head revision exists to parse",
                attribution="file",
            )

        return self._attribute(adapter, changed)

    def _attribute(self, adapter: LanguageAdapter, changed: ChangedFile) -> FileChange:
        """Walk the tier ladder, stopping at the first tier that resolves symbols.

        The cascade lives here rather than inside the adapter on purpose. An
        adapter's job is to parse what it is handed; deciding *what* to hand it,
        and in what order, is stage 2's. Keeping it here means the tier that
        actually succeeded is a fact the analyzer observed rather than something
        it has to infer, and the locked `changed_symbols` interface does not have
        to grow a way to report it.
        """
        head = self._head_source(adapter, changed)

        # Tier 1 — the real head revision.
        if head is not None:
            symbols = self._ask(adapter, changed, source=head)
            if symbols and not _is_file_level(symbols, changed.path):
                return FileChange(
                    file=changed,
                    language=adapter.language,
                    symbols=symbols,
                    degraded=False,
                    reason=None,
                    attribution="exact",
                    source_available=True,
                )

        # Tier 2 — the patch, rebuilt at its true line numbers.
        symbols = self._ask(adapter, changed)
        if symbols and not _is_file_level(symbols, changed.path):
            return FileChange(
                file=changed,
                language=adapter.language,
                symbols=symbols,
                degraded=False,
                reason=(
                    None
                    if head is None
                    else "head revision did not parse; resolved from the patch"
                ),
                attribution="reconstructed",
                source_available=head is not None,
            )

        # Tier 3 — the safety net.
        return FileChange(
            file=changed,
            language=adapter.language,
            symbols=symbols or [_whole_file(changed)],
            degraded=True,
            reason="the adapter could not resolve symbols from this patch",
            attribution="file",
            source_available=head is not None,
        )

    def _head_source(
        self, adapter: LanguageAdapter, changed: ChangedFile
    ) -> str | None:
        """Head content for a file, if a provider has it and the adapter can use it."""
        if self._read_source is None or not _accepts_source(type(adapter)):
            return None
        try:
            return self._read_source(changed.path)
        except Exception as exc:  # pragma: no cover - a provider is not trusted
            logger.warning(
                "change_source_failed path=%s error=%s",
                changed.path,
                exc,
                extra={
                    "event": "change_source_failed",
                    "path": changed.path,
                    "error": str(exc),
                },
            )
            return None

    def _ask(
        self,
        adapter: LanguageAdapter,
        changed: ChangedFile,
        *,
        source: str | None = None,
    ) -> list[Symbol]:
        """One adapter call. An adapter that raises degrades, it does not abort."""
        try:
            if source is not None:
                return list(
                    adapter.changed_symbols(  # type: ignore[call-arg]
                        changed.path, changed.patch, source=source
                    )
                )
            return list(adapter.changed_symbols(changed.path, changed.patch))
        except Exception as exc:
            logger.warning(
                "change_adapter_failed path=%s adapter=%s error=%s",
                changed.path,
                adapter.language,
                exc,
                extra={
                    "event": "change_adapter_failed",
                    "path": changed.path,
                    "language": adapter.language,
                    "error": str(exc),
                },
            )
            return []

    def _adapter_for(self, path: str) -> LanguageAdapter | None:
        for adapter in self.adapters:
            if adapter.matches(path):
                return adapter
        return None


def analyze(
    pull_request: PullRequest,
    adapters: Sequence[LanguageAdapter] | None = None,
    *,
    source: SourceProvider | None = None,
) -> ChangeAnalysis:
    """Convenience entry point over :class:`ChangeAnalyzer`."""
    return ChangeAnalyzer(adapters, source=source).analyze(pull_request)


@lru_cache(maxsize=None)
def _accepts_source(adapter_type: type[Any]) -> bool:
    """Whether an adapter's `changed_symbols` takes the optional `source` argument.

    Inspected once per adapter class. Phase 4 probed this by catching `TypeError`
    from the call itself, which would also swallow a `TypeError` raised *inside* a
    conforming adapter and silently drop to a lower tier. Asking the signature
    separates "cannot accept source" from "failed while using it".
    """
    try:
        signature = inspect.signature(adapter_type.changed_symbols)
    except (TypeError, ValueError, AttributeError):  # pragma: no cover - exotic adapter
        return False
    return "source" in signature.parameters


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


def _group(changes: Sequence[FileChange]) -> list[ChangeGroup]:
    """Partition changed files into coherent review units.

    The heuristic is ordered, and each file joins the first group that claims it:

    1. **dependencies** — every manifest and lockfile, together. A dependency
       bump is one decision however many files record it.
    2. **schema** — migrations and schema definitions, together. Ordering and
       reversibility are properties of the set, not of one file.
    3. **module + tests** — a source file with the tests that exercise it,
       matched by filename stem. Reviewing a change without its test change is
       reviewing half of it.
    4. **package** — whatever is left, grouped by top-level directory, so a
       reviewer sees one area at a time.

    Deterministic throughout: groups come out in a fixed order and so do their
    members.
    """
    remaining = {change.path: change for change in changes}
    groups: list[ChangeGroup] = []

    manifests = sorted(path for path in remaining if _is_manifest(path))
    if manifests:
        groups.append(
            _build_group(
                "dependencies",
                "Dependency manifests",
                "dependency manifests and lockfiles change as one decision",
                manifests,
                remaining,
            )
        )

    schema = sorted(path for path in remaining if _is_schema(path))
    if schema:
        groups.append(
            _build_group(
                "schema",
                "Schema and migrations",
                "schema changes must be reviewed as an ordered set, not file by file",
                schema,
                remaining,
            )
        )

    for path in sorted(p for p in remaining if not _is_test(p)):
        if path not in remaining:
            continue
        partners = sorted(
            other
            for other in remaining
            if other != path and _is_test(other) and _pairs_with(other, path)
        )
        if not partners:
            continue
        groups.append(
            _build_group(
                f"module:{path}",
                _module_label(path),
                "a source file reviewed together with the tests that exercise it",
                [path, *partners],
                remaining,
            )
        )

    by_package: dict[str, list[str]] = {}
    for path in sorted(remaining):
        by_package.setdefault(_package_of(path), []).append(path)

    for package, paths in sorted(by_package.items()):
        groups.append(
            _build_group(
                f"package:{package}",
                package or "(repository root)",
                "remaining files grouped by the area of the tree they sit in",
                paths,
                remaining,
            )
        )

    return groups


def _build_group(
    key: str,
    label: str,
    reason: str,
    paths: Sequence[str],
    remaining: dict[str, FileChange],
) -> ChangeGroup:
    """Claim ``paths`` out of ``remaining`` and build the group."""
    claimed = [path for path in paths if path in remaining]
    symbols = [
        symbol for path in claimed for symbol in remaining[path].symbols
    ]
    for path in claimed:
        del remaining[path]
    return ChangeGroup(
        key=key, label=label, reason=reason, paths=claimed, symbols=symbols
    )


def _is_manifest(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    return name in MANIFEST_NAMES or name.endswith(MANIFEST_SUFFIXES)


def _is_schema(path: str) -> bool:
    segments = {segment.lower() for segment in path.split("/")}
    if segments & set(MIGRATION_MARKERS):
        return True
    return path.lower().endswith(".sql")


def _is_test(path: str) -> bool:
    segments = [segment.lower() for segment in path.split("/")]
    if set(segments[:-1]) & set(TEST_MARKERS):
        return True
    name = segments[-1]
    stem = name.rsplit(".", 1)[0]
    return (
        stem.startswith("test_")
        or stem.endswith("_test")
        or stem.endswith(".test")
        or stem.endswith(".spec")
        or stem.endswith("_spec")
    )


def _pairs_with(test_path: str, source_path: str) -> bool:
    """Whether ``test_path`` looks like the tests for ``source_path``."""
    return _test_subject(test_path) == _stem(source_path)


def _test_subject(test_path: str) -> str:
    """The stem a test file appears to be testing."""
    stem = _stem(test_path)
    for prefix in ("test_", "spec_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    for suffix in ("_test", "_spec", ".test", ".spec"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _stem(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


def _module_label(path: str) -> str:
    return f"{_stem(path)} and its tests"


def _package_of(path: str) -> str:
    """Top-level directory, the coarsest useful cohesion signal."""
    head, separator, _ = path.partition("/")
    return head if separator else ""


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------


def _scope(changes: Sequence[FileChange], groups: Sequence[ChangeGroup]) -> ChangeScope:
    directories = {change.file.directory for change in changes}
    languages = sorted(
        {change.language for change in changes if change.language is not None}
    )
    return ChangeScope(
        files_touched=len(changes),
        symbols_touched=sum(len(change.symbols) for change in changes),
        additions=sum(change.file.additions for change in changes),
        deletions=sum(change.file.deletions for change in changes),
        directories_touched=len(directories),
        languages=languages,
        unsupported_files=sum(1 for change in changes if change.language is None),
        renames=sum(1 for change in changes if change.file.is_rename),
        deletions_of_files=sum(
            1 for change in changes if change.file.status == "removed"
        ),
        groups=len(groups),
        exact_files=sum(1 for change in changes if change.attribution == "exact"),
        reconstructed_files=sum(
            1 for change in changes if change.attribution == "reconstructed"
        ),
        file_level_files=sum(1 for change in changes if change.attribution == "file"),
    )


def _whole_file(changed: ChangedFile) -> Symbol:
    """A file-level symbol, for changes no adapter could resolve further."""
    return Symbol(
        name=changed.path,
        kind="file",
        file=changed.path,
        line_start=1,
        line_end=max(changed.additions + changed.deletions, 1),
    )


def _is_file_level(symbols: Sequence[Symbol], path: str) -> bool:
    """Whether the adapter's answer was the file itself rather than symbols."""
    return len(symbols) == 1 and symbols[0].kind == "file" and symbols[0].name == path
