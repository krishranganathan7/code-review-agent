"""The path chokepoint: every tool resolves paths here and nowhere else.

Hard constraint (CLAUDE.md §2.7): search tools are read-only, never reach the
network, and are scoped to the checkout root. A path that escapes the root is
refused rather than resolved.

Paths arriving here are **model-supplied, untrusted input**. The refusal rule is
deliberately one rule, applied after full resolution:

    resolve the candidate completely — every ``..``, every symlink, every
    Windows junction, at every level — then require the result to be inside the
    resolved root.

Resolving first is what makes the check total. Detecting links instead would be
unsound: an intermediate directory link is followed before the final component
is ever examined, and on Windows ``os.path.islink()`` is *False* for a directory
junction, which is a working escape vector. Resolution has no such blind spot,
so this module never inspects link-ness at all.

The root itself is resolved once at construction, so a checkout that lives under
a symlink compares consistently.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

__all__ = ["Sandbox", "SandboxViolation"]


class SandboxViolation(Exception):
    """A path was refused because it escapes the repository root.

    Tools convert this into a failed :class:`~reviewer.types.ToolResult`; it is
    never raised across a tool's ``run()`` boundary. The message names what was
    requested but never where the root actually is — the agent does not need the
    absolute layout of the machine to correct its own call.
    """

    def __init__(self, requested: str, reason: str) -> None:
        self.requested = requested
        self.reason = reason
        super().__init__(f"refused path {requested!r}: {reason}")


class Sandbox:
    """A repository root, and the only thing permitted to turn a path into a path.

    Construct one per run and hand it to every tool::

        sandbox = Sandbox("/checkout")
        target = sandbox.resolve("src/pr.py")     # absolute, verified inside root
        name = sandbox.relative(target)           # "src/pr.py", posix-style
    """

    def __init__(self, repo_root: str | Path) -> None:
        root = Path(repo_root).expanduser()
        try:
            resolved = root.resolve()
        except OSError as exc:  # pragma: no cover - unreadable mount
            raise SandboxViolation(str(repo_root), f"unusable root ({exc})") from exc
        if not resolved.is_dir():
            raise SandboxViolation(
                str(repo_root), "repository root is not an existing directory"
            )
        self.root = resolved

    def __repr__(self) -> str:
        return f"Sandbox({str(self.root)!r})"

    # -- the one resolution path ------------------------------------------

    def resolve(self, path: str | None = None) -> Path:
        """Resolve ``path`` against the root, or raise :class:`SandboxViolation`.

        ``None`` and ``""`` mean the root itself. A relative path is taken
        relative to the root; an absolute path is accepted only if it lands
        inside the root. Existence is *not* required — a missing file resolves
        fine and the caller reports "no such file", which is a more useful error
        than a refusal.
        """
        requested = "." if path is None or path == "" else str(path)

        if "\x00" in requested:
            raise SandboxViolation(requested, "path contains a null byte")

        candidate = Path(requested).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate

        try:
            # Full resolution: every '..', symlink and junction, at every level.
            resolved = candidate.resolve()
        except (OSError, RuntimeError) as exc:
            # RuntimeError is a symlink loop on some platforms.
            raise SandboxViolation(requested, f"could not be resolved ({exc})") from exc

        if not self.contains(resolved):
            raise SandboxViolation(
                requested, "resolves outside the repository root"
            )
        return resolved

    def contains(self, target: Path) -> bool:
        """Whether an already-resolved path lies at or inside the root."""
        return target == self.root or target.is_relative_to(self.root)

    def relative(self, target: Path) -> str:
        """A resolved path as a repo-relative, forward-slash string.

        Forward slashes regardless of platform: these strings become `Finding.file`
        values and appear in review output, which must not vary by the OS the
        review happened to run on.
        """
        if target == self.root:
            return "."
        return str(PurePosixPath(*target.relative_to(self.root).parts))

    def is_inside(self, path: str | None) -> bool:
        """Whether ``path`` would be accepted. Never raises."""
        try:
            self.resolve(path)
        except SandboxViolation:
            return False
        return True
