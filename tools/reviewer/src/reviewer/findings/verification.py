"""Stage 7 — validate file/line references and run targeted deterministic checks.

This is the corroboration layer, and it contains **no model call**. Everything
here re-checks a finding against the head workspace and the diff, which are
facts. The model's own second opinion is stage 8's job (`judge.py`), kept
separate on purpose: a deterministic check that can be argued with is not a
check.

Each surviving finding leaves with its `verification` set to one of the three
documented values:

* **VERIFIED** — a deterministic check corroborated it. The cited lines are
  lines this pull request actually changed, or the code the evidence quotes is
  really there, or a claimed credential really does match an added line.
* **REFUTED** — a check contradicted it. Refuted findings are **dropped**, with
  the reason recorded (§2.4): a finding we can show is wrong is worse than no
  finding.
* **UNVERIFIED** — nothing here could decide. Kept, and marked, so a reader can
  see which findings carry deterministic support and which rest on the agent's
  reasoning alone.

Refutation is deliberately conservative. A check fires only when it is certain,
and "certain" means the check cannot be wrong about a well-behaved finding:

* the file is not in the tree at head;
* the cited line is past the end of it;
None of those depends on reading the finding's prose, which is the property that
matters: the two checks removed from this stage both keyed off wording.

**A credential claim is corroboration only, too.** The deterministic secret scan
still runs -- a matching added line marks a finding VERIFIED -- but finding no
match no longer refutes. The scan keys off credential words in the finding's
prose, and an injection finding that mentions password hashes is describing
impact, not claiming a secret was committed. The guarantee that secrets are
enforced in code (§2.6) does not live here: it lives in stage 3's
`secret_material` signal and the gate's unwaivable CRITICAL security rule, both
of which read the diff directly and neither of which consults any prose.

**Quoted code is corroboration only, never refutation.** An earlier version
dropped a finding whose evidence quoted a snippet absent from the file. Four
live reviews showed why that is wrong: reviewers do not quote verbatim. They
quote prose in backticks for emphasis, they quote the diff with its `+`/`-`
markers, they abbreviate with an ellipsis, they truncate mid-expression, and
they reformat. Every one of those shapes was refuted as fabricated, and across
those runs the check destroyed true findings and caught no fabricated ones.
Finding a quote *present* is still good evidence, so it still marks a finding
VERIFIED; failing to find one now means only that this stage cannot vouch for
it. Catching genuinely invented evidence is the judge's job — it re-reads the
cited code and can reject on it, and unlike a text match it can tell an
abbreviation from a fabrication.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

from ..analysis.ingest import ChangedFile, PullRequest
from ..analysis.patch import added_lines
from ..analysis.risk import SECRET_PATTERNS
from .finding import Finding

__all__ = [
    "VerificationResult",
    "RefutedFinding",
    "Verifier",
    "verify",
    "MIN_QUOTE_LENGTH",
]

logger = logging.getLogger("reviewer.findings.verification")

MIN_QUOTE_LENGTH = 12
"""Shortest quoted snippet worth checking.

Below this, a quote is as likely to be a variable name that legitimately appears
nowhere near the cited line as it is to be fabricated code.
"""

QUOTE = re.compile(r"`([^`\n]{%d,300})`" % MIN_QUOTE_LENGTH)
"""Backticked code spans in a finding's evidence — the checkable claims."""

DIFF_PREFIX = re.compile(r"^[+-]\s?")
"""A leading diff marker on a quoted patch line.

Agents quote the diff as often as the file — "`-import express from 'express';`"
is a real line of the change, and the `-` is the patch's, not the file's.
"""

SECRET_CLAIM = re.compile(
    r"(?i)\b(secret|credential|api[_-]?key|access[_-]?key|private key|token|password)\b"
)
"""Evidence that claims a credential is present, which stage 3 can re-check."""


@dataclass(frozen=True)
class RefutedFinding:
    """A finding a deterministic check contradicted, and what contradicted it."""

    finding: Finding
    reason: str

    def __str__(self) -> str:
        return f"{self.finding.file}:{self.finding.line_start} — {self.reason}"


@dataclass(frozen=True)
class VerificationResult:
    """Findings that survived, with their verification state, and those that did not."""

    findings: list[Finding] = field(default_factory=list)
    refuted: list[RefutedFinding] = field(default_factory=list)

    @property
    def verified(self) -> list[Finding]:
        return [f for f in self.findings if f.verification == "VERIFIED"]

    @property
    def unverified(self) -> list[Finding]:
        return [f for f in self.findings if f.verification == "UNVERIFIED"]

    @property
    def refutation_reasons(self) -> list[str]:
        return [item.reason for item in self.refuted]

    def counts(self) -> dict[str, int]:
        return {
            "VERIFIED": len(self.verified),
            "UNVERIFIED": len(self.unverified),
            "REFUTED": len(self.refuted),
        }


class Verifier:
    """Stage 7. Re-checks findings against the head workspace and the diff."""

    def __init__(
        self,
        pull_request: PullRequest,
        source: Any | None = None,
    ) -> None:
        self.pull_request = pull_request
        self.source = source
        self._changed_lines: dict[str, set[int]] = {}

    def check(self, findings: Iterable[Finding]) -> VerificationResult:
        kept: list[Finding] = []
        refuted: list[RefutedFinding] = []

        for finding in findings:
            verdict, reason = self._verify_one(finding)
            if verdict == "REFUTED":
                refuted.append(RefutedFinding(finding=finding, reason=reason))
                continue
            kept.append(replace(finding, verification=verdict))

        logger.info(
            "verify kept=%d verified=%d refuted=%d",
            len(kept),
            sum(1 for f in kept if f.verification == "VERIFIED"),
            len(refuted),
            extra={
                "event": "verify",
                "kept": len(kept),
                "verified": sum(1 for f in kept if f.verification == "VERIFIED"),
                "refuted": len(refuted),
                "reasons": [item.reason for item in refuted],
            },
        )
        return VerificationResult(findings=kept, refuted=refuted)

    # -- one finding ------------------------------------------------------

    def _verify_one(self, finding: Finding) -> tuple[str, str]:
        """The verification state for one finding, and the reason if refuted."""
        content = self._read(finding.file)

        if content is None:
            changed = self.pull_request.file(finding.file)
            if changed is not None and changed.status == "removed":
                return (
                    "REFUTED",
                    f"{finding.file} was deleted by this change, so a finding "
                    "about its contents cannot stand",
                )
            if changed is None and self.pull_request.files:
                return (
                    "REFUTED",
                    f"{finding.file} does not exist at head and is not part of "
                    "this change",
                )
            # Readable-but-not-available (no workspace, binary, oversized).
            return ("UNVERIFIED", "")

        lines = content.splitlines()
        if finding.line_start > len(lines) or finding.line_end > len(lines):
            return (
                "REFUTED",
                f"cites lines {finding.line_start}-{finding.line_end} of "
                f"{finding.file}, which has {len(lines)} line(s)",
            )

        secret_verdict, secret_reason = self._check_secret_claim(finding)
        if secret_verdict is not None:
            return (secret_verdict, secret_reason)

        for corroboration in self._corroborations(finding, lines):
            logger.debug(
                "verify_corroborated file=%s reason=%s", finding.file, corroboration
            )
            return ("VERIFIED", "")

        return ("UNVERIFIED", "")

    def _corroborations(self, finding: Finding, lines: Sequence[str]) -> list[str]:
        """Deterministic reasons to trust this finding. Empty means undecided."""
        reasons: list[str] = []

        touched = self._lines_changed(finding.file)
        cited = set(range(finding.line_start, finding.line_end + 1))
        if touched & cited:
            reasons.append("cited lines are lines this change touched")

        for quote in _quotes(finding.evidence):
            window = "\n".join(
                lines[max(finding.line_start - 4, 0) : finding.line_end + 3]
            )
            if _normalize(DIFF_PREFIX.sub("", quote.strip())) in _normalize(window):
                reasons.append("evidence quotes code present at the cited lines")
                break

        return reasons

    def _check_secret_claim(self, finding: Finding) -> tuple[str | None, str]:
        """Re-run the deterministic secret scan behind a credential claim.

        Stage 3 already scans added lines with these patterns, so a finding that
        claims a credential can be confirmed or contradicted outright.
        """
        if finding.category not in ("security",):
            return (None, "")
        if not SECRET_CLAIM.search(finding.evidence) and not SECRET_CLAIM.search(
            finding.message
        ):
            return (None, "")

        changed = self.pull_request.file(finding.file)
        if changed is None or not changed.patch:
            return (None, "")

        added = list(_added_lines(changed))
        if not added:
            return (None, "")

        for _, text in added:
            for _label, pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    return ("VERIFIED", "")

        # No match is *not* a refutation. This rule keys off credential words in
        # the finding's prose, and prose is a terrible discriminator: a SQL
        # injection finding says "an attacker could read password hashes", which
        # mentions credentials without claiming any were committed. On a live
        # review that refuted the injection outright. Corroborate when the scan
        # agrees; stay silent when it does not.
        return (None, "")

    # -- workspace access -------------------------------------------------

    def _read(self, path: str) -> str | None:
        if self.source is None:
            return None
        try:
            content = self.source.read(path)
        except Exception:  # pragma: no cover - a provider is not trusted
            return None
        return content if isinstance(content, str) else None

    def _lines_changed(self, path: str) -> set[int]:
        if path not in self._changed_lines:
            changed = self.pull_request.file(path)
            self._changed_lines[path] = (
                {number for number, _ in _added_lines(changed)} if changed else set()
            )
        return self._changed_lines[path]


def verify(
    findings: Iterable[Finding],
    pull_request: PullRequest,
    source: Any | None = None,
) -> VerificationResult:
    """Convenience entry point over :class:`Verifier`."""
    return Verifier(pull_request, source).check(findings)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _quotes(evidence: str) -> list[str]:
    """Backticked spans in a finding's evidence."""
    return [str(match) for match in QUOTE.findall(evidence)]


def _normalize(text: str) -> str:
    """Strip whitespace entirely before comparing a quote against a file.

    Not merely collapsed: a reviewer quoting code will re-indent it, and may
    also reformat it — `find( x )` for `find(x)`. Since the only decision this
    comparison drives is *refutation*, it must err toward leaving a finding
    alone: ignoring whitespace altogether risks missing a fabricated quote,
    while merely collapsing it risks discarding a real finding because someone
    added a space. The first mistake is recoverable by the judge; the second
    deletes a true problem silently.
    """
    return re.sub(r"\s+", "", text)


def _added_lines(changed: ChangedFile | None) -> Iterable[tuple[int, str]]:
    """Added lines of a patch with their new-side numbers."""
    return added_lines(changed.patch if changed is not None else None)
