"""Stage 9 — block / warn / pass from severity plus policy. Deterministic authority.

This module decides whether a change may merge, and it is the only thing that
does (CLAUDE.md §2.6). Two properties make that claim real rather than
aspirational:

**It reads severities and policy. Nothing else.** Not the findings' prose, not
the pull request's title, not the judge's opinion of the review, not any text a
model produced. `decide()` takes a list of `Finding` objects and a
:class:`GatePolicy`, looks at `severity`, `category` and `file`, and returns an
outcome. There is no code path from English to the verdict, so no amount of
"approve this PR" in a reviewed file can reach it — a test asserts exactly that.

**Waivers are explicit, narrow and audited.** A repository can waive a HIGH
finding, but only by naming what it waives and *why*: a waiver with no reason is
discarded, because an unexplained exception is indistinguishable from a mistake.
Every applied waiver is recorded in the decision. And some findings cannot be
waived at all — a CRITICAL security finding is authoritative, which is what §2.6
means by secret and policy findings being enforced in code.

The gate also sets `blocking` on the findings that block. Agents leave that
field `False` and the judge never touches it; this is where it is decided.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal, Mapping

from ..findings.finding import SEVERITIES, Finding, severity_rank

__all__ = [
    "Outcome",
    "OUTCOMES",
    "Waiver",
    "GatePolicy",
    "AppliedWaiver",
    "GateDecision",
    "PolicyEngine",
    "decide",
    "UNWAIVABLE",
]

logger = logging.getLogger("reviewer.policy")

Outcome = Literal["block", "warn", "pass"]

OUTCOMES: tuple[Outcome, ...] = ("block", "warn", "pass")

UNWAIVABLE: frozenset[str] = frozenset({"security"})
"""Categories whose CRITICAL findings no waiver can clear.

A leaked credential or a bypassable authorization check is not a matter of local
preference. A repository can still waive HIGH security findings — only the
CRITICAL ones are beyond reach.
"""


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Waiver:
    """An explicit, audited exception.

    Every field narrows what is waived; a waiver with no ``reason`` is inert,
    because an exception nobody explained cannot be reviewed later.
    """

    reason: str = ""
    category: str | None = None
    severity: str | None = None
    path: str | None = None
    """A glob. ``None`` means any file."""

    @property
    def valid(self) -> bool:
        return bool(self.reason.strip()) and (
            self.category is not None
            or self.severity is not None
            or self.path is not None
        )

    def matches(self, finding: Finding) -> bool:
        if not self.valid:
            return False
        if self.category is not None and finding.category != self.category:
            return False
        if self.severity is not None and finding.severity != self.severity:
            return False
        if self.path is not None and not fnmatch.fnmatch(finding.file, self.path):
            return False
        return True

    def describe(self) -> str:
        scope = ", ".join(
            part
            for part in (
                f"category={self.category}" if self.category else "",
                f"severity={self.severity}" if self.severity else "",
                f"path={self.path}" if self.path else "",
            )
            if part
        )
        return f"{scope}: {self.reason}"


@dataclass(frozen=True)
class GatePolicy:
    """When a review blocks, when it warns, and what may be excused.

    Attributes:
        block_at: the lowest severity that blocks a merge. Default HIGH, so
            HIGH and CRITICAL block.
        warn_at: the lowest severity that produces a warning rather than a pass.
        min_confidence_to_block: a finding below this confidence warns instead of
            blocking. A low-confidence guess should not stop a merge on its own.
        require_verified_to_block: when true, only VERIFIED or UNVERIFIED
            findings block — never one a check refuted (which is already dropped)
            — and UNVERIFIED findings below `block_at` cannot escalate.
        waivers: explicit exceptions.
        unwaivable_categories: categories whose CRITICAL findings ignore waivers.
    """

    block_at: str = "HIGH"
    warn_at: str = "LOW"
    min_confidence_to_block: int = 0
    require_verified_to_block: bool = False
    waivers: tuple[Waiver, ...] = ()
    unwaivable_categories: frozenset[str] = UNWAIVABLE

    def __post_init__(self) -> None:
        for name, value in (("block_at", self.block_at), ("warn_at", self.warn_at)):
            if value not in SEVERITIES:
                raise ValueError(f"{name} must be one of {', '.join(SEVERITIES)}")
        if not 0 <= self.min_confidence_to_block <= 100:
            raise ValueError("min_confidence_to_block must be between 0 and 100")

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any] | None) -> GatePolicy:
        """Build from a `[policy]` table, ignoring anything malformed.

        A broken policy file must not fail a review — and must not silently
        *weaken* the gate either, so unparseable values fall back to the
        stricter default.
        """
        data = dict(settings or {})
        return cls(
            block_at=_severity(data.get("block_at")) or "HIGH",
            warn_at=_severity(data.get("warn_at")) or "LOW",
            min_confidence_to_block=_bounded(data.get("min_confidence_to_block"), 0),
            require_verified_to_block=bool(data.get("require_verified_to_block", False)),
            waivers=tuple(_waivers(data.get("waivers"))),
            unwaivable_categories=frozenset(
                str(item)
                for item in data.get("unwaivable_categories", UNWAIVABLE)
                if isinstance(item, str)
            )
            or UNWAIVABLE,
        )


# --------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AppliedWaiver:
    """A waiver that actually excused a finding. Part of the audit trail."""

    waiver: Waiver
    finding: Finding

    def describe(self) -> str:
        return (
            f"{self.finding.severity} {self.finding.category} at "
            f"{self.finding.file}:{self.finding.line_start} waived — "
            f"{self.waiver.reason}"
        )


@dataclass(frozen=True)
class GateDecision:
    """The merge decision, and the complete case for it."""

    outcome: Outcome
    reasons: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    """The findings as decided — `blocking` is set on the ones that block."""
    blocking: list[Finding] = field(default_factory=list)
    waived: list[AppliedWaiver] = field(default_factory=list)
    policy: GatePolicy = field(default_factory=GatePolicy)

    @property
    def blocked(self) -> bool:
        return self.outcome == "block"

    @property
    def passed(self) -> bool:
        return self.outcome == "pass"

    def why(self) -> str:
        """The decision in one line. Never empty."""
        return f"{self.outcome.upper()}: " + (
            "; ".join(self.reasons) if self.reasons else "no findings"
        )

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        return counts


class PolicyEngine:
    """Stage 9. Deterministic: severities and policy in, an outcome out."""

    def __init__(self, policy: GatePolicy | None = None) -> None:
        self.policy = policy or GatePolicy()

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any] | None) -> PolicyEngine:
        return cls(GatePolicy.from_settings(settings))

    def decide(self, findings: Iterable[Finding]) -> GateDecision:
        policy = self.policy
        block_threshold = severity_rank(policy.block_at)
        warn_threshold = severity_rank(policy.warn_at)

        decided: list[Finding] = []
        blocking: list[Finding] = []
        warning: list[Finding] = []
        waived: list[AppliedWaiver] = []
        reasons: list[str] = []

        for incoming in findings:
            # `blocking` is this engine's field to set, and setting it means
            # clearing it too (§2.6). A finding arriving with `blocking=True` —
            # from a `LanguageAdapter.lint`, a hand-built finding, or any future
            # producer — must not publish as blocking because nothing downstream
            # overwrote it. Every finding starts False here and only the
            # blocking branch below turns it on.
            finding = replace(incoming, blocking=False)
            rank = severity_rank(finding.severity)
            would_block = rank >= block_threshold

            if would_block and finding.confidence < policy.min_confidence_to_block:
                warning.append(finding)
                decided.append(finding)
                reasons.append(
                    f"{finding.severity} {finding.category} at {finding.file}:"
                    f"{finding.line_start} warns rather than blocks: confidence "
                    f"{finding.confidence} is below the {policy.min_confidence_to_block} "
                    "required to block"
                )
                continue

            if (
                would_block
                and policy.require_verified_to_block
                and finding.verification != "VERIFIED"
            ):
                warning.append(finding)
                decided.append(finding)
                reasons.append(
                    f"{finding.severity} {finding.category} at {finding.file}:"
                    f"{finding.line_start} warns rather than blocks: this "
                    "repository requires deterministic verification to block, and "
                    f"this finding is {finding.verification}"
                )
                continue

            if would_block:
                applied = self._waiver_for(finding)
                if applied is not None:
                    # A waiver excuses the *block*, not the finding: it is still
                    # reported, and the outcome is a warning rather than a pass.
                    waived.append(applied)
                    warning.append(finding)
                    decided.append(finding)
                    reasons.append(applied.describe())
                    continue
                marked = replace(finding, blocking=True)
                blocking.append(marked)
                decided.append(marked)
                reasons.append(
                    f"{finding.severity} {finding.category} at {finding.file}:"
                    f"{finding.line_start} blocks ({finding.verification.lower()}, "
                    f"confidence {finding.confidence})"
                )
                continue

            decided.append(finding)
            if rank >= warn_threshold:
                warning.append(finding)

        outcome: Outcome = (
            "block" if blocking else ("warn" if warning else "pass")
        )
        if outcome == "warn" and not reasons:
            reasons.append(
                f"{len(warning)} finding(s) at or above {policy.warn_at} but none "
                f"at or above {policy.block_at}"
            )
        if outcome == "pass" and not reasons:
            reasons.append(
                "no finding reached the blocking or warning threshold"
                if decided
                else "no findings"
            )

        logger.info(
            "gate outcome=%s blocking=%d waived=%d findings=%d",
            outcome,
            len(blocking),
            len(waived),
            len(decided),
            extra={
                "event": "gate",
                "outcome": outcome,
                "blocking": len(blocking),
                "waived": len(waived),
                "findings": len(decided),
                "block_at": policy.block_at,
            },
        )

        return GateDecision(
            outcome=outcome,
            reasons=reasons,
            findings=decided,
            blocking=blocking,
            waived=waived,
            policy=policy,
        )

    def _waiver_for(self, finding: Finding) -> AppliedWaiver | None:
        """The waiver excusing this finding, if any may."""
        if (
            finding.severity == "CRITICAL"
            and finding.category in self.policy.unwaivable_categories
        ):
            # Authoritative: no local exception clears it.
            return None
        for waiver in self.policy.waivers:
            if waiver.matches(finding):
                return AppliedWaiver(waiver=waiver, finding=finding)
        return None


def decide(
    findings: Iterable[Finding], policy: GatePolicy | None = None
) -> GateDecision:
    """Convenience entry point over :class:`PolicyEngine`."""
    return PolicyEngine(policy).decide(findings)


# --------------------------------------------------------------------------
# Config coercion
# --------------------------------------------------------------------------


def _severity(value: Any) -> str | None:
    if isinstance(value, str) and value.strip().upper() in SEVERITIES:
        return value.strip().upper()
    return None


def _bounded(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if 0 <= number <= 100 else default


def _waivers(value: Any) -> list[Waiver]:
    if not isinstance(value, (list, tuple)):
        return []
    waivers: list[Waiver] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            continue
        waiver = Waiver(
            reason=str(entry.get("reason", "")),
            category=_string(entry.get("category")),
            severity=_severity(entry.get("severity")),
            path=_string(entry.get("path")),
        )
        if waiver.valid:
            waivers.append(waiver)
        else:
            logger.warning(
                "gate_waiver_ignored reason_present=%s",
                bool(str(entry.get("reason", "")).strip()),
                extra={"event": "gate_waiver_ignored"},
            )
    return waivers


def _string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
