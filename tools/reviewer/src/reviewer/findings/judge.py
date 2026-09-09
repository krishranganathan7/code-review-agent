"""Stage 8 — an independent re-check of findings on a fresh context.

The judge is the only model call in the back half of the pipeline, and it is
deliberately the *last* opinion rather than the first: everything deterministic
has already run, so the judge is arguing against facts, not producing them.

**Fresh context is the whole design.** The judge never sees the producing
agent's transcript — not its system prompt, not its tool results, not its
reasoning. It is given only the finding's own fields and the cited code re-read
from the workspace. Three things follow:

* content that talked its way past one agent gets a second, uncontaminated
  reading, so a single-pass manipulation has to work twice on two different
  prompts;
* the judge cannot inherit an agent's mistaken premise, because it cannot see
  the premise;
* a finding stands or falls on what would actually be published about it.

**The judge can only lower.** Severity may be downgraded, never raised;
confidence may be reduced, never increased; a finding may be rejected outright.
Raising is refused in code rather than by prompt, because "the evidence supports
more than the agent claimed" is not a judgement a fresh reader is positioned to
make — it has less context than the agent did, by construction.

**The judge has no authority over the gate** (§2.6). It never touches
`blocking`, and its verdicts feed the same deterministic policy engine as
everything else. A judge that approved of a CRITICAL finding cannot make it
merge, and one that disliked it cannot either.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from ..agent.context import untrusted
from ..findings.finding import SEVERITIES, Finding, severity_rank
from ..providers.base import LLMProvider, ProviderError
from ..types import Message

__all__ = [
    "Judge",
    "JudgeResult",
    "JudgeVerdict",
    "RejectedFinding",
    "JUDGE_SYSTEM_PROMPT",
    "VERDICTS",
    "CODE_WINDOW",
]

logger = logging.getLogger("reviewer.findings.judge")

VERDICTS = ("keep", "downgrade", "reject")

CODE_WINDOW = 12
"""Lines of context either side of a finding, re-read for the judge."""

MAX_JUDGE_TOKENS = 4096

JUDGE_SYSTEM_PROMPT = """
You are an independent reviewer of code-review findings. You did not produce
these findings and you cannot see how they were produced — only the claim, the
evidence offered, and the code it points at.

For each finding, decide:

  * "keep" — the finding is correct, the severity is right, and a maintainer
    could act on it.
  * "downgrade" — the finding is real but overstated. Give the severity it
    deserves, and lower the confidence.
  * "reject" — the finding is incorrect, unsupported by the code shown, or not
    actionable (a restatement of what the change does, a style preference, a
    hypothetical with no path to it).

Judge only what you are shown. If the code does not support the claim, reject
it — do not assume there is context you are missing that would justify it.

You may lower a severity or a confidence. You may not raise either: you have
less context than the reviewer who filed the finding, so you are not positioned
to argue a problem is worse than claimed. Attempts to raise are ignored.

Reply with a single JSON object and nothing else:

{
  "verdicts": [
    {
      "id": <the finding's id, as given>,
      "verdict": "keep" | "downgrade" | "reject",
      "severity": "LOW | MEDIUM | HIGH | CRITICAL",
      "confidence": <0-100>,
      "reason": "<one sentence: why>"
    }
  ]
}

Every finding you were given must appear exactly once. A finding you omit is
kept unchanged.
"""


@dataclass(frozen=True)
class JudgeVerdict:
    """What the judge decided about one finding."""

    finding_id: int
    verdict: str
    reason: str = ""
    severity: str | None = None
    confidence: int | None = None

    @property
    def rejected(self) -> bool:
        return self.verdict == "reject"


@dataclass(frozen=True)
class RejectedFinding:
    """A finding the judge threw out, and why."""

    finding: Finding
    reason: str

    def __str__(self) -> str:
        return f"{self.finding.file}:{self.finding.line_start} — {self.reason}"


@dataclass(frozen=True)
class JudgeResult:
    """Surviving findings after the judge, and the record of what it changed."""

    findings: list[Finding] = field(default_factory=list)
    rejected: list[RejectedFinding] = field(default_factory=list)
    downgraded: list[str] = field(default_factory=list)
    verdicts: list[JudgeVerdict] = field(default_factory=list)
    skipped: str | None = None
    """Set when the judge could not run, and the findings passed through."""

    @property
    def rejection_reasons(self) -> list[str]:
        return [item.reason for item in self.rejected]

    def counts(self) -> dict[str, int]:
        return {
            "kept": len(self.findings),
            "rejected": len(self.rejected),
            "downgraded": len(self.downgraded),
        }


class Judge:
    """Stage 8. One model call, on a context built only from the findings."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        source: Any | None = None,
        max_tokens: int = MAX_JUDGE_TOKENS,
        code_window: int = CODE_WINDOW,
    ) -> None:
        self.provider = provider
        self.source = source
        self.max_tokens = max_tokens
        self.code_window = code_window

    def review(self, findings: Iterable[Finding]) -> JudgeResult:
        candidates = list(findings)
        if not candidates:
            return JudgeResult()

        messages = self.build_context(candidates)

        try:
            response = self.provider.complete(
                messages, tools=None, max_tokens=self.max_tokens
            )
        except ProviderError as exc:
            # A judge that cannot run must not delete a review. Findings pass
            # through unchanged, and the failure is recorded.
            logger.warning(
                "judge_unavailable error=%s",
                exc,
                extra={"event": "judge_unavailable", "error": str(exc)},
            )
            return JudgeResult(
                findings=list(candidates), skipped=f"judge unavailable: {exc}"
            )

        verdicts = _parse_verdicts(response.text)
        return self._apply(candidates, verdicts)

    # -- the fresh context ------------------------------------------------

    def build_context(self, findings: Sequence[Finding]) -> list[Message]:
        """Build the judge's conversation. Deliberately, this is all of it.

        Nothing from the producing agent's run appears here: no transcript, no
        tool results, no system prompt, no agent name. Only the finding's own
        published fields, plus the cited code re-read from the workspace — which
        is a fact, not the agent's account of a fact.

        The finding's own prose is model-authored and quotes repository text, so
        it passes through the same `untrusted()` boundary as everything else: the
        judge is a second reader, and a claim that talked its way past the first
        one must not get a free pass here.
        """
        blocks: list[str] = []
        for index, finding in enumerate(findings, start=1):
            # The trusted half: fields this pipeline computed.
            block = [
                f"--- finding {index} ---",
                f"category: {finding.category}",
                f"claimed severity: {finding.severity}",
                f"claimed confidence: {finding.confidence}",
                f"location: {finding.file}:{finding.line_start}-{finding.line_end}",
                f"verification: {finding.verification}",
                "claim:",
                # The untrusted half: a reviewer's words, quoting repository text.
                untrusted(finding.message, label="reviewer claim"),
                "evidence offered:",
                untrusted(finding.evidence, label="reviewer evidence"),
            ]
            code = self._cited_code(finding)
            if code:
                block += [
                    "code at that location:",
                    untrusted(code, label="repository code"),
                ]
            else:
                block.append(
                    "code at that location: unavailable — judge the claim on its "
                    "evidence alone, and reject it if the evidence does not stand up."
                )
            blocks.append("\n".join(block))

        task = (
            f"Judge the following {len(findings)} finding(s). "
            "Return one verdict for each, by id.\n\n" + "\n\n".join(blocks)
        )
        return [
            Message(role="system", content=JUDGE_SYSTEM_PROMPT.strip()),
            Message(role="user", content=task),
        ]

    def _cited_code(self, finding: Finding) -> str:
        """The lines a finding points at, numbered, read from the workspace."""
        if self.source is None:
            return ""
        try:
            content = self.source.read(finding.file)
        except Exception:  # pragma: no cover - a provider is not trusted
            return ""
        if not isinstance(content, str):
            return ""

        lines = content.splitlines()
        start = max(finding.line_start - self.code_window, 1)
        end = min(finding.line_end + self.code_window, len(lines))
        if start > len(lines):
            return ""
        return "\n".join(
            f"{number:>5}  {lines[number - 1]}" for number in range(start, end + 1)
        )

    # -- applying verdicts ------------------------------------------------

    def _apply(
        self, findings: Sequence[Finding], verdicts: Mapping[int, JudgeVerdict]
    ) -> JudgeResult:
        kept: list[Finding] = []
        rejected: list[RejectedFinding] = []
        downgraded: list[str] = []

        for index, finding in enumerate(findings, start=1):
            verdict = verdicts.get(index)
            if verdict is None:
                # Silence is not rejection: an omitted finding is kept as filed.
                kept.append(finding)
                continue

            if verdict.rejected:
                rejected.append(
                    RejectedFinding(
                        finding=finding,
                        reason=verdict.reason
                        or "the judge rejected this finding without giving a reason",
                    )
                )
                continue

            adjusted = self._lower_only(finding, verdict)
            if adjusted is not finding:
                downgraded.append(
                    f"{finding.file}:{finding.line_start} "
                    f"{finding.severity}->{adjusted.severity} "
                    f"({verdict.reason or 'no reason given'})"
                )
            kept.append(adjusted)

        logger.info(
            "judge kept=%d rejected=%d downgraded=%d",
            len(kept),
            len(rejected),
            len(downgraded),
            extra={
                "event": "judge",
                "kept": len(kept),
                "rejected": len(rejected),
                "downgraded": len(downgraded),
            },
        )
        return JudgeResult(
            findings=kept,
            rejected=rejected,
            downgraded=downgraded,
            verdicts=list(verdicts.values()),
        )

    @staticmethod
    def _lower_only(finding: Finding, verdict: JudgeVerdict) -> Finding:
        """Apply a verdict, discarding any attempt to raise severity or confidence."""
        severity = finding.severity
        if (
            verdict.severity is not None
            and verdict.severity in SEVERITIES
            and severity_rank(verdict.severity) < severity_rank(severity)
        ):
            severity = verdict.severity

        confidence = finding.confidence
        if verdict.confidence is not None and verdict.confidence < confidence:
            confidence = max(0, verdict.confidence)

        if severity == finding.severity and confidence == finding.confidence:
            return finding
        return replace(finding, severity=severity, confidence=confidence)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)


def _parse_verdicts(text: str) -> dict[int, JudgeVerdict]:
    """Read verdicts out of the judge's reply, tolerating shape and prose.

    A judge whose answer cannot be parsed changes nothing: every finding is
    kept as filed. Failing open here is the safe direction — the judge can only
    remove findings, so an unparseable judge must not silently remove them all.
    """
    payload = _load(text)
    if payload is None:
        logger.warning(
            "judge_unparseable chars=%d",
            len(text or ""),
            extra={"event": "judge_unparseable"},
        )
        return {}

    entries: Sequence[Any] = ()
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, Mapping):
        for key in ("verdicts", "findings", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                entries = value
                break

    verdicts: dict[int, JudgeVerdict] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        finding_id = _int(entry.get("id"))
        if finding_id is None:
            continue
        verdict = str(entry.get("verdict", "")).strip().lower()
        if verdict not in VERDICTS:
            continue
        verdicts[finding_id] = JudgeVerdict(
            finding_id=finding_id,
            verdict=verdict,
            reason=str(entry.get("reason", "")).strip(),
            severity=(
                str(entry["severity"]).strip().upper()
                if isinstance(entry.get("severity"), str)
                else None
            ),
            confidence=_int(entry.get("confidence")),
        )
    return verdicts


def _load(text: str) -> Any | None:
    if not text or not text.strip():
        return None
    # The whole reply first: a well-formed answer *is* the JSON, and slicing
    # from the first "{" would find the inner object of a bare list instead.
    candidates = [text.strip()]
    candidates += [match.group(1).strip() for match in _FENCE.finditer(text)]
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None
