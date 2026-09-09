"""The security review agent: vulnerabilities and unsafe handling.

Summoned by signal, not by level (see :mod:`reviewer.agents.selection`): a change
that touches authentication, cryptography, payments, credential-bearing config,
or that adds a line matching a credential pattern. A change with no security
signal does not pay for this agent.

The deterministic secret scan in stage 3 has already run, and its findings are
enforced in code (§2.6). This agent is for what a pattern cannot see: a check in
the wrong order, a boundary that trusts its input, an authorization test that
passes for the wrong reason.
"""

from __future__ import annotations

from typing import ClassVar

from .base import ReviewAgent

__all__ = ["SecurityAgent"]


SECURITY_SIGNALS = frozenset(
    {
        "auth",
        "crypto",
        "secret_material",
        "config_secrets_file",
        "payments",
        "request_surface",
    }
)
"""Signals that summon the security agent.

`request_surface` is here because of a live miss: a pull request adding a route
with a SQL injection fired only `public_api`, which is an architecture signal, so
the security agent was never summoned on the one file that needed it. Route,
handler and controller directories are precisely where injection and missing
authorization live, so they now raise a security signal of their own.

Each is a place where a deterministic rule found *something security-shaped* but
cannot tell whether it is wrong. `secret_material` is included even though stage
3 already enforces it in code: a leaked credential usually arrives with other
mishandling around it, and that is what an agent can see.
"""


class SecurityAgent(ReviewAgent):
    """Vulnerabilities: authz flaws, unsafe data handling, injection, exposure."""

    name: ClassVar[str] = "security-agent"
    triggers: ClassVar[frozenset[str]] = SECURITY_SIGNALS
    concern: ClassVar[str] = "security vulnerabilities"
    categories: ClassVar[tuple[str, ...]] = ("security", "dependency")
    prompt: ClassVar[str] = """
You are a security reviewer on a pull request. You find exploitable weaknesses
and show how they would be reached.

Look for:
  * authentication or authorization that can be bypassed: a check that runs
    after the effect, a check on the wrong subject, an admin path that trusts a
    client-supplied field, a default that fails open;
  * missing authorization entirely — a new endpoint, handler or command that
    performs a privileged action with no check at all;
  * untrusted input reaching a dangerous sink: SQL or command construction by
    concatenation, path traversal, deserialization, template or expression
    evaluation, a redirect built from a request parameter;
  * secrets exposed: credentials in code or config, tokens written to logs or
    error messages, keys committed to the repository;
  * weakened cryptography: a hardcoded key or IV, a broken or fast hash for
    passwords, verification disabled, randomness that is not cryptographic;
  * a dependency change that pulls in a package with known problems, or that
    relaxes a version constraint on something security-relevant;
  * data crossing a trust boundary without validation, or leaving one without
    redaction.

How to work:
  * for each candidate, establish the path an attacker controls. Read the
    caller, the route, the handler. State what an attacker supplies and what
    they get.
  * check whether an existing control already stops it before you report it —
    grep for the guard, do not assume its absence.
  * a missing check is only a finding if you can show the privileged action it
    fails to protect.

Do NOT report: theoretical weaknesses with no reachable path, defence-in-depth
suggestions, or dependency updates with no security dimension.
"""
