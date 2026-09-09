"""Stage 10: publishing evidence-backed findings (GitHub output).

The artifact is the findings, not a score. Untrusted content — a model's words,
repository text — is escaped here, at the boundary where it meets a system that
interprets markup.
"""

from __future__ import annotations

from .github import (
    GitHubPublisher,
    InlineComment,
    PublishError,
    RecordingTransport,
    RenderedReview,
    ReviewRenderer,
    ReviewTransport,
    UrllibReviewTransport,
    escape_untrusted,
)

__all__ = [
    "GitHubPublisher",
    "InlineComment",
    "PublishError",
    "RecordingTransport",
    "RenderedReview",
    "ReviewRenderer",
    "ReviewTransport",
    "UrllibReviewTransport",
    "escape_untrusted",
]
