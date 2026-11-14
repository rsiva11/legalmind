import logging
from contextvars import ContextVar
from dataclasses import dataclass

from app.config import get_settings

logger = logging.getLogger(__name__)


class TokenBudgetExceeded(Exception):
    """The current request has used up its token budget."""


@dataclass
class UsageTracker:
    budget: int
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


_current: ContextVar[UsageTracker | None] = ContextVar("usage_tracker", default=None)


def start_tracking(budget: int | None = None) -> UsageTracker:
    """Begin tracking for the current request. Call once, at the start of a request."""
    tracker = UsageTracker(budget=budget or get_settings().request_token_budget)
    _current.set(tracker)
    return tracker


def get_tracker() -> UsageTracker | None:
    return _current.get()


def ensure_budget() -> None:
    """Raise before an LLM call if the request already used its budget."""
    tracker = _current.get()
    if tracker and tracker.total_tokens >= tracker.budget:
        logger.warning("Token budget exceeded: %d/%d", tracker.total_tokens, tracker.budget)
        raise TokenBudgetExceeded(f"Token budget of {tracker.budget} exceeded")


def record_usage(prompt_tokens: int, completion_tokens: int, latency_ms: int) -> None:
    tracker = _current.get()
    if tracker is None:
        return
    tracker.calls += 1
    tracker.prompt_tokens += prompt_tokens
    tracker.completion_tokens += completion_tokens
    tracker.latency_ms += latency_ms


def estimate_tokens(text: str) -> int:
    """Rough estimate (about 4 characters per token) for pre-flight size checks."""
    return len(text) // 4 + 1