import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable


class RequestTooLarge(Exception):
    """One request needs more tokens than the per-minute limit allows, so waiting cannot help."""


class TokenRateLimiter:
    """Sliding-window token limiter. Waits for room instead of letting the provider answer 429."""

    def __init__(
        self,
        limit: int,
        *,
        window: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.limit = limit
        self.window = window
        self._clock = clock
        self._sleep = sleep
        self._events: deque[list[float]] = deque()  # [timestamp, tokens]
        self._lock = asyncio.Lock()

    def used(self) -> int:
        now = self._clock()
        while self._events and now - self._events[0][0] >= self.window:
            self._events.popleft()
        return int(sum(e[1] for e in self._events))

    async def acquire(self, tokens: int) -> list[float]:
        """Reserve `tokens`, waiting if the last minute is too full. Returns a ticket for reconcile()."""
        if self.limit <= 0:
            return [0.0, 0.0]
        if tokens > self.limit:
            raise RequestTooLarge(f"Request needs about {tokens} tokens but the per-minute limit is {self.limit}")
        async with self._lock:  # one waiter at a time, in arrival order
            while True:
                if self.used() + tokens <= self.limit:
                    ticket = [self._clock(), float(tokens)]
                    self._events.append(ticket)
                    return ticket
                wait = self.window - (self._clock() - self._events[0][0])
                await self._sleep(max(wait, 0.05))

    def reconcile(self, ticket: list[float], actual_tokens: int) -> None:
        """Replace the reservation with what the call really used."""
        if self.limit > 0:
            ticket[1] = float(actual_tokens)