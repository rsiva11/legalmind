import asyncio

import pytest

from app.llm.rate_limiter import RequestTooLarge, TokenRateLimiter


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(round(seconds, 3))
        self.now += seconds
        await asyncio.sleep(0)


def make(limit=1000):
    ft = FakeTime()
    return ft, TokenRateLimiter(limit, clock=ft.clock, sleep=ft.sleep)


def test_requests_within_the_limit_do_not_wait():
    async def main():
        ft, lim = make()
        await lim.acquire(400)
        await lim.acquire(400)
        return ft, lim

    ft, lim = asyncio.run(main())
    assert ft.sleeps == [] and lim.used() == 800


def test_waits_until_the_oldest_reservation_expires():
    async def main():
        ft, lim = make()
        await lim.acquire(600)
        ft.now = 10.0
        await lim.acquire(600)  # 600 + 600 > 1000: must wait for the first to leave the 60s window
        return ft, lim

    ft, lim = asyncio.run(main())
    assert ft.sleeps == [50.0] and lim.used() == 600


def test_reconcile_frees_the_unused_reservation():
    async def main():
        ft, lim = make()
        ticket = await lim.acquire(800)
        lim.reconcile(ticket, 100)  # the call really used 100
        await lim.acquire(800)
        return ft, lim

    ft, lim = asyncio.run(main())
    assert ft.sleeps == [] and lim.used() == 900


def test_a_request_that_can_never_fit_fails_immediately():
    async def main():
        ft, lim = make()
        with pytest.raises(RequestTooLarge, match=r"1001.*1000"):
            await lim.acquire(1001)
        return ft, lim

    ft, lim = asyncio.run(main())
    assert ft.sleeps == [] and lim.used() == 0


def test_limit_zero_disables_the_limiter():
    async def main():
        ft, lim = make(limit=0)
        ticket = await lim.acquire(10**9)
        lim.reconcile(ticket, 5)
        return ft, lim

    ft, lim = asyncio.run(main())
    assert ft.sleeps == [] and lim.used() == 0


def test_waiters_are_served_in_arrival_order():
    async def main():
        ft, lim = make()
        await lim.acquire(600)
        order = []

        async def waiter(name, tokens):
            await lim.acquire(tokens)
            order.append(name)

        await asyncio.gather(waiter("A", 600), waiter("B", 300))
        return ft, order

    ft, order = asyncio.run(main())
    assert order == ["A", "B"] and ft.sleeps == [60.0]