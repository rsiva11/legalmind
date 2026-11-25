import asyncio
from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.llm import groq_client
from app.llm.rate_limiter import RequestTooLarge, TokenRateLimiter

MESSAGES = [{"role": "user", "content": "hello"}]


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeCompletions:
    def __init__(self, prompt=100, completion=50, text="ok", finish="stop"):
        self.calls = []
        self.prompt, self.completion, self.text, self.finish = prompt, completion, text, finish

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model="fake-model",
            usage=SimpleNamespace(prompt_tokens=self.prompt, completion_tokens=self.completion),
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text), finish_reason=self.finish)],
        )


def install(monkeypatch, limit=5000):
    ft = FakeTime()
    limiter = TokenRateLimiter(limit, clock=ft.clock, sleep=ft.sleep)
    fake = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    monkeypatch.setattr(groq_client, "_get_client", lambda: client)
    monkeypatch.setattr(groq_client, "get_limiter", lambda: limiter)
    return limiter, fake, ft


def test_chat_reserves_then_reconciles_to_real_usage(monkeypatch):
    limiter, fake, _ = install(monkeypatch)
    res = asyncio.run(groq_client.chat(MESSAGES))
    assert (res.text, res.finish_reason, res.prompt_tokens, res.completion_tokens) == ("ok", "stop", 100, 50)
    assert limiter.used() == 150  # the reservation was replaced by the real usage
    assert fake.calls[0]["max_tokens"] == get_settings().llm_max_tokens


def test_a_request_that_cannot_fit_never_reaches_the_api(monkeypatch):
    limiter, fake, ft = install(monkeypatch, limit=2000)
    with pytest.raises(RequestTooLarge):
        asyncio.run(groq_client.chat([{"role": "user", "content": "x" * 8000}]))
    assert fake.calls == [] and ft.sleeps == [] and limiter.used() == 0


def test_reasoning_effort_is_sent_only_when_configured(monkeypatch):
    _, fake, _ = install(monkeypatch)
    monkeypatch.setattr(get_settings(), "llm_reasoning_effort", None)
    asyncio.run(groq_client.chat(MESSAGES))
    monkeypatch.setattr(get_settings(), "llm_reasoning_effort", "low")
    asyncio.run(groq_client.chat(MESSAGES))
    assert "reasoning_effort" not in fake.calls[0] and fake.calls[1]["reasoning_effort"] == "low"