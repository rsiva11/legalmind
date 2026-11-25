import asyncio
import logging

import pytest
from pydantic import BaseModel

from app.llm import structured_output as so
from app.llm.groq_client import LLMResult


class Facts(BaseModel):
    days: int


def reply(text, finish="stop", completion=5):
    return LLMResult(text=text, model="fake", prompt_tokens=10, completion_tokens=completion,
                     latency_ms=1, finish_reason=finish)


def script(monkeypatch, replies):
    calls = []

    async def fake_chat(messages, **kwargs):
        calls.append((messages, kwargs))
        return replies[len(calls) - 1]

    monkeypatch.setattr(so, "chat", fake_chat)
    return calls


def run(**kw):
    return asyncio.run(so.chat_structured([{"role": "user", "content": "q"}], Facts, **kw))


def test_valid_reply_is_returned_in_one_call(monkeypatch):
    calls = script(monkeypatch, [reply('{"days": 30}')])
    assert run() == Facts(days=30) and len(calls) == 1
    assert calls[0][0][0]["role"] == "system" and "JSON Schema" in calls[0][0][0]["content"]


def test_json_inside_prose_and_fences_is_extracted(monkeypatch):
    script(monkeypatch, [reply('Sure! ```json\n{"days": 30}\n```')])
    assert run() == Facts(days=30)


def test_one_retry_with_the_correction(monkeypatch):
    calls = script(monkeypatch, [reply("no json here"), reply('{"days": 7}')])
    assert run() == Facts(days=7) and len(calls) == 2
    retry = calls[1][0]
    assert retry[-2] == {"role": "assistant", "content": "no json here"}
    assert retry[-1]["role"] == "user" and "That reply was invalid" in retry[-1]["content"]


def test_two_failures_raise_without_echoing_output(monkeypatch):
    calls = script(monkeypatch, [reply("nope"), reply("still nope")])
    with pytest.raises(so.StructuredOutputError) as exc:
        run()
    assert not isinstance(exc.value, so.OutputTruncated)
    assert "2 attempts" in str(exc.value) and "nope" not in str(exc.value) and len(calls) == 2


def test_truncated_reply_stops_immediately(monkeypatch):
    calls = script(monkeypatch, [reply("", finish="length", completion=3000)])
    with pytest.raises(so.OutputTruncated) as exc:
        run()
    assert isinstance(exc.value, so.StructuredOutputError) and len(calls) == 1

    script(monkeypatch, [reply('{"days": 4}', finish="length")])  # complete JSON at the limit is still accepted
    assert run() == Facts(days=4)


def test_diagnostics_are_logged_without_values(monkeypatch, caplog):
    script(monkeypatch, [reply('{"days": "SECRET-TEXT"}'), reply('{"days": 3}')])
    with caplog.at_level(logging.WARNING, logger="app.llm.structured_output"):
        assert run() == Facts(days=3)
    assert "attempt 1/2" in caplog.text and "ValidationError" in caplog.text
    assert "finish_reason=stop" in caplog.text and "reply_chars=" in caplog.text
    assert "days:int_parsing" in caplog.text and "SECRET-TEXT" not in caplog.text


def test_chat_options_are_passed_through(monkeypatch):
    calls = script(monkeypatch, [reply('{"days": 1}')])
    run(max_tokens=123, temperature=0.0)
    assert calls[0][1] == {"max_tokens": 123, "temperature": 0.0}