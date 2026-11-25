import logging
import time
from functools import lru_cache

from groq import AsyncGroq, GroqError
from pydantic import BaseModel

from app.config import get_settings
from app.llm.rate_limiter import TokenRateLimiter
from app.llm.token_manager import ensure_budget, estimate_tokens, record_usage

logger = logging.getLogger(__name__)


class LLMResult(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    finish_reason: str | None = None
    waited_ms: int = 0  # time spent waiting for the per-minute token budget


@lru_cache
def _get_client() -> AsyncGroq:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Check backend/.env")
    return AsyncGroq(
        api_key=settings.groq_api_key,
        timeout=settings.groq_timeout_seconds,
        max_retries=settings.groq_max_retries,
    )


@lru_cache
def get_limiter() -> TokenRateLimiter:
    s = get_settings()
    return TokenRateLimiter(int(s.groq_tpm_limit * s.groq_tpm_safety))


async def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMResult:
    settings = get_settings()
    model_name = model or settings.groq_model
    ensure_budget()

    cap = max_tokens or settings.llm_max_tokens
    reserved = sum(estimate_tokens(m["content"]) + 4 for m in messages) + cap  # prompt + output cap
    waited_from = time.perf_counter()
    ticket = await get_limiter().acquire(reserved)  # waits for room; RequestTooLarge if it can never fit
    waited_ms = int((time.perf_counter() - waited_from) * 1000)

    extra: dict = {}
    if settings.llm_reasoning_effort:  # only sent when configured
        extra["reasoning_effort"] = settings.llm_reasoning_effort

    start = time.perf_counter()
    try:
        resp = await _get_client().chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=settings.llm_temperature if temperature is None else temperature,
            max_tokens=cap,
            **extra,
        )
    except GroqError as exc:
        logger.error("Groq call failed (model=%s): %s", model_name, type(exc).__name__)
        raise

    latency_ms = int((time.perf_counter() - start) * 1000)
    usage = resp.usage
    result = LLMResult(
        text=resp.choices[0].message.content or "",
        model=resp.model,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        latency_ms=latency_ms,
        finish_reason=resp.choices[0].finish_reason,
        waited_ms=waited_ms,
    )
    if usage:
        get_limiter().reconcile(ticket, result.prompt_tokens + result.completion_tokens)
    record_usage(result.prompt_tokens, result.completion_tokens, result.latency_ms)
    logger.info(
        "LLM call model=%s prompt_tokens=%d completion_tokens=%d finish_reason=%s latency_ms=%d waited_ms=%d",
        result.model, result.prompt_tokens, result.completion_tokens, result.finish_reason,
        result.latency_ms, result.waited_ms,
    )
    return result