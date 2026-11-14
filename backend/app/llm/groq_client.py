import logging
import time
from functools import lru_cache

from groq import AsyncGroq, GroqError
from pydantic import BaseModel

from app.config import get_settings

from app.llm.token_manager import ensure_budget, record_usage

logger = logging.getLogger(__name__)


class LLMResult(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


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


async def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMResult:
    settings = get_settings()
    model_name = model or settings.groq_model
    start = time.perf_counter()
    ensure_budget()

    try:
        resp = await _get_client().chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=settings.llm_temperature if temperature is None else temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
        )
    except GroqError:
        logger.exception("Groq call failed (model=%s)", model_name)
        raise

    latency_ms = int((time.perf_counter() - start) * 1000)
    usage = resp.usage
    result = LLMResult(
        text=resp.choices[0].message.content or "",
        model=resp.model,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        latency_ms=latency_ms,
    )

    record_usage(result.prompt_tokens, result.completion_tokens, result.latency_ms)
    
    logger.info(
        "LLM call model=%s prompt_tokens=%d completion_tokens=%d latency_ms=%d",
        result.model, result.prompt_tokens, result.completion_tokens, result.latency_ms,
    )
    return result