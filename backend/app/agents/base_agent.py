import logging
import time
from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, SerializeAsAny

from app.llm import token_manager
from app.llm.groq_client import LLMResult, chat
from app.llm.structured_output import chat_structured

logger = logging.getLogger(__name__)

InT = TypeVar("InT", bound=BaseModel)
OutT = TypeVar("OutT", bound=BaseModel)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class AgentUsage(BaseModel):
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class AgentResult(BaseModel):
    agent: str
    status: Literal["ok", "failed"]
    output: SerializeAsAny[BaseModel] | None = None
    error_type: str | None = None
    duration_ms: int
    usage: AgentUsage


class BaseAgent(ABC, Generic[InT, OutT]):
    name: str
    description: str
    input_schema: type[InT]
    output_schema: type[OutT]

    @abstractmethod
    async def execute(self, data: InT) -> OutT:
        """The agent's own logic. Subclasses implement this."""

    async def run(self, data: InT) -> AgentResult:
        """Standard entry point: validate, execute, validate, record metadata."""
        start = time.perf_counter()
        before = self._usage_snapshot()
        output: OutT | None = None
        error_type: str | None = None

        try:
            if not isinstance(data, self.input_schema):
                raise TypeError(f"{self.name} expects {self.input_schema.__name__}")
            output = await self.execute(data)
            if not isinstance(output, self.output_schema):
                raise TypeError(f"{self.name} must return {self.output_schema.__name__}")
        except Exception as exc:
            output = None
            error_type = type(exc).__name__
            # Messages can echo contract text, so log the type here and details at DEBUG only
            logger.error("Agent %s failed: %s", self.name, error_type)
            logger.debug("Agent %s failure details", self.name, exc_info=True)

        duration_ms = int((time.perf_counter() - start) * 1000)
        usage = self._usage_since(before)
        logger.info(
            "Agent %s status=%s duration_ms=%d llm_calls=%d tokens=%d",
            self.name,
            "failed" if error_type else "ok",
            duration_ms,
            usage.llm_calls,
            usage.prompt_tokens + usage.completion_tokens,
        )
        return AgentResult(
            agent=self.name,
            status="failed" if error_type else "ok",
            output=output,
            error_type=error_type,
            duration_ms=duration_ms,
            usage=usage,
        )

    async def ask(self, messages: list[dict[str, str]], **kwargs) -> LLMResult:
        return await chat(messages, **kwargs)

    async def ask_structured(
        self, messages: list[dict[str, str]], schema: type[SchemaT], **kwargs
    ) -> SchemaT:
        return await chat_structured(messages, schema, **kwargs)

    @staticmethod
    def _usage_snapshot() -> tuple[int, int, int]:
        t = token_manager.get_tracker()
        return (t.calls, t.prompt_tokens, t.completion_tokens) if t else (0, 0, 0)

    def _usage_since(self, before: tuple[int, int, int]) -> AgentUsage:
        calls, prompt, completion = self._usage_snapshot()
        return AgentUsage(
            llm_calls=calls - before[0],
            prompt_tokens=prompt - before[1],
            completion_tokens=completion - before[2],
        )