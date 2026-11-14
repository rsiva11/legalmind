import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.groq_client import chat

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """The model never produced valid JSON for the requested schema."""


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in the reply")
    return json.loads(text[start : end + 1])


def _with_schema_instruction(messages: list[dict[str, str]], schema: type[BaseModel]) -> list[dict[str, str]]:
    instruction = (
        "Respond with ONE JSON object and nothing else: no prose, no markdown fences. "
        "It must validate against this JSON Schema:\n"
        f"{json.dumps(schema.model_json_schema())}"
    )
    out = [dict(m) for m in messages]
    if out and out[0]["role"] == "system":
        out[0]["content"] += "\n\n" + instruction
    else:
        out.insert(0, {"role": "system", "content": instruction})
    return out


async def chat_structured(
    messages: list[dict[str, str]],
    schema: type[T],
    *,
    max_attempts: int = 2,
    **chat_kwargs,
) -> T:
    convo = _with_schema_instruction(messages, schema)
    last_error = "unknown"

    for attempt in range(1, max_attempts + 1):
        result = await chat(convo, **chat_kwargs)
        try:
            return schema.model_validate(_extract_json(result.text))
        except (ValueError, ValidationError) as exc:  # JSONDecodeError is a ValueError
            last_error = str(exc)[:500]
            logger.warning("Structured output invalid (attempt %d/%d): %s", attempt, max_attempts, type(exc).__name__)
            convo = convo + [
                {"role": "assistant", "content": result.text},
                {
                    "role": "user",
                    "content": f"That reply was invalid: {last_error}\nReturn only the corrected JSON object.",
                },
            ]

    raise StructuredOutputError(f"No valid output after {max_attempts} attempts: {last_error}")