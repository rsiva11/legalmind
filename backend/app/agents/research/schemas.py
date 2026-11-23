from typing import Literal

from pydantic import BaseModel, Field

from app.guardrails.result import GuardrailFinding
from app.guardrails.tool_guardrail import ToolCallRecord
from app.schemas.knowledge import KnowledgePassage, SourceRef


class ResearchInput(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class ResearchResult(BaseModel):
    """Stored with the job, so it never contains the user's question."""

    status: Literal["found", "not_found", "unavailable"]
    unavailable_reason: str | None = None
    passages: list[KnowledgePassage] = []
    sources: list[SourceRef] = []
    context: str = ""
    truncated: bool = False
    withheld_count: int = 0
    dropped_count: int = 0
    tool_calls: list[ToolCallRecord] = []
    findings: list[GuardrailFinding] = []