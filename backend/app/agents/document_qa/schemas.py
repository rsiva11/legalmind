from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.document import BuiltContext, Citation


class DocumentQuestionInput(BaseModel):
    document_id: str
    question: str = Field(min_length=3, max_length=2000)
    context: BuiltContext


class DocumentAnswer(BaseModel):
    document_id: str
    status: Literal["answered", "not_found", "unverified"]
    answer: str
    citations: list[Citation]
    rejected_reason: str | None = None  # for reviewers and logs, not for end users
    evidence_labels: list[str]
    context_truncated: bool