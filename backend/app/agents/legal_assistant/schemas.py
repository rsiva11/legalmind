from typing import Literal

from pydantic import BaseModel, Field

from app.agents.research.schemas import ResearchResult
from app.schemas.knowledge import KnowledgeCitation


class LegalQuestionInput(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    knowledge: ResearchResult | None = None


class LegalQuote(BaseModel):
    label: str = Field(pattern=r"^K\d+$")
    quote: str = Field(min_length=3, max_length=300)


class LegalAnswerDraft(BaseModel):
    """What the LLM produces. Code verifies it before anything is shown."""

    in_scope: bool = Field(description="True if the question is about contracts or general legal information")
    answered_from_knowledge: bool = Field(description="True only if the <knowledge> passages answer the question")
    answer: str = Field(max_length=1500, description="Short answer; every statement ends with a label like [K1]")
    key_points: list[str] = Field(default_factory=list, max_length=5)
    quotes: list[LegalQuote] = Field(
        default_factory=list, max_length=6, description="Exact words (at most 25) copied from the cited passages"
    )
    jurisdiction_dependent: bool
    seek_professional_advice: bool


class LegalAnswer(BaseModel):
    """The agent's result. status and source are set by code."""

    status: Literal["answered", "not_found", "unverified", "out_of_scope"] = "answered"
    in_scope: bool = True
    answer: str
    key_points: list[str] = []
    quotes: list[LegalQuote] = []
    citations: list[KnowledgeCitation] = []
    jurisdiction_dependent: bool = False
    seek_professional_advice: bool = False
    source: Literal["knowledge_base", "none"] = "knowledge_base"
    rejected_reason: str | None = None  # for reviewers and logs, not for end users
    evidence_labels: list[str] = []
    disclaimer: str