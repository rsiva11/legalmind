from typing import Literal

from pydantic import BaseModel, Field


class LegalQuestionInput(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class LegalAnswerDraft(BaseModel):
    """What the LLM produces."""

    in_scope: bool = Field(
        description="True if the question is about contracts or general legal information"
    )
    answer: str = Field(description="Plain-language answer, or a short redirect if out of scope")
    key_points: list[str] = Field(
        default_factory=list, description="At most 5 short takeaways; empty if out of scope"
    )
    jurisdiction_dependent: bool = Field(
        description="True if the answer varies by country or state"
    )
    seek_professional_advice: bool = Field(
        description="True if the question concerns the user's own specific situation or a decision with legal consequences"
    )


class LegalAnswer(LegalAnswerDraft):
    """What the agent returns: the draft plus fields set by code."""

    source: Literal["general_knowledge"] = "general_knowledge"
    disclaimer: str