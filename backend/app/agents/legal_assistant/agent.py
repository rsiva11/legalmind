from app.agents.base_agent import BaseAgent
from app.agents.legal_assistant.prompts import SYSTEM_PROMPT
from app.agents.legal_assistant.schemas import (
    LegalAnswer,
    LegalAnswerDraft,
    LegalQuestionInput,
)

DISCLAIMER = (
    "This is general legal information, not legal advice. "
    "For decisions about your own situation, consult a qualified lawyer."
)


class LegalAssistantAgent(BaseAgent[LegalQuestionInput, LegalAnswer]):
    name = "legal_assistant"
    description = "Answers general legal-information questions in plain language"
    input_schema = LegalQuestionInput
    output_schema = LegalAnswer

    async def execute(self, data: LegalQuestionInput) -> LegalAnswer:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<question>\n{data.question}\n</question>"},
        ]
        draft = await self.ask_structured(messages, LegalAnswerDraft)
        return LegalAnswer(**draft.model_dump(), disclaimer=DISCLAIMER)