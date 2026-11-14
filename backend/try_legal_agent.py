import asyncio

from app.agents.legal_assistant.agent import LegalAssistantAgent
from app.agents.legal_assistant.schemas import LegalQuestionInput
from app.llm.token_manager import start_tracking

QUESTIONS = [
    "What is an NDA?",
    "My landlord kept my deposit. Should I sue him?",
    "What is agreement",
]


async def main():
    start_tracking()
    agent = LegalAssistantAgent()
    for q in QUESTIONS:
        res = await agent.run(LegalQuestionInput(question=q))
        print("Q:", q)
        print("  status:", res.status, "| tokens:", res.usage.prompt_tokens + res.usage.completion_tokens)
        if res.output:
            o = res.output
            print("  in_scope:", o.in_scope, "| jurisdiction_dependent:", o.jurisdiction_dependent,
                  "| seek_professional_advice:", o.seek_professional_advice)
            print("  answer:", o.answer[:160])
            print("  key_points:", len(o.key_points), "| source:", o.source)
        print()
    print("disclaimer:", res.output.disclaimer if res.output else None)


asyncio.run(main())