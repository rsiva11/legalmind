import asyncio

from factories import make_context
from app.agents.document_qa.agent import UNVERIFIED_MESSAGE, DocumentQAAgent
from app.agents.document_qa.schemas import DocumentQuestionInput
from app.llm.groq_client import LLMResult
from app.llm.prompt_manager import NOT_FOUND_MESSAGE
from app.schemas.document import BuiltContext

CTX = make_context()


class FakeAgent(DocumentQAAgent):
    """Replaces the model with a canned reply. Counts calls."""

    def __init__(self, reply=""):
        self.reply, self.calls = reply, 0

    async def ask(self, messages, **kwargs):
        self.calls += 1
        return LLMResult(text=self.reply, model="fake", prompt_tokens=0, completion_tokens=0, latency_ms=0)


def ask(reply, ctx=CTX):
    agent = FakeAgent(reply)
    res = asyncio.run(
        agent.run(DocumentQuestionInput(document_id="d", question="What is the notice period?", context=ctx))
    )
    assert res.status == "ok", res.error_type
    return res.output, agent.calls


def test_answered_with_valid_citation():
    text = "Either party may terminate on thirty (30) days written notice. [E2]"
    out, calls = ask(text)
    assert out.status == "answered" and out.answer == text and calls == 1
    assert [(c.label, c.page_number) for c in out.citations] == [("E2", 2)]


def test_not_found():
    out, _ = ask(NOT_FOUND_MESSAGE)
    assert (out.status, out.answer, out.citations) == ("not_found", NOT_FOUND_MESSAGE, [])


def test_unknown_label_is_unverified():
    out, _ = ask("Notice is 30 days [E2]. See also [E7].")
    assert out.status == "unverified" and out.answer == UNVERIFIED_MESSAGE
    assert "E7" in out.rejected_reason


def test_no_citation_is_unverified():
    out, _ = ask("The notice period is thirty days.")
    assert out.status == "unverified" and out.rejected_reason == "answer has no citations"


def test_number_not_in_evidence_is_unverified():
    out, _ = ask("The notice period is sixty (60) days [E2].")
    assert out.status == "unverified" and "60" in out.rejected_reason


def test_number_from_the_wrong_chunk_is_unverified():
    out, _ = ask("Fees are due within 45 days [E2].")  # 45 is in E3, not E2
    assert out.status == "unverified" and "45" in out.rejected_reason


def test_mixed_not_found_and_claim_is_unverified():
    out, _ = ask(NOT_FOUND_MESSAGE + " Also, fees are due in 45 days [E3].")
    assert out.status == "unverified"


def test_empty_reply_is_unverified():
    out, _ = ask("   ")
    assert out.status == "unverified" and out.rejected_reason == "empty answer"


def test_no_evidence_skips_the_llm():
    out, calls = ask("should never be used", BuiltContext(text="", evidence=[], truncated=False))
    assert out.status == "not_found" and calls == 0