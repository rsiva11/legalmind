import asyncio

from knowledge_fixture import passage, research_with
from app.agents.legal_assistant.agent import (
    DISCLAIMER, NOT_FOUND_ANSWER, UNVERIFIED_ANSWER, LegalAssistantAgent,
)
from app.agents.legal_assistant.schemas import LegalAnswerDraft, LegalQuestionInput, LegalQuote
from app.agents.research.schemas import ResearchResult
from app.guardrails.output_guardrail import check_text

P1 = "A member who resigns must give 14 days written notice to the Club Secretary."
P2 = "Late fees accrue a surcharge of 2 percent after 30 days, and meetings are held every 6 months."
QUESTION = "How much notice must a member give?"


class FakeLegal(LegalAssistantAgent):
    """Replaces the model with a canned draft. Counts calls and records the messages."""

    def __init__(self, draft=None):
        self.draft, self.calls, self.messages = draft, 0, []

    async def ask_structured(self, messages, schema, **kwargs):
        self.calls += 1
        self.messages.append(messages)
        return self.draft


def draft(answer="A member must give 14 days written notice [K1].", quotes=(("K1", "14 days written notice"),), **kw):
    base = dict(
        in_scope=True, answered_from_knowledge=True, answer=answer, key_points=[],
        quotes=[LegalQuote(label=label, quote=q) for label, q in quotes],
        jurisdiction_dependent=True, seek_professional_advice=False,
    )
    base.update(kw)
    return LegalAnswerDraft(**base)


def one():
    return research_with(passage("K1", P1))


def two():
    return research_with(
        passage("K1", P1, source_id="src-a", title="Synthetic Source A", page=3, index=4),
        passage("K2", P2, source_id="src-b", title="Synthetic Source B", page=7, index=9),
    )


def ask(draft_obj, knowledge):
    agent = FakeLegal(draft_obj)
    res = asyncio.run(agent.run(LegalQuestionInput(question=QUESTION, knowledge=knowledge)))
    return res, agent


def test_answered_with_a_verified_citation_and_quote():
    res, agent = ask(draft(), one())
    out = res.output
    assert res.status == "ok" and agent.calls == 1
    assert (out.status, out.source, out.disclaimer, out.evidence_labels) == (
        "answered", "knowledge_base", DISCLAIMER, ["K1"],
    )
    assert [(c.label, c.source_id, c.page_number) for c in out.citations] == [("K1", "src-a", 1)]
    assert [q.label for q in out.quotes] == ["K1"] and out.rejected_reason is None


def test_citations_map_to_sources_pages_and_chunks():
    d = draft(
        answer="Late fees add 2 percent after 30 days [K2]. A member gives 14 days notice [K1].",
        quotes=(("K2", "surcharge of 2 percent after 30 days"), ("K1", "14 days written notice")),
    )
    out = ask(d, two())[0].output
    assert out.status == "answered"
    assert [(c.label, c.source_id, c.source_title, c.source_url, c.page_number, c.chunk_index) for c in out.citations] == [
        ("K2", "src-b", "Synthetic Source B", "https://example.gov/src-b", 7, 9),
        ("K1", "src-a", "Synthetic Source A", "https://example.gov/src-a", 3, 4),
    ]


def test_no_knowledge_means_no_llm_call_and_no_fallback():
    for knowledge in (None, research_with()):
        res, agent = ask(draft(), knowledge)
        out = res.output
        assert res.status == "ok" and agent.calls == 0
        assert (out.status, out.source, out.answer, out.citations) == ("not_found", "none", NOT_FOUND_ANSWER, [])
        assert out.disclaimer == DISCLAIMER


def test_unavailable_knowledge_fails_the_run():
    res, agent = ask(draft(), ResearchResult(status="unavailable", unavailable_reason="tool_timeout"))
    assert (res.status, res.error_type, agent.calls) == ("failed", "RuntimeError", 0)


def test_model_reports_the_passages_do_not_answer():
    res, agent = ask(draft(answered_from_knowledge=False, answer="x"), one())
    out = res.output
    assert (out.status, out.answer, agent.calls) == ("not_found", NOT_FOUND_ANSWER, 1)


def test_unknown_label_is_unverified():
    out = ask(draft(answer="A member must give 14 days written notice [K1]. See also [K9]."), one())[0].output
    assert (out.status, out.answer) == ("unverified", UNVERIFIED_ANSWER)
    assert "K9" in out.rejected_reason and out.citations == []


def test_answer_without_a_citation_is_unverified():
    out = ask(draft(answer="A member must give 14 days written notice."), one())[0].output
    assert (out.status, out.rejected_reason) == ("unverified", "answer has no citations")


def test_unverifiable_quote_is_unverified():
    out = ask(draft(quotes=(("K1", "a quote that is not in the passage"),)), one())[0].output
    assert (out.status, out.rejected_reason) == ("unverified", "no supporting quote could be verified")


def test_quote_taken_from_a_different_passage_does_not_count():
    out = ask(draft(quotes=(("K1", "surcharge of 2 percent after 30 days"),)), two())[0].output
    assert (out.status, out.rejected_reason) == ("unverified", "no supporting quote could be verified")


def test_quote_for_an_uncited_label_does_not_count():
    out = ask(draft(quotes=(("K2", "surcharge of 2 percent after 30 days"),)), two())[0].output
    assert (out.status, out.rejected_reason) == ("unverified", "no supporting quote could be verified")


def test_number_not_in_the_cited_passage_is_unverified():
    out = ask(draft(answer="A member must give 30 days written notice [K1]."), one())[0].output
    assert (out.status, out.rejected_reason) == ("unverified", "number(s) not in the cited passages: 30")


def test_one_bad_quote_is_dropped_when_another_verifies():
    out = ask(draft(quotes=(("K1", "14 days written notice"), ("K1", "a made up quote"))), one())[0].output
    assert out.status == "answered" and [q.quote for q in out.quotes] == ["14 days written notice"]


def test_key_points_are_checked_too():
    bad_number = ask(draft(key_points=["It is due in 99 days [K1]."]), one())[0].output
    assert (bad_number.status, bad_number.rejected_reason) == (
        "unverified", "number(s) not in the cited passages: 99",
    )
    bad_label = ask(draft(key_points=["See the other rule [K5]."]), one())[0].output
    assert bad_label.status == "unverified" and "K5" in bad_label.rejected_reason
    good = ask(draft(key_points=["Notice is written [K1]."]), one())[0].output
    assert good.status == "answered" and good.key_points == ["Notice is written [K1]."]


def test_out_of_scope_question():
    res, agent = ask(
        draft(in_scope=False, answered_from_knowledge=False,
              answer="I can only help with contracts and legal information.", quotes=()),
        one(),
    )
    out = res.output
    assert (out.status, out.in_scope, out.source, agent.calls) == ("out_of_scope", False, "none", 1)
    assert out.answer == "I can only help with contracts and legal information." and out.citations == []


def test_fixed_answers_pass_every_output_profile():
    for profile in ("general", "document", "commentary"):
        for text in (NOT_FOUND_ANSWER, UNVERIFIED_ANSWER):
            assert check_text("f", text, profile, "", set()) == [], (profile, text)


def test_knowledge_goes_in_the_user_message_not_the_system_prompt():
    knowledge = one()
    _, agent = ask(draft(), knowledge)
    system, user = agent.messages[0]
    assert (system["role"], user["role"]) == ("system", "user")
    assert P1 not in system["content"]
    assert knowledge.context in user["content"] and "<question>" in user["content"] and QUESTION in user["content"]


def test_only_cited_passages_supply_numbers():
    out = ask(draft(answer="A member gives 14 days notice [K1]. Meetings are every 6 months [K1]."), two())[0].output
    assert (out.status, out.rejected_reason) == ("unverified", "number(s) not in the cited passages: 6")