from knowledge_fixture import passage, research_with
from test_workflow import Fake, run, start
from app.agents.legal_assistant.agent import NOT_FOUND_ANSWER, LegalAssistantAgent
from app.agents.legal_assistant.schemas import LegalAnswer
from app.agents.research.schemas import ResearchResult
from app.config import get_settings
from app.guardrails.result import GuardrailFinding
from app.orchestration.workflow import WorkflowDeps

QUESTION = "How much notice must a member give?"
RES = research_with(passage("K1", "A member who resigns must give 14 days written notice."))


def answered(text="A member must give 14 days written notice [K1]."):
    return LegalAnswer(answer=text, disclaimer="d")


def go(research, legal):
    return run(start(question=QUESTION), WorkflowDeps(research=research, legal=legal))


def test_research_then_legal_assistant():
    research = Fake("research", RES)
    legal = Fake("legal_assistant", answered(), tokens=120, calls=1)
    state = go(research, legal)
    assert (state.status, state.failed_step) == ("completed", None)
    assert [s.agent for s in state.steps] == ["research", "legal_assistant"]
    assert state.research == RES and state.legal_answer.status == "answered"
    assert research.inputs[0].question == QUESTION
    assert legal.inputs[0].knowledge == RES and legal.inputs[0].question == QUESTION
    assert (state.document_id, state.guardrail_findings, state.warnings) == (None, [], [])
    assert research.budgets == legal.budgets == [get_settings().request_token_budget]


def test_unavailable_knowledge_base_fails_the_job_without_calling_the_assistant():
    research = Fake("research", ResearchResult(status="unavailable", unavailable_reason="tool_timeout"))
    legal = Fake("legal_assistant", answered(), calls=1)
    state = go(research, legal)
    assert (state.status, state.failed_step) == ("failed", "research")
    assert [(s.agent, s.status, s.error_type) for s in state.steps] == [
        ("research", "ok", None), ("research", "failed", "tool_timeout"),
    ]
    assert legal.inputs == [] and state.legal_answer is None


def test_research_agent_failure_is_fatal():
    legal = Fake("legal_assistant", answered(), calls=1)
    state = go(Fake("research", status="failed", error="ManifestError"), legal)
    assert (state.status, state.failed_step) == ("failed", "research")
    assert state.steps[0].error_type == "ManifestError" and legal.inputs == [] and state.research is None


def test_not_found_completes_with_the_fixed_answer_and_no_llm_call():
    state = go(Fake("research", research_with()), LegalAssistantAgent())
    assert state.status == "completed"
    assert (state.legal_answer.status, state.legal_answer.answer) == ("not_found", NOT_FOUND_ANSWER)
    assert [(s.agent, s.llm_calls) for s in state.steps] == [("research", 0), ("legal_assistant", 0)]
    assert state.needs_human_review is False


def test_withheld_knowledge_passages_warn_and_escalate():
    finding = GuardrailFinding(
        guardrail="retrieval", action="block", code="suspicious_content", field="knowledge",
        detail="1 knowledge passage(s) were withheld from the model",
    )
    res = research_with(
        passage("K1", "A member who resigns must give 14 days written notice."), withheld_count=1, findings=[finding]
    )
    state = go(Fake("research", res), Fake("legal_assistant", answered(), calls=1))
    assert (state.status, state.warnings) == (
        "completed", ["1 knowledge passage(s) were withheld as suspicious and not used"],
    )
    assert state.guardrail_findings == [finding]
    assert (state.needs_human_review, state.review_codes) == (True, ["suspicious_document_content"])


def test_unknown_citation_label_blocks_the_answer():
    legal = Fake("legal_assistant", answered("Notice is 14 days [K9]."), calls=1)
    state = go(Fake("research", RES), legal)
    assert (state.status, state.failed_step) == ("failed", "guardrail:output")
    assert state.legal_answer is None and state.research is not None
    assert [(f.code, f.field) for f in state.guardrail_findings] == [("unknown_citation", "legal_answer.answer")]
    assert (state.needs_human_review, state.review_codes) == (True, ["guardrail_failure"])


def test_out_of_scope_text_is_screened():
    bad = LegalAnswer(
        status="out_of_scope", in_scope=False, source="none", answer="You should sign this contract.", disclaimer="d"
    )
    state = go(Fake("research", RES), Fake("legal_assistant", bad, calls=1))
    assert (state.status, state.failed_step) == ("failed", "guardrail:output")
    assert [(f.code, f.field) for f in state.guardrail_findings] == [("legal_advice", "legal_answer.answer")]


def test_code_written_answers_are_not_screened():
    fixed = LegalAnswer(status="unverified", source="none", answer="You should sign this contract.", disclaimer="d")
    state = go(Fake("research", RES), Fake("legal_assistant", fixed, calls=1))
    assert state.status == "completed" and state.guardrail_findings == []