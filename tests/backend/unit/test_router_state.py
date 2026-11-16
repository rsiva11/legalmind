import pytest
from pydantic import ValidationError

from app.agents.base_agent import AgentResult, AgentUsage
from app.orchestration.router import RoutingError, route
from app.orchestration.state import WorkflowRequest, WorkflowState


def decide(**kw):
    return route(WorkflowRequest(**kw)).workflow


def rejects(**kw):
    with pytest.raises(RoutingError):
        route(WorkflowRequest(**kw))


def result(agent, status="ok", calls=1, prompt=100, completion=50, error=None):
    return AgentResult(
        agent=agent, status=status, output=None, error_type=error, duration_ms=10,
        usage=AgentUsage(llm_calls=calls, prompt_tokens=prompt, completion_tokens=completion),
    )


def test_general_question_without_document():
    assert decide(question="What is an NDA?") == "legal_question"


def test_general_mode_without_document():
    assert decide(question="What is an NDA?", mode="legal_question") == "legal_question"


def test_document_mode_without_document_is_rejected():
    rejects(question="What is the notice period?", mode="document_qa")
    rejects(mode="full_analysis")


def test_blank_question_without_document_is_rejected():
    rejects()
    rejects(question="")
    rejects(question="   \n ")


def test_document_requires_explicit_mode():
    rejects(document_id="d1", question="What is the notice period?")


def test_document_question_routes_to_qa():
    assert decide(document_id="d1", question="What is the notice period?", mode="document_qa") == "document_qa"


def test_document_question_needs_text():
    rejects(document_id="d1", mode="document_qa")
    rejects(document_id="d1", question="  ", mode="document_qa")


def test_full_analysis_needs_no_question():
    assert decide(document_id="d1", mode="full_analysis") == "full_analysis"


def test_general_question_with_document_selected_ignores_document():
    req = WorkflowRequest(document_id="d1", question="  What is an NDA?  ", mode="legal_question")
    decision = route(req)
    state = WorkflowState.start(req, decision)
    assert decision.workflow == "legal_question"
    assert state.document_id is None
    assert state.question == "What is an NDA?"


def test_empty_document_id_is_invalid():
    with pytest.raises(ValidationError):
        WorkflowRequest(document_id="", mode="full_analysis")


def test_new_state_defaults():
    req = WorkflowRequest(document_id="d1", mode="full_analysis")
    state = WorkflowState.start(req, route(req))
    assert (state.status, state.failed_step, state.steps) == ("running", None, [])
    assert state.document_id == "d1" and state.needs_human_review is False
    assert len(state.request_id) == 32


def test_step_records_accumulate_usage():
    req = WorkflowRequest(document_id="d1", mode="full_analysis")
    state = WorkflowState.start(req, route(req))
    state.record_step(result("contract", calls=1, prompt=100, completion=50))
    state.record_step(result("compliance", calls=2, prompt=200, completion=100))
    assert (state.total_tokens, state.total_llm_calls) == (450, 3)
    assert state.status == "running"
    state.complete()
    assert state.status == "completed"


def test_failed_step_is_recorded():
    req = WorkflowRequest(document_id="d1", mode="full_analysis")
    state = WorkflowState.start(req, route(req))
    state.record_step(result("contract", status="failed", calls=0, prompt=0, completion=0, error="StructuredOutputError"))
    assert (state.status, state.failed_step) == ("failed", "contract")
    assert state.steps[0].error_type == "StructuredOutputError"
    state.complete()  # a failed state must never flip to completed
    assert state.status == "failed"

def test_nonfatal_step_failure_is_a_warning():
    req = WorkflowRequest(document_id="d1", mode="full_analysis")
    state = WorkflowState.start(req, route(req))
    state.record_step(
        result("risk", status="failed", calls=0, prompt=0, completion=0, error="TimeoutError"), fatal=False
    )
    assert (state.status, state.failed_step) == ("running", None)
    assert state.warnings == ["risk step failed (TimeoutError); its results are missing"]
    assert state.steps[0].status == "failed"


def test_record_error_fails_the_state():
    req = WorkflowRequest(document_id="d1", mode="full_analysis")
    state = WorkflowState.start(req, route(req))
    state.record_error("retrieval", "ConnectionError")
    assert (state.status, state.failed_step) == ("failed", "retrieval")
    assert (state.steps[0].error_type, state.steps[0].llm_calls) == ("ConnectionError", 0)

def test_question_is_not_persisted():
    req = WorkflowRequest(question="My PAN is ABCPE1234F")
    state = WorkflowState.start(req, route(req))
    assert state.question == "My PAN is ABCPE1234F"  # still available in memory
    dumped = state.model_dump(mode="json")
    assert "question" not in dumped and dumped["question_masked"] is None
    assert "ABCPE1234F" not in str(dumped)