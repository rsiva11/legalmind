import contextvars

import pytest

from app.config import get_settings
from app.orchestration.execution_context import ExecutionContext


@pytest.mark.parametrize(
    "workflow, setting",
    [
        ("full_analysis", "analysis_token_budget"),
        ("document_qa", "request_token_budget"),
        ("legal_question", "request_token_budget"),
    ],
)
def test_budget_depends_on_workflow(workflow, setting):
    # copy_context keeps the tracker from leaking into other tests
    ctx = contextvars.copy_context().run(ExecutionContext.begin, "r1", workflow)
    assert ctx.tracker.budget == getattr(get_settings(), setting)
    assert ctx.request_id == "r1" and ctx.elapsed_ms >= 0