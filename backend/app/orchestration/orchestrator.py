import logging
from typing import Any

from pydantic import BaseModel

from app.database.repositories.job_repository import MongoJobStore
from app.guardrails.guardrail_manager import GuardrailManager
from app.guardrails.input_guardrail import GUARDRAIL_ERROR_MESSAGE, InputCheck
from app.guardrails.result import GuardrailBlocked, GuardrailFinding
from app.orchestration.router import route
from app.orchestration.state import Workflow, WorkflowRequest, WorkflowState
from app.orchestration.task_manager import Job, TaskManager

logger = logging.getLogger(__name__)

PII_NOTICE = (
    "Personal identifiers in your question were masked in the stored copy. "
    "The original text is still sent to the language model to answer you."
)


class Submission(BaseModel):
    job_id: str
    workflow: Workflow
    reason: str
    notices: list[str] = []


class Orchestrator:
    def __init__(self, tasks: TaskManager, guardrails: Any = None) -> None:
        self._tasks = tasks
        self._guardrails = guardrails if guardrails is not None else GuardrailManager()

    def _check_input(self, request: WorkflowRequest) -> InputCheck:
        try:
            check = self._guardrails.check_input(request.question, request.document_id)
        except Exception as exc:  # fail closed: an unchecked request is not processed
            logger.error("Input guardrail raised %s", type(exc).__name__)
            raise GuardrailBlocked(
                GUARDRAIL_ERROR_MESSAGE,
                [
                    GuardrailFinding(
                        guardrail="input", action="block", code="guardrail_error", field="request",
                        detail="The input check failed, so the request was blocked",
                    )
                ],
            ) from exc
        if check.action == "block":
            blocked = sorted({f.code for f in check.result.findings if f.action == "block"})
            logger.warning("Request blocked by input guardrail: %s", blocked)
            raise GuardrailBlocked(check.user_message or GUARDRAIL_ERROR_MESSAGE, check.result.findings)
        return check

    async def submit(self, request: WorkflowRequest, deps: Any = None) -> Submission:
        check = self._check_input(request)  # a blocked request creates no job
        decision = route(request)  # RoutingError propagates: a bad request creates no job
        state = WorkflowState.start(request, decision)
        state.question_masked = check.masked_question
        state.guardrail_findings.extend(check.result.findings)
        job_id = await self._tasks.submit(state, deps)
        notices = [PII_NOTICE] if any(f.code == "pii_masked" for f in check.result.findings) else []
        return Submission(job_id=job_id, workflow=decision.workflow, reason=decision.reason, notices=notices)

    async def get_job(self, job_id: str) -> Job | None:
        return await self._tasks.get(job_id)

    async def startup(self) -> int:
        return await self._tasks.recover_orphans()

    async def wait_idle(self) -> None:
        await self._tasks.wait_all()

    async def shutdown(self) -> None:
        await self._tasks.shutdown()


def build_orchestrator() -> Orchestrator:
    return Orchestrator(TaskManager(MongoJobStore()))