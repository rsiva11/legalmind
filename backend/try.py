import asyncio

from app.orchestration.router import route
from app.orchestration.state import WorkflowRequest, WorkflowState
from app.orchestration.workflow import run_workflow

QUESTIONS = [
    "What is an NDA?",
    "My landlord kept my deposit. Should I sue him?",
    "Is a contract signed under threat valid?",
    "Should I sign this contract?",
]


async def main():
    for q in QUESTIONS:
        req = WorkflowRequest(question=q)
        state = await run_workflow(WorkflowState.start(req, route(req)))
        print("Q:", q)
        print("   status:", state.status, "| failed_step:", state.failed_step, "| tokens:", state.total_tokens)
        print("   findings:", [(f.action, f.code, f.field) for f in state.guardrail_findings])
        print("   answer shown:", state.legal_answer is not None)


asyncio.run(main())