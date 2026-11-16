import asyncio

import pytest

from test_task_manager import FakeStore, ok_runner
from app.orchestration.orchestrator import Orchestrator
from app.orchestration.router import RoutingError
from app.orchestration.state import WorkflowRequest
from app.orchestration.task_manager import TaskManager


def make(store=None):
    store = store or FakeStore()
    return store, Orchestrator(TaskManager(store, runner=ok_runner, max_concurrent=1, timeout_seconds=5))


def test_submit_routes_and_runs():
    async def main():
        store, orch = make()
        sub = await orch.submit(WorkflowRequest(document_id="d1", mode="full_analysis"))
        assert sub.workflow == "full_analysis" and sub.reason
        await orch.wait_idle()
        job = await orch.get_job(sub.job_id)
        assert (job.status, job.workflow, job.document_id) == ("completed", "full_analysis", "d1")

    asyncio.run(main())


def test_routing_error_creates_no_job():
    async def main():
        store, orch = make()
        with pytest.raises(RoutingError):
            await orch.submit(WorkflowRequest(document_id="d1"))  # a mode is required
        assert store.docs == {} and store.saves == 0

    asyncio.run(main())


def test_unknown_job_is_none():
    async def main():
        _, orch = make()
        assert await orch.get_job("nope") is None

    asyncio.run(main())