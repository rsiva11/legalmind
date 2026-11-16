import asyncio
import json

import pytest

from factories import comp_report, good_analysis, risk_report
from app.orchestration.router import route
from app.orchestration.state import WorkflowRequest, WorkflowState
from app.orchestration.task_manager import Job, TaskManager
from app.utils.time_utils import utc_now


class FakeStore:
    """In-memory job store. Proves every saved document is JSON-safe."""

    def __init__(self, fail_from=None):
        self.docs, self.saves, self.fail_from = {}, 0, fail_from

    async def save(self, job_id, doc):
        self.saves += 1
        if self.fail_from is not None and self.saves >= self.fail_from:
            raise ConnectionError("store down")
        self.docs[job_id] = json.loads(json.dumps(doc))

    async def get(self, job_id):
        doc = self.docs.get(job_id)
        return json.loads(json.dumps(doc)) if doc else None

    async def fail_orphans(self, finished_at):
        n = 0
        for doc in self.docs.values():
            if doc["status"] in ("queued", "running"):
                doc.update(status="failed", failed_step="server_restarted", finished_at=finished_at)
                n += 1
        return n


def new_state():
    req = WorkflowRequest(document_id="d1", mode="full_analysis")
    return WorkflowState.start(req, route(req))


async def ok_runner(state, deps=None):
    state.duration_ms = 5
    state.complete()
    return state


async def failing_runner(state, deps=None):
    state.record_error("contract", "StructuredOutputError")
    state.complete()
    return state


async def slow_runner(state, deps=None):
    await asyncio.sleep(10)


def test_successful_job_lifecycle():
    async def main():
        store = FakeStore()
        mgr = TaskManager(store, runner=ok_runner, max_concurrent=2, timeout_seconds=5)
        state = new_state()
        job_id = await mgr.submit(state)
        assert job_id == state.request_id
        assert store.docs[job_id]["status"] == "queued"  # saved before the task has run
        await mgr.wait_all()
        job = await mgr.get(job_id)
        assert (job.status, job.failed_step) == ("completed", None)
        assert job.result.status == "completed"
        assert job.started_at is not None and job.finished_at >= job.started_at

    asyncio.run(main())


def test_failed_workflow_job():
    async def main():
        mgr = TaskManager(FakeStore(), runner=failing_runner, max_concurrent=2, timeout_seconds=5)
        job_id = await mgr.submit(new_state())
        await mgr.wait_all()
        job = await mgr.get(job_id)
        assert (job.status, job.failed_step) == ("failed", "contract")
        assert job.result.steps[0].error_type == "StructuredOutputError"

    asyncio.run(main())


def test_timeout_marks_timed_out():
    async def main():
        mgr = TaskManager(FakeStore(), runner=slow_runner, max_concurrent=2, timeout_seconds=0.05)
        job_id = await mgr.submit(new_state())
        await mgr.wait_all()
        job = await mgr.get(job_id)
        assert (job.status, job.failed_step) == ("timed_out", "timeout")
        assert job.result.status == "failed"

    asyncio.run(main())


def test_concurrency_limit_queues_jobs():
    async def main():
        store, gate = FakeStore(), asyncio.Event()

        async def gated(state, deps=None):
            await gate.wait()
            state.complete()

        mgr = TaskManager(store, runner=gated, max_concurrent=1, timeout_seconds=5)
        a, b = new_state(), new_state()
        await mgr.submit(a)
        await mgr.submit(b)
        for _ in range(5):
            await asyncio.sleep(0)
        assert store.docs[a.request_id]["status"] == "running"
        assert store.docs[b.request_id]["status"] == "queued"
        gate.set()
        await mgr.wait_all()
        assert store.docs[a.request_id]["status"] == "completed"
        assert store.docs[b.request_id]["status"] == "completed"

    asyncio.run(main())


def test_orphans_are_failed_on_startup():
    async def main():
        store = FakeStore()
        store.docs = {
            "q": {"job_id": "q", "status": "queued"},
            "r": {"job_id": "r", "status": "running"},
            "c": {"job_id": "c", "status": "completed"},
        }
        mgr = TaskManager(store, runner=ok_runner, max_concurrent=1, timeout_seconds=5)
        assert await mgr.recover_orphans() == 2
        assert [store.docs[k]["status"] for k in ("q", "r", "c")] == ["failed", "failed", "completed"]
        assert store.docs["q"]["failed_step"] == "server_restarted"

    asyncio.run(main())


def test_store_failure_does_not_crash_the_task():
    async def main():
        store = FakeStore(fail_from=2)  # the first save works, every later one fails
        mgr = TaskManager(store, runner=ok_runner, max_concurrent=1, timeout_seconds=5)
        job_id = await mgr.submit(new_state())
        await mgr.wait_all()  # must not raise
        assert store.docs[job_id]["status"] == "queued"  # last good save; recover_orphans fixes it at restart

    asyncio.run(main())


def test_submit_fails_if_job_cannot_be_created():
    async def main():
        mgr = TaskManager(FakeStore(fail_from=1), runner=ok_runner, max_concurrent=1, timeout_seconds=5)
        with pytest.raises(ConnectionError):
            await mgr.submit(new_state())
        assert mgr._tasks == set()  # nothing was started

    asyncio.run(main())


def test_shutdown_saves_cancelled_jobs():
    async def main():
        started = asyncio.Event()

        async def forever(state, deps=None):
            started.set()
            await asyncio.sleep(60)

        mgr = TaskManager(FakeStore(), runner=forever, max_concurrent=1, timeout_seconds=100)
        state = new_state()
        await mgr.submit(state)
        await started.wait()
        await mgr.shutdown()
        job = await mgr.get(state.request_id)
        assert (job.status, job.failed_step) == ("failed", "cancelled")

    asyncio.run(main())


def test_job_round_trips_through_json():
    a = good_analysis()
    state = new_state()
    state.contract, state.compliance, state.risk = a, comp_report(a), risk_report(a)
    state.complete()
    job = Job(
        job_id=state.request_id, workflow=state.workflow, document_id="d1",
        status="completed", created_at=utc_now(), result=state,
    )
    doc = job.model_dump(mode="json")
    json.dumps(doc)  # must be JSON-safe: this is what MongoDB receives
    assert Job.model_validate(doc) == job