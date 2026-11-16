import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from app.config import get_settings
from app.orchestration.state import Workflow, WorkflowState
from app.orchestration.workflow import run_workflow
from app.utils.time_utils import utc_now

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "completed", "failed", "timed_out"]


class JobStore(Protocol):
    async def save(self, job_id: str, doc: dict) -> None: ...
    async def get(self, job_id: str) -> dict | None: ...
    async def fail_orphans(self, finished_at: str) -> int: ...


class Job(BaseModel):
    job_id: str
    workflow: Workflow
    document_id: str | None = None
    status: JobStatus = "queued"
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    failed_step: str | None = None
    result: WorkflowState | None = None


class TaskManager:
    def __init__(
        self,
        store: JobStore,
        *,
        runner: Any = run_workflow,
        max_concurrent: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        settings = get_settings()
        self._store = store
        self._runner = runner
        self._sem = asyncio.Semaphore(max_concurrent or settings.max_concurrent_jobs)
        self._timeout = timeout_seconds or settings.workflow_timeout_seconds
        self._tasks: set[asyncio.Task] = set()  # strong references, so tasks are not garbage collected

    async def submit(self, state: WorkflowState, deps: Any = None) -> str:
        job = Job(
            job_id=state.request_id,
            workflow=state.workflow,
            document_id=state.document_id,
            created_at=utc_now(),
        )
        await self._save(job, required=True)  # no record means no job: fail loudly
        task = asyncio.create_task(self._run(job, state, deps), name=f"job-{job.job_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job.job_id

    async def get(self, job_id: str) -> Job | None:
        doc = await self._store.get(job_id)
        return Job.model_validate(doc) if doc else None

    async def recover_orphans(self) -> int:
        count = await self._store.fail_orphans(utc_now().isoformat())
        if count:
            logger.warning("Marked %d unfinished jobs from a previous run as failed", count)
        return count

    async def wait_all(self) -> None:
        await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def shutdown(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _save(self, job: Job, *, required: bool = False) -> None:
        try:
            await self._store.save(job.job_id, job.model_dump(mode="json"))
        except Exception as exc:
            if required:
                raise
            # A lost update leaves the job as it was; recover_orphans cleans it up at the next start
            logger.error("Could not save job %s: %s", job.job_id, type(exc).__name__)

    async def _run(self, job: Job, state: WorkflowState, deps: Any) -> None:
        cancelled = False
        started = time.perf_counter()
        try:
            async with self._sem:  # waiting here is "queued"; the timeout starts after
                job.status, job.started_at = "running", utc_now()
                await self._save(job)
                try:
                    await asyncio.wait_for(self._runner(state, deps), self._timeout)
                except TimeoutError:
                    state.fail("timeout")
                    state.duration_ms = int((time.perf_counter() - started) * 1000)
                    job.status = "timed_out"
                else:
                    job.status = "completed" if state.status == "completed" else "failed"
        except asyncio.CancelledError:
            state.fail("cancelled")
            job.status = "failed"
            cancelled = True

        job.finished_at = utc_now()
        job.failed_step = state.failed_step
        job.result = state
        await self._save(job)
        logger.info("Job %s finished status=%s", job.job_id, job.status)
        if cancelled:
            raise asyncio.CancelledError()