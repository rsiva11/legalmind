import time
from dataclasses import dataclass, field

from app.config import get_settings
from app.llm.token_manager import UsageTracker, start_tracking
from app.orchestration.state import Workflow


@dataclass
class ExecutionContext:
    request_id: str
    workflow: Workflow
    tracker: UsageTracker
    _started: float = field(default_factory=time.perf_counter)

    @classmethod
    def begin(cls, request_id: str, workflow: Workflow) -> "ExecutionContext":
        """Start tracking for the current request. Call once, from the task that runs the workflow."""
        settings = get_settings()
        budget = settings.analysis_token_budget if workflow == "full_analysis" else settings.request_token_budget
        return cls(request_id=request_id, workflow=workflow, tracker=start_tracking(budget))

    @property
    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._started) * 1000)