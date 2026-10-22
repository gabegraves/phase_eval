"""In-memory application state management."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from .models import AnalysisResult, PatternData, PipelineStateEnum


@dataclass
class JobState:
    """State of a batch analysis job."""

    job_id: str
    file_ids: list[str]
    total: int
    completed: int = 0
    failed: int = 0
    current_file: str | None = None
    state: PipelineStateEnum = PipelineStateEnum.INIT
    progress: float = 0.0
    error_message: str | None = None
    results: dict[str, AnalysisResult] = field(default_factory=dict)
    hypotheses: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None


class AppState:
    """Singleton application state manager."""

    _instance: Optional["AppState"] = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize state storage."""
        self._patterns: dict[str, PatternData] = {}
        self._jobs: dict[str, JobState] = {}
        self._temp_files: dict[str, Path] = {}
        self._state_lock = Lock()

    # Pattern management

    def add_pattern(self, file_id: str, pattern: PatternData, temp_path: Path | None = None):
        """Add uploaded pattern to state."""
        with self._state_lock:
            self._patterns[file_id] = pattern
            if temp_path:
                self._temp_files[file_id] = temp_path

    def get_pattern(self, file_id: str) -> PatternData | None:
        """Get pattern by file ID."""
        with self._state_lock:
            return self._patterns.get(file_id)

    def remove_pattern(self, file_id: str) -> bool:
        """Remove pattern from state."""
        with self._state_lock:
            if file_id in self._patterns:
                del self._patterns[file_id]
                # Clean up temp file if exists
                if file_id in self._temp_files:
                    temp_path = self._temp_files[file_id]
                    if temp_path.exists():
                        temp_path.unlink()
                    del self._temp_files[file_id]
                return True
            return False

    def list_patterns(self) -> list[str]:
        """List all uploaded pattern IDs."""
        with self._state_lock:
            return list(self._patterns.keys())

    def get_temp_file(self, file_id: str) -> Path | None:
        """Get temporary file path for a pattern."""
        with self._state_lock:
            return self._temp_files.get(file_id)

    # Job management

    def create_job(self, job_id: str, file_ids: list[str]) -> JobState:
        """Create a new batch analysis job."""
        with self._state_lock:
            job = JobState(job_id=job_id, file_ids=file_ids, total=len(file_ids))
            self._jobs[job_id] = job
            return job

    def get_job(self, job_id: str) -> JobState | None:
        """Get job state by ID."""
        with self._state_lock:
            return self._jobs.get(job_id)

    def update_job_progress(
        self,
        job_id: str,
        current_file: str | None = None,
        state: PipelineStateEnum | None = None,
        progress: float | None = None,
        error_message: str | None = None,
    ):
        """Update job progress."""
        with self._state_lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                if current_file is not None:
                    job.current_file = current_file
                if state is not None:
                    job.state = state
                if progress is not None:
                    job.progress = progress
                if error_message is not None:
                    job.error_message = error_message

    def add_job_result(self, job_id: str, sample_id: str, result: AnalysisResult):
        """Add a completed result to job."""
        with self._state_lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.results[sample_id] = result
                job.completed += 1
                job.progress = job.completed / job.total

    def add_job_failure(self, job_id: str):
        """Mark a pattern as failed in job."""
        with self._state_lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.failed += 1
                job.completed += 1  # Count failures toward completion
                job.progress = job.completed / job.total

    def add_job_hypotheses(self, job_id: str, sample_id: str, hypotheses: list[dict[str, Any]]):
        """Add multi-hypothesis results for a pattern."""
        with self._state_lock:
            if job_id in self._jobs:
                self._jobs[job_id].hypotheses[sample_id] = hypotheses

    def complete_job(self, job_id: str):
        """Mark job as completed."""
        with self._state_lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.state = PipelineStateEnum.COMPLETED
                job.completed_at = datetime.now()

    def fail_job(self, job_id: str, error_message: str):
        """Mark job as failed."""
        with self._state_lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.state = PipelineStateEnum.FAILED
                job.error_message = error_message
                job.completed_at = datetime.now()

    def list_jobs(self) -> list[str]:
        """List all job IDs."""
        with self._state_lock:
            return list(self._jobs.keys())

    # Statistics

    def get_stats(self) -> dict[str, int]:
        """Get application statistics."""
        with self._state_lock:
            return {
                "num_patterns": len(self._patterns),
                "num_jobs": len(self._jobs),
                "num_active_jobs": sum(
                    1
                    for j in self._jobs.values()
                    if j.state not in [PipelineStateEnum.COMPLETED, PipelineStateEnum.FAILED]
                ),
                "num_completed_jobs": sum(
                    1 for j in self._jobs.values() if j.state == PipelineStateEnum.COMPLETED
                ),
            }


# Global singleton instance
app_state = AppState()
