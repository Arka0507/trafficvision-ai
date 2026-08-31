from __future__ import annotations

import json
import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from backend.config import JOBS_ROOT, settings
from backend.exceptions import ProcessingCancelled
from backend.schemas import ProcessingOptions

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    filename: str
    input_path: str
    options: dict
    status: str = "queued"
    progress: float = 0.0
    phase: str = "Queued"
    message: str = "Waiting for the processor"
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    cancel_requested: bool = False

    def public_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("input_path", None)
        payload.pop("options", None)
        payload.pop("cancel_requested", None)
        payload["artifacts"] = {
            key: f"/api/jobs/{self.id}/artifacts/{key}" for key in self.artifacts
        }
        return payload


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers, thread_name_prefix="trafficvision")
        self._load_existing()

    def _job_dir(self, job_id: str) -> Path:
        return JOBS_ROOT / job_id

    def _metadata_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def _save(self, job: Job) -> None:
        path = self._metadata_path(job.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")
        temporary.replace(path)

    def _load_existing(self) -> None:
        for metadata_path in JOBS_ROOT.glob("*/job.json"):
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                job = Job(**payload)
                if job.status in {"queued", "processing", "cancelling"}:
                    job.status = "failed"
                    job.phase = "Interrupted"
                    job.error = "The server stopped before this job finished. Please upload the video again."
                    job.completed_at = utc_now()
                    self._save(job)
                self.jobs[job.id] = job
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Ignoring invalid job metadata at %s: %s", metadata_path, exc)

    def register(self, job_id: str, filename: str, input_path: Path, options: ProcessingOptions) -> Job:
        job = Job(id=job_id, filename=filename, input_path=str(input_path), options=options.model_dump())
        with self.lock:
            self.jobs[job.id] = job
            self._save(job)
        self.executor.submit(self._run, job.id)
        return job

    def _run(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            job.status = "processing"
            job.started_at = utc_now()
            job.phase = "Starting"
            job.message = "Preparing models and video"
            self._save(job)

        def progress(progress_value: float, phase: str, message: str, stats_patch: dict) -> None:
            with self.lock:
                current = self.jobs[job_id]
                current.progress = round(float(progress_value), 2)
                current.phase = phase
                current.message = message
                current.stats.update(stats_patch)
                self._save(current)

        def cancelled() -> bool:
            with self.lock:
                return self.jobs[job_id].cancel_requested

        try:
            from backend.processing.pipeline import VideoProcessor

            options = ProcessingOptions.model_validate(job.options)
            processor = VideoProcessor(options)
            result = processor.process(
                input_path=Path(job.input_path),
                output_dir=self._job_dir(job_id),
                callback=progress,
                cancel_check=cancelled,
            )
            with self.lock:
                current = self.jobs[job_id]
                current.status = "completed"
                current.progress = 100.0
                current.phase = "Completed"
                current.message = "Annotated video and reports are ready"
                current.completed_at = utc_now()
                current.stats.update(result["summary"]["results"])
                current.stats["input"] = result["summary"]["input"]
                current.stats["measurement"] = result["summary"]["measurement"]
                current.warnings = result["warnings"]
                current.artifacts = result["artifacts"]
                self._save(current)
        except ProcessingCancelled as exc:
            with self.lock:
                current = self.jobs[job_id]
                current.status = "cancelled"
                current.phase = "Cancelled"
                current.message = str(exc)
                current.completed_at = utc_now()
                self._save(current)
        except Exception as exc:  # noqa: BLE001 - a worker must persist any model/runtime failure.
            with self.lock:
                current = self.jobs[job_id]
                current.status = "failed"
                current.phase = "Failed"
                current.message = "Video processing failed"
                current.error = f"{type(exc).__name__}: {exc}"
                current.completed_at = utc_now()
                current.stats["debug_trace"] = traceback.format_exc(limit=8)
                self._save(current)

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def recent(self, limit: int = 20) -> list[Job]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda job: job.created_at, reverse=True)[:limit]

    def request_cancel(self, job_id: str) -> Job | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            if job.status in {"queued", "processing"}:
                job.cancel_requested = True
                job.status = "cancelling"
                job.phase = "Cancelling"
                job.message = "Stopping safely after the current frame"
                self._save(job)
            return job


job_manager = JobManager()
