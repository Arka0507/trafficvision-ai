from __future__ import annotations

import json
import mimetypes
import re
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from backend.config import FRONTEND_ROOT, JOBS_ROOT, settings
from backend.jobs import job_manager
from backend.schemas import JobPublic, ProcessingOptions

app = FastAPI(
    title="TrafficVision AI API",
    version="1.0.0",
    description="YOLOv8n, ByteTrack, vehicle recognition, distance and speed video analytics.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "capabilities": [
            "YOLO object detection",
            "ByteTrack multi-object tracking",
            "fine-grained car classification",
            "monocular distance estimation",
            "camera-relative speed estimation",
        ],
    }


@app.get("/api/jobs", response_model=list[JobPublic])
def list_jobs() -> list[dict]:
    return [job.public_dict() for job in job_manager.recent()]


@app.post("/api/jobs", response_model=JobPublic, status_code=202)
async def create_job(
    video: Annotated[UploadFile, File()],
    options_json: Annotated[str, Form()] = "{}",
) -> dict:
    original_name = Path(video.filename or "video.mp4").name
    extension = Path(original_name).suffix.lower()
    if extension not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video type. Allowed: {', '.join(settings.allowed_extensions)}",
        )
    try:
        options = ProcessingOptions.model_validate(json.loads(options_json or "{}"))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid processing options: {exc}") from exc

    job_id = uuid.uuid4().hex
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    input_path = job_dir / f"input{extension}"
    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with input_path.open("wb") as destination:
            while chunk := await video.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video exceeds the {settings.max_upload_mb} MB upload limit",
                    )
                destination.write(chunk)
    except Exception:
        input_path.unlink(missing_ok=True)
        try:
            job_dir.rmdir()
        except OSError:
            pass
        raise
    finally:
        await video.close()

    job = job_manager.register(job_id, original_name, input_path, options)
    return job.public_dict()


@app.get("/api/jobs/{job_id}", response_model=JobPublic)
def get_job(job_id: str) -> dict:
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.public_dict()


@app.post("/api/jobs/{job_id}/cancel", response_model=JobPublic)
def cancel_job(job_id: str) -> dict:
    job = job_manager.request_cancel(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.public_dict()


@app.get("/api/jobs/{job_id}/artifacts/{artifact_key}")
def download_artifact(job_id: str, artifact_key: str, download: bool = False):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    path_string = job.artifacts.get(artifact_key)
    if not path_string:
        raise HTTPException(status_code=404, detail="Artifact not available")
    path = Path(path_string).resolve()
    expected_root = (JOBS_ROOT / job_id).resolve()
    if expected_root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file is missing")

    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    disposition = "attachment" if download or artifact_key != "video" else "inline"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(job.filename).stem)[:80] or "video"
    friendly_names = {
        "video": f"trafficvision_{safe_stem}.mp4",
        "tracks_csv": f"trafficvision_{safe_stem}_tracks.csv",
        "detections_csv": f"trafficvision_{safe_stem}_detections.csv",
        "summary_json": f"trafficvision_{safe_stem}_summary.json",
    }
    filename = friendly_names.get(artifact_key, path.name)
    return FileResponse(
        path,
        media_type=media_type,
        headers={"Content-Disposition": f'{disposition}; filename="{filename}"'},
    )


if FRONTEND_ROOT.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_ROOT, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
