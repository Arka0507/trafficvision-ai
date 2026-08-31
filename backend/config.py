from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
DATA_ROOT = Path(os.getenv("TRAFFICVISION_DATA_DIR", PROJECT_ROOT / "data")).resolve()
JOBS_ROOT = DATA_ROOT / "jobs"
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
TRACKER_CONFIG = PROJECT_ROOT / "config" / "bytetrack_traffic.yaml"


@dataclass(frozen=True)
class AppSettings:
    app_name: str = "TrafficVision AI"
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "2048"))
    max_workers: int = int(os.getenv("MAX_PROCESSING_WORKERS", "1"))
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173",
        ).split(",")
        if origin.strip()
    )
    allowed_extensions: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


settings = AppSettings()
JOBS_ROOT.mkdir(parents=True, exist_ok=True)
