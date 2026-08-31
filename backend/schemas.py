from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProcessingOptions(BaseModel):
    """User-tunable processing options with conservative defaults."""

    model_config = ConfigDict(extra="forbid")

    detector_model: str = Field(default="yolov8n.pt", description="Ultralytics model name or local weights path")
    confidence: float = Field(default=0.05, ge=0.01, le=0.95)
    iou: float = Field(default=0.45, ge=0.10, le=0.95)
    image_size: int = Field(default=960, ge=320, le=1280)
    device: str = Field(default="auto")

    horizontal_fov_degrees: float = Field(default=70.0, ge=25.0, le=140.0)
    distance_scale: float = Field(default=1.0, ge=0.10, le=10.0)
    speed_scale: float = Field(default=1.0, ge=0.10, le=10.0)
    calibration_mode: Literal["fov", "homography"] = "fov"
    image_points: list[list[float]] | None = None
    world_points: list[list[float]] | None = None

    enable_vehicle_classifier: bool = True
    vehicle_classifier_model: str = "twincar-group2/twincar-classifier"
    vehicle_classifier_confidence: float = Field(default=0.45, ge=0.01, le=0.99)
    vehicle_sample_interval_frames: int = Field(default=20, ge=1, le=600)
    vehicle_max_samples_per_track: int = Field(default=5, ge=1, le=10)
    vehicle_min_crop_pixels: int = Field(default=32, ge=16, le=512)

    draw_trails: bool = True
    trail_length: int = Field(default=32, ge=5, le=180)
    show_model_confidence: bool = False
    save_frame_detections: bool = True

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        value = value.strip().lower()
        if value in {"auto", "cpu", "mps"}:
            return value
        if value.isdigit() or (value.startswith("cuda:") and value[5:].isdigit()):
            return value
        raise ValueError("device must be auto, cpu, mps, a GPU index, or cuda:N")

    @field_validator("image_points", "world_points")
    @classmethod
    def validate_points(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return value
        if len(value) != 4 or any(len(point) != 2 for point in value):
            raise ValueError("calibration points must contain exactly four [x, y] pairs")
        return value


class JobPublic(BaseModel):
    id: str
    filename: str
    status: Literal["queued", "processing", "completed", "failed", "cancelling", "cancelled"]
    progress: float
    phase: str
    message: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    stats: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
