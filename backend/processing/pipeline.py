from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from backend.config import TRACKER_CONFIG
from backend.exceptions import ProcessingCancelled
from backend.schemas import ProcessingOptions

from .drawing import class_color, draw_footer, draw_hud, draw_label, draw_trail
from .geometry import DistanceSpeedEstimator, MotionReading
from .vehicle_classifier import FineGrainedVehicleClassifier, VehicleConsensus

ProgressCallback = Callable[[float, str, str, dict], None]
CancelCheck = Callable[[], bool]


@dataclass
class TrackRecord:
    track_id: int
    class_name: str
    first_seen_s: float
    last_seen_s: float
    frame_count: int = 0
    confidence_sum: float = 0.0
    min_distance_m: float = math.inf
    max_distance_m: float = 0.0
    speed_samples: list[float] = field(default_factory=list)
    vehicle_consensus: VehicleConsensus = field(default_factory=VehicleConsensus)
    last_vehicle_sample_frame: int = -100000
    trail: deque = field(default_factory=lambda: deque(maxlen=32))

    def update(self, timestamp_s: float, confidence: float, motion: MotionReading) -> None:
        self.last_seen_s = timestamp_s
        self.frame_count += 1
        self.confidence_sum += confidence
        self.min_distance_m = min(self.min_distance_m, motion.distance_m)
        self.max_distance_m = max(self.max_distance_m, motion.distance_m)
        if motion.speed_kmh is not None:
            self.speed_samples.append(motion.speed_kmh)

    @property
    def average_confidence(self) -> float:
        return self.confidence_sum / max(self.frame_count, 1)

    @property
    def average_speed_kmh(self) -> float | None:
        if not self.speed_samples:
            return None
        return float(np.median(self.speed_samples))

    @property
    def max_speed_kmh(self) -> float | None:
        if not self.speed_samples:
            return None
        return float(np.percentile(self.speed_samples, 95))


class VideoProcessor:
    """YOLO + ByteTrack video analytics pipeline."""

    def __init__(self, options: ProcessingOptions) -> None:
        self.options = options
        self.detector: YOLO | None = None
        self.vehicle_classifier: FineGrainedVehicleClassifier | None = None
        self.warnings: list[str] = []

    @staticmethod
    def _class_name(names, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    @staticmethod
    def _safe_crop(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x = int((x2 - x1) * 0.04)
        pad_y = int((y2 - y1) * 0.04)
        left = max(0, int(x1) - pad_x)
        top = max(0, int(y1) - pad_y)
        right = min(width, int(x2) + pad_x)
        bottom = min(height, int(y2) + pad_y)
        if right - left < 2 or bottom - top < 2:
            return None
        return frame[top:bottom, left:right].copy()

    @staticmethod
    def _format_vehicle_label(record: TrackRecord, threshold: float) -> tuple[str | None, float | None]:
        best = record.vehicle_consensus.best()
        agreement = record.vehicle_consensus.counts.get(best.label, 0) if best else 0
        if best is None or best.confidence < threshold or agreement < 2:
            return None, best.confidence if best else None
        return best.label, best.confidence

    def _load_models(self, callback: ProgressCallback) -> None:
        import os
        import torch
        if os.cpu_count() and torch.get_num_threads() < os.cpu_count():
            torch.set_num_threads(os.cpu_count())

        callback(0.5, "Loading detector", "Loading lightweight YOLO weights", {})
        self.detector = YOLO(self.options.detector_model)
        if self.options.enable_vehicle_classifier:
            callback(
                1.0,
                "Loading vehicle model",
                "Loading fine-grained car make/model classifier (first run downloads weights)",
                {},
            )
            self.vehicle_classifier = FineGrainedVehicleClassifier(
                self.options.vehicle_classifier_model,
                self.options.vehicle_classifier_confidence,
                self.options.device,
            )
            self.vehicle_classifier.load()
            if self.vehicle_classifier.load_error:
                self.warnings.append(self.vehicle_classifier.load_error)

    def _transcode_for_browser(self, raw_path: Path, input_path: Path, final_path: Path) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg = None

        if not ffmpeg:
            shutil.move(str(raw_path), str(final_path))
            self.warnings.append(
                "FFmpeg was not found. Output uses the OpenCV MP4 codec and may not play in every browser."
            )
            return

        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_path),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(final_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            shutil.move(str(raw_path), str(final_path))
            self.warnings.append(f"H.264 conversion failed; retained OpenCV output: {result.stderr[-300:]}")
            return
        raw_path.unlink(missing_ok=True)

    def process(
        self,
        input_path: Path,
        output_dir: Path,
        callback: ProgressCallback,
        cancel_check: CancelCheck,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._load_models(callback)
        if cancel_check():
            raise ProcessingCancelled("Processing cancelled before video analysis started")

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open the input video: {input_path.name}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0 or not math.isfinite(fps):
            fps = 30.0
            self.warnings.append("Input FPS metadata was invalid; 30 FPS was used for timing.")
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_width <= 0 or frame_height <= 0:
            capture.release()
            raise ValueError("Input video has invalid frame dimensions")

        raw_video_path = output_dir / "annotated_raw.mp4"
        final_video_path = output_dir / "annotated_video.mp4"
        tracks_csv_path = output_dir / "tracks.csv"
        detections_csv_path = output_dir / "frame_detections.csv"
        summary_json_path = output_dir / "summary.json"

        writer = cv2.VideoWriter(
            str(raw_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("OpenCV could not create the output video")

        estimator = DistanceSpeedEstimator(
            frame_width=frame_width,
            frame_height=frame_height,
            horizontal_fov_degrees=self.options.horizontal_fov_degrees,
            distance_scale=self.options.distance_scale,
            speed_scale=self.options.speed_scale,
            image_points=self.options.image_points if self.options.calibration_mode == "homography" else None,
            world_points=self.options.world_points if self.options.calibration_mode == "homography" else None,
        )
        calibration_label = "ROAD-PLANE CALIBRATED" if estimator.homography is not None else "MONOCULAR ESTIMATE"

        tracks: dict[int, TrackRecord] = {}
        class_detection_counts: Counter[str] = Counter()
        unique_class_tracks: dict[str, set[int]] = {}
        frame_index = 0
        started = time.perf_counter()
        update_interval = max(1, total_frames // 200) if total_frames else 10

        detection_file = detections_csv_path.open("w", newline="", encoding="utf-8") if self.options.save_frame_detections else None
        detection_writer = None
        if detection_file:
            detection_writer = csv.DictWriter(
                detection_file,
                fieldnames=[
                    "frame",
                    "timestamp_s",
                    "track_id",
                    "class",
                    "detection_confidence",
                    "vehicle_model",
                    "vehicle_model_confidence",
                    "distance_m",
                    "speed_kmh",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                ],
            )
            detection_writer.writeheader()

        try:
            while True:
                if cancel_check():
                    raise ProcessingCancelled("Processing cancelled by user")
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                timestamp_s = (frame_index - 1) / fps
                elapsed = max(time.perf_counter() - started, 1e-6)
                processing_fps = frame_index / elapsed

                yolo_device = None if self.options.device == "auto" else self.options.device
                results = self.detector.track(
                    frame,
                    persist=True,
                    tracker=str(TRACKER_CONFIG),
                    conf=self.options.confidence,
                    iou=self.options.iou,
                    imgsz=self.options.image_size,
                    device=yolo_device,
                    verbose=False,
                )
                result = results[0]
                detections: list[dict] = []
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.detach().cpu().numpy()
                    confidences = boxes.conf.detach().cpu().numpy()
                    classes = boxes.cls.detach().cpu().numpy().astype(int)
                    track_ids = (
                        boxes.id.detach().cpu().numpy().astype(int)
                        if boxes.id is not None
                        else np.full(len(xyxy), -1, dtype=int)
                    )

                    for detection_index, (coords, confidence, class_id, track_id) in enumerate(
                        zip(xyxy, confidences, classes, track_ids)
                    ):
                        bbox = tuple(float(value) for value in coords)
                        class_name = self._class_name(self.detector.names, int(class_id))
                        class_detection_counts[class_name] += 1
                        if track_id >= 0:
                            unique_class_tracks.setdefault(class_name, set()).add(int(track_id))
                            motion = estimator.update(int(track_id), class_name, bbox, timestamp_s)
                            if track_id not in tracks:
                                tracks[int(track_id)] = TrackRecord(
                                    track_id=int(track_id),
                                    class_name=class_name,
                                    first_seen_s=timestamp_s,
                                    last_seen_s=timestamp_s,
                                    trail=deque(maxlen=self.options.trail_length),
                                )
                            record = tracks[int(track_id)]
                            record.update(timestamp_s, float(confidence), motion)
                            record.trail.append((int((bbox[0] + bbox[2]) / 2), int(bbox[3])))
                        else:
                            distance = estimator.distance_only(class_name, bbox)
                            motion = MotionReading(distance, None, 0.0, distance, "fov-estimate")
                            record = None

                        detections.append(
                            {
                                "bbox": bbox,
                                "confidence": float(confidence),
                                "class_id": int(class_id),
                                "class_name": class_name,
                                "track_id": int(track_id),
                                "motion": motion,
                                "record": record,
                                "detection_index": detection_index,
                            }
                        )

                # Classify selected car crops in one batch and aggregate predictions over each track.
                classifier_targets: list[tuple[TrackRecord, np.ndarray]] = []
                if self.vehicle_classifier and self.vehicle_classifier.available:
                    for detection in detections:
                        record = detection["record"]
                        if detection["class_name"] != "car" or record is None:
                            continue
                        bbox = detection["bbox"]
                        crop_width = bbox[2] - bbox[0]
                        crop_height = bbox[3] - bbox[1]
                        due = frame_index - record.last_vehicle_sample_frame >= self.options.vehicle_sample_interval_frames
                        has_capacity = record.vehicle_consensus.samples < self.options.vehicle_max_samples_per_track
                        large_enough = min(crop_width, crop_height) >= self.options.vehicle_min_crop_pixels
                        if due and has_capacity and large_enough:
                            crop = self._safe_crop(frame, bbox)
                            if crop is not None:
                                classifier_targets.append((record, crop))
                                record.last_vehicle_sample_frame = frame_index

                    if classifier_targets:
                        predictions = self.vehicle_classifier.predict_batch(crop for _, crop in classifier_targets)
                        for (record, _), prediction in zip(classifier_targets, predictions):
                            record.vehicle_consensus.add(prediction)

                for detection in detections:
                    bbox = detection["bbox"]
                    motion: MotionReading = detection["motion"]
                    record: TrackRecord | None = detection["record"]
                    track_id = detection["track_id"]
                    class_name = detection["class_name"]
                    confidence = detection["confidence"]
                    color = class_color(detection["class_id"])
                    vehicle_label = None
                    vehicle_confidence = None
                    if record is not None:
                        vehicle_label, vehicle_confidence = self._format_vehicle_label(
                            record, self.options.vehicle_classifier_confidence
                        )

                    primary = f"#{track_id} {class_name}" if track_id >= 0 else class_name
                    if vehicle_label:
                        primary = f"#{track_id} {vehicle_label}"
                    if self.options.show_model_confidence:
                        primary += f"  {confidence:.0%}"
                    estimate_prefix = "~" if estimator.homography is None else ""
                    speed_text = (
                        f"{estimate_prefix}{motion.speed_kmh:.1f} km/h"
                        if motion.speed_kmh is not None
                        else "speed: acquiring"
                    )
                    secondary = f"{estimate_prefix}{motion.distance_m:.1f} m  |  {speed_text}"
                    int_bbox = tuple(round(value) for value in bbox)
                    if self.options.draw_trails and record is not None:
                        draw_trail(frame, record.trail, color)
                    draw_label(frame, int_bbox, [primary, secondary], color)

                    if detection_writer:
                        detection_writer.writerow(
                            {
                                "frame": frame_index,
                                "timestamp_s": round(timestamp_s, 3),
                                "track_id": track_id if track_id >= 0 else "",
                                "class": class_name,
                                "detection_confidence": round(confidence, 5),
                                "vehicle_model": vehicle_label or "",
                                "vehicle_model_confidence": round(vehicle_confidence, 5) if vehicle_confidence else "",
                                "distance_m": round(motion.distance_m, 3),
                                "speed_kmh": round(motion.speed_kmh, 3) if motion.speed_kmh is not None else "",
                                "x1": round(bbox[0], 1),
                                "y1": round(bbox[1], 1),
                                "x2": round(bbox[2], 1),
                                "y2": round(bbox[3], 1),
                            }
                        )

                draw_hud(
                    frame,
                    frame_index,
                    total_frames,
                    len(detections),
                    len(tracks),
                    processing_fps,
                    calibration_label,
                )
                draw_footer(frame)
                writer.write(frame)

                if frame_index == 1 or frame_index % update_interval == 0:
                    progress = min(94.0, (frame_index / total_frames * 93.0 + 2.0) if total_frames else 2.0)
                    vehicle_tracks = sum(1 for track in tracks.values() if track.class_name in {"car", "bus", "truck", "motorcycle"})
                    callback(
                        progress,
                        "Detecting and tracking",
                        f"Processed {frame_index:,} of {total_frames:,} frames",
                        {
                            "frames_processed": frame_index,
                            "total_frames": total_frames,
                            "processing_fps": round(processing_fps, 2),
                            "active_objects": len(detections),
                            "unique_tracks": len(tracks),
                            "vehicle_tracks": vehicle_tracks,
                        },
                    )
        finally:
            capture.release()
            writer.release()
            if detection_file:
                detection_file.close()

        callback(95.0, "Encoding output", "Creating a browser-compatible H.264 video", {})
        self._transcode_for_browser(raw_video_path, input_path, final_video_path)

        track_rows: list[dict] = []
        all_speeds: list[float] = []
        for track_id in sorted(tracks):
            record = tracks[track_id]
            vehicle_label, vehicle_confidence = self._format_vehicle_label(
                record, self.options.vehicle_classifier_confidence
            )
            average_speed = record.average_speed_kmh
            max_speed = record.max_speed_kmh
            if average_speed is not None:
                all_speeds.append(average_speed)
            track_rows.append(
                {
                    "track_id": record.track_id,
                    "class": record.class_name,
                    "vehicle_model": vehicle_label or "Uncertain" if record.class_name == "car" else "",
                    "vehicle_model_confidence": round(vehicle_confidence, 5) if vehicle_confidence else "",
                    "first_seen_s": round(record.first_seen_s, 3),
                    "last_seen_s": round(record.last_seen_s, 3),
                    "visible_duration_s": round(record.last_seen_s - record.first_seen_s, 3),
                    "frames_tracked": record.frame_count,
                    "average_detection_confidence": round(record.average_confidence, 5),
                    "nearest_distance_m": round(record.min_distance_m, 3) if math.isfinite(record.min_distance_m) else "",
                    "farthest_distance_m": round(record.max_distance_m, 3),
                    "average_speed_kmh": round(average_speed, 3) if average_speed is not None else "",
                    "max_speed_kmh": round(max_speed, 3) if max_speed is not None else "",
                }
            )

        with tracks_csv_path.open("w", newline="", encoding="utf-8") as track_file:
            fieldnames = list(track_rows[0].keys()) if track_rows else [
                "track_id",
                "class",
                "vehicle_model",
                "vehicle_model_confidence",
                "first_seen_s",
                "last_seen_s",
                "visible_duration_s",
                "frames_tracked",
                "average_detection_confidence",
                "nearest_distance_m",
                "farthest_distance_m",
                "average_speed_kmh",
                "max_speed_kmh",
            ]
            writer_csv = csv.DictWriter(track_file, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(track_rows)

        duration_s = frame_index / fps if fps else 0.0
        processing_seconds = max(time.perf_counter() - started, 1e-6)
        unique_by_class = {name: len(ids) for name, ids in sorted(unique_class_tracks.items())}
        summary = {
            "input": {
                "filename": input_path.name,
                "width": frame_width,
                "height": frame_height,
                "fps": round(fps, 4),
                "frames": frame_index,
                "duration_s": round(duration_s, 3),
            },
            "models": {
                "detector": self.options.detector_model,
                "tracker": "ByteTrack",
                "vehicle_classifier": self.options.vehicle_classifier_model
                if self.options.enable_vehicle_classifier
                else None,
            },
            "measurement": {
                "mode": "road-plane" if estimator.homography is not None else "monocular-fov-estimate",
                "horizontal_fov_degrees": self.options.horizontal_fov_degrees,
                "distance_scale": self.options.distance_scale,
                "speed_scale": self.options.speed_scale,
                "speed_reference": "camera-relative",
            },
            "results": {
                "unique_tracks": len(tracks),
                "vehicle_tracks": sum(
                    1 for track in tracks.values() if track.class_name in {"car", "bus", "truck", "motorcycle"}
                ),
                "unique_by_class": unique_by_class,
                "detection_events_by_class": dict(sorted(class_detection_counts.items())),
                "average_track_speed_kmh": round(float(np.median(all_speeds)), 2) if all_speeds else None,
                "max_track_speed_kmh": round(max((row["max_speed_kmh"] for row in track_rows if row["max_speed_kmh"] != ""), default=0.0), 2),
                "processing_seconds": round(processing_seconds, 2),
                "processing_fps": round(frame_index / processing_seconds, 2),
            },
            "warnings": self.warnings,
            "tracks": track_rows,
        }
        summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        callback(100.0, "Completed", "Annotated video and reports are ready", summary["results"])

        artifacts = {
            "video": str(final_video_path),
            "tracks_csv": str(tracks_csv_path),
            "summary_json": str(summary_json_path),
        }
        if self.options.save_frame_detections:
            artifacts["detections_csv"] = str(detections_csv_path)
        return {"summary": summary, "artifacts": artifacts, "warnings": self.warnings}
