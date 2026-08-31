from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

# Typical physical heights in metres. These are priors, not measurements.
REFERENCE_HEIGHTS_M: dict[str, float] = {
    "person": 1.70,
    "bicycle": 1.10,
    "car": 1.50,
    "motorcycle": 1.20,
    "airplane": 4.00,
    "bus": 3.20,
    "train": 3.50,
    "truck": 3.20,
    "boat": 2.00,
    "traffic light": 0.90,
    "fire hydrant": 0.75,
    "stop sign": 0.75,
    "parking meter": 1.30,
    "bench": 0.85,
    "bird": 0.25,
    "cat": 0.30,
    "dog": 0.60,
    "horse": 1.60,
    "sheep": 1.00,
    "cow": 1.40,
    "elephant": 3.00,
    "bear": 1.50,
    "zebra": 1.40,
    "giraffe": 4.50,
    "backpack": 0.45,
    "umbrella": 0.90,
    "handbag": 0.35,
    "tie": 0.45,
    "suitcase": 0.65,
    "frisbee": 0.27,
    "skis": 1.70,
    "snowboard": 1.50,
    "sports ball": 0.22,
    "kite": 0.80,
    "baseball bat": 0.85,
    "baseball glove": 0.30,
    "skateboard": 0.15,
    "surfboard": 0.50,
    "tennis racket": 0.70,
    "bottle": 0.25,
    "wine glass": 0.20,
    "cup": 0.12,
    "fork": 0.20,
    "knife": 0.25,
    "spoon": 0.20,
    "bowl": 0.12,
    "banana": 0.20,
    "apple": 0.08,
    "sandwich": 0.10,
    "orange": 0.08,
    "broccoli": 0.20,
    "carrot": 0.15,
    "hot dog": 0.10,
    "pizza": 0.35,
    "donut": 0.10,
    "cake": 0.20,
    "chair": 0.90,
    "couch": 0.90,
    "potted plant": 0.60,
    "bed": 0.60,
    "dining table": 0.75,
    "toilet": 0.75,
    "tv": 0.60,
    "laptop": 0.25,
    "mouse": 0.05,
    "remote": 0.18,
    "keyboard": 0.15,
    "cell phone": 0.15,
    "microwave": 0.35,
    "oven": 0.90,
    "toaster": 0.25,
    "sink": 0.35,
    "refrigerator": 1.80,
    "book": 0.25,
    "clock": 0.30,
    "vase": 0.30,
    "scissors": 0.20,
    "teddy bear": 0.40,
    "hair drier": 0.25,
    "toothbrush": 0.18,
}

MOBILE_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}

PLAUSIBLE_MAX_SPEED_KMH = {
    "person": 45.0,
    "bicycle": 90.0,
    "car": 220.0,
    "motorcycle": 260.0,
    "airplane": 950.0,
    "bus": 150.0,
    "train": 350.0,
    "truck": 160.0,
    "boat": 180.0,
    "bird": 160.0,
    "cat": 55.0,
    "dog": 65.0,
    "horse": 90.0,
    "sheep": 40.0,
    "cow": 45.0,
    "elephant": 45.0,
    "bear": 65.0,
    "zebra": 70.0,
    "giraffe": 60.0,
}


@dataclass
class MotionReading:
    distance_m: float
    speed_kmh: float | None
    world_x_m: float
    world_y_m: float
    method: str


@dataclass
class _MotionState:
    smoothed_distance: float | None = None
    smoothed_world: tuple[float, float] | None = None
    positions: deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=180))
    recent_speeds: deque[float] = field(default_factory=lambda: deque(maxlen=21))


class DistanceSpeedEstimator:
    """Estimate monocular range and camera-relative ground speed per track."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        horizontal_fov_degrees: float = 70.0,
        distance_scale: float = 1.0,
        speed_scale: float = 1.0,
        image_points: list[list[float]] | None = None,
        world_points: list[list[float]] | None = None,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focal_length_px = frame_width / (
            2.0 * math.tan(math.radians(horizontal_fov_degrees) / 2.0)
        )
        self.distance_scale = distance_scale
        self.speed_scale = speed_scale
        self.states: dict[int, _MotionState] = defaultdict(_MotionState)
        self.homography: np.ndarray | None = None
        if image_points and world_points:
            source = np.asarray(image_points, dtype=np.float32)
            destination = np.asarray(world_points, dtype=np.float32)
            self.homography = cv2.getPerspectiveTransform(source, destination)

    def distance_only(self, class_name: str, bbox: tuple[float, float, float, float]) -> float:
        _, y1, _, y2 = bbox
        pixel_height = max(float(y2 - y1), 2.0)
        reference_height = REFERENCE_HEIGHTS_M.get(class_name, 1.0)
        distance = (reference_height * self.focal_length_px / pixel_height) * self.distance_scale
        return float(np.clip(distance, 0.5, 300.0))

    def _world_position(
        self,
        center_x: float,
        bottom_y: float,
        distance_m: float,
    ) -> tuple[float, float, str]:
        if self.homography is not None:
            point = np.asarray([[[center_x, bottom_y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, self.homography)[0, 0]
            return float(transformed[0]), float(transformed[1]), "road-plane"

        lateral = distance_m * ((center_x - self.frame_width / 2.0) / self.focal_length_px)
        return float(lateral), float(distance_m), "fov-estimate"

    def update(
        self,
        track_id: int,
        class_name: str,
        bbox: tuple[float, float, float, float],
        timestamp_s: float,
    ) -> MotionReading:
        x1, _, x2, y2 = bbox
        state = self.states[track_id]
        raw_distance = self.distance_only(class_name, bbox)
        alpha_distance = 0.16
        if state.smoothed_distance is None:
            state.smoothed_distance = raw_distance
        else:
            state.smoothed_distance = (
                alpha_distance * raw_distance + (1.0 - alpha_distance) * state.smoothed_distance
            )

        center_x = (x1 + x2) / 2.0
        world_x, world_y, method = self._world_position(center_x, y2, state.smoothed_distance)
        alpha_world = 0.20 if self.homography is None else 0.35
        if state.smoothed_world is None:
            state.smoothed_world = (world_x, world_y)
        else:
            state.smoothed_world = (
                alpha_world * world_x + (1.0 - alpha_world) * state.smoothed_world[0],
                alpha_world * world_y + (1.0 - alpha_world) * state.smoothed_world[1],
            )

        current_x, current_y = state.smoothed_world
        state.positions.append((timestamp_s, current_x, current_y))
        speed_kmh: float | None = None

        if class_name not in MOBILE_CLASSES:
            return MotionReading(
                distance_m=state.smoothed_distance,
                speed_kmh=0.0,
                world_x_m=current_x,
                world_y_m=current_y,
                method=method,
            )

        # A longer baseline suppresses frame-to-frame bounding-box jitter.
        if len(state.positions) >= 4:
            target_window = 1.50 if self.homography is None else 0.55
            previous = None
            for sample in state.positions:
                if timestamp_s - sample[0] >= target_window:
                    previous = sample
                else:
                    break
            if previous is not None:
                dt = timestamp_s - previous[0]
                displacement = math.hypot(current_x - previous[1], current_y - previous[2])
                instantaneous = displacement / max(dt, 1e-6) * 3.6 * self.speed_scale
                maximum_speed = PLAUSIBLE_MAX_SPEED_KMH.get(class_name, 240.0)
                existing_median = statistics.median(state.recent_speeds) if state.recent_speeds else None
                is_outlier = (
                    existing_median is not None
                    and instantaneous > max(existing_median * 3.0, existing_median + 35.0)
                )
                if 0.0 <= instantaneous <= maximum_speed and not is_outlier:
                    if instantaneous < 0.8:
                        instantaneous = 0.0
                    state.recent_speeds.append(instantaneous)
                if len(state.recent_speeds) >= 3:
                    speed_kmh = float(statistics.median(state.recent_speeds))

        return MotionReading(
            distance_m=state.smoothed_distance,
            speed_kmh=speed_kmh,
            world_x_m=current_x,
            world_y_m=current_y,
            method=method,
        )
