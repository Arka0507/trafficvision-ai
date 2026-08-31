from __future__ import annotations

import colorsys
from collections import deque

import cv2
import numpy as np


def class_color(class_id: int) -> tuple[int, int, int]:
    hue = (class_id * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
    return int(blue * 255), int(green * 255), int(red * 255)


def draw_label(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    lines: list[str],
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    thickness = max(2, round(width / 800))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.42, min(0.72, width / 2100))
    text_thickness = max(1, round(width / 1500))
    line_height = int(21 * font_scale / 0.55)
    text_width = max(cv2.getTextSize(line, font, font_scale, text_thickness)[0][0] for line in lines)
    block_height = line_height * len(lines) + 8
    top = y1 - block_height
    if top < 0:
        top = y1
    right = min(width - 1, x1 + text_width + 12)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, top), (right, top + block_height), (8, 16, 27), -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
    cv2.rectangle(frame, (x1, top), (right, top + block_height), color, 1)
    for index, line in enumerate(lines):
        baseline_y = top + 6 + line_height * (index + 1) - 4
        cv2.putText(
            frame,
            line,
            (x1 + 6, baseline_y),
            font,
            font_scale,
            (245, 249, 255),
            text_thickness,
            cv2.LINE_AA,
        )


def draw_trail(frame: np.ndarray, points: deque, color: tuple[int, int, int]) -> None:
    if len(points) < 2:
        return
    point_list = list(points)
    for index in range(1, len(point_list)):
        fade = index / len(point_list)
        trail_color = tuple(int(channel * (0.35 + 0.65 * fade)) for channel in color)
        thickness = max(1, int(3 * fade))
        cv2.line(frame, point_list[index - 1], point_list[index], trail_color, thickness, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    frame_index: int,
    total_frames: int,
    active_objects: int,
    unique_tracks: int,
    processing_fps: float,
    calibration_label: str,
) -> None:
    _, width = frame.shape[:2]
    panel_width = min(520, int(width * 0.34))
    panel_height = 112
    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (18 + panel_width, 18 + panel_height), (5, 13, 24), -1)
    cv2.addWeighted(overlay, 0.83, frame, 0.17, 0, frame)
    cv2.rectangle(frame, (18, 18), (18 + panel_width, 18 + panel_height), (241, 181, 42), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.48, min(0.72, width / 2100))
    cv2.putText(frame, "TRAFFICVISION AI", (34, 48), font, scale + 0.12, (60, 232, 255), 2, cv2.LINE_AA)
    progress = (frame_index / total_frames * 100.0) if total_frames else 0.0
    cv2.putText(
        frame,
        f"Frame {frame_index:,}/{total_frames:,}  |  {progress:5.1f}%  |  {processing_fps:4.1f} FPS",
        (34, 76),
        font,
        scale,
        (236, 242, 250),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Active {active_objects}  |  Tracks {unique_tracks}  |  {calibration_label}",
        (34, 103),
        font,
        scale,
        (184, 201, 222),
        1,
        cv2.LINE_AA,
    )


def draw_footer(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    text = "Distance/speed are monocular estimates unless road-plane calibration is supplied"
    scale = max(0.38, min(0.56, width / 2600))
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max(12, width - text_width - 18)
    y = height - 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, y - text_height - 7), (width - 8, height - 7), (5, 13, 24), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (211, 220, 234), 1, cv2.LINE_AA)
