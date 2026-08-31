import math

from backend.processing.geometry import DistanceSpeedEstimator


def test_pinhole_distance_uses_reference_height_and_fov():
    estimator = DistanceSpeedEstimator(1920, 1080, horizontal_fov_degrees=70)
    reading = estimator.distance_only("car", (100, 200, 400, 350))
    focal = 1920 / (2 * math.tan(math.radians(70) / 2))
    expected = 1.5 * focal / 150
    assert reading == pytest.approx(expected, rel=1e-5)


def test_speed_becomes_available_after_temporal_window():
    estimator = DistanceSpeedEstimator(1920, 1080, horizontal_fov_degrees=70)
    result = None
    for index in range(100):
        # Move the same-size car laterally over two seconds.
        x = 300 + index * 4
        result = estimator.update(7, "car", (x, 300, x + 220, 450), index / 50)
    assert result is not None
    assert result.speed_kmh is not None
    assert 0 < result.speed_kmh < 240


def test_four_point_homography_uses_road_plane_mode():
    estimator = DistanceSpeedEstimator(
        100,
        100,
        image_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
        world_points=[[0, 0], [10, 0], [10, 10], [0, 10]],
    )
    result = estimator.update(1, "person", (40, 20, 60, 50), 0.0)
    assert result.method == "road-plane"
    assert result.world_x_m == pytest.approx(5.0, abs=0.1)
    assert result.world_y_m == pytest.approx(5.0, abs=0.1)


import pytest
