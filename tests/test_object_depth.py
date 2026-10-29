from types import SimpleNamespace

import numpy as np
import pytest

from src.drivesense.object_depth import ObjectDepthEstimator


@pytest.fixture
def estimator():
    return ObjectDepthEstimator()


@pytest.fixture
def tracked_object():
    return SimpleNamespace(
        track_id=7,
        label="car",
        x1=10,
        y1=10,
        x2=50,
        y2=50,
        confidence=0.95,
    )


def test_estimates_depth_statistics(estimator, tracked_object):
    depth_map = np.full((100, 100), 0.5, dtype=np.float32)

    result = estimator.estimate_object_depth(depth_map, tracked_object)

    assert result is not None
    assert result.track_id == 7
    assert result.label == "car"
    assert result.median_depth == pytest.approx(0.5)
    assert result.mean_depth == pytest.approx(0.5)
    assert result.valid_pixels > 0


def test_ignores_nonfinite_depth_values(estimator, tracked_object):
    depth_map = np.full((100, 100), 0.4, dtype=np.float32)

    # The object's central region includes some invalid pixels.
    depth_map[20:40, 20:40] = np.nan

    result = estimator.estimate_object_depth(depth_map, tracked_object)

    assert result is not None
    assert np.isfinite(result.median_depth)
    assert np.isfinite(result.mean_depth)


def test_returns_none_for_empty_depth_map(estimator, tracked_object):
    depth_map = np.empty((0, 0), dtype=np.float32)

    assert estimator.estimate_object_depth(depth_map, tracked_object) is None


def test_returns_none_for_tiny_bounding_box(estimator):
    obj = SimpleNamespace(
        track_id=1,
        label="person",
        x1=10,
        y1=10,
        x2=12,
        y2=12,
    )

    depth_map = np.ones((100, 100), dtype=np.float32)

    assert estimator.estimate_object_depth(depth_map, obj) is None


def test_returns_none_for_excessively_large_box(estimator):
    obj = SimpleNamespace(
        track_id=1,
        label="car",
        x1=0,
        y1=0,
        x2=90,
        y2=90,
    )

    depth_map = np.ones((100, 100), dtype=np.float32)

    assert estimator.estimate_object_depth(depth_map, obj) is None


def test_returns_none_when_depth_region_has_no_finite_values(
    estimator,
    tracked_object,
):
    depth_map = np.full((100, 100), np.nan, dtype=np.float32)

    assert estimator.estimate_object_depth(depth_map, tracked_object) is None
