"""Control-point extraction tests for runtime_cuda."""

from __future__ import annotations

import numpy as np
import pytest
from facefx.runtime_cuda.landmarks import (
    CONTROL_POINT_INDICES,
    LandmarkCadenceState,
    extract_control_points,
    select_landmark_indices,
    should_update_landmarks,
    smooth_landmarks,
)


def _make_landmarks(n: int = 478) -> np.ndarray:
    xs = np.linspace(320.0, 980.0, n, dtype=np.float32)
    ys = np.linspace(120.0, 620.0, n, dtype=np.float32)
    return np.stack([xs, ys], axis=1)


def test_extract_control_points_shape_and_count():
    landmarks = _make_landmarks()
    points = extract_control_points(landmarks)
    assert points.shape == (len(CONTROL_POINT_INDICES), 2)
    assert points.dtype == np.float32


def test_extract_control_points_deterministic_order():
    landmarks = _make_landmarks()
    points1 = extract_control_points(landmarks)
    points2 = extract_control_points(landmarks)
    assert np.array_equal(points1, points2)


def test_extract_control_points_rejects_short_input():
    landmarks = _make_landmarks(32)
    with pytest.raises(ValueError):
        extract_control_points(landmarks)


def test_extract_control_points_supports_extended_count():
    landmarks = _make_landmarks()
    points = extract_control_points(landmarks, max_count=64)
    assert points.shape == (64, 2)
    assert points.dtype == np.float32


def test_select_landmark_indices_supports_extended_count():
    selected = select_landmark_indices(478, max_count=96)
    assert selected.shape == (96,)
    assert selected.dtype == np.int32
    assert tuple(selected[: len(CONTROL_POINT_INDICES)]) == CONTROL_POINT_INDICES
    assert len(np.unique(selected)) == 96


def test_extract_control_points_rejects_invalid_max_count():
    landmarks = _make_landmarks()
    with pytest.raises(ValueError):
        extract_control_points(landmarks, max_count=0)


def test_should_update_landmarks_every_third_frame_with_hold():
    flags = [should_update_landmarks(i, 3, has_held_landmarks=True) for i in range(7)]
    assert flags == [True, False, False, True, False, False, True]


def test_should_update_landmarks_without_hold_is_true():
    assert should_update_landmarks(1, 3, has_held_landmarks=False) is True


def test_should_update_landmarks_force_overrides_cadence():
    assert should_update_landmarks(2, 3, has_held_landmarks=True, force=True) is True


def test_smooth_landmarks_blends_previous_and_current():
    previous = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    current = np.array([[30.0, 30.0], [50.0, 50.0]], dtype=np.float32)
    out = smooth_landmarks(previous, current, 0.5)
    assert np.allclose(out, np.array([[20.0, 20.0], [35.0, 35.0]], dtype=np.float32))


def test_landmark_cadence_state_hold_and_smoothing():
    state = LandmarkCadenceState(every=3, smooth=0.5)
    a = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    b = np.array([[30.0, 30.0], [50.0, 50.0]], dtype=np.float32)

    assert state.needs_update() is True
    r0 = state.update(a)
    assert r0.updated is True
    assert np.allclose(r0.landmarks_xy, a)

    assert state.needs_update() is False
    r1 = state.update(None)
    assert r1.updated is False
    assert np.allclose(r1.landmarks_xy, a)

    assert state.needs_update() is False
    r2 = state.update(None)
    assert r2.updated is False
    assert np.allclose(r2.landmarks_xy, a)

    assert state.needs_update() is True
    r3 = state.update(b)
    assert r3.updated is True
    assert np.allclose(
        r3.landmarks_xy,
        np.array([[20.0, 20.0], [35.0, 35.0]], dtype=np.float32),
    )
