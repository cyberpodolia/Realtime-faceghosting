"""Adaptive ROI tests for runtime_cuda."""

from __future__ import annotations

import numpy as np
import pytest
from facefx.runtime_cuda.landmarks import extract_control_points
from facefx.runtime_cuda.roi import Roi, compute_adaptive_roi, draw_roi_overlay


def _make_landmarks(n: int = 478) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x = 360.0 + 520.0 * t
    y = 140.0 + 420.0 * t
    return np.stack([x, y], axis=1)


def test_compute_adaptive_roi_within_frame_bounds():
    frame_hw = (720, 1280)
    control = extract_control_points(_make_landmarks())
    roi = compute_adaptive_roi(control, frame_hw, expand=0.20, min_size=96)
    assert isinstance(roi, Roi)
    assert roi.x >= 0 and roi.y >= 0
    assert roi.w >= 96 and roi.h >= 96
    assert roi.x + roi.w <= frame_hw[1]
    assert roi.y + roi.h <= frame_hw[0]


def test_compute_adaptive_roi_deterministic():
    frame_hw = (720, 1280)
    control = extract_control_points(_make_landmarks())
    roi1 = compute_adaptive_roi(control, frame_hw, expand=0.18, min_size=96)
    roi2 = compute_adaptive_roi(control, frame_hw, expand=0.18, min_size=96)
    assert roi1 == roi2


def test_draw_roi_overlay_outputs_same_shape():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    control = extract_control_points(_make_landmarks())
    roi = compute_adaptive_roi(control, frame.shape[:2], expand=0.2, min_size=96)
    out = draw_roi_overlay(frame, roi, control)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


def test_compute_adaptive_roi_rejects_empty_points():
    with pytest.raises(ValueError):
        compute_adaptive_roi(np.zeros((0, 2), dtype=np.float32), (720, 1280))
