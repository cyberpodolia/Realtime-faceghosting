"""ROI helper tests for region-bounded pipeline."""

from __future__ import annotations

import numpy as np

from facefx.main import _roi_from_polys, _shift_poly


def test_roi_from_polys_clips_to_frame_and_applies_padding():
    poly_a = np.array([[10, 10], [20, 10], [20, 20]], dtype=np.float32)
    poly_b = np.array([[30, 15], [35, 20], [25, 25]], dtype=np.float32)
    roi = _roi_from_polys([poly_a, poly_b], (40, 50), pad=2)
    assert roi == (8, 8, 30, 20)


def test_roi_from_polys_handles_edges():
    poly = np.array([[0, 0], [5, 0], [5, 5]], dtype=np.float32)
    roi = _roi_from_polys([poly], (10, 10), pad=10)
    assert roi == (0, 0, 10, 10)


def test_shift_poly_translates_points():
    poly = np.array([[10, 12], [15, 17]], dtype=np.float32)
    shifted = _shift_poly(poly, 3, 4)
    assert np.array_equal(shifted, np.array([[7, 8], [12, 13]], dtype=np.float32))
