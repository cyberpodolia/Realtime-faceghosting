"""ROI compositing tests for runtime_cuda."""

from __future__ import annotations

import numpy as np
from facefx.runtime_cuda.composite import CompositeStats, composite_roi
from facefx.runtime_cuda.roi import Roi


def test_composite_roi_overwrites_with_full_alpha():
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    patch = np.full((10, 12, 3), 180, dtype=np.uint8)
    alpha = np.ones((10, 12), dtype=np.float32)
    roi = Roi(x=8, y=6, w=12, h=10)
    out, stats = composite_roi(frame, patch, alpha, roi, use_cuda=False)
    assert isinstance(stats, CompositeStats)
    assert stats.backend == "cpu"
    assert np.all(out[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] == 180)


def test_composite_roi_keeps_base_with_zero_alpha():
    frame = np.full((24, 24, 3), 99, dtype=np.uint8)
    patch = np.full((8, 8, 3), 220, dtype=np.uint8)
    alpha = np.zeros((8, 8), dtype=np.float32)
    roi = Roi(x=4, y=5, w=8, h=8)
    out, stats = composite_roi(frame, patch, alpha, roi, use_cuda=False)
    assert isinstance(stats, CompositeStats)
    assert np.all(out == frame)


def test_composite_roi_partial_alpha_blends():
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    patch = np.full((6, 6, 3), 200, dtype=np.uint8)
    alpha = np.full((6, 6), 0.5, dtype=np.float32)
    roi = Roi(x=3, y=4, w=6, h=6)
    out, _ = composite_roi(frame, patch, alpha, roi, use_cuda=False)
    roi_out = out[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
    assert np.all((roi_out >= 99) & (roi_out <= 101))


def test_composite_roi_cleans_low_alpha_edge_bleed():
    frame = np.full((20, 20, 3), (50, 60, 70), dtype=np.uint8)
    patch = np.full((8, 8, 3), (220, 140, 90), dtype=np.uint8)
    alpha = np.full((8, 8), 0.2, dtype=np.float32)
    roi = Roi(x=6, y=6, w=8, h=8)
    out, _ = composite_roi(frame, patch, alpha, roi, use_cuda=False)
    roi_out = out[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
    base_roi = frame[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
    naive = np.clip(
        base_roi.astype(np.float32) * (1.0 - alpha[:, :, None]) + patch.astype(np.float32) * alpha[:, :, None],
        0.0,
        255.0,
    ).astype(np.uint8)
    # Edge-cleaning should pull low-alpha output toward base compared to naive blend.
    err_clean = float(np.mean(np.abs(roi_out.astype(np.int16) - base_roi.astype(np.int16))))
    err_naive = float(np.mean(np.abs(naive.astype(np.int16) - base_roi.astype(np.int16))))
    assert err_clean < err_naive
