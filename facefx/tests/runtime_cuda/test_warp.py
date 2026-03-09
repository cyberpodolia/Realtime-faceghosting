"""Dense warp map tests for runtime_cuda."""

from __future__ import annotations

import numpy as np
from facefx.runtime_cuda.roi import Roi
from facefx.runtime_cuda.topology import mediapipe_simplices_for_count, mediapipe_simplices_for_indices
from facefx.runtime_cuda.warp import (
    DenseMapStats,
    RemapStats,
    build_dense_remap_idw,
    remap_patch_and_alpha,
    warp_patch_and_alpha_triangles,
)


def _sample_points() -> tuple[np.ndarray, np.ndarray]:
    dst = np.array(
        [
            [30.0, 20.0],
            [70.0, 20.0],
            [50.0, 45.0],
            [35.0, 60.0],
            [65.0, 60.0],
        ],
        dtype=np.float32,
    )
    src = dst + np.array([2.0, -1.5], dtype=np.float32)
    return src, dst


def test_build_dense_remap_shapes_dtype_and_stats():
    roi = Roi(x=20, y=10, w=80, h=64)
    src, dst = _sample_points()
    map_x, map_y, stats = build_dense_remap_idw(src, dst, roi, src_shape_hw=(120, 160))
    assert map_x.shape == (roi.h, roi.w)
    assert map_y.shape == (roi.h, roi.w)
    assert map_x.dtype == np.float32
    assert map_y.dtype == np.float32
    assert isinstance(stats, DenseMapStats)
    assert stats.width == roi.w
    assert stats.height == roi.h
    assert stats.build_ms >= 0.0
    assert stats.builder_backend in {"numpy", "native", "numpy+sim", "native+sim"}


def test_build_dense_remap_identity_when_src_equals_dst():
    roi = Roi(x=0, y=0, w=32, h=24)
    dst = np.array([[5.0, 5.0], [25.0, 5.0], [16.0, 18.0]], dtype=np.float32)
    map_x, map_y, _ = build_dense_remap_idw(dst, dst, roi, src_shape_hw=(24, 32))
    grid_x, grid_y = np.meshgrid(
        np.arange(roi.w, dtype=np.float32), np.arange(roi.h, dtype=np.float32)
    )
    assert np.allclose(map_x, grid_x, atol=1e-4)
    assert np.allclose(map_y, grid_y, atol=1e-4)


def test_build_dense_remap_clamps_to_source_shape():
    roi = Roi(x=10, y=10, w=40, h=30)
    dst = np.array([[12.0, 12.0], [45.0, 12.0], [30.0, 35.0]], dtype=np.float32)
    src = dst + np.array([300.0, -200.0], dtype=np.float32)
    map_x, map_y, _ = build_dense_remap_idw(src, dst, roi, src_shape_hw=(50, 60))
    assert float(np.min(map_x)) >= 0.0
    assert float(np.max(map_x)) <= 59.0
    assert float(np.min(map_y)) >= 0.0
    assert float(np.max(map_y)) <= 49.0


def test_build_dense_remap_deterministic():
    roi = Roi(x=8, y=6, w=64, h=48)
    src, dst = _sample_points()
    map_x1, map_y1, _ = build_dense_remap_idw(src, dst, roi)
    map_x2, map_y2, _ = build_dense_remap_idw(src, dst, roi)
    assert np.array_equal(map_x1, map_x2)
    assert np.array_equal(map_y1, map_y2)


def test_build_dense_remap_force_numpy_backend():
    roi = Roi(x=10, y=8, w=40, h=32)
    src, dst = _sample_points()
    map_x, map_y, stats = build_dense_remap_idw(
        src,
        dst,
        roi,
        prefer_native=False,
        similarity_prealign=False,
    )
    assert map_x.shape == (roi.h, roi.w)
    assert map_y.shape == (roi.h, roi.w)
    assert stats.builder_backend == "numpy"


def test_remap_patch_and_alpha_identity_cpu():
    src = np.zeros((32, 48, 3), dtype=np.uint8)
    for y in range(src.shape[0]):
        src[y, :, 0] = y
    alpha = np.zeros((32, 48), dtype=np.uint8)
    alpha[8:24, 10:30] = 255
    roi = Roi(x=10, y=8, w=20, h=16)
    map_x, map_y = np.meshgrid(
        np.arange(roi.x, roi.x + roi.w, dtype=np.float32),
        np.arange(roi.y, roi.y + roi.h, dtype=np.float32),
    )
    patch_roi, alpha_roi, stats = remap_patch_and_alpha(
        src,
        alpha,
        map_x,
        map_y,
        use_cuda=False,
    )
    assert isinstance(stats, RemapStats)
    assert stats.backend == "cpu"
    assert np.array_equal(patch_roi, src[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w])
    assert np.allclose(alpha_roi, 1.0)


def test_warp_patch_and_alpha_triangles_identity_cpu():
    roi = Roi(x=8, y=6, w=40, h=32)
    source = np.zeros((48, 64, 3), dtype=np.uint8)
    source[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w, 2] = 180
    alpha = np.zeros((48, 64), dtype=np.uint8)
    alpha[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = 255
    src_dst = np.array(
        [
            [8.0, 6.0],
            [47.0, 6.0],
            [47.0, 37.0],
            [8.0, 37.0],
        ],
        dtype=np.float32,
    )
    simplices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    patch_roi, alpha_roi, stats = warp_patch_and_alpha_triangles(
        source,
        alpha,
        src_dst,
        src_dst,
        simplices,
        roi,
        use_native=False,
    )
    assert isinstance(stats, RemapStats)
    assert stats.backend == "cpu"
    assert np.array_equal(patch_roi[:, :, 2], np.full((roi.h, roi.w), 180, dtype=np.uint8))
    assert np.allclose(alpha_roi, 1.0, atol=1e-3)


def test_mediapipe_simplices_for_count_returns_triangles():
    simplices = mediapipe_simplices_for_count(468)
    assert simplices.ndim == 2
    assert simplices.shape[1] == 3
    assert simplices.shape[0] > 100


def test_mediapipe_simplices_for_indices_reindexes_subset():
    indices = (10, 33, 61, 133, 152, 234)
    simplices = mediapipe_simplices_for_indices(indices)
    assert simplices.ndim == 2
    assert simplices.shape[1] == 3
    if simplices.shape[0] > 0:
        assert int(np.min(simplices)) >= 0
        assert int(np.max(simplices)) < len(indices)
