"""Triangulation topology behavior tests."""

from __future__ import annotations

import numpy as np

from facefx.src import triangulation as tri_mod


def _grid_points(nx: int = 20, ny: int = 24) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    return pts


def test_frozen_topology_reuses_cached_simplices(monkeypatch):
    calls = {"count": 0}
    real_delaunay = tri_mod._delaunay_simplices

    def counted(points):
        calls["count"] += 1
        return real_delaunay(points)

    monkeypatch.setattr(tri_mod, "_delaunay_simplices", counted)
    cache = tri_mod.TopologyCache()
    pts = _grid_points()

    a = tri_mod.triangulate(pts, topology="frozen", cache=cache)
    b = tri_mod.triangulate(pts + 0.1, topology="frozen", cache=cache)

    assert calls["count"] == 1
    assert a.simplices.shape == b.simplices.shape
    assert np.array_equal(a.simplices, b.simplices)


def test_mediapipe_topology_works_with_base_and_refined_landmark_counts():
    base_pts = np.zeros((468, 2), dtype=np.float32)
    refined_pts = np.zeros((478, 2), dtype=np.float32)

    tri_base = tri_mod.triangulate(base_pts, topology="mediapipe")
    tri_refined = tri_mod.triangulate(refined_pts, topology="mediapipe")

    assert tri_base.simplices.ndim == 2
    assert tri_base.simplices.shape[1] == 3
    assert tri_base.simplices.shape[0] > 0
    assert int(tri_base.simplices.max()) < 468
    assert np.array_equal(tri_base.simplices, tri_refined.simplices)
