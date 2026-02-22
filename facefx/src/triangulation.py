"""Delaunay triangulation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay


@dataclass
class Triangulation:
    points: np.ndarray
    simplices: np.ndarray


@dataclass
class TopologyCache:
    frozen_simplices: np.ndarray | None = None
    frozen_point_count: int | None = None


def _delaunay_simplices(points: np.ndarray) -> np.ndarray:
    tri = Delaunay(points)
    return np.asarray(tri.simplices, dtype=np.int32)


@lru_cache(maxsize=1)
def _mediapipe_tessellation_simplices() -> np.ndarray:
    edges = {tuple(sorted(edge)) for edge in mp.solutions.face_mesh.FACEMESH_TESSELATION}
    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    triangles: set[tuple[int, int, int]] = set()
    for a, neighbors in adj.items():
        for b in neighbors:
            if b <= a:
                continue
            common = adj[a].intersection(adj[b])
            for c in common:
                if c <= b:
                    continue
                triangles.add((a, b, c))

    if not triangles:
        raise RuntimeError("Failed to derive triangles from MediaPipe tessellation")
    return np.asarray(sorted(triangles), dtype=np.int32)


def _mediapipe_simplices_for_count(n_points: int) -> np.ndarray:
    simplices = _mediapipe_tessellation_simplices()
    if n_points <= 0:
        return np.empty((0, 3), dtype=np.int32)
    # refine_landmarks adds iris points beyond 468; tessellation covers base landmarks only.
    keep = np.all(simplices < n_points, axis=1)
    return simplices[keep]


def triangulate(
    points: np.ndarray,
    *,
    topology: str = "delaunay",
    cache: TopologyCache | None = None,
) -> Triangulation:
    """Return triangulation for (N,2) float32 points."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be shape (N,2)")
    if topology == "delaunay":
        simplices = _delaunay_simplices(points)
    elif topology == "frozen":
        if (
            cache is not None
            and cache.frozen_simplices is not None
            and cache.frozen_point_count == int(points.shape[0])
        ):
            simplices = cache.frozen_simplices
        else:
            simplices = _delaunay_simplices(points)
            if cache is not None:
                cache.frozen_simplices = simplices
                cache.frozen_point_count = int(points.shape[0])
    elif topology == "mediapipe":
        simplices = _mediapipe_simplices_for_count(int(points.shape[0]))
    else:
        raise ValueError(f"Unsupported topology: {topology}")
    return Triangulation(points=points, simplices=simplices)
