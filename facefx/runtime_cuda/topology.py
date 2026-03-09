"""Topology helpers for runtime_cuda face warp."""

from __future__ import annotations

from functools import lru_cache

import mediapipe as mp
import numpy as np


@lru_cache(maxsize=1)
def _mediapipe_tessellation_simplices() -> np.ndarray:
    edges = {tuple(sorted(edge)) for edge in mp.solutions.face_mesh.FACEMESH_TESSELATION}
    adjacency: dict[int, set[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    triangles: set[tuple[int, int, int]] = set()
    for a, neighbors in adjacency.items():
        for b in neighbors:
            if b <= a:
                continue
            common = adjacency[a].intersection(adjacency[b])
            for c in common:
                if c <= b:
                    continue
                triangles.add((a, b, c))

    if not triangles:
        raise RuntimeError("failed to derive triangles from MediaPipe tessellation")
    return np.asarray(sorted(triangles), dtype=np.int32)


def mediapipe_simplices_for_count(n_points: int) -> np.ndarray:
    """Return MediaPipe tessellation triangles clipped to the first `n_points` landmarks."""
    count = int(n_points)
    if count <= 0:
        return np.empty((0, 3), dtype=np.int32)
    simplices = _mediapipe_tessellation_simplices()
    return simplices[np.all(simplices < count, axis=1)]


@lru_cache(maxsize=32)
def _mediapipe_simplices_for_index_tuple(indices: tuple[int, ...]) -> np.ndarray:
    if len(indices) < 3:
        return np.empty((0, 3), dtype=np.int32)
    index_map = {src_idx: dst_idx for dst_idx, src_idx in enumerate(indices)}
    simplices = _mediapipe_tessellation_simplices()
    keep: list[tuple[int, int, int]] = []
    for tri in simplices:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if a in index_map and b in index_map and c in index_map:
            keep.append((index_map[a], index_map[b], index_map[c]))
    if not keep:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(keep, dtype=np.int32)


def mediapipe_simplices_for_indices(indices: np.ndarray | tuple[int, ...] | list[int]) -> np.ndarray:
    """Return MediaPipe tessellation triangles reindexed to an arbitrary landmark subset."""
    arr = tuple(int(i) for i in indices)
    if any(i < 0 for i in arr):
        raise ValueError("indices must be >= 0")
    return _mediapipe_simplices_for_index_tuple(arr)
