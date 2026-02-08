"""Delaunay triangulation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay


@dataclass
class Triangulation:
    points: np.ndarray
    simplices: np.ndarray


def triangulate(points: np.ndarray) -> Triangulation:
    """Return triangulation for (N,2) float32 points."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be shape (N,2)")
    tri = Delaunay(points)
    return Triangulation(points=points, simplices=tri.simplices)
