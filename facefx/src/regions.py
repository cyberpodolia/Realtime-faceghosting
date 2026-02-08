"""Region mask helpers."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np


def region_polygon(landmarks_xy: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    """Return convex hull polygon for selected landmark indices."""
    pts = landmarks_xy[list(indices)]
    hull = cv2.convexHull(pts.astype(np.float32))
    return hull.reshape(-1, 2)


def feather_mask(shape_hw: tuple[int, int], poly: np.ndarray, feather_px: int) -> np.ndarray:
    """Return feathered mask in [0,1] for polygon."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly.astype(np.int32), 255, lineType=cv2.LINE_AA)
    if feather_px > 0:
        k = feather_px * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask.astype(np.float32) / 255.0
