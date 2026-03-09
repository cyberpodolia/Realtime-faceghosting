"""Adaptive ROI helpers for runtime_cuda."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    w: int
    h: int


def _clip(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def _expand_axis(lo: float, hi: float, limit: int, expand: float, min_size: int) -> tuple[int, int]:
    span = max(1.0, hi - lo)
    pad = span * expand
    lo_i = int(np.floor(lo - pad))
    hi_i = int(np.ceil(hi + pad))
    size = hi_i - lo_i + 1
    if size < min_size:
        center = 0.5 * (lo_i + hi_i)
        half = 0.5 * (min_size - 1)
        lo_i = int(np.floor(center - half))
        hi_i = int(np.ceil(center + half))

    lo_i = _clip(lo_i, 0, max(0, limit - 1))
    hi_i = _clip(hi_i, lo_i, max(0, limit - 1))

    # If clipping reduced ROI below min_size and there is room, grow inward.
    while hi_i - lo_i + 1 < min_size and (lo_i > 0 or hi_i < limit - 1):
        if lo_i > 0:
            lo_i -= 1
        if hi_i < limit - 1 and hi_i - lo_i + 1 < min_size:
            hi_i += 1
    return lo_i, hi_i


def compute_adaptive_roi(
    points_xy: np.ndarray,
    frame_shape_hw: tuple[int, int],
    *,
    expand: float = 0.18,
    min_size: int = 96,
) -> Roi:
    """Compute deterministic adaptive ROI from 2D points."""
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("expected points with shape [N, 2]")
    if points.shape[0] == 0:
        raise ValueError("points must not be empty")
    h, w = int(frame_shape_hw[0]), int(frame_shape_hw[1])
    if h < 1 or w < 1:
        raise ValueError("frame shape must be positive")
    if min_size < 1:
        raise ValueError("min_size must be >= 1")

    x0, x1 = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
    y0, y1 = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
    xx0, xx1 = _expand_axis(x0, x1, w, expand, min_size)
    yy0, yy1 = _expand_axis(y0, y1, h, expand, min_size)
    return Roi(x=xx0, y=yy0, w=xx1 - xx0 + 1, h=yy1 - yy0 + 1)


def draw_roi_overlay(frame_bgr: np.ndarray, roi: Roi, anchors_xy: np.ndarray) -> np.ndarray:
    """Draw ROI rectangle and anchor points for debug review."""
    frame = frame_bgr.copy()
    cv2.rectangle(
        frame,
        (roi.x, roi.y),
        (roi.x + roi.w - 1, roi.y + roi.h - 1),
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    anchors = np.asarray(anchors_xy, dtype=np.float32)
    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError("expected anchors with shape [N, 2]")
    for i, pt in enumerate(anchors):
        cx = int(round(float(pt[0])))
        cy = int(round(float(pt[1])))
        cv2.circle(frame, (cx, cy), 2, (0, 180, 255), -1, cv2.LINE_AA)
        if i % 4 == 0:
            cv2.putText(
                frame,
                str(i),
                (cx + 4, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 200, 0),
                1,
                cv2.LINE_AA,
            )
    return frame
