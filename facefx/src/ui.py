"""Simple UI helpers."""

from __future__ import annotations

import cv2
import numpy as np


def draw_label(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int] = (10, 30),
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def wait_key(delay_ms: int = 1) -> int:
    return cv2.waitKey(delay_ms) & 0xFF
