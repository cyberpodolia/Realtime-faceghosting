"""Simple UI helpers."""

from __future__ import annotations

import cv2
import numpy as np


def draw_label(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int] = (10, 30),
    color: tuple[int, int, int] = (0, 0, 0),
    *,
    font_face: int = cv2.FONT_HERSHEY_DUPLEX,
    font_scale: float = 0.7,
    thickness: int = 2,
    bg: bool = True,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    bg_pad: int = 6,
) -> None:
    """Draw a HUD-style label.

    OpenCV only ships Hershey vector fonts; `FONT_HERSHEY_DUPLEX` is a readable
    sans-serif option that looks closer to Helvetica than SIMPLEX for HUD text.
    """

    x, y = pos
    if bg:
        (tw, th), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
        x0 = max(0, x - bg_pad)
        y0 = max(0, y - th - bg_pad)
        x1 = min(frame.shape[1] - 1, x + tw + bg_pad)
        y1 = min(frame.shape[0] - 1, y + baseline + bg_pad)
        cv2.rectangle(frame, (x0, y0), (x1, y1), bg_color, thickness=-1)

    cv2.putText(
        frame,
        text,
        (x, y),
        font_face,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def wait_key(delay_ms: int = 1) -> int:
    return cv2.waitKey(delay_ms) & 0xFF
