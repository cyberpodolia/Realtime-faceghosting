"""Production mask preset for runtime_cuda."""

from __future__ import annotations

import cv2
import numpy as np

from .roi import Roi

MASK_PRESET_NAME = "face_hull_eye_mouth_cutout"

FACE_OVAL_IDX: tuple[int, ...] = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)
FOREHEAD_IDX: tuple[int, ...] = (
    10,
    109,
    67,
    103,
    54,
    21,
    71,
    68,
    104,
    69,
    108,
    151,
    338,
    297,
    332,
    284,
    251,
    301,
    298,
    333,
    299,
    337,
    336,
    296,
    334,
    293,
    300,
    107,
    66,
    105,
    63,
    70,
)
LEFT_EYE_IDX: tuple[int, ...] = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_IDX: tuple[int, ...] = (362, 385, 387, 263, 373, 380)
MOUTH_IDX: tuple[int, ...] = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
REQUIRED_MAX_INDEX = max(*FACE_OVAL_IDX, *FOREHEAD_IDX, *LEFT_EYE_IDX, *RIGHT_EYE_IDX, *MOUTH_IDX)


def _ensure_landmarks(landmarks_xy: np.ndarray) -> np.ndarray:
    lm = np.asarray(landmarks_xy, dtype=np.float32)
    if lm.ndim != 2 or lm.shape[1] != 2:
        raise ValueError("landmarks_xy must have shape [N, 2]")
    if lm.shape[0] <= REQUIRED_MAX_INDEX:
        raise ValueError(
            f"landmarks_xy requires at least {REQUIRED_MAX_INDEX + 1} points for mask preset"
        )
    return lm


def _expand_poly(poly: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0.0:
        raise ValueError("scale must be > 0")
    center = np.mean(poly, axis=0, keepdims=True)
    return (center + (poly - center) * scale).astype(np.float32)


def _to_roi_poly(poly_xy: np.ndarray, roi: Roi) -> np.ndarray:
    out = poly_xy.astype(np.float32).copy()
    out[:, 0] -= float(roi.x)
    out[:, 1] -= float(roi.y)
    return out


def _blur_mask_u8(mask_u8: np.ndarray, feather_px: int) -> np.ndarray:
    if feather_px <= 0:
        return mask_u8
    k = max(3, int(feather_px) * 2 + 1)
    return cv2.GaussianBlur(mask_u8, (k, k), 0)


def _edge_falloff(mask_f32: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0")
    return np.clip(np.power(np.clip(mask_f32, 0.0, 1.0), gamma), 0.0, 1.0).astype(np.float32)


def build_production_mask_preset(
    landmarks_xy: np.ndarray,
    roi: Roi,
    frame_shape_hw: tuple[int, int],
    *,
    eye_scale: float = 1.35,
    mouth_scale: float = 1.20,
    feather_px: int = 5,
) -> np.ndarray:
    """Build broad face-coverage mask with eye/mouth cutouts in ROI space."""
    if roi.w < 1 or roi.h < 1:
        raise ValueError("roi dimensions must be positive")
    h, w = int(frame_shape_hw[0]), int(frame_shape_hw[1])
    if h < 1 or w < 1:
        raise ValueError("frame_shape_hw must be positive")
    if roi.x < 0 or roi.y < 0 or roi.x + roi.w > w or roi.y + roi.h > h:
        raise ValueError("roi is out of frame bounds")

    lm = _ensure_landmarks(landmarks_xy)
    face_mask_u8 = np.zeros((roi.h, roi.w), dtype=np.uint8)
    face_oval = lm[np.array(FACE_OVAL_IDX, dtype=np.int32)].astype(np.float32)
    cv2.fillPoly(
        face_mask_u8,
        [np.round(_to_roi_poly(face_oval, roi)).astype(np.int32)],
        255,
        cv2.LINE_AA,
    )

    if feather_px > 0:
        # Pull the outer contour slightly inward before feathering to hide the visible seam.
        erode_px = max(1, int(np.ceil(feather_px * 0.5)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erode_px * 2 + 1, erode_px * 2 + 1),
        )
        face_mask_u8 = cv2.erode(face_mask_u8, kernel, iterations=1)
    face_mask_u8 = _blur_mask_u8(face_mask_u8, feather_px * 2 if feather_px > 0 else 0)

    eye_l = _expand_poly(lm[np.array(LEFT_EYE_IDX, dtype=np.int32)], eye_scale)
    eye_r = _expand_poly(lm[np.array(RIGHT_EYE_IDX, dtype=np.int32)], eye_scale)
    mouth = _expand_poly(lm[np.array(MOUTH_IDX, dtype=np.int32)], mouth_scale)
    cutout_mask_u8 = np.zeros((roi.h, roi.w), dtype=np.uint8)
    for poly in (eye_l, eye_r, mouth):
        cv2.fillConvexPoly(
            cutout_mask_u8,
            np.round(_to_roi_poly(poly, roi)).astype(np.int32),
            255,
            cv2.LINE_AA,
        )
    cutout_mask_u8 = _blur_mask_u8(cutout_mask_u8, feather_px)

    mask = np.clip(
        face_mask_u8.astype(np.float32) / 255.0 - cutout_mask_u8.astype(np.float32) / 255.0,
        0.0,
        1.0,
    )
    # Compress the low-alpha outer ring so the face edge falls into the webcam
    # image more seamlessly instead of leaving a bright contour.
    return _edge_falloff(mask, 1.6)
