"""Production mask preset tests for runtime_cuda."""

from __future__ import annotations

import numpy as np
import pytest
from facefx.runtime_cuda.mask import FACE_OVAL_IDX, FOREHEAD_IDX, build_production_mask_preset
from facefx.runtime_cuda.roi import Roi


def _resample_poly(points_xy: np.ndarray, count: int) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    pos = np.linspace(0.0, float(len(pts)), num=count, endpoint=False, dtype=np.float32)
    out = np.empty((count, 2), dtype=np.float32)
    for i, p in enumerate(pos):
        idx0 = int(np.floor(p)) % len(pts)
        idx1 = (idx0 + 1) % len(pts)
        t = p - np.floor(p)
        out[i] = pts[idx0] * (1.0 - t) + pts[idx1] * t
    return out


def _make_landmarks() -> np.ndarray:
    n = 478
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    cx, cy = 640.0, 360.0
    rx, ry = 220.0, 280.0
    lm = np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1).astype(np.float32)

    # Eye and mouth points inside the face hull.
    left_eye = np.array(
        [[560, 320], [570, 310], [585, 308], [600, 320], [585, 332], [570, 330]],
        dtype=np.float32,
    )
    face_oval = _resample_poly(
        np.array(
            [
                [520, 320],
                [540, 280],
                [580, 250],
                [620, 232],
                [660, 232],
                [700, 250],
                [740, 280],
                [760, 320],
                [750, 380],
                [735, 445],
                [715, 500],
                [690, 540],
                [660, 570],
                [640, 585],
                [620, 570],
                [590, 540],
                [565, 500],
                [545, 445],
                [530, 380],
            ],
            dtype=np.float32,
        ),
        len(FACE_OVAL_IDX),
    )
    right_eye = np.array(
        [[680, 320], [695, 310], [710, 308], [724, 320], [710, 332], [695, 330]],
        dtype=np.float32,
    )
    mouth = np.array(
        [
            [570, 430],
            [590, 418],
            [615, 412],
            [640, 410],
            [665, 412],
            [690, 420],
            [705, 440],
            [690, 458],
            [665, 468],
            [640, 472],
            [610, 468],
        ],
        dtype=np.float32,
    )
    forehead = _resample_poly(
        np.array(
            [
                [520, 320],
                [540, 290],
                [580, 265],
                [620, 248],
                [660, 248],
                [700, 265],
                [740, 290],
                [760, 320],
                [720, 332],
                [680, 338],
                [640, 340],
                [600, 338],
                [560, 332],
            ],
            dtype=np.float32,
        ),
        len(FOREHEAD_IDX),
    )

    # Indices used by preset.
    lm[np.array(FACE_OVAL_IDX, dtype=np.int32)] = face_oval
    lm[np.array(FOREHEAD_IDX, dtype=np.int32)] = forehead
    lm[[33, 160, 158, 133, 153, 144]] = left_eye
    lm[[362, 385, 387, 263, 373, 380]] = right_eye
    lm[[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]] = mouth
    return lm


def test_build_production_mask_shape_range():
    lm = _make_landmarks()
    roi = Roi(x=360, y=120, w=560, h=520)
    mask = build_production_mask_preset(lm, roi, (720, 1280))
    assert mask.shape == (roi.h, roi.w)
    assert mask.dtype == np.float32
    assert float(np.min(mask)) >= 0.0
    assert float(np.max(mask)) <= 1.0


def test_build_production_mask_is_broad_face_cover():
    lm = _make_landmarks()
    roi = Roi(x=360, y=120, w=560, h=520)
    mask = build_production_mask_preset(lm, roi, (720, 1280), feather_px=0)

    left_eye_center = np.mean(lm[[33, 133]], axis=0)
    mouth_center = np.mean(lm[[61, 291]], axis=0)
    forehead_point = np.array([640.0, 340.0], dtype=np.float32)
    jaw_point = np.array([640.0, 560.0], dtype=np.float32)

    le = mask[int(left_eye_center[1] - roi.y), int(left_eye_center[0] - roi.x)]
    mc = mask[int(mouth_center[1] - roi.y), int(mouth_center[0] - roi.x)]
    fh = mask[int(forehead_point[1] - roi.y), int(forehead_point[0] - roi.x)]
    jaw = mask[int(jaw_point[1] - roi.y), int(jaw_point[0] - roi.x)]

    assert float(le) < 0.05
    assert float(mc) < 0.05
    assert float(fh) > 0.5
    assert float(jaw) > 0.5


def test_build_production_mask_rejects_short_landmark_array():
    lm = np.zeros((100, 2), dtype=np.float32)
    roi = Roi(x=0, y=0, w=100, h=100)
    with pytest.raises(ValueError):
        build_production_mask_preset(lm, roi, (100, 100))
