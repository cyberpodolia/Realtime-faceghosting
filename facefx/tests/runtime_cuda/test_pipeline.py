"""Pipeline tests for runtime_cuda scaffold."""

import numpy as np
import pytest
from facefx.runtime_cuda.config import RuntimeConfig
from facefx.runtime_cuda.mask import FACE_OVAL_IDX, FOREHEAD_IDX
from facefx.runtime_cuda.pipeline import RuntimePipeline, RuntimeStageTimings
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


def test_pipeline_passthrough_returns_copy():
    cfg = RuntimeConfig()
    pipe = RuntimePipeline(cfg)
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    out = pipe.process(frame)
    assert np.array_equal(out, frame)
    assert out is not frame
    assert pipe.frame_index == 1


def test_pipeline_rejects_non_bgr_shape():
    cfg = RuntimeConfig()
    pipe = RuntimePipeline(cfg)
    with pytest.raises(ValueError):
        pipe.process(np.zeros((32, 48), dtype=np.uint8))


def test_pipeline_dense_roi_identity_path():
    cfg = RuntimeConfig(color_match_enabled=False, shading_enabled=False)
    pipe = RuntimePipeline(cfg)
    frame = np.zeros((64, 80, 3), dtype=np.uint8)
    source = np.zeros((64, 80, 3), dtype=np.uint8)
    source[:, :, 1] = 150
    alpha = np.zeros((64, 80), dtype=np.uint8)
    roi = Roi(x=20, y=16, w=24, h=18)
    alpha[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = 255

    dst = np.array(
        [[22.0, 18.0], [40.0, 18.0], [32.0, 30.0], [24.0, 32.0], [38.0, 32.0]],
        dtype=np.float32,
    )
    src = dst.copy()
    out, timings = pipe.process_dense_roi(
        frame,
        source,
        alpha,
        src,
        dst,
        roi,
        use_cuda=False,
    )
    assert isinstance(timings, RuntimeStageTimings)
    assert timings.remap_backend == "cpu"
    assert timings.composite_backend == "cpu"
    assert timings.map_build_ms >= 0.0
    assert timings.remap_ms >= 0.0
    assert timings.color_ms == 0.0
    assert timings.shading_ms == 0.0
    assert timings.composite_ms >= 0.0
    roi_out = out[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
    assert np.all(roi_out[:, :, 1] >= 140)


def test_pipeline_dense_roi_applies_broad_composite_mask():
    cfg = RuntimeConfig(
        mask_feather_px=0,
        color_match_enabled=False,
        shading_enabled=False,
    )
    pipe = RuntimePipeline(cfg)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    source = np.full((720, 1280, 3), 200, dtype=np.uint8)
    alpha = np.ones((720, 1280), dtype=np.uint8) * 255
    roi = Roi(x=360, y=120, w=560, h=520)

    n = 478
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    cx, cy = 640.0, 360.0
    rx, ry = 220.0, 280.0
    lm = np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1).astype(np.float32)
    lm[[33, 160, 158, 133, 153, 144]] = np.array(
        [[560, 320], [570, 310], [585, 308], [600, 320], [585, 332], [570, 330]], dtype=np.float32
    )
    lm[[362, 385, 387, 263, 373, 380]] = np.array(
        [[680, 320], [695, 310], [710, 308], [724, 320], [710, 332], [695, 330]], dtype=np.float32
    )
    lm[[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]] = np.array(
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
    lm[np.array(FOREHEAD_IDX, dtype=np.int32)] = _resample_poly(
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
    lm[np.array(FACE_OVAL_IDX, dtype=np.int32)] = _resample_poly(
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

    dst = np.array(
        [[520.0, 250.0], [760.0, 250.0], [640.0, 360.0], [560.0, 460.0], [720.0, 460.0]],
        dtype=np.float32,
    )
    src = dst.copy()
    out, _timings = pipe.process_dense_roi(
        frame,
        source,
        alpha,
        src,
        dst,
        roi,
        landmarks_xy=lm,
        use_cuda=False,
    )

    eye_center = np.mean(lm[[33, 133]], axis=0).astype(int)
    mouth_center = np.mean(lm[[61, 291]], axis=0).astype(int)
    forehead = np.array([640, 280], dtype=int)
    jaw = np.array([640, 560], dtype=int)
    eye_val = out[eye_center[1], eye_center[0], 0]
    mouth_val = out[mouth_center[1], mouth_center[0], 0]
    forehead_val = out[forehead[1], forehead[0], 0]
    jaw_val = out[jaw[1], jaw[0], 0]
    assert int(eye_val) < 20
    assert int(mouth_val) < 20
    assert int(forehead_val) > 150
    assert int(jaw_val) > 150


def test_pipeline_triangle_roi_identity_path():
    cfg = RuntimeConfig(color_match_enabled=False, shading_enabled=False)
    pipe = RuntimePipeline(cfg)
    frame = np.zeros((72, 96, 3), dtype=np.uint8)
    source = np.zeros((72, 96, 3), dtype=np.uint8)
    source[:, :, 0] = 200
    alpha = np.zeros((72, 96), dtype=np.uint8)
    roi = Roi(x=16, y=12, w=32, h=24)
    alpha[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = 255
    points = np.array(
        [
            [16.0, 12.0],
            [47.0, 12.0],
            [47.0, 35.0],
            [16.0, 35.0],
        ],
        dtype=np.float32,
    )
    simplices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    out, timings = pipe.process_triangle_roi(
        frame,
        source,
        alpha,
        points,
        points,
        simplices,
        roi,
        use_native=False,
        use_cuda=False,
    )
    assert isinstance(timings, RuntimeStageTimings)
    assert timings.map_build_ms == 0.0
    assert timings.remap_backend == "cpu"
    assert timings.composite_backend == "cpu"
    roi_out = out[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
    assert np.all(roi_out[:, :, 0] >= 190)


def test_pipeline_triangle_roi_holds_postprocess_state_between_refreshes():
    cfg = RuntimeConfig(color_match_enabled=False, shading_enabled=True, shading_kernel=21)
    pipe = RuntimePipeline(cfg)
    frame = np.zeros((72, 96, 3), dtype=np.uint8)
    source = np.full((72, 96, 3), 120, dtype=np.uint8)
    alpha = np.zeros((72, 96), dtype=np.uint8)
    roi = Roi(x=16, y=12, w=32, h=24)
    alpha[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = 255
    frame[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = np.linspace(
        10, 220, roi.w, dtype=np.uint8
    )[None, :, None]
    points = np.array(
        [
            [16.0, 12.0],
            [47.0, 12.0],
            [47.0, 35.0],
            [16.0, 35.0],
        ],
        dtype=np.float32,
    )
    simplices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    out_refresh, timings_refresh = pipe.process_triangle_roi(
        frame,
        source,
        alpha,
        points,
        points,
        simplices,
        roi,
        enable_color_match=False,
        enable_shading=True,
        use_native=False,
        use_cuda=False,
    )
    out_hold, timings_hold = pipe.process_triangle_roi(
        frame,
        source,
        alpha,
        points,
        points,
        simplices,
        roi,
        enable_color_match=False,
        enable_shading=False,
        use_native=False,
        use_cuda=False,
    )
    assert np.array_equal(out_refresh, out_hold)
    assert timings_refresh.shading_ms >= 0.0
    assert timings_hold.shading_ms == 0.0


def test_pipeline_dense_roi_includes_color_and_shading_timings():
    cfg = RuntimeConfig(color_match_enabled=True, shading_enabled=True, shading_kernel=21)
    pipe = RuntimePipeline(cfg)
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    frame[:, :, 2] = 90
    source = np.zeros((96, 96, 3), dtype=np.uint8)
    source[:, :, 1] = 180
    alpha = np.zeros((96, 96), dtype=np.uint8)
    roi = Roi(x=24, y=20, w=40, h=36)
    alpha[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = 255

    dst = np.array(
        [[30.0, 24.0], [58.0, 24.0], [44.0, 36.0], [32.0, 50.0], [56.0, 50.0]],
        dtype=np.float32,
    )
    src = dst.copy()
    _out, timings = pipe.process_dense_roi(
        frame,
        source,
        alpha,
        src,
        dst,
        roi,
        use_cuda=False,
    )
    assert timings.color_ms >= 0.0
    assert timings.shading_ms >= 0.0
    assert timings.color_backend == "cpu"


def test_pipeline_landmark_cadence_hold_and_smoothing():
    cfg = RuntimeConfig(landmark_every=3, landmark_smooth=0.5)
    pipe = RuntimePipeline(cfg)
    a = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    b = np.array([[30.0, 30.0], [50.0, 50.0]], dtype=np.float32)

    assert pipe.should_update_landmarks() is True
    r0 = pipe.update_landmarks(a)
    assert r0.updated is True
    assert np.allclose(r0.landmarks_xy, a)

    assert pipe.should_update_landmarks() is False
    r1 = pipe.update_landmarks(None)
    assert r1.updated is False
    assert np.allclose(r1.landmarks_xy, a)

    assert pipe.should_update_landmarks() is False
    r2 = pipe.update_landmarks(None)
    assert r2.updated is False
    assert np.allclose(r2.landmarks_xy, a)

    assert pipe.should_update_landmarks() is True
    r3 = pipe.update_landmarks(b)
    assert r3.updated is True
    assert np.allclose(
        r3.landmarks_xy,
        np.array([[20.0, 20.0], [35.0, 35.0]], dtype=np.float32),
    )


def test_pipeline_landmark_cadence_force_update():
    pipe = RuntimePipeline(RuntimeConfig(landmark_every=3))
    pipe.update_landmarks(np.array([[1.0, 1.0]], dtype=np.float32))
    assert pipe.should_update_landmarks() is False
    assert pipe.should_update_landmarks(force=True) is True
