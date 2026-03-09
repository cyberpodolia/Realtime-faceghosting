"""Config tests for runtime_cuda scaffold."""

from argparse import Namespace

import pytest
from facefx.runtime_cuda.config import RuntimeConfig, config_from_namespace


def test_config_from_namespace_defaults():
    args = Namespace(
        camera=7,
        device="auto",
        input_width=1280,
        input_height=720,
        camera_fps=30.0,
        debug_overlay="off",
        max_frames=0,
        roi_expand=0.18,
        roi_min_size=96,
        control_point_count=24,
        mask_preset="face_hull_eye_mouth_cutout",
        mask_feather_px=5,
        mask_eye_scale=1.35,
        mask_mouth_scale=1.2,
        landmark_every=3,
        landmark_scale=1.0,
        landmark_smooth=0.25,
        profile=False,
        dump_roi_overlay=False,
        roi_overlay_path="scripts/baseline_artifacts/runtime_cuda_roi_overlay.png",
    )
    cfg = config_from_namespace(args)
    assert isinstance(cfg, RuntimeConfig)
    assert cfg.camera == 1
    assert cfg.input_width == 1280
    assert cfg.input_height == 720
    assert cfg.camera_fps == 30.0
    assert cfg.control_point_count == 24
    assert cfg.mask_preset == "face_hull_eye_mouth_cutout"
    assert cfg.mask_feather_px == 5
    assert cfg.mask_eye_scale == 1.35
    assert cfg.mask_mouth_scale == 1.2
    assert cfg.landmark_every == 3
    assert cfg.landmark_scale == 1.0
    assert cfg.landmark_smooth == 0.25
    assert cfg.color_match_enabled is True
    assert cfg.color_ab_strength == 0.5
    assert cfg.shading_enabled is True
    assert cfg.shading_every == 1
    assert cfg.shading_kernel == 51
    assert cfg.shading_clamp_min == 0.6
    assert cfg.shading_clamp_max == 1.6
    assert cfg.composite_backend == "auto"
    assert cfg.composite_cuda_min_area == 180000


@pytest.mark.parametrize("width,height", [(0, 720), (1280, 0), (-1, 720)])
def test_config_from_namespace_rejects_invalid_dimensions(width: int, height: int):
    args = Namespace(
        camera=1,
        device="auto",
        input_width=width,
        input_height=height,
        camera_fps=30.0,
        debug_overlay="off",
        max_frames=0,
        roi_expand=0.18,
        roi_min_size=96,
        control_point_count=24,
        mask_preset="face_hull_eye_mouth_cutout",
        mask_feather_px=5,
        mask_eye_scale=1.35,
        mask_mouth_scale=1.2,
        landmark_every=1,
        landmark_scale=1.0,
        landmark_smooth=0.25,
        profile=False,
        dump_roi_overlay=False,
        roi_overlay_path="scripts/baseline_artifacts/runtime_cuda_roi_overlay.png",
    )
    with pytest.raises(ValueError):
        config_from_namespace(args)


@pytest.mark.parametrize("expand", [-0.1, 1.2])
def test_config_from_namespace_rejects_invalid_roi_expand(expand: float):
    args = Namespace(
        camera=1,
        device="auto",
        input_width=1280,
        input_height=720,
        camera_fps=30.0,
        debug_overlay="off",
        max_frames=0,
        roi_expand=expand,
        roi_min_size=96,
        control_point_count=24,
        mask_preset="face_hull_eye_mouth_cutout",
        mask_feather_px=5,
        mask_eye_scale=1.35,
        mask_mouth_scale=1.2,
        landmark_every=1,
        landmark_scale=1.0,
        landmark_smooth=0.25,
        profile=False,
        dump_roi_overlay=False,
        roi_overlay_path="scripts/baseline_artifacts/runtime_cuda_roi_overlay.png",
    )
    with pytest.raises(ValueError):
        config_from_namespace(args)


@pytest.mark.parametrize("every", [0, -1])
def test_config_from_namespace_rejects_invalid_landmark_every(every: int):
    args = Namespace(
        camera=1,
        device="auto",
        input_width=1280,
        input_height=720,
        camera_fps=30.0,
        debug_overlay="off",
        max_frames=0,
        roi_expand=0.18,
        roi_min_size=96,
        control_point_count=24,
        mask_preset="face_hull_eye_mouth_cutout",
        mask_feather_px=5,
        mask_eye_scale=1.35,
        mask_mouth_scale=1.2,
        landmark_every=every,
        landmark_scale=1.0,
        landmark_smooth=0.25,
        profile=False,
        dump_roi_overlay=False,
        roi_overlay_path="scripts/baseline_artifacts/runtime_cuda_roi_overlay.png",
    )
    with pytest.raises(ValueError):
        config_from_namespace(args)


def test_config_from_namespace_rejects_invalid_camera_fps():
    args = Namespace(
        camera=1,
        device="auto",
        input_width=1280,
        input_height=720,
        camera_fps=0.0,
        debug_overlay="off",
        max_frames=0,
        roi_expand=0.18,
        roi_min_size=96,
        control_point_count=24,
        mask_preset="face_hull_eye_mouth_cutout",
        mask_feather_px=5,
        mask_eye_scale=1.35,
        mask_mouth_scale=1.2,
        landmark_every=1,
        landmark_scale=1.0,
        landmark_smooth=0.25,
        profile=False,
        dump_roi_overlay=False,
        roi_overlay_path="scripts/baseline_artifacts/runtime_cuda_roi_overlay.png",
    )
    with pytest.raises(ValueError):
        config_from_namespace(args)


@pytest.mark.parametrize("ab_strength", [-0.1, 1.2])
def test_config_from_namespace_rejects_invalid_color_ab_strength(ab_strength: float):
    args = Namespace(
        camera=1,
        device="auto",
        input_width=1280,
        input_height=720,
        camera_fps=30.0,
        debug_overlay="off",
        max_frames=0,
        roi_expand=0.18,
        roi_min_size=96,
        control_point_count=24,
        mask_preset="face_hull_eye_mouth_cutout",
        mask_feather_px=5,
        mask_eye_scale=1.35,
        mask_mouth_scale=1.2,
        landmark_every=3,
        landmark_scale=1.0,
        landmark_smooth=0.25,
        color_ab_strength=ab_strength,
        profile=False,
        dump_roi_overlay=False,
        roi_overlay_path="scripts/baseline_artifacts/runtime_cuda_roi_overlay.png",
    )
    with pytest.raises(ValueError):
        config_from_namespace(args)


def test_config_from_namespace_rejects_invalid_shading_clamp():
    args = Namespace(
        camera=1,
        device="auto",
        input_width=1280,
        input_height=720,
        camera_fps=30.0,
        debug_overlay="off",
        max_frames=0,
        roi_expand=0.18,
        roi_min_size=96,
        control_point_count=24,
        mask_preset="face_hull_eye_mouth_cutout",
        mask_feather_px=5,
        mask_eye_scale=1.35,
        mask_mouth_scale=1.2,
        landmark_every=3,
        landmark_scale=1.0,
        landmark_smooth=0.25,
        shading_clamp_min=1.2,
        shading_clamp_max=0.8,
        profile=False,
        dump_roi_overlay=False,
        roi_overlay_path="scripts/baseline_artifacts/runtime_cuda_roi_overlay.png",
    )
    with pytest.raises(ValueError):
        config_from_namespace(args)


@pytest.mark.parametrize("shading_every", [0, -2])
def test_config_from_namespace_rejects_invalid_shading_every(shading_every: int):
    args = Namespace(
        camera=1,
        device="auto",
        input_width=1280,
        input_height=720,
        camera_fps=30.0,
        debug_overlay="off",
        max_frames=0,
        roi_expand=0.18,
        roi_min_size=96,
        control_point_count=24,
        mask_preset="face_hull_eye_mouth_cutout",
        mask_feather_px=5,
        mask_eye_scale=1.35,
        mask_mouth_scale=1.2,
        landmark_every=3,
        landmark_scale=1.0,
        landmark_smooth=0.25,
        shading_every=shading_every,
        profile=False,
        dump_roi_overlay=False,
        roi_overlay_path="scripts/baseline_artifacts/runtime_cuda_roi_overlay.png",
    )
    with pytest.raises(ValueError):
        config_from_namespace(args)


def test_config_from_namespace_rejects_invalid_composite_backend():
    args = Namespace(
        camera=1,
        device="auto",
        input_width=1280,
        input_height=720,
        camera_fps=30.0,
        debug_overlay="off",
        max_frames=0,
        roi_expand=0.18,
        roi_min_size=96,
        control_point_count=24,
        mask_preset="face_hull_eye_mouth_cutout",
        mask_feather_px=5,
        mask_eye_scale=1.35,
        mask_mouth_scale=1.2,
        landmark_every=3,
        landmark_scale=1.0,
        landmark_smooth=0.25,
        composite_backend="metal",
        profile=False,
        dump_roi_overlay=False,
        roi_overlay_path="scripts/baseline_artifacts/runtime_cuda_roi_overlay.png",
    )
    with pytest.raises(ValueError):
        config_from_namespace(args)


def test_config_from_namespace_rejects_invalid_composite_cuda_min_area():
    args = Namespace(
        camera=1,
        device="auto",
        input_width=1280,
        input_height=720,
        camera_fps=30.0,
        debug_overlay="off",
        max_frames=0,
        roi_expand=0.18,
        roi_min_size=96,
        control_point_count=24,
        mask_preset="face_hull_eye_mouth_cutout",
        mask_feather_px=5,
        mask_eye_scale=1.35,
        mask_mouth_scale=1.2,
        landmark_every=3,
        landmark_scale=1.0,
        landmark_smooth=0.25,
        composite_cuda_min_area=0,
        profile=False,
        dump_roi_overlay=False,
        roi_overlay_path="scripts/baseline_artifacts/runtime_cuda_roi_overlay.png",
    )
    with pytest.raises(ValueError):
        config_from_namespace(args)
