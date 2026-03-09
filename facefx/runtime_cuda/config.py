"""Configuration models for runtime_cuda."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration for the standalone runtime_cuda path."""

    camera: int = 1
    device: str = "auto"
    input_width: int = 1280
    input_height: int = 720
    camera_fps: float = 30.0
    debug_overlay: str = "off"
    max_frames: int = 0
    roi_expand: float = 0.18
    roi_min_size: int = 96
    control_point_count: int = 24
    warp_point_count: int = 0
    mask_preset: str = "face_hull_eye_mouth_cutout"
    mask_feather_px: int = 5
    mask_eye_scale: float = 1.35
    mask_mouth_scale: float = 1.20
    landmark_every: int = 3
    landmark_scale: float = 1.0
    landmark_smooth: float = 0.25
    color_match_enabled: bool = True
    color_match_every: int = 1
    color_ab_strength: float = 0.5
    shading_enabled: bool = True
    shading_every: int = 1
    shading_kernel: int = 51
    shading_clamp_min: float = 0.6
    shading_clamp_max: float = 1.6
    shading_strength: float = 0.35
    profile: bool = False
    dump_roi_overlay: bool = False
    roi_overlay_path: str = "scripts/baseline_artifacts/runtime_cuda_roi_overlay.png"
    composite_backend: str = "auto"
    composite_cuda_min_area: int = 180000


def config_from_namespace(args: Namespace) -> RuntimeConfig:
    """Build validated RuntimeConfig from argparse namespace."""
    if args.input_width < 1 or args.input_height < 1:
        raise ValueError("input dimensions must be positive")
    if args.max_frames < 0:
        raise ValueError("max_frames must be >= 0")
    camera_fps = float(getattr(args, "camera_fps", 30.0))
    if camera_fps <= 0.0:
        raise ValueError("camera_fps must be > 0")
    roi_expand = float(getattr(args, "roi_expand", 0.18))
    if not (0.0 <= roi_expand <= 1.0):
        raise ValueError("roi_expand must be in [0, 1]")
    roi_min_size = int(getattr(args, "roi_min_size", 96))
    if roi_min_size < 1:
        raise ValueError("roi_min_size must be >= 1")
    control_point_count = int(getattr(args, "control_point_count", 24))
    if control_point_count < 1:
        raise ValueError("control_point_count must be >= 1")
    warp_point_count = int(getattr(args, "warp_point_count", 0))
    if warp_point_count < 0:
        raise ValueError("warp_point_count must be >= 0")
    mask_preset = str(getattr(args, "mask_preset", "face_hull_eye_mouth_cutout"))
    if not mask_preset:
        raise ValueError("mask_preset must not be empty")
    mask_feather_px = int(getattr(args, "mask_feather_px", 5))
    if mask_feather_px < 0:
        raise ValueError("mask_feather_px must be >= 0")
    mask_eye_scale = float(getattr(args, "mask_eye_scale", 1.35))
    if mask_eye_scale <= 0.0:
        raise ValueError("mask_eye_scale must be > 0")
    mask_mouth_scale = float(getattr(args, "mask_mouth_scale", 1.20))
    if mask_mouth_scale <= 0.0:
        raise ValueError("mask_mouth_scale must be > 0")
    landmark_every = int(getattr(args, "landmark_every", 3))
    if landmark_every < 1:
        raise ValueError("landmark_every must be >= 1")
    landmark_scale = float(getattr(args, "landmark_scale", 1.0))
    if not (0.1 <= landmark_scale <= 1.0):
        raise ValueError("landmark_scale must be in [0.1, 1.0]")
    landmark_smooth = float(getattr(args, "landmark_smooth", 0.25))
    if not (0.0 <= landmark_smooth <= 0.95):
        raise ValueError("landmark_smooth must be in [0.0, 0.95]")
    color_match_enabled = bool(getattr(args, "color_match_enabled", True))
    color_match_every = int(getattr(args, "color_match_every", 1))
    if color_match_every < 1:
        raise ValueError("color_match_every must be >= 1")
    color_ab_strength = float(getattr(args, "color_ab_strength", 0.5))
    if not (0.0 <= color_ab_strength <= 1.0):
        raise ValueError("color_ab_strength must be in [0.0, 1.0]")
    shading_enabled = bool(getattr(args, "shading_enabled", True))
    shading_every = int(getattr(args, "shading_every", 1))
    if shading_every < 1:
        raise ValueError("shading_every must be >= 1")
    shading_kernel = int(getattr(args, "shading_kernel", 51))
    if shading_kernel < 1:
        raise ValueError("shading_kernel must be >= 1")
    shading_clamp_min = float(getattr(args, "shading_clamp_min", 0.6))
    shading_clamp_max = float(getattr(args, "shading_clamp_max", 1.6))
    if shading_clamp_min <= 0.0 or shading_clamp_min > shading_clamp_max:
        raise ValueError("invalid shading clamp range")
    shading_strength = float(getattr(args, "shading_strength", 0.35))
    if not (0.0 <= shading_strength <= 1.0):
        raise ValueError("shading_strength must be in [0.0, 1.0]")
    profile = bool(getattr(args, "profile", False))
    dump_roi_overlay = bool(getattr(args, "dump_roi_overlay", False))
    roi_overlay_path = str(
        getattr(args, "roi_overlay_path", "scripts/baseline_artifacts/runtime_cuda_roi_overlay.png")
    )
    if not roi_overlay_path:
        raise ValueError("roi_overlay_path must not be empty")
    composite_backend = str(getattr(args, "composite_backend", "auto")).lower()
    if composite_backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("composite_backend must be one of: auto|cpu|cuda")
    composite_cuda_min_area = int(getattr(args, "composite_cuda_min_area", 180000))
    if composite_cuda_min_area < 1:
        raise ValueError("composite_cuda_min_area must be >= 1")
    return RuntimeConfig(
        camera=1,
        device=str(args.device),
        input_width=int(args.input_width),
        input_height=int(args.input_height),
        camera_fps=camera_fps,
        debug_overlay=str(args.debug_overlay),
        max_frames=int(args.max_frames),
        roi_expand=roi_expand,
        roi_min_size=roi_min_size,
        control_point_count=control_point_count,
        warp_point_count=warp_point_count,
        mask_preset=mask_preset,
        mask_feather_px=mask_feather_px,
        mask_eye_scale=mask_eye_scale,
        mask_mouth_scale=mask_mouth_scale,
        landmark_every=landmark_every,
        landmark_scale=landmark_scale,
        landmark_smooth=landmark_smooth,
        color_match_enabled=color_match_enabled,
        color_match_every=color_match_every,
        color_ab_strength=color_ab_strength,
        shading_enabled=shading_enabled,
        shading_every=shading_every,
        shading_kernel=shading_kernel,
        shading_clamp_min=shading_clamp_min,
        shading_clamp_max=shading_clamp_max,
        shading_strength=shading_strength,
        profile=profile,
        dump_roi_overlay=dump_roi_overlay,
        roi_overlay_path=roi_overlay_path,
        composite_backend=composite_backend,
        composite_cuda_min_area=composite_cuda_min_area,
    )
