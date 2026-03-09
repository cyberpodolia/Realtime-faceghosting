"""Runtime pipeline for runtime_cuda."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .color import (
    ColorTransferStateLab,
    ShadingStateLab,
    match_color_and_shading_roi_cached,
)
from .composite import composite_roi
from .config import RuntimeConfig
from .landmarks import LandmarkCadenceResult, LandmarkCadenceState
from .mask import MASK_PRESET_NAME, build_production_mask_preset
from .roi import Roi
from .warp import build_dense_remap_idw, remap_patch_and_alpha, warp_patch_and_alpha_triangles


@dataclass(frozen=True)
class RuntimeStageTimings:
    map_build_ms: float
    remap_ms: float
    color_ms: float
    shading_ms: float
    composite_ms: float
    total_ms: float
    remap_backend: str
    color_backend: str
    composite_backend: str


@dataclass(frozen=True)
class RuntimeDebugPass:
    name: str
    frame_bgr: np.ndarray


class RuntimePipeline:
    """Runtime pipeline with passthrough, dense, and triangle ROI paths."""

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._frame_index = 0
        self._landmark_state = LandmarkCadenceState(
            every=config.landmark_every,
            smooth=config.landmark_smooth,
        )
        self._color_state_curr: ColorTransferStateLab | None = None
        self._shading_state_curr: ShadingStateLab | None = None
        self._last_debug_passes: list[RuntimeDebugPass] = []
        self._debug_passes_enabled = False

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def config(self) -> RuntimeConfig:
        return self._config

    def reset_postprocess_state(self) -> None:
        self._color_state_curr = None
        self._shading_state_curr = None

    def debug_passes(self) -> list[RuntimeDebugPass]:
        return [RuntimeDebugPass(p.name, p.frame_bgr.copy()) for p in self._last_debug_passes]

    def set_debug_passes_enabled(self, enabled: bool) -> None:
        self._debug_passes_enabled = bool(enabled)

    def _build_debug_passes(
        self,
        frame_bgr: np.ndarray,
        roi: Roi,
        warped_patch_roi: np.ndarray,
        warped_alpha_roi: np.ndarray,
        corrected_patch_roi: np.ndarray,
        out_bgr: np.ndarray,
    ) -> list[RuntimeDebugPass]:
        base = frame_bgr.copy()
        warped_view = frame_bgr.copy()
        warped_view[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = warped_patch_roi

        alpha_u8 = np.clip(warped_alpha_roi * 255.0, 0.0, 255.0).astype(np.uint8)
        alpha_heat = np.zeros_like(frame_bgr)
        alpha_col = cv2.applyColorMap(alpha_u8, cv2.COLORMAP_TURBO)
        alpha_heat[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = alpha_col
        alpha_view = np.clip(
            frame_bgr.astype(np.float32) * 0.55 + alpha_heat.astype(np.float32) * 0.45,
            0.0,
            255.0,
        ).astype(np.uint8)

        color_view = frame_bgr.copy()
        color_view[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = corrected_patch_roi

        return [
            RuntimeDebugPass("input", base),
            RuntimeDebugPass("warp", warped_view),
            RuntimeDebugPass("alpha", alpha_view),
            RuntimeDebugPass("color", color_view),
            RuntimeDebugPass("final", out_bgr.copy()),
        ]

    def should_update_landmarks(self, *, force: bool = False) -> bool:
        return self._landmark_state.needs_update(force=force)

    def update_landmarks(self, latest_landmarks_xy: np.ndarray | None) -> LandmarkCadenceResult:
        return self._landmark_state.update(latest_landmarks_xy)

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("expected BGR frame with shape [H, W, 3]")
        self._frame_index += 1
        out = frame_bgr.copy()
        self._last_debug_passes = [RuntimeDebugPass("final", out.copy())]
        return out

    def process_dense_roi(
        self,
        frame_bgr: np.ndarray,
        source_bgr: np.ndarray,
        source_alpha: np.ndarray,
        src_points_xy: np.ndarray,
        dst_points_xy: np.ndarray,
        roi: Roi,
        *,
        landmarks_xy: np.ndarray | None = None,
        enable_color_match: bool | None = None,
        enable_shading: bool | None = None,
        use_cuda: bool = True,
    ) -> tuple[np.ndarray, RuntimeStageTimings]:
        """Run dense-map remap + ROI compositing path."""

        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must have shape [H, W, 3]")
        if source_bgr.ndim != 3 or source_bgr.shape[2] != 3:
            raise ValueError("source_bgr must have shape [H, W, 3]")

        total_t0 = time.perf_counter()
        map_x, map_y, map_stats = build_dense_remap_idw(
            src_points_xy,
            dst_points_xy,
            roi,
            src_shape_hw=source_bgr.shape[:2],
        )
        warped_patch_roi, warped_alpha_roi, remap_stats = remap_patch_and_alpha(
            source_bgr,
            source_alpha,
            map_x,
            map_y,
            use_cuda=use_cuda,
        )
        if landmarks_xy is not None and self._config.mask_preset == MASK_PRESET_NAME:
            preset_mask = build_production_mask_preset(
                landmarks_xy,
                roi,
                frame_bgr.shape[:2],
                eye_scale=self._config.mask_eye_scale,
                mouth_scale=self._config.mask_mouth_scale,
                feather_px=self._config.mask_feather_px,
            )
            warped_alpha_roi = np.clip(warped_alpha_roi * preset_mask, 0.0, 1.0)
        target_roi = frame_bgr[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
        refresh_color = (
            self._config.color_match_enabled if enable_color_match is None else bool(enable_color_match)
        )
        refresh_shading = (
            self._config.shading_enabled if enable_shading is None else bool(enable_shading)
        )
        corrected_patch_roi, color_stats, self._color_state_curr, self._shading_state_curr = (
            match_color_and_shading_roi_cached(
            warped_patch_roi,
            target_roi,
            warped_alpha_roi,
            refresh_color=refresh_color,
            refresh_shading=refresh_shading,
            color_state=self._color_state_curr,
            shading_state=self._shading_state_curr,
            color_match_enabled=self._config.color_match_enabled,
            ab_strength=self._config.color_ab_strength,
            shading_enabled=self._config.shading_enabled,
            shading_kernel=self._config.shading_kernel,
            shading_clamp=(
                self._config.shading_clamp_min,
                self._config.shading_clamp_max,
            ),
            use_cuda=use_cuda,
        ))
        out, composite_stats = composite_roi(
            frame_bgr,
            corrected_patch_roi,
            warped_alpha_roi,
            roi,
            use_cuda=use_cuda,
            backend_pref=self._config.composite_backend,
            cuda_min_area=self._config.composite_cuda_min_area,
        )
        if self._debug_passes_enabled:
            self._last_debug_passes = self._build_debug_passes(
                frame_bgr,
                roi,
                warped_patch_roi,
                warped_alpha_roi,
                corrected_patch_roi,
                out,
            )
        else:
            self._last_debug_passes = [RuntimeDebugPass("final", out.copy())]
        self._frame_index += 1
        timings = RuntimeStageTimings(
            map_build_ms=map_stats.build_ms,
            remap_ms=remap_stats.remap_ms,
            color_ms=color_stats.color_ms,
            shading_ms=color_stats.shading_ms,
            composite_ms=composite_stats.composite_ms,
            total_ms=(time.perf_counter() - total_t0) * 1000.0,
            remap_backend=remap_stats.backend,
            color_backend=color_stats.backend,
            composite_backend=composite_stats.backend,
        )
        return out, timings

    def process_triangle_roi(
        self,
        frame_bgr: np.ndarray,
        source_bgr: np.ndarray,
        source_alpha: np.ndarray,
        src_points_xy: np.ndarray,
        dst_points_xy: np.ndarray,
        simplices: np.ndarray,
        roi: Roi,
        *,
        landmarks_xy: np.ndarray | None = None,
        enable_color_match: bool | None = None,
        enable_shading: bool | None = None,
        use_native: bool = True,
        use_cuda: bool = True,
    ) -> tuple[np.ndarray, RuntimeStageTimings]:
        """Run production triangle-warp + ROI compositing path."""

        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must have shape [H, W, 3]")
        if source_bgr.ndim != 3 or source_bgr.shape[2] != 3:
            raise ValueError("source_bgr must have shape [H, W, 3]")

        total_t0 = time.perf_counter()
        warped_patch_roi, warped_alpha_roi, remap_stats = warp_patch_and_alpha_triangles(
            source_bgr,
            source_alpha,
            src_points_xy,
            dst_points_xy,
            simplices,
            roi,
            use_native=use_native,
        )
        if landmarks_xy is not None and self._config.mask_preset == MASK_PRESET_NAME:
            preset_mask = build_production_mask_preset(
                landmarks_xy,
                roi,
                frame_bgr.shape[:2],
                eye_scale=self._config.mask_eye_scale,
                mouth_scale=self._config.mask_mouth_scale,
                feather_px=self._config.mask_feather_px,
            )
            warped_alpha_roi = np.clip(warped_alpha_roi * preset_mask, 0.0, 1.0)
        target_roi = frame_bgr[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
        refresh_color = (
            self._config.color_match_enabled if enable_color_match is None else bool(enable_color_match)
        )
        refresh_shading = (
            self._config.shading_enabled if enable_shading is None else bool(enable_shading)
        )
        corrected_patch_roi, color_stats, self._color_state_curr, self._shading_state_curr = (
            match_color_and_shading_roi_cached(
            warped_patch_roi,
            target_roi,
            warped_alpha_roi,
            refresh_color=refresh_color,
            refresh_shading=refresh_shading,
            color_state=self._color_state_curr,
            shading_state=self._shading_state_curr,
            color_match_enabled=self._config.color_match_enabled,
            ab_strength=self._config.color_ab_strength,
            shading_enabled=self._config.shading_enabled,
            shading_kernel=self._config.shading_kernel,
            shading_clamp=(
                self._config.shading_clamp_min,
                self._config.shading_clamp_max,
            ),
            use_cuda=use_cuda,
        ))
        out, composite_stats = composite_roi(
            frame_bgr,
            corrected_patch_roi,
            warped_alpha_roi,
            roi,
            use_cuda=use_cuda,
            backend_pref=self._config.composite_backend,
            cuda_min_area=self._config.composite_cuda_min_area,
        )
        if self._debug_passes_enabled:
            self._last_debug_passes = self._build_debug_passes(
                frame_bgr,
                roi,
                warped_patch_roi,
                warped_alpha_roi,
                corrected_patch_roi,
                out,
            )
        else:
            self._last_debug_passes = [RuntimeDebugPass("final", out.copy())]
        self._frame_index += 1
        timings = RuntimeStageTimings(
            map_build_ms=0.0,
            remap_ms=remap_stats.remap_ms,
            color_ms=color_stats.color_ms,
            shading_ms=color_stats.shading_ms,
            composite_ms=composite_stats.composite_ms,
            total_ms=(time.perf_counter() - total_t0) * 1000.0,
            remap_backend=remap_stats.backend,
            color_backend=color_stats.backend,
            composite_backend=composite_stats.backend,
        )
        return out, timings
