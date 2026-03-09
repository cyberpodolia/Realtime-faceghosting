"""Color match and shading helpers for runtime_cuda."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

_CUDA_GAUSS_FILTERS_F32: dict[int, object] = {}
_EDGE_TRANSITION_FADE_PX = 14.0
_INTERIOR_BLEND_STRENGTH = 0.82


@dataclass(frozen=True)
class ColorStageStats:
    """Summary diagnostics for color and shading stage."""

    color_ms: float
    shading_ms: float
    total_ms: float
    backend: str


@dataclass(frozen=True)
class ColorTransferStateLab:
    src_mean: np.ndarray
    src_std: np.ndarray
    tgt_mean: np.ndarray
    tgt_std: np.ndarray


@dataclass(frozen=True)
class ShadingStateLab:
    ratio_l: np.ndarray
    backend: str


def _overlay_blend_l(base_l: np.ndarray, blend_l: np.ndarray) -> np.ndarray:
    base = np.clip(base_l / 255.0, 0.0, 1.0)
    blend = np.clip(blend_l / 255.0, 0.0, 1.0)
    out = np.where(
        base <= 0.5,
        2.0 * base * blend,
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend),
    )
    return np.clip(out * 255.0, 0.0, 255.0).astype(np.float32)


def _mask_to_u8(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        if m.shape[2] == 1:
            m = m[:, :, 0]
        else:
            raise ValueError("alpha mask with 3 dims must have a single channel")
    if m.ndim != 2:
        raise ValueError("alpha mask must have shape [H, W] or [H, W, 1]")
    if m.dtype == np.uint8:
        return m
    if np.issubdtype(m.dtype, np.floating):
        if float(np.max(m)) <= 1.0 + 1e-6:
            return (np.clip(m, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.clip(m, 0.0, 255.0).astype(np.uint8)
    return np.clip(m, 0, 255).astype(np.uint8)


def _masked_mean_std_lab(lab: np.ndarray, mask_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, std = cv2.meanStdDev(lab, mask=mask_u8)
    return mean.reshape(3).astype(np.float32), std.reshape(3).astype(np.float32)


def _cuda_gaussian_blur_f32(channel_f32: np.ndarray, k: int) -> tuple[np.ndarray, bool]:
    if not hasattr(cv2, "cuda"):
        return cv2.GaussianBlur(channel_f32, (k, k), 0), False
    getter = getattr(cv2.cuda, "getCudaEnabledDeviceCount", None)
    if getter is None:
        return cv2.GaussianBlur(channel_f32, (k, k), 0), False
    try:
        if int(getter()) <= 0:
            return cv2.GaussianBlur(channel_f32, (k, k), 0), False
    except Exception:
        return cv2.GaussianBlur(channel_f32, (k, k), 0), False

    try:
        filt = _CUDA_GAUSS_FILTERS_F32.get(k)
        if filt is None:
            filt = cv2.cuda.createGaussianFilter(cv2.CV_32F, cv2.CV_32F, (k, k), 0)
            _CUDA_GAUSS_FILTERS_F32[k] = filt
        gpu = cv2.cuda_GpuMat()
        gpu.upload(channel_f32)
        out = filt.apply(gpu)
        return out.download(), True
    except Exception:
        return cv2.GaussianBlur(channel_f32, (k, k), 0), False


def _masked_gaussian_blur_f32(
    channel_f32: np.ndarray,
    mask_f32: np.ndarray,
    k: int,
    *,
    eps: float = 1e-6,
    use_cuda: bool = True,
) -> tuple[np.ndarray, str]:
    if channel_f32.shape != mask_f32.shape:
        raise ValueError("channel_f32 and mask_f32 must have identical shape")
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    weighted = channel_f32 * mask_f32
    if use_cuda:
        blur_num, used_num = _cuda_gaussian_blur_f32(weighted, k)
        blur_den, used_den = _cuda_gaussian_blur_f32(mask_f32, k)
        backend = "cuda" if used_num and used_den else "cpu"
    else:
        blur_num = cv2.GaussianBlur(weighted, (k, k), 0)
        blur_den = cv2.GaussianBlur(mask_f32, (k, k), 0)
        backend = "cpu"
    return blur_num / (blur_den + eps), backend


def _as_u8_bgr(img: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{name} must have shape [H, W, 3]")
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)


def _mask_bbox(mask_u8: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def _build_postprocess_core_mask(
    mask_u8: np.ndarray,
    *,
    min_alpha: int = 160,
    erode_px: int = 1,
) -> np.ndarray:
    core = np.where(mask_u8 >= int(min_alpha), 255, 0).astype(np.uint8)
    if cv2.countNonZero(core) == 0:
        return core
    if erode_px > 0:
        k = max(1, int(erode_px) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        core = cv2.erode(core, kernel, iterations=1)
    return core


def _mix_with_mask(base_bgr: np.ndarray, corrected_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    if base_bgr.shape != corrected_bgr.shape:
        raise ValueError("base_bgr and corrected_bgr must have identical shape")
    if mask_u8.shape != base_bgr.shape[:2]:
        raise ValueError("mask shape must match image shape")
    alpha = mask_u8.astype(np.float32) / 255.0
    alpha3 = alpha[:, :, None]
    mixed = base_bgr.astype(np.float32) * (1.0 - alpha3) + corrected_bgr.astype(np.float32) * alpha3
    return np.clip(mixed, 0.0, 255.0).astype(np.uint8)


def _edge_transition_weight(mask_u8: np.ndarray, *, fade_px: float = 8.0) -> np.ndarray:
    """Return interior weight [0,1] that is 0 near mask edge and 1 deeper inside."""
    if fade_px <= 0.0:
        raise ValueError("fade_px must be > 0")
    binary = (mask_u8 > 0).astype(np.uint8)
    if cv2.countNonZero(binary) == 0:
        return np.zeros(mask_u8.shape, dtype=np.float32)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    w = np.clip(dist / float(fade_px), 0.0, 1.0).astype(np.float32)
    return w


def build_color_transfer_state_lab(
    src_bgr: np.ndarray,
    tgt_bgr: np.ndarray,
    mask: np.ndarray,
) -> ColorTransferStateLab | None:
    src_u8 = _as_u8_bgr(src_bgr, "src_bgr")
    tgt_u8 = _as_u8_bgr(tgt_bgr, "tgt_bgr")
    if src_u8.shape != tgt_u8.shape:
        raise ValueError("src_bgr and tgt_bgr must have identical shape")
    mask_u8 = _mask_to_u8(mask)
    if mask_u8.shape != src_u8.shape[:2]:
        raise ValueError("mask shape must match image shape")
    if cv2.countNonZero(mask_u8) == 0:
        return None
    core_mask_u8 = _build_postprocess_core_mask(mask_u8)
    stats_mask_u8 = core_mask_u8 if cv2.countNonZero(core_mask_u8) > 0 else mask_u8
    src_lab = cv2.cvtColor(src_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_mean, src_std = _masked_mean_std_lab(src_lab, stats_mask_u8)
    tgt_mean, tgt_std = _masked_mean_std_lab(tgt_lab, stats_mask_u8)
    return ColorTransferStateLab(
        src_mean=src_mean,
        src_std=src_std,
        tgt_mean=tgt_mean,
        tgt_std=tgt_std,
    )


def apply_color_transfer_state_lab(
    src_bgr: np.ndarray,
    state: ColorTransferStateLab,
    *,
    ab_strength: float = 0.5,
    eps: float = 1e-6,
) -> np.ndarray:
    if not (0.0 <= ab_strength <= 1.0):
        raise ValueError("ab_strength must be in [0, 1]")
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    src_u8 = _as_u8_bgr(src_bgr, "src_bgr")
    src_lab = cv2.cvtColor(src_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    scale = state.tgt_std / (state.src_std + eps)
    out_lab = (src_lab - state.src_mean.reshape(1, 1, 3)) * scale.reshape(
        1, 1, 3
    ) + state.tgt_mean.reshape(1, 1, 3)
    if ab_strength < 1.0:
        target_a = float(state.tgt_mean[1])
        target_b = float(state.tgt_mean[2])
        out_lab[:, :, 1] = target_a + (out_lab[:, :, 1] - target_a) * ab_strength
        out_lab[:, :, 2] = target_b + (out_lab[:, :, 2] - target_b) * ab_strength
    out_lab_u8 = np.clip(out_lab, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(out_lab_u8, cv2.COLOR_LAB2BGR)


def build_shading_state_lab(
    src_bgr: np.ndarray,
    tgt_bgr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    shading_kernel: int = 51,
    shading_clamp: tuple[float, float] = (0.6, 1.6),
    eps: float = 1e-6,
    use_cuda: bool = True,
) -> ShadingStateLab:
    if shading_kernel < 1:
        raise ValueError("shading_kernel must be >= 1")
    if shading_clamp[0] <= 0.0 or shading_clamp[0] > shading_clamp[1]:
        raise ValueError("invalid shading_clamp range")
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    src_u8 = _as_u8_bgr(src_bgr, "src_bgr")
    tgt_u8 = _as_u8_bgr(tgt_bgr, "tgt_bgr")
    if src_u8.shape != tgt_u8.shape:
        raise ValueError("src_bgr and tgt_bgr must have identical shape")

    src_lab = cv2.cvtColor(src_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_l = src_lab[:, :, 0]
    tgt_l = tgt_lab[:, :, 0]
    k = max(3, int(shading_kernel) | 1)
    backend = "cpu"
    if mask is not None:
        mask_u8 = _mask_to_u8(mask)
        if mask_u8.shape != src_u8.shape[:2]:
            raise ValueError("mask shape must match source shape")
        mask_f32 = mask_u8.astype(np.float32) / 255.0
        blur_t, backend_t = _masked_gaussian_blur_f32(tgt_l, mask_f32, k, eps=eps, use_cuda=use_cuda)
        blur_s, backend_s = _masked_gaussian_blur_f32(src_l, mask_f32, k, eps=eps, use_cuda=use_cuda)
        backend = "cuda" if backend_t == "cuda" and backend_s == "cuda" else "cpu"
    else:
        if use_cuda:
            blur_t, used_t = _cuda_gaussian_blur_f32(tgt_l, k)
            blur_s, used_s = _cuda_gaussian_blur_f32(src_l, k)
            if used_t and used_s:
                backend = "cuda"
        else:
            blur_t = cv2.GaussianBlur(tgt_l, (k, k), 0)
            blur_s = cv2.GaussianBlur(src_l, (k, k), 0)

    ratio = (blur_t + eps) / (blur_s + eps)
    ratio = np.clip(ratio, shading_clamp[0], shading_clamp[1]).astype(np.float32)
    return ShadingStateLab(ratio_l=ratio, backend=backend)


def apply_shading_state_lab(
    src_bgr: np.ndarray,
    state: ShadingStateLab,
    *,
    strength: float = 0.35,
) -> np.ndarray:
    if not (0.0 <= strength <= 1.0):
        raise ValueError("strength must be in [0, 1]")
    src_u8 = _as_u8_bgr(src_bgr, "src_bgr")
    src_lab = cv2.cvtColor(src_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    if src_lab.shape[:2] != state.ratio_l.shape:
        raise ValueError("shading state shape must match source shape")
    src_l = src_lab[:, :, 0]
    shaded_l = np.clip(src_l * state.ratio_l, 0.0, 255.0)
    overlay_l = _overlay_blend_l(src_l, shaded_l)
    src_lab[:, :, 0] = np.clip(src_l * (1.0 - strength) + overlay_l * strength, 0.0, 255.0)
    out_lab_u8 = np.clip(src_lab, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(out_lab_u8, cv2.COLOR_LAB2BGR)


def match_color_and_shading_roi(
    patch_roi_bgr: np.ndarray,
    target_roi_bgr: np.ndarray,
    alpha_roi: np.ndarray,
    *,
    enable_color_match: bool = True,
    ab_strength: float = 0.5,
    enable_shading: bool = True,
    shading_kernel: int = 51,
    shading_clamp: tuple[float, float] = (0.6, 1.6),
    eps: float = 1e-6,
    use_cuda: bool = True,
) -> tuple[np.ndarray, ColorStageStats]:
    """Match patch colors to target ROI and apply optional luminance shading."""
    if not (0.0 <= ab_strength <= 1.0):
        raise ValueError("ab_strength must be in [0, 1]")
    if shading_kernel < 1:
        raise ValueError("shading_kernel must be >= 1")
    if shading_clamp[0] <= 0.0 or shading_clamp[0] > shading_clamp[1]:
        raise ValueError("invalid shading_clamp range")
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    patch_u8 = _as_u8_bgr(patch_roi_bgr, "patch_roi_bgr")
    target_u8 = _as_u8_bgr(target_roi_bgr, "target_roi_bgr")
    if patch_u8.shape != target_u8.shape:
        raise ValueError("patch_roi_bgr and target_roi_bgr must have identical shape")

    mask_u8 = _mask_to_u8(alpha_roi)
    if mask_u8.shape != patch_u8.shape[:2]:
        raise ValueError("alpha_roi shape must match ROI shape")

    t0 = time.perf_counter()
    if (not enable_color_match and not enable_shading) or cv2.countNonZero(mask_u8) == 0:
        stats = ColorStageStats(
            color_ms=0.0,
            shading_ms=0.0,
            total_ms=(time.perf_counter() - t0) * 1000.0,
            backend="off",
        )
        return patch_u8.copy(), stats

    bbox = _mask_bbox(mask_u8)
    if bbox is None:
        stats = ColorStageStats(
            color_ms=0.0,
            shading_ms=0.0,
            total_ms=(time.perf_counter() - t0) * 1000.0,
            backend="off",
        )
        return patch_u8.copy(), stats

    x0, y0, x1, y1 = bbox
    patch_work = patch_u8[y0:y1, x0:x1]
    target_work = target_u8[y0:y1, x0:x1]
    mask_work = mask_u8[y0:y1, x0:x1]

    src_lab = cv2.cvtColor(patch_work, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_work, cv2.COLOR_BGR2LAB).astype(np.float32)

    color_ms = 0.0
    if enable_color_match:
        t_color = time.perf_counter()
        src_mean, src_std = _masked_mean_std_lab(src_lab, mask_work)
        tgt_mean, tgt_std = _masked_mean_std_lab(tgt_lab, mask_work)
        scale = tgt_std / (src_std + eps)
        out_lab = (src_lab - src_mean.reshape(1, 1, 3)) * scale.reshape(1, 1, 3) + tgt_mean.reshape(
            1, 1, 3
        )
        if ab_strength < 1.0:
            out_lab[:, :, 1] = (
                tgt_lab[:, :, 1] + (out_lab[:, :, 1] - tgt_lab[:, :, 1]) * ab_strength
            )
            out_lab[:, :, 2] = (
                tgt_lab[:, :, 2] + (out_lab[:, :, 2] - tgt_lab[:, :, 2]) * ab_strength
            )
        color_ms = (time.perf_counter() - t_color) * 1000.0
    else:
        out_lab = src_lab.copy()

    shading_ms = 0.0
    backend = "cpu"
    if enable_shading:
        t_shading = time.perf_counter()
        k = max(3, int(shading_kernel) | 1)
        tgt_l = tgt_lab[:, :, 0]
        src_l = out_lab[:, :, 0]

        if use_cuda:
            blur_t, used_t = _cuda_gaussian_blur_f32(tgt_l, k)
            blur_s, used_s = _cuda_gaussian_blur_f32(src_l, k)
            backend = "cuda" if used_t and used_s else "cpu"
        else:
            blur_t = cv2.GaussianBlur(tgt_l, (k, k), 0)
            blur_s = cv2.GaussianBlur(src_l, (k, k), 0)
            backend = "cpu"

        ratio = (blur_t + eps) / (blur_s + eps)
        ratio = np.clip(ratio, shading_clamp[0], shading_clamp[1])
        out_lab[:, :, 0] = src_l * ratio
        shading_ms = (time.perf_counter() - t_shading) * 1000.0

    out_lab_u8 = np.clip(out_lab, 0.0, 255.0).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_lab_u8, cv2.COLOR_LAB2BGR)
    out_full = patch_u8.copy()
    out_full[y0:y1, x0:x1] = out_bgr
    stats = ColorStageStats(
        color_ms=color_ms,
        shading_ms=shading_ms,
        total_ms=(time.perf_counter() - t0) * 1000.0,
        backend=backend,
    )
    return out_full, stats


def match_color_and_shading_roi_cached(
    patch_roi_bgr: np.ndarray,
    target_roi_bgr: np.ndarray,
    alpha_roi: np.ndarray,
    *,
    refresh_color: bool = True,
    refresh_shading: bool = True,
    color_state: ColorTransferStateLab | None = None,
    shading_state: ShadingStateLab | None = None,
    color_match_enabled: bool = True,
    ab_strength: float = 0.5,
    shading_enabled: bool = True,
    shading_kernel: int = 51,
    shading_clamp: tuple[float, float] = (0.6, 1.6),
    shading_strength: float = 0.35,
    use_cuda: bool = True,
) -> tuple[np.ndarray, ColorStageStats, ColorTransferStateLab | None, ShadingStateLab | None]:
    """Apply refresh-or-hold color/shading behavior matching the legacy pipeline."""
    patch_u8 = _as_u8_bgr(patch_roi_bgr, "patch_roi_bgr")
    target_u8 = _as_u8_bgr(target_roi_bgr, "target_roi_bgr")
    if patch_u8.shape != target_u8.shape:
        raise ValueError("patch_roi_bgr and target_roi_bgr must have identical shape")

    mask_u8 = _mask_to_u8(alpha_roi)
    if mask_u8.shape != patch_u8.shape[:2]:
        raise ValueError("alpha_roi shape must match ROI shape")

    t0 = time.perf_counter()
    if cv2.countNonZero(mask_u8) == 0:
        stats = ColorStageStats(0.0, 0.0, (time.perf_counter() - t0) * 1000.0, "off")
        return patch_u8.copy(), stats, None, None

    bbox = _mask_bbox(mask_u8)
    if bbox is None:
        stats = ColorStageStats(0.0, 0.0, (time.perf_counter() - t0) * 1000.0, "off")
        return patch_u8.copy(), stats, None, None

    x0, y0, x1, y1 = bbox
    patch_work = patch_u8[y0:y1, x0:x1]
    target_work = target_u8[y0:y1, x0:x1]
    mask_work = mask_u8[y0:y1, x0:x1]
    core_mask_work = _build_postprocess_core_mask(mask_work)
    stats_mask_work = core_mask_work if cv2.countNonZero(core_mask_work) > 0 else mask_work

    color_ms = 0.0
    shading_ms = 0.0
    backend = "off"

    next_color_state = color_state if color_match_enabled else None
    if color_match_enabled and (refresh_color or next_color_state is None):
        t_color = time.perf_counter()
        next_color_state = build_color_transfer_state_lab(patch_work, target_work, stats_mask_work)
        color_ms = (time.perf_counter() - t_color) * 1000.0
        backend = "cpu"

    corrected = patch_work.copy()
    if color_match_enabled and next_color_state is not None:
        corrected = apply_color_transfer_state_lab(
            corrected,
            next_color_state,
            ab_strength=ab_strength,
        )
        backend = "cpu"

    next_shading_state = shading_state if shading_enabled else None
    if shading_enabled:
        shape_mismatch = (
            next_shading_state is not None and next_shading_state.ratio_l.shape != corrected.shape[:2]
        )
        if refresh_shading or next_shading_state is None or shape_mismatch:
            t_shading = time.perf_counter()
            next_shading_state = build_shading_state_lab(
                corrected,
                target_work,
                mask=stats_mask_work,
                shading_kernel=shading_kernel,
                shading_clamp=shading_clamp,
                use_cuda=use_cuda,
            )
            shading_ms = (time.perf_counter() - t_shading) * 1000.0
        if next_shading_state is not None:
            corrected = apply_shading_state_lab(
                corrected,
                next_shading_state,
                strength=shading_strength,
            )
            backend = next_shading_state.backend
    else:
        next_shading_state = None

    if not color_match_enabled and not shading_enabled:
        backend = "off"

    out_full = patch_u8.copy()
    if color_match_enabled or shading_enabled:
        # 1) Soften interior contrast by mixing corrected patch with live target.
        interior = np.clip(
            target_work.astype(np.float32) * (1.0 - _INTERIOR_BLEND_STRENGTH)
            + corrected.astype(np.float32) * _INTERIOR_BLEND_STRENGTH,
            0.0,
            255.0,
        ).astype(np.uint8)
        # 2) Force seamless edge transition: near mask boundary prefer live target color.
        edge_w = _edge_transition_weight(mask_work, fade_px=_EDGE_TRANSITION_FADE_PX)
        edge_w3 = edge_w[:, :, None]
        corrected_soft = np.clip(
            target_work.astype(np.float32) * (1.0 - edge_w3) + interior.astype(np.float32) * edge_w3,
            0.0,
            255.0,
        ).astype(np.uint8)
    else:
        corrected_soft = corrected
    # Keep ROI content as fully corrected patch and apply alpha only once
    # in final composite stage to avoid edge halos from double blending.
    out_full[y0:y1, x0:x1] = corrected_soft
    stats = ColorStageStats(
        color_ms=color_ms,
        shading_ms=shading_ms,
        total_ms=(time.perf_counter() - t0) * 1000.0,
        backend=backend,
    )
    return out_full, stats, next_color_state, next_shading_state
