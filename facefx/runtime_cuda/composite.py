"""ROI compositing helpers for runtime_cuda."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .roi import Roi


@dataclass(frozen=True)
class CompositeStats:
    """Summary diagnostics for compositing stage."""

    composite_ms: float
    backend: str


def _cuda_blend_available() -> bool:
    if not hasattr(cv2, "cuda"):
        return False
    getter = getattr(cv2.cuda, "getCudaEnabledDeviceCount", None)
    add_fn = getattr(cv2.cuda, "add", None)
    mul_fn = getattr(cv2.cuda, "multiply", None)
    if getter is None or add_fn is None or mul_fn is None:
        return False
    try:
        return int(getter()) > 0
    except Exception:
        return False


def _alpha_to_f32(alpha_roi: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    alpha = np.asarray(alpha_roi)
    if alpha.ndim == 3:
        if alpha.shape[2] == 1:
            alpha = alpha[:, :, 0]
        else:
            raise ValueError("alpha_roi with 3 dims must have a single channel")
    if alpha.ndim != 2:
        raise ValueError("alpha_roi must have shape [H, W] or [H, W, 1]")
    if alpha.shape != shape_hw:
        raise ValueError("alpha_roi shape must match ROI shape")
    if np.issubdtype(alpha.dtype, np.floating):
        if float(np.max(alpha)) <= 1.0 + 1e-6:
            return np.clip(alpha, 0.0, 1.0).astype(np.float32)
        return (np.clip(alpha, 0.0, 255.0) / 255.0).astype(np.float32)
    return (np.clip(alpha, 0, 255).astype(np.float32) / 255.0).astype(np.float32)


def _edge_clean_weight(
    alpha_f32: np.ndarray,
    *,
    start: float = 0.02,
    end: float = 0.45,
) -> np.ndarray:
    """Return edge-clean weight in [0,1] derived from alpha."""
    if not (0.0 <= start < end <= 1.0):
        raise ValueError("alpha clean range must satisfy 0 <= start < end <= 1")
    a = np.clip(alpha_f32, 0.0, 1.0)
    return np.clip((a - start) / max(1e-6, end - start), 0.0, 1.0).astype(np.float32)


def composite_roi(
    frame_bgr: np.ndarray,
    warped_patch_roi_bgr: np.ndarray,
    warped_alpha_roi: np.ndarray,
    roi: Roi,
    *,
    use_cuda: bool = True,
    backend_pref: str = "auto",
    cuda_min_area: int = 180000,
) -> tuple[np.ndarray, CompositeStats]:
    """Composite remapped patch over destination ROI."""

    frame = np.asarray(frame_bgr)
    patch = np.asarray(warped_patch_roi_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame_bgr must have shape [H, W, 3]")
    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError("warped_patch_roi_bgr must have shape [H, W, 3]")
    if patch.shape[:2] != (roi.h, roi.w):
        raise ValueError("warped_patch_roi_bgr shape must match ROI")

    h, w = frame.shape[:2]
    if roi.x < 0 or roi.y < 0 or roi.x + roi.w > w or roi.y + roi.h > h:
        raise ValueError("roi is out of frame bounds")
    pref = str(backend_pref).lower()
    if pref not in {"auto", "cpu", "cuda"}:
        raise ValueError("backend_pref must be one of: auto|cpu|cuda")
    if cuda_min_area < 1:
        raise ValueError("cuda_min_area must be >= 1")

    alpha = _alpha_to_f32(warped_alpha_roi, (roi.h, roi.w))
    edge_w = _edge_clean_weight(alpha)
    # Equivalent to:
    # patch_clean = base*(1-edge_w) + patch*edge_w
    # out = base*(1-alpha) + patch_clean*alpha
    # => out = base*(1-effective_alpha) + patch*effective_alpha
    effective_alpha3 = (alpha * edge_w)[:, :, None].astype(np.float32)

    out = frame.copy()
    base_roi_u8 = out[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
    t0 = time.perf_counter()
    backend = "cpu"
    roi_area = int(roi.w) * int(roi.h)
    allow_cuda = use_cuda and pref != "cpu"
    if pref == "auto":
        allow_cuda = allow_cuda and roi_area >= int(cuda_min_area)

    if allow_cuda and _cuda_blend_available():
        try:
            base_f32 = base_roi_u8.astype(np.float32) / 255.0
            patch_f32 = patch.astype(np.float32) / 255.0
            inv_alpha3 = 1.0 - effective_alpha3

            gpu_base = cv2.cuda_GpuMat()
            gpu_base.upload(base_f32)
            gpu_patch = cv2.cuda_GpuMat()
            gpu_patch.upload(patch_f32)
            gpu_alpha = cv2.cuda_GpuMat()
            gpu_alpha.upload(effective_alpha3)
            gpu_inv_alpha = cv2.cuda_GpuMat()
            gpu_inv_alpha.upload(inv_alpha3)

            left = cv2.cuda.multiply(gpu_base, gpu_inv_alpha)
            right = cv2.cuda.multiply(gpu_patch, gpu_alpha)
            merged = cv2.cuda.add(left, right)
            out_roi = np.clip(merged.download() * 255.0, 0.0, 255.0).astype(np.uint8)
            backend = "cuda"
        except Exception:
            out_roi = np.clip(
                base_roi_u8.astype(np.float32) * (1.0 - effective_alpha3)
                + patch.astype(np.float32) * effective_alpha3,
                0.0,
                255.0,
            ).astype(np.uint8)
            backend = "cpu"
    else:
        out_roi = np.clip(
            base_roi_u8.astype(np.float32) * (1.0 - effective_alpha3)
            + patch.astype(np.float32) * effective_alpha3,
            0.0,
            255.0,
        ).astype(np.uint8)
        backend = "cpu"

    out[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = out_roi
    stats = CompositeStats(composite_ms=(time.perf_counter() - t0) * 1000.0, backend=backend)
    return out, stats
