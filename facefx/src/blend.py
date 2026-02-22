"""Blend helpers."""

from __future__ import annotations

import cv2
import numpy as np


_CUDA_GAUSS_FILTERS_F32: dict[int, object] = {}


def _cuda_gaussian_blur_f32(channel_f32: np.ndarray, k: int) -> np.ndarray:
    """Gaussian blur using OpenCV CUDA if available; falls back to CPU on failure."""
    if not hasattr(cv2, "cuda"):
        return cv2.GaussianBlur(channel_f32, (k, k), 0)
    getter = getattr(cv2.cuda, "getCudaEnabledDeviceCount", None)
    if getter is None or int(getter()) <= 0:
        return cv2.GaussianBlur(channel_f32, (k, k), 0)

    try:
        filt = _CUDA_GAUSS_FILTERS_F32.get(k)
        if filt is None:
            # CV_32F lets us keep shading math in float and avoids repeated dtype changes.
            filt = cv2.cuda.createGaussianFilter(cv2.CV_32F, cv2.CV_32F, (k, k), 0)
            _CUDA_GAUSS_FILTERS_F32[k] = filt

        gpu = cv2.cuda_GpuMat()
        gpu.upload(channel_f32)
        out = filt.apply(gpu)
        return out.download()
    except Exception:
        # Safety: CUDA support varies across OpenCV builds; never make this path fatal.
        return cv2.GaussianBlur(channel_f32, (k, k), 0)


def blend_with_mask(
    base: np.ndarray,
    overlay: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Blend overlay onto base using mask in [0,1]."""
    if mask.ndim == 2:
        mask = mask[..., None]
    mask = np.clip(mask, 0.0, 1.0)
    out = base.astype(np.float32) * (1.0 - mask) + overlay.astype(np.float32) * mask
    return out.astype(np.uint8)


def _masked_mean_std(channel: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    mean, std = cv2.meanStdDev(channel, mask=mask)
    return float(mean[0][0]), float(std[0][0])


def _mask_to_u8(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.dtype == np.uint8:
        return mask
    return (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)


def _masked_mean_std_lab(lab: np.ndarray, mask_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, std = cv2.meanStdDev(lab, mask=mask_u8)
    return mean.reshape(3).astype(np.float32), std.reshape(3).astype(np.float32)


def color_match_lab(
    src_bgr: np.ndarray,
    tgt_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    ab_strength: float = 0.5,
    shading: bool = True,
    shading_kernel: int = 51,
    shading_clamp: tuple[float, float] = (0.6, 1.6),
    eps: float = 1e-6,
    use_cuda: bool = False,
) -> np.ndarray:
    """Match src to target colors in LAB using masked mean/std and optional shading."""
    mask_u8 = _mask_to_u8(mask)

    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_mean, src_std = _masked_mean_std_lab(src_lab, mask_u8)
    tgt_mean, tgt_std = _masked_mean_std_lab(tgt_lab, mask_u8)

    scale = tgt_std / (src_std + eps)
    out_lab = (src_lab - src_mean.reshape(1, 1, 3)) * scale.reshape(1, 1, 3) + tgt_mean.reshape(
        1, 1, 3
    )

    tgt_l = tgt_lab[:, :, 0]
    src_l = out_lab[:, :, 0]
    tgt_a = tgt_lab[:, :, 1]
    tgt_b = tgt_lab[:, :, 2]
    src_a = out_lab[:, :, 1]
    src_b = out_lab[:, :, 2]

    if ab_strength < 1.0:
        src_a = tgt_a + (src_a - tgt_a) * ab_strength
        src_b = tgt_b + (src_b - tgt_b) * ab_strength

    if shading:
        k = max(3, int(shading_kernel) | 1)
        if use_cuda:
            blur_t = _cuda_gaussian_blur_f32(tgt_l, k)
            blur_s = _cuda_gaussian_blur_f32(src_l, k)
        else:
            blur_t = cv2.GaussianBlur(tgt_l, (k, k), 0)
            blur_s = cv2.GaussianBlur(src_l, (k, k), 0)
        ratio = (blur_t + eps) / (blur_s + eps)
        ratio = np.clip(ratio, shading_clamp[0], shading_clamp[1])
        src_l = src_l * ratio

    out_lab = cv2.merge(
        [
            np.clip(src_l, 0, 255),
            np.clip(src_a, 0, 255),
            np.clip(src_b, 0, 255),
        ]
    ).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
