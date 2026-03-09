"""Warp helpers for runtime_cuda."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .native_backend import load_native_api
from .roi import Roi


@dataclass(frozen=True)
class DenseMapStats:
    """Summary diagnostics for dense map generation."""

    build_ms: float
    width: int
    height: int
    map_x_min: float
    map_x_max: float
    map_y_min: float
    map_y_max: float
    builder_backend: str = "numpy"


@dataclass(frozen=True)
class RemapStats:
    """Summary diagnostics for remap stage."""

    remap_ms: float
    backend: str


def _validate_triangle_inputs(
    src_points_xy: np.ndarray,
    dst_points_xy: np.ndarray,
    simplices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = np.asarray(src_points_xy, dtype=np.float32)
    dst = np.asarray(dst_points_xy, dtype=np.float32)
    tris = np.asarray(simplices, dtype=np.int32)
    if src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src_points_xy must have shape [N, 2]")
    if dst.ndim != 2 or dst.shape[1] != 2:
        raise ValueError("dst_points_xy must have shape [N, 2]")
    if src.shape != dst.shape:
        raise ValueError("src_points_xy and dst_points_xy must have identical shape")
    if src.shape[0] < 3:
        raise ValueError("at least 3 points are required")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("simplices must have shape [T, 3]")
    if tris.size and (int(np.min(tris)) < 0 or int(np.max(tris)) >= src.shape[0]):
        raise ValueError("simplices reference out-of-range point indices")
    return src, dst, tris


def _warp_triangle_into(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    t_src: np.ndarray,
    t_dst: np.ndarray,
) -> None:
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

    t1_rect = np.float32([[pt[0] - r1[0], pt[1] - r1[1]] for pt in t_src])
    t2_rect = np.float32([[pt[0] - r2[0], pt[1] - r2[1]] for pt in t_dst])

    src_crop = src_img[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
    if src_crop.size == 0:
        return

    mtx = cv2.getAffineTransform(t1_rect, t2_rect)
    warped = cv2.warpAffine(
        src_crop,
        mtx,
        (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((r2[3], r2[2]), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(np.round(t2_rect)), 1.0, lineType=cv2.LINE_AA)

    x, y, w, h = r2
    roi = dst_img[y : y + h, x : x + w]
    if roi.shape[:2] != warped.shape[:2]:
        return

    if roi.ndim == 2:
        dst_img[y : y + h, x : x + w] = roi * (1.0 - mask) + warped * mask
        return

    mask3 = mask[:, :, None]
    dst_img[y : y + h, x : x + w] = roi * (1.0 - mask3) + warped * mask3


def _warp_triangles_numpy(
    source_img: np.ndarray,
    alpha_mask: np.ndarray,
    src_points_xy: np.ndarray,
    dst_points_xy: np.ndarray,
    simplices: np.ndarray,
    roi: Roi,
) -> tuple[np.ndarray, np.ndarray]:
    dst_points_local = dst_points_xy - np.array([roi.x, roi.y], dtype=np.float32)
    source_f32 = source_img.astype(np.float32, copy=False)
    alpha_f32 = alpha_mask.astype(np.float32, copy=False)
    warped_patch = np.zeros((roi.h, roi.w, 3), dtype=np.float32)
    warped_alpha_u8 = np.zeros((roi.h, roi.w), dtype=np.float32)

    for tri in simplices:
        t_src = src_points_xy[tri]
        t_dst = dst_points_local[tri]
        _warp_triangle_into(source_f32, warped_patch, t_src, t_dst)
        _warp_triangle_into(alpha_f32, warped_alpha_u8, t_src, t_dst)

    return (
        np.clip(warped_patch, 0.0, 255.0).astype(np.uint8),
        np.clip(warped_alpha_u8, 0.0, 255.0).astype(np.uint8),
    )


def _validate_points(
    src_points_xy: np.ndarray, dst_points_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(src_points_xy, dtype=np.float32)
    dst = np.asarray(dst_points_xy, dtype=np.float32)
    if src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src_points_xy must have shape [N, 2]")
    if dst.ndim != 2 or dst.shape[1] != 2:
        raise ValueError("dst_points_xy must have shape [N, 2]")
    if src.shape != dst.shape:
        raise ValueError("src_points_xy and dst_points_xy must have identical shape")
    if src.shape[0] < 3:
        raise ValueError("at least 3 control points are required")
    return src, dst


def _roi_grid_xy(roi: Roi) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(roi.x, roi.x + roi.w, dtype=np.float32)
    y = np.arange(roi.y, roi.y + roi.h, dtype=np.float32)
    return np.meshgrid(x, y)


def _build_dense_remap_idw_numpy(
    src: np.ndarray,
    dst: np.ndarray,
    roi: Roi,
    *,
    power: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    grid_x, grid_y = _roi_grid_xy(roi)
    grid = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1).astype(np.float32)

    delta = (src - dst).astype(np.float32)  # [N,2]
    diff = grid[:, None, :] - dst[None, :, :]  # [M,N,2]
    dist2 = np.sum(diff * diff, axis=2) + float(eps)  # [M,N]
    weights = (1.0 / np.power(dist2, power * 0.5)).astype(np.float32)
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights = weights / weights_sum

    disp = weights @ delta  # [M,2]
    map_flat = grid + disp
    map_x = map_flat[:, 0].reshape(roi.h, roi.w).astype(np.float32)
    map_y = map_flat[:, 1].reshape(roi.h, roi.w).astype(np.float32)
    return map_x, map_y


def _estimate_similarity_affine(
    src_points_xy: np.ndarray, dst_points_xy: np.ndarray
) -> np.ndarray | None:
    try:
        affine, _inliers = cv2.estimateAffinePartial2D(
            src_points_xy.astype(np.float32),
            dst_points_xy.astype(np.float32),
            method=cv2.LMEDS,
        )
    except Exception:
        return None
    if affine is None or affine.shape != (2, 3):
        return None
    return affine.astype(np.float32)


def _ensure_alpha_u8(alpha_mask: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha_mask)
    if alpha.ndim == 3:
        if alpha.shape[2] == 1:
            alpha = alpha[:, :, 0]
        elif alpha.shape[2] == 3:
            alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("alpha_mask with 3 dims must have 1 or 3 channels")
    if alpha.ndim != 2:
        raise ValueError("alpha_mask must have shape [H, W] or [H, W, C]")

    if alpha.dtype == np.uint8:
        return alpha
    if np.issubdtype(alpha.dtype, np.floating):
        if float(np.max(alpha)) <= 1.0 + 1e-6:
            return (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.clip(alpha, 0.0, 255.0).astype(np.uint8)
    return np.clip(alpha, 0, 255).astype(np.uint8)


def _cuda_remap_available() -> bool:
    if not hasattr(cv2, "cuda"):
        return False
    getter = getattr(cv2.cuda, "getCudaEnabledDeviceCount", None)
    remap_fn = getattr(cv2.cuda, "remap", None)
    if getter is None or remap_fn is None:
        return False
    try:
        return int(getter()) > 0
    except Exception:
        return False


def build_dense_remap_idw(
    src_points_xy: np.ndarray,
    dst_points_xy: np.ndarray,
    roi: Roi,
    *,
    src_shape_hw: tuple[int, int] | None = None,
    power: float = 2.0,
    eps: float = 1e-3,
    prefer_native: bool = True,
    similarity_prealign: bool = True,
) -> tuple[np.ndarray, np.ndarray, DenseMapStats]:
    """Build dense remap `map_x/map_y` for ROI using vectorized IDW interpolation.

    The map defines a transform from destination ROI pixels to source coordinates.
    No per-triangle loops are used.
    """

    if roi.w < 1 or roi.h < 1:
        raise ValueError("roi dimensions must be positive")
    if power <= 0:
        raise ValueError("power must be > 0")
    if eps <= 0:
        raise ValueError("eps must be > 0")

    src, dst = _validate_points(src_points_xy, dst_points_xy)
    t0 = time.perf_counter()
    backend = "numpy"
    src_eff = src
    inv_affine: np.ndarray | None = None
    if similarity_prealign:
        affine = _estimate_similarity_affine(src, dst)
        if affine is not None:
            src_eff = cv2.transform(src[None, :, :], affine)[0].astype(np.float32)
            inv_affine = cv2.invertAffineTransform(affine).astype(np.float32)

    native_api = load_native_api() if prefer_native else None
    if native_api is not None:
        try:
            map_x, map_y = native_api.build_dense_remap_idw(
                src_eff,
                dst,
                roi_x=roi.x,
                roi_y=roi.y,
                roi_w=roi.w,
                roi_h=roi.h,
                power=power,
                eps=eps,
            )
            backend = "native"
        except Exception:
            map_x, map_y = _build_dense_remap_idw_numpy(src_eff, dst, roi, power=power, eps=eps)
            backend = "numpy"
    else:
        map_x, map_y = _build_dense_remap_idw_numpy(src_eff, dst, roi, power=power, eps=eps)
        backend = "numpy"

    if inv_affine is not None:
        mx = map_x.astype(np.float32)
        my = map_y.astype(np.float32)
        map_x = inv_affine[0, 0] * mx + inv_affine[0, 1] * my + inv_affine[0, 2]
        map_y = inv_affine[1, 0] * mx + inv_affine[1, 1] * my + inv_affine[1, 2]
        backend = f"{backend}+sim"

    if src_shape_hw is not None:
        h, w = int(src_shape_hw[0]), int(src_shape_hw[1])
        if h < 1 or w < 1:
            raise ValueError("src_shape_hw must be positive")
        map_x = np.clip(map_x, 0.0, float(w - 1))
        map_y = np.clip(map_y, 0.0, float(h - 1))

    stats = DenseMapStats(
        build_ms=(time.perf_counter() - t0) * 1000.0,
        width=roi.w,
        height=roi.h,
        map_x_min=float(np.min(map_x)),
        map_x_max=float(np.max(map_x)),
        map_y_min=float(np.min(map_y)),
        map_y_max=float(np.max(map_y)),
        builder_backend=backend,
    )
    return map_x, map_y, stats


def remap_patch_and_alpha(
    source_bgr: np.ndarray,
    alpha_mask: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    *,
    use_cuda: bool = True,
) -> tuple[np.ndarray, np.ndarray, RemapStats]:
    """Remap patch and alpha using CUDA when available, else CPU."""

    src = np.asarray(source_bgr)
    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError("source_bgr must have shape [H, W, 3]")
    mx = np.asarray(map_x, dtype=np.float32)
    my = np.asarray(map_y, dtype=np.float32)
    if mx.shape != my.shape:
        raise ValueError("map_x and map_y must have identical shape")
    if mx.ndim != 2:
        raise ValueError("map_x/map_y must be 2D")

    alpha_u8 = _ensure_alpha_u8(alpha_mask)
    if alpha_u8.shape != src.shape[:2]:
        raise ValueError("alpha_mask shape must match source image shape")

    t0 = time.perf_counter()
    backend = "cpu"

    if use_cuda and _cuda_remap_available():
        try:
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(src)
            gpu_alpha = cv2.cuda_GpuMat()
            gpu_alpha.upload(alpha_u8)
            gpu_mx = cv2.cuda_GpuMat()
            gpu_mx.upload(mx)
            gpu_my = cv2.cuda_GpuMat()
            gpu_my.upload(my)

            warped_src_gpu = cv2.cuda.remap(
                gpu_src,
                gpu_mx,
                gpu_my,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            warped_alpha_gpu = cv2.cuda.remap(
                gpu_alpha,
                gpu_mx,
                gpu_my,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            warped_src = warped_src_gpu.download()
            warped_alpha_u8 = warped_alpha_gpu.download()
            backend = "cuda"
        except Exception:
            warped_src = cv2.remap(
                src,
                mx,
                my,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            warped_alpha_u8 = cv2.remap(
                alpha_u8,
                mx,
                my,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            backend = "cpu"
    else:
        warped_src = cv2.remap(
            src,
            mx,
            my,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        warped_alpha_u8 = cv2.remap(
            alpha_u8,
            mx,
            my,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        backend = "cpu"

    warped_alpha = warped_alpha_u8.astype(np.float32) / 255.0
    stats = RemapStats(remap_ms=(time.perf_counter() - t0) * 1000.0, backend=backend)
    return warped_src, warped_alpha, stats


def warp_patch_and_alpha_triangles(
    source_bgr: np.ndarray,
    alpha_mask: np.ndarray,
    src_points_xy: np.ndarray,
    dst_points_xy: np.ndarray,
    simplices: np.ndarray,
    roi: Roi,
    *,
    use_native: bool = True,
) -> tuple[np.ndarray, np.ndarray, RemapStats]:
    """Warp patch/alpha into ROI using triangle topology."""

    src = np.asarray(source_bgr)
    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError("source_bgr must have shape [H, W, 3]")
    alpha_u8 = _ensure_alpha_u8(alpha_mask)
    if alpha_u8.shape != src.shape[:2]:
        raise ValueError("alpha_mask shape must match source image shape")
    if roi.w < 1 or roi.h < 1:
        raise ValueError("roi dimensions must be positive")

    src_pts, dst_pts, tris = _validate_triangle_inputs(src_points_xy, dst_points_xy, simplices)
    if len(tris) == 0:
        raise ValueError("simplices must not be empty")

    t0 = time.perf_counter()
    backend = "cpu"

    native_api = load_native_api() if use_native else None
    if native_api is not None:
        try:
            dst_local = dst_pts - np.array([roi.x, roi.y], dtype=np.float32)
            warped_patch, _patch_mask = native_api.warp_triangles_u8(
                src,
                src_pts,
                dst_local,
                tris,
                dst_h=roi.h,
                dst_w=roi.w,
            )
            alpha_img, _alpha_mask = native_api.warp_triangles_u8(
                alpha_u8[:, :, None],
                src_pts,
                dst_local,
                tris,
                dst_h=roi.h,
                dst_w=roi.w,
            )
            warped_alpha_u8 = alpha_img[:, :, 0]
            backend = "native"
        except Exception:
            warped_patch, warped_alpha_u8 = _warp_triangles_numpy(
                src,
                alpha_u8,
                src_pts,
                dst_pts,
                tris,
                roi,
            )
            backend = "cpu"
    else:
        warped_patch, warped_alpha_u8 = _warp_triangles_numpy(
            src,
            alpha_u8,
            src_pts,
            dst_pts,
            tris,
            roi,
        )
        backend = "cpu"

    warped_alpha = warped_alpha_u8.astype(np.float32) / 255.0
    stats = RemapStats(remap_ms=(time.perf_counter() - t0) * 1000.0, backend=backend)
    return warped_patch, warped_alpha, stats
