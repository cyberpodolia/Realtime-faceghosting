"""Optional native backend bindings for runtime_cuda hot paths."""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

_DEFAULT_DLL_NAME = "facefx_runtime_cuda_native.dll"


def _native_dir() -> Path:
    return Path(__file__).resolve().parent / "native"


def _candidate_dll_paths() -> list[Path]:
    env_path = os.environ.get("FACEFX_RUNTIME_NATIVE_DLL", "").strip()
    paths: list[Path] = []
    if env_path:
        paths.append(Path(env_path))
    paths.append(_native_dir() / _DEFAULT_DLL_NAME)
    return paths


class _NativeApi:
    def __init__(self, dll: ctypes.CDLL) -> None:
        func = dll.build_dense_remap_idw_f32
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # src_xy [N*2]
            ctypes.POINTER(ctypes.c_float),  # dst_xy [N*2]
            ctypes.c_int,  # n_points
            ctypes.c_int,  # roi_x
            ctypes.c_int,  # roi_y
            ctypes.c_int,  # roi_w
            ctypes.c_int,  # roi_h
            ctypes.c_float,  # power
            ctypes.c_float,  # eps
            ctypes.POINTER(ctypes.c_float),  # out_map_x [roi_w*roi_h]
            ctypes.POINTER(ctypes.c_float),  # out_map_y [roi_w*roi_h]
        ]
        func.restype = ctypes.c_int
        self._build_dense = func

        warp_func = dll.warp_triangles_u8
        warp_func.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),  # src_img
            ctypes.c_int,  # src_h
            ctypes.c_int,  # src_w
            ctypes.c_int,  # channels
            ctypes.POINTER(ctypes.c_ubyte),  # dst_img
            ctypes.c_int,  # dst_h
            ctypes.c_int,  # dst_w
            ctypes.POINTER(ctypes.c_float),  # src_points [N*2]
            ctypes.POINTER(ctypes.c_float),  # dst_points [N*2]
            ctypes.c_int,  # n_points
            ctypes.POINTER(ctypes.c_int),  # simplices [T*3]
            ctypes.c_int,  # n_tris
            ctypes.POINTER(ctypes.c_float),  # dst_mask [dst_h*dst_w]
        ]
        warp_func.restype = ctypes.c_int
        self._warp_triangles = warp_func

    def build_dense_remap_idw(
        self,
        src_points_xy: np.ndarray,
        dst_points_xy: np.ndarray,
        *,
        roi_x: int,
        roi_y: int,
        roi_w: int,
        roi_h: int,
        power: float,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        src = np.ascontiguousarray(src_points_xy, dtype=np.float32)
        dst = np.ascontiguousarray(dst_points_xy, dtype=np.float32)
        if src.ndim != 2 or src.shape[1] != 2:
            raise ValueError("src_points_xy must have shape [N, 2]")
        if dst.ndim != 2 or dst.shape[1] != 2:
            raise ValueError("dst_points_xy must have shape [N, 2]")
        if src.shape != dst.shape:
            raise ValueError("src_points_xy and dst_points_xy must have identical shape")
        if src.shape[0] < 3:
            raise ValueError("at least 3 control points are required")

        n_points = int(src.shape[0])
        out_size = int(roi_w * roi_h)
        out_x = np.empty(out_size, dtype=np.float32)
        out_y = np.empty(out_size, dtype=np.float32)

        rc = int(
            self._build_dense(
                src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(n_points),
                ctypes.c_int(int(roi_x)),
                ctypes.c_int(int(roi_y)),
                ctypes.c_int(int(roi_w)),
                ctypes.c_int(int(roi_h)),
                ctypes.c_float(float(power)),
                ctypes.c_float(float(eps)),
                out_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        )
        if rc != 0:
            raise RuntimeError(f"native build_dense_remap_idw_f32 failed with code {rc}")
        return out_x.reshape(roi_h, roi_w), out_y.reshape(roi_h, roi_w)

    def warp_triangles_u8(
        self,
        src_img: np.ndarray,
        src_points_xy: np.ndarray,
        dst_points_xy: np.ndarray,
        simplices: np.ndarray,
        *,
        dst_h: int,
        dst_w: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        src = np.ascontiguousarray(src_img)
        if src.ndim != 3 or src.shape[2] not in (1, 3):
            raise ValueError("src_img must have shape [H, W, 1|3]")
        if src.dtype != np.uint8:
            raise ValueError("src_img must be uint8")

        src_pts = np.ascontiguousarray(src_points_xy, dtype=np.float32)
        dst_pts = np.ascontiguousarray(dst_points_xy, dtype=np.float32)
        tris = np.ascontiguousarray(simplices, dtype=np.int32)
        if src_pts.ndim != 2 or src_pts.shape[1] != 2:
            raise ValueError("src_points_xy must have shape [N, 2]")
        if dst_pts.ndim != 2 or dst_pts.shape[1] != 2:
            raise ValueError("dst_points_xy must have shape [N, 2]")
        if src_pts.shape != dst_pts.shape:
            raise ValueError("src_points_xy and dst_points_xy must have identical shape")
        if tris.ndim != 2 or tris.shape[1] != 3:
            raise ValueError("simplices must have shape [T, 3]")
        if dst_h < 1 or dst_w < 1:
            raise ValueError("destination shape must be positive")

        channels = int(src.shape[2])
        dst_img = np.zeros((dst_h, dst_w, channels), dtype=np.uint8)
        dst_mask = np.zeros((dst_h, dst_w), dtype=np.float32)
        rc = int(
            self._warp_triangles(
                src.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                ctypes.c_int(int(src.shape[0])),
                ctypes.c_int(int(src.shape[1])),
                ctypes.c_int(channels),
                dst_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                ctypes.c_int(int(dst_h)),
                ctypes.c_int(int(dst_w)),
                src_pts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                dst_pts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(int(src_pts.shape[0])),
                tris.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                ctypes.c_int(int(tris.shape[0])),
                dst_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        )
        if rc != 0:
            raise RuntimeError(f"native warp_triangles_u8 failed with code {rc}")
        return dst_img, dst_mask


@lru_cache(maxsize=1)
def load_native_api() -> _NativeApi | None:
    for path in _candidate_dll_paths():
        if not path.exists():
            continue
        try:
            dll = ctypes.CDLL(str(path))
            return _NativeApi(dll)
        except Exception:
            continue
    return None


def native_idw_available() -> bool:
    return load_native_api() is not None
