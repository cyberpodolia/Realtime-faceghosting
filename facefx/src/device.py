"""Device selection helpers (CPU vs CUDA).

This project runs end-to-end on CPU by default. CUDA acceleration is optional
and only applies to a small subset of operations (when available) because the
MediaPipe FaceMesh Python API is typically CPU-bound.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    name: str  # "cpu" or "cuda"
    cuda_available: bool
    cuda_device_count: int


def _cuda_device_count(cv2) -> int:
    """Return CUDA device count or 0 when CUDA APIs are unavailable."""
    try:
        cuda_mod = getattr(cv2, "cuda")
        getter = getattr(cuda_mod, "getCudaEnabledDeviceCount", None)
        if getter is None:
            return 0
        return int(getter())
    except Exception:
        return 0


def resolve_device(*, device: str, cv2) -> DeviceInfo:
    """Resolve a user-facing device mode into an execution device.

    Args:
        device: One of "auto", "cpu", "cuda".
        cv2: OpenCV module (injected to keep import-time failures contained).
    """
    normalized = (device or "auto").strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")

    count = _cuda_device_count(cv2)
    cuda_available = count > 0
    if normalized == "cpu":
        return DeviceInfo(name="cpu", cuda_available=cuda_available, cuda_device_count=count)
    if normalized == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "CUDA was requested but OpenCV is not CUDA-enabled (device count is 0)."
            )
        return DeviceInfo(name="cuda", cuda_available=True, cuda_device_count=count)

    # auto
    return DeviceInfo(
        name="cuda" if cuda_available else "cpu",
        cuda_available=cuda_available,
        cuda_device_count=count,
    )
