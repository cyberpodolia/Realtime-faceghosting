"""Deterministic control-point extraction for runtime_cuda."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

# Stable subset chosen for dense runtime deformation anchors.
# Count remains within the WI target range (16..32) and is deterministic.
CONTROL_POINT_INDICES: tuple[int, ...] = (
    10,
    67,
    109,
    338,
    297,
    332,
    152,
    234,
    454,
    127,
    356,
    33,
    133,
    362,
    263,
    61,
    291,
    78,
    308,
    168,
    197,
    5,
    4,
    0,
)


def ensure_landmark_array(landmarks_xy: np.ndarray) -> np.ndarray:
    """Validate and normalize landmark array to float32 [N,2]."""
    arr = np.asarray(landmarks_xy, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("expected landmark array with shape [N, 2]")
    return arr


def should_update_landmarks(
    frame_index: int,
    every: int,
    *,
    has_held_landmarks: bool,
    force: bool = False,
) -> bool:
    """Decide whether to run an expensive landmark solve on this frame."""
    if frame_index < 0:
        raise ValueError("frame_index must be >= 0")
    if every < 1:
        raise ValueError("every must be >= 1")
    if force or not has_held_landmarks:
        return True
    return (frame_index % every) == 0


def smooth_landmarks(
    previous_landmarks_xy: np.ndarray | None,
    current_landmarks_xy: np.ndarray,
    smooth: float,
) -> np.ndarray:
    """Blend previous and current landmarks for stable hold frames."""
    if not (0.0 <= smooth <= 0.95):
        raise ValueError("smooth must be in [0.0, 0.95]")
    current = ensure_landmark_array(current_landmarks_xy)
    if previous_landmarks_xy is None:
        return current.copy()
    previous = ensure_landmark_array(previous_landmarks_xy)
    if previous.shape != current.shape:
        return current.copy()
    return (previous * smooth + current * (1.0 - smooth)).astype(np.float32, copy=False)


@dataclass(frozen=True)
class LandmarkCadenceResult:
    landmarks_xy: np.ndarray | None
    updated: bool
    frame_index: int


class LandmarkCadenceState:
    """Track cadence/hold state for runtime landmark updates."""

    def __init__(self, *, every: int = 3, smooth: float = 0.25) -> None:
        if every < 1:
            raise ValueError("every must be >= 1")
        if not (0.0 <= smooth <= 0.95):
            raise ValueError("smooth must be in [0.0, 0.95]")
        self._every = int(every)
        self._smooth = float(smooth)
        self._frame_index = 0
        self._held_landmarks: np.ndarray | None = None

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def every(self) -> int:
        return self._every

    @property
    def smooth(self) -> float:
        return self._smooth

    def needs_update(self, *, force: bool = False) -> bool:
        return should_update_landmarks(
            self._frame_index,
            self._every,
            has_held_landmarks=self._held_landmarks is not None,
            force=force,
        )

    def update(self, latest_landmarks_xy: np.ndarray | None) -> LandmarkCadenceResult:
        updated = False
        if latest_landmarks_xy is not None:
            self._held_landmarks = smooth_landmarks(
                self._held_landmarks,
                latest_landmarks_xy,
                self._smooth,
            )
            updated = True

        result_landmarks = None if self._held_landmarks is None else self._held_landmarks.copy()
        result = LandmarkCadenceResult(
            landmarks_xy=result_landmarks,
            updated=updated,
            frame_index=self._frame_index,
        )
        self._frame_index += 1
        return result


def extract_control_points(
    landmarks_xy: np.ndarray,
    *,
    indices: Iterable[int] = CONTROL_POINT_INDICES,
    max_count: int | None = None,
) -> np.ndarray:
    """Extract deterministic control points from full landmark set."""
    landmarks = ensure_landmark_array(landmarks_xy)
    selected = select_landmark_indices(
        int(landmarks.shape[0]),
        indices=indices,
        max_count=max_count,
    )
    return landmarks[selected].copy()


def select_landmark_indices(
    total_count: int,
    *,
    indices: Iterable[int] = CONTROL_POINT_INDICES,
    max_count: int | None = None,
) -> np.ndarray:
    """Return a deterministic landmark index subset with stable feature coverage."""
    if total_count < 1:
        raise ValueError("total_count must be >= 1")
    idx = tuple(int(i) for i in indices)
    if not idx:
        raise ValueError("indices must not be empty")
    if max_count is not None and int(max_count) < 1:
        raise ValueError("max_count must be >= 1")
    max_idx = max(idx)
    if total_count <= max_idx:
        raise ValueError(
            f"landmark array too short for requested index {max_idx} (N={total_count})"
        )
    if max_count is None or max_count <= len(idx):
        return np.array(idx[: max_count if max_count is not None else len(idx)], dtype=np.int32)

    target_count = min(int(max_count), int(total_count))
    selected_list = list(idx)
    needed = target_count - len(selected_list)
    if needed > 0:
        selected_set = set(selected_list)
        candidates = np.array(
            [i for i in range(int(total_count)) if i not in selected_set],
            dtype=np.int32,
        )
        if needed >= len(candidates):
            extra = candidates
        else:
            pos = np.linspace(0, len(candidates) - 1, num=needed, dtype=np.int32)
            extra = candidates[pos]
        selected = np.concatenate(
            [np.array(selected_list, dtype=np.int32), extra.astype(np.int32)],
            axis=0,
        )
    else:
        selected = np.array(selected_list, dtype=np.int32)
    return selected
