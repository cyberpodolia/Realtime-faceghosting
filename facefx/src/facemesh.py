"""FaceMesh wrapper utilities."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FaceMeshTracker:
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    def __post_init__(self) -> None:
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "mediapipe.solutions is missing. Install mediapipe==0.10.14 and reinstall deps."
            )
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._face_mesh_static = self._mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def process_bgr(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Return landmark array (N,2) in pixel coords, or None if no face."""
        return self._process(frame_bgr, self._face_mesh)

    def process_bgr_static(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Return landmarks for a still image, or None if no face."""
        return self._process(image_bgr, self._face_mesh_static)

    def _process(self, image_bgr: np.ndarray, mesh) -> np.ndarray | None:
        if image_bgr is None or image_bgr.size == 0:
            return None
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        face = res.multi_face_landmarks[0]
        h, w = image_bgr.shape[:2]
        pts = np.array([(p.x * w, p.y * h) for p in face.landmark], dtype=np.float32)
        return pts
