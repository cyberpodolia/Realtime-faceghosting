"""Patch loading, face detection, and selection."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import cv2
import numpy as np

from facefx.src.facemesh import FaceMeshTracker


@dataclass
class PatchFace:
    image: np.ndarray
    landmarks: np.ndarray


def load_patch_faces(patch_dir: str, tracker: FaceMeshTracker) -> list[PatchFace]:
    patch_faces: list[PatchFace] = []
    if not os.path.isdir(patch_dir):
        return patch_faces
    for name in sorted(os.listdir(patch_dir)):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(patch_dir, name)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            bgr = img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img
            landmarks = tracker.process_bgr_static(bgr)
            if landmarks is None:
                continue
            patch_faces.append(PatchFace(image=img, landmarks=landmarks))
    return patch_faces


@dataclass
class PatchBank:
    patch_faces: list[PatchFace]

    def random_patch(self) -> PatchFace | None:
        if self.patch_faces:
            return random.choice(self.patch_faces)
        return None


def fallback_patch() -> PatchFace:
    h, w = 256, 256
    img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    landmarks = None
    return PatchFace(image=img, landmarks=landmarks)
