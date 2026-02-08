"""FaceFX entry point (MVP pipeline wiring)."""

from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import cv2
import numpy as np

from facefx.src.blend import blend_with_mask
from facefx.src.facemesh import FaceMeshTracker
from facefx.src.patchbank import PatchBank, PatchFace, fallback_patch, load_patch_faces
from facefx.src.regions import feather_mask, region_polygon
from facefx.src.triangulation import triangulate
from facefx.src.ui import draw_label, wait_key
from facefx.src.warp import warp_triangle

WANTED_FPS = 30
PATCH_DIR = "patches"
WINDOW_NAME = "FaceFX (Window Capture this in OBS)"
FEATHER_PX = 25

REGIONS = {
    "forehead": [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ],
    "eyes": [33, 133, 159, 145, 153, 154, 155, 133, 362, 263, 386, 374, 380, 381, 382, 263],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FaceFX MVP")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    return parser.parse_args()


def _resize_patch_to_face(
    patch: np.ndarray, face_hull: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, bw, bh = cv2.boundingRect(face_hull)
    x = max(0, x)
    y = max(0, y)
    bw = max(1, bw)
    bh = max(1, bh)
    patch_bgr = patch[:, :, :3] if patch.ndim == 3 and patch.shape[2] == 4 else patch
    patch_resized = cv2.resize(patch_bgr, (bw, bh), interpolation=cv2.INTER_LINEAR)
    return patch_resized, (x, y, bw, bh)


def _select_patch(bank: PatchBank) -> PatchFace:
    patch_face = bank.random_patch()
    if patch_face is None:
        return fallback_patch()
    return patch_face


def main() -> int:
    args = _parse_args()

    tracker = FaceMeshTracker()

    patch_dir = os.path.join(os.path.dirname(__file__), PATCH_DIR)
    patch_faces = load_patch_faces(patch_dir, tracker)
    bank = PatchBank(patch_faces)
    current_patch = _select_patch(bank)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, WANTED_FPS)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        landmarks = tracker.process_bgr(frame)
        out = frame.copy()

        if landmarks is not None:
            face_hull = cv2.convexHull(landmarks.astype(np.float32))

            patch_img = current_patch.image
            patch_landmarks = current_patch.landmarks

            if patch_landmarks is None:
                patch_resized, (x, y, bw, bh) = _resize_patch_to_face(patch_img, face_hull)

                dst_pts = landmarks.copy()
                dst_pts[:, 0] = np.clip(dst_pts[:, 0], x, x + bw - 1)
                dst_pts[:, 1] = np.clip(dst_pts[:, 1], y, y + bh - 1)

                src_pts = dst_pts.copy()
                src_pts[:, 0] = dst_pts[:, 0] - x
                src_pts[:, 1] = dst_pts[:, 1] - y
            else:
                patch_bgr = (
                    patch_img[:, :, :3]
                    if patch_img.ndim == 3 and patch_img.shape[2] == 4
                    else patch_img
                )
                patch_resized = patch_bgr
                src_pts = patch_landmarks.copy()
                dst_pts = landmarks.copy()

            tri = triangulate(dst_pts)

            warped_canvas = out.copy()
            paint_mask = np.zeros((h, w), dtype=np.float32)

            region_masks: dict[str, np.ndarray] = {}
            for name, idxs in REGIONS.items():
                poly = region_polygon(landmarks, idxs)
                region_masks[name] = feather_mask((h, w), poly, FEATHER_PX)

            combined_region = np.zeros((h, w), dtype=np.float32)
            for rname in ("forehead", "eyes", "mouth"):
                combined_region = np.maximum(combined_region, region_masks[rname])

            for tri_idx in tri.simplices:
                t_dst = dst_pts[tri_idx]
                cx = float(np.mean(t_dst[:, 0]))
                cy = float(np.mean(t_dst[:, 1]))
                if cx < 0 or cy < 0 or cx >= w or cy >= h:
                    continue
                if combined_region[int(cy), int(cx)] < 0.05:
                    continue

                t_src = src_pts[tri_idx].copy()

                if patch_landmarks is None:
                    t_src[:, 0] = np.clip(t_src[:, 0], 0, patch_resized.shape[1] - 1)
                    t_src[:, 1] = np.clip(t_src[:, 1], 0, patch_resized.shape[0] - 1)

                warp_triangle(patch_resized, warped_canvas, t_src, t_dst, dst_mask_accum=paint_mask)

            blend_mask = np.clip(combined_region * paint_mask, 0.0, 1.0)
            out = blend_with_mask(out, warped_canvas, blend_mask)

        draw_label(out, "Mode: patch-swap | [SPACE]=new patch | [ESC]=quit")
        cv2.imshow(WINDOW_NAME, out)

        key = wait_key(1)
        if key == 27:
            break
        if key == ord(" "):
            current_patch = _select_patch(bank)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
