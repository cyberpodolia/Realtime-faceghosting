"""FaceFX entry point (MVP pipeline wiring)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field

if __package__ in (None, ""):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import cv2
import numpy as np

from facefx.src.blend import blend_with_mask, color_match_lab
from facefx.src.facemesh import FaceMeshTracker
from facefx.src.patchbank import PatchBank, PatchFace, fallback_patch, load_patch_faces
from facefx.src.regions import feather_mask, region_polygon
from facefx.src.device import resolve_device
from facefx.src.triangulation import TopologyCache, triangulate
from facefx.src.ui import draw_label, wait_key
from facefx.src.warp import warp_triangle

WANTED_FPS = 30
PATCH_DIR = "patches"
WINDOW_NAME = "FaceFX (Window Capture this in OBS)"
FEATHER_PX = 25
COLOR_MATCH_AB = 0.5
SHADING_KERNEL = 51
SHADING_CLAMP = (0.6, 1.6)

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


def _active_region_names(region_arg: str) -> tuple[str, ...]:
    if region_arg == "all":
        return ("forehead", "eyes", "mouth")
    return (region_arg,)


def _expand_region_indices(
    seed_indices: set[int],
    simplices: np.ndarray,
    *,
    hops: int = 1,
) -> set[int]:
    if not seed_indices or simplices.size == 0 or hops <= 0:
        return set(seed_indices)
    expanded = set(seed_indices)
    frontier = set(seed_indices)
    for _ in range(hops):
        nxt: set[int] = set()
        for tri_idx in simplices:
            tri_set = {int(tri_idx[0]), int(tri_idx[1]), int(tri_idx[2])}
            if tri_set.intersection(frontier):
                nxt.update(tri_set)
        frontier = nxt - expanded
        if not frontier:
            break
        expanded.update(nxt)
    return expanded


def _filter_simplices_by_indices(simplices: np.ndarray, active_indices: set[int]) -> np.ndarray:
    if not active_indices or simplices.size == 0:
        return simplices
    active = np.fromiter(sorted(active_indices), dtype=np.int32)
    # Keep only triangles fully contained in the expanded active landmark set.
    keep = np.all(np.isin(simplices, active), axis=1)
    return simplices[keep]


def _choice_on_off(value: str) -> str:
    value = value.lower()
    if value not in {"on", "off"}:
        raise argparse.ArgumentTypeError("expected 'on' or 'off'")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FaceFX MVP")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate setup and exit without camera/UI"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Show per-stage timings/FPS overlay",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="FaceMesh input scale factor (0<scale<=1.0). Landmarks are scaled back to frame.",
    )
    parser.add_argument(
        "--refine-landmarks",
        type=_choice_on_off,
        default="on",
        help="Use MediaPipe refined landmarks (on/off, default: on)",
    )
    parser.add_argument(
        "--topology",
        choices=("frozen", "mediapipe"),
        default="frozen",
        help="Triangle topology mode (wired in later perf steps)",
    )
    parser.add_argument(
        "--region",
        choices=("forehead", "eyes", "mouth", "all"),
        default="all",
        help="Region subset mode (wired in later perf steps)",
    )
    parser.add_argument(
        "--color-match-every",
        type=int,
        default=1,
        help="Run color matching every N frames (default: 1)",
    )
    parser.add_argument(
        "--shading",
        type=_choice_on_off,
        default="on",
        help="Enable L-channel shading match (on/off, default: on)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device for optional accelerations (auto|cpu|cuda).",
    )
    return parser


def _parse_args() -> argparse.Namespace:
    args = _build_parser().parse_args()
    if not (0 < args.scale <= 1.0):
        raise SystemExit("--scale must be in (0, 1.0]")
    if args.color_match_every < 1:
        raise SystemExit("--color-match-every must be >= 1")
    return args


@dataclass
class FrameProfiler:
    enabled: bool
    _frame_start: float = 0.0
    _last_mark: float = 0.0
    _dur_ms: dict[str, float] = field(default_factory=dict)
    _ema_ms: dict[str, float] = field(default_factory=dict)
    _ema_frame_ms: float | None = None

    def start_frame(self) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self._frame_start = now
        self._last_mark = now
        self._dur_ms.clear()

    def mark(self, stage: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self._dur_ms[stage] = (now - self._last_mark) * 1000.0
        self._last_mark = now

    def end_frame(self) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        frame_ms = (now - self._frame_start) * 1000.0
        alpha = 0.2
        self._ema_frame_ms = (
            frame_ms
            if self._ema_frame_ms is None
            else (1.0 - alpha) * self._ema_frame_ms + alpha * frame_ms
        )
        for stage, dur in self._dur_ms.items():
            prev = self._ema_ms.get(stage)
            self._ema_ms[stage] = dur if prev is None else (1.0 - alpha) * prev + alpha * dur

    def overlay_lines(self) -> list[str]:
        if not self.enabled:
            return []
        frame_ms = self._ema_frame_ms or 0.0
        fps = (1000.0 / frame_ms) if frame_ms > 0 else 0.0
        parts = [f"FPS {fps:4.1f}", f"frame {frame_ms:5.1f}ms"]
        stage_order = ("capture", "facemesh", "masks", "warp", "color_match", "blend", "display")
        for stage in stage_order:
            if stage in self._ema_ms:
                parts.append(f"{stage}:{self._ema_ms[stage]:.1f}")
        return [" | ".join(parts)]


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


def _roi_from_mask(mask: np.ndarray, thresh: float = 0.05) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > thresh)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _roi_from_polys(
    polys: list[np.ndarray],
    shape_hw: tuple[int, int],
    *,
    pad: int = 0,
) -> tuple[int, int, int, int] | None:
    if not polys:
        return None
    h, w = shape_hw
    pts = np.concatenate([p.reshape(-1, 2) for p in polys], axis=0)
    x0 = max(0, int(np.floor(np.min(pts[:, 0]))) - pad)
    y0 = max(0, int(np.floor(np.min(pts[:, 1]))) - pad)
    x1 = min(w - 1, int(np.ceil(np.max(pts[:, 0]))) + pad)
    y1 = min(h - 1, int(np.ceil(np.max(pts[:, 1]))) + pad)
    if x1 < x0 or y1 < y0:
        return None
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _shift_poly(poly: np.ndarray, dx: float, dy: float) -> np.ndarray:
    shifted = poly.astype(np.float32).copy()
    shifted[:, 0] -= dx
    shifted[:, 1] -= dy
    return shifted


def main() -> int:
    args = _parse_args()

    # Import-time OpenCV failures should not break `--help` or test collection.
    device_info = resolve_device(device=args.device, cv2=cv2)
    use_cuda = device_info.name == "cuda"
    device_tag = device_info.name.upper()

    tracker = FaceMeshTracker(refine_landmarks=args.refine_landmarks == "on")

    patch_dir = os.path.join(os.path.dirname(__file__), PATCH_DIR)
    patch_faces = load_patch_faces(patch_dir, tracker)
    bank = PatchBank(patch_faces)
    current_patch = _select_patch(bank)

    if args.dry_run:
        print(
            "FaceFX dry-run OK "
            f"(patches={len(patch_faces)}, scale={args.scale}, "
            f"refine_landmarks={args.refine_landmarks}, topology={args.topology}, "
            "region="
            f"{args.region}, color_match_every={args.color_match_every}, "
            f"shading={args.shading}, device={device_info.name}, cuda_available={device_info.cuda_available})"
        )
        return 0

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, WANTED_FPS)
    profiler = FrameProfiler(enabled=args.profile)
    frame_index = 0
    topology_cache = TopologyCache()

    while True:
        profiler.start_frame()
        ok, frame = cap.read()
        profiler.mark("capture")
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        landmarks = tracker.process_bgr(frame, scale=args.scale)
        profiler.mark("facemesh")
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

            tri = triangulate(dst_pts, topology=args.topology, cache=topology_cache)

            active_regions = _active_region_names(args.region)
            region_polys: dict[str, np.ndarray] = {}
            for name in active_regions:
                idxs = REGIONS[name]
                region_polys[name] = region_polygon(landmarks, idxs)

            region_roi = _roi_from_polys(
                list(region_polys.values()),
                (h, w),
                pad=FEATHER_PX + 4,
            )
            if region_roi is None:
                profiler.mark("masks")
                profiler.mark("warp")
                profiler.mark("color_match")
                profiler.mark("blend")
                draw_label(out, f"[{device_tag}] Mode: patch-swap | [SPACE]=new patch | [ESC]=quit")
                for i, line in enumerate(profiler.overlay_lines(), start=1):
                    draw_label(out, line, pos=(10, 30 + 30 * i))
                cv2.imshow(WINDOW_NAME, out)
                key = wait_key(1)
                profiler.mark("display")
                profiler.end_frame()
                if key == 27:
                    break
                if key == ord(" "):
                    current_patch = _select_patch(bank)
                frame_index += 1
                continue

            rx, ry, rw, rh = region_roi
            region_masks: dict[str, np.ndarray] = {}
            for name in active_regions:
                poly = _shift_poly(region_polys[name], rx, ry)
                region_masks[name] = feather_mask((rh, rw), poly, FEATHER_PX)
            profiler.mark("masks")

            combined_region = np.zeros((rh, rw), dtype=np.float32)
            for rname in active_regions:
                combined_region = np.maximum(combined_region, region_masks[rname])

            if args.region == "all":
                # Preserve legacy visual coverage for the default mode.
                active_simplices = tri.simplices
            else:
                seed_indices = {idx for rname in active_regions for idx in REGIONS[rname]}
                active_indices = _expand_region_indices(seed_indices, tri.simplices, hops=1)
                active_simplices = _filter_simplices_by_indices(tri.simplices, active_indices)

            warped_canvas_roi = out[ry : ry + rh, rx : rx + rw].copy()
            paint_mask_roi = np.zeros((rh, rw), dtype=np.float32)

            for tri_idx in active_simplices:
                t_dst_full = dst_pts[tri_idx]
                cx = float(np.mean(t_dst_full[:, 0]))
                cy = float(np.mean(t_dst_full[:, 1]))
                if cx < rx or cy < ry or cx >= rx + rw or cy >= ry + rh:
                    continue
                if combined_region[int(cy - ry), int(cx - rx)] < 0.05:
                    continue

                t_src = src_pts[tri_idx].copy()
                t_dst = t_dst_full.copy()
                t_dst[:, 0] -= rx
                t_dst[:, 1] -= ry

                if patch_landmarks is None:
                    t_src[:, 0] = np.clip(t_src[:, 0], 0, patch_resized.shape[1] - 1)
                    t_src[:, 1] = np.clip(t_src[:, 1], 0, patch_resized.shape[0] - 1)

                warp_triangle(
                    patch_resized,
                    warped_canvas_roi,
                    t_src,
                    t_dst,
                    dst_mask_accum=paint_mask_roi,
                )
            profiler.mark("warp")

            blend_mask = np.clip(combined_region * paint_mask_roi, 0.0, 1.0)
            roi = _roi_from_mask(blend_mask)
            if roi is not None:
                x, y, cw, ch = roi
                src_roi = warped_canvas_roi[y : y + ch, x : x + cw]
                tgt_roi = out[ry + y : ry + y + ch, rx + x : rx + x + cw]
                mask_roi = blend_mask[y : y + ch, x : x + cw]
                if frame_index % args.color_match_every == 0:
                    matched = color_match_lab(
                        src_roi,
                        tgt_roi,
                        mask_roi,
                        ab_strength=COLOR_MATCH_AB,
                        shading=args.shading == "on",
                        shading_kernel=SHADING_KERNEL,
                        shading_clamp=SHADING_CLAMP,
                        use_cuda=use_cuda,
                    )
                    warped_canvas_roi[y : y + ch, x : x + cw] = matched
            profiler.mark("color_match")

            out_roi = out[ry : ry + rh, rx : rx + rw]
            out[ry : ry + rh, rx : rx + rw] = blend_with_mask(
                out_roi,
                warped_canvas_roi,
                blend_mask,
            )
            profiler.mark("blend")
        else:
            profiler.mark("masks")
            profiler.mark("warp")
            profiler.mark("color_match")
            profiler.mark("blend")

        draw_label(out, f"[{device_tag}] Mode: patch-swap | [SPACE]=new patch | [ESC]=quit")
        for i, line in enumerate(profiler.overlay_lines(), start=1):
            draw_label(out, line, pos=(10, 30 + 30 * i))
        cv2.imshow(WINDOW_NAME, out)

        key = wait_key(1)
        profiler.mark("display")
        profiler.end_frame()
        if key == 27:
            break
        if key == ord(" "):
            current_patch = _select_patch(bank)
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
