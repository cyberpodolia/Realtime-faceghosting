"""CLI entrypoint for the runtime_cuda pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from facefx.src.facemesh import FaceMeshTracker
from facefx.src.patchbank import PatchBank, PatchFace, fallback_patch, load_patch_faces

from .config import RuntimeConfig, config_from_namespace
from .landmarks import extract_control_points, select_landmark_indices, smooth_landmarks
from .pipeline import RuntimePipeline
from .roi import compute_adaptive_roi, draw_roi_overlay
from .topology import mediapipe_simplices_for_indices

WINDOW_NAME = "FaceFX Runtime CUDA"
CAMERA_INDEX = 1
PATCH_DIR = "patches"


def _open_camera(camera_index: int) -> tuple[cv2.VideoCapture, str] | tuple[None, str]:
    attempts = (
        (cv2.CAP_DSHOW, "dshow"),
        (cv2.CAP_MSMF, "msmf"),
        (None, "default"),
    )
    for backend, name in attempts:
        cap = cv2.VideoCapture(camera_index) if backend is None else cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            return cap, name
        cap.release()
    return None, "unavailable"


def _choice_overlay(value: str) -> str:
    lowered = value.lower()
    if lowered not in {"off", "basic", "roi"}:
        raise argparse.ArgumentTypeError("debug overlay must be one of: off, basic, roi")
    return lowered


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FaceFX runtime_cuda")
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help="Camera index (ignored: runtime uses camera=1).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Requested execution device tag for runtime path.",
    )
    parser.add_argument("--input-width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--input-height", type=int, default=720, help="Capture height.")
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=30.0,
        help="Requested camera FPS.",
    )
    parser.add_argument(
        "--debug-overlay",
        type=_choice_overlay,
        default="off",
        help="Debug overlay mode: off|basic|roi",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Show lightweight frame-time / FPS debug overlay.",
    )
    parser.add_argument(
        "--dump-roi-overlay",
        action="store_true",
        help="Save one ROI+anchor debug frame to --roi-overlay-path.",
    )
    parser.add_argument(
        "--roi-overlay-path",
        default="scripts/baseline_artifacts/runtime_cuda_roi_overlay.png",
        help="Output path used by --dump-roi-overlay.",
    )
    parser.add_argument(
        "--roi-expand",
        type=float,
        default=0.18,
        help="Adaptive ROI expansion ratio (0..1).",
    )
    parser.add_argument(
        "--roi-min-size",
        type=int,
        default=96,
        help="Minimum adaptive ROI size in pixels.",
    )
    parser.add_argument(
        "--control-point-count",
        type=int,
        default=24,
        help="Control point count cap for debug visualization.",
    )
    parser.add_argument(
        "--warp-point-count",
        type=int,
        default=0,
        help="Landmark count used by triangle warp path (0 = full mesh).",
    )
    parser.add_argument(
        "--landmark-every",
        type=int,
        default=3,
        help="Run MediaPipe once every N frames and reuse between updates.",
    )
    parser.add_argument(
        "--landmark-scale",
        type=float,
        default=1.0,
        help="Downscale factor for MediaPipe input (0.1..1.0).",
    )
    parser.add_argument(
        "--landmark-smooth",
        type=float,
        default=0.25,
        help="Hold smoothing factor (0=no smoothing, 0.6 keeps more previous state).",
    )
    parser.add_argument(
        "--color-match-every",
        type=int,
        default=1,
        help="Run color+shading every N frames (1 means every frame).",
    )
    parser.add_argument(
        "--shading-every",
        type=int,
        default=1,
        help="Run shading every N frames and hold between updates (1 means every frame).",
    )
    parser.add_argument(
        "--shading-strength",
        type=float,
        default=0.35,
        help="Shading blend strength (0..1).",
    )
    parser.add_argument(
        "--composite-backend",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Composite backend selection.",
    )
    parser.add_argument(
        "--composite-cuda-min-area",
        type=int,
        default=180000,
        help="In auto mode use CUDA composite only when ROI area >= this threshold.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 means unlimited).",
    )
    parser.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="Smoke mode: run for N frames then exit (0 disables).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate runtime_cuda setup and exit without opening camera.",
    )
    parser.add_argument(
        "--experimental-dense",
        action="store_true",
        help="Enable experimental dense face-warp path (can be unstable).",
    )
    parser.add_argument(
        "--debug-flat-patch",
        action="store_true",
        help="Debug view: replace patch texture with a flat color to inspect color/shading base.",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if args.input_width < 1 or args.input_height < 1:
        raise SystemExit("--input-width and --input-height must be >= 1")
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be >= 0")
    if args.smoke < 0:
        raise SystemExit("--smoke must be >= 0")
    if args.camera_fps <= 0.0:
        raise SystemExit("--camera-fps must be > 0")
    if args.roi_min_size < 1:
        raise SystemExit("--roi-min-size must be >= 1")
    if args.control_point_count < 1:
        raise SystemExit("--control-point-count must be >= 1")
    if args.warp_point_count < 0:
        raise SystemExit("--warp-point-count must be >= 0")
    if args.landmark_every < 1:
        raise SystemExit("--landmark-every must be >= 1")
    if not (0.1 <= args.landmark_scale <= 1.0):
        raise SystemExit("--landmark-scale must be in [0.1, 1.0]")
    if not (0.0 <= args.landmark_smooth <= 0.95):
        raise SystemExit("--landmark-smooth must be in [0.0, 0.95]")
    if args.color_match_every < 1:
        raise SystemExit("--color-match-every must be >= 1")
    if args.shading_every < 1:
        raise SystemExit("--shading-every must be >= 1")
    if not (0.0 <= args.shading_strength <= 1.0):
        raise SystemExit("--shading-strength must be in [0.0, 1.0]")
    if args.composite_cuda_min_area < 1:
        raise SystemExit("--composite-cuda-min-area must be >= 1")
    return args


def _draw_overlay(frame: object, cfg: RuntimeConfig, frame_index: int) -> None:
    if cfg.debug_overlay != "basic":
        return
    cv2.putText(
        frame,
        f"runtime_cuda | frame={frame_index} | device={cfg.device}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_profile_overlay(
    frame: object, frame_ms: float, fps: float, requested_fps: float, reported_fps: float
) -> None:
    cv2.putText(
        frame,
        (
            f"FPS {fps:4.1f} | frame {frame_ms:5.1f}ms | "
            f"cam_req={requested_fps:4.1f} cam_rep={reported_fps:4.1f}"
        ),
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 255, 120),
        1,
        cv2.LINE_AA,
    )


def _smooth_landmarks(prev: np.ndarray | None, current: np.ndarray, smooth: float) -> np.ndarray:
    return smooth_landmarks(prev, current, smooth)


def _runtime_patch_dir() -> Path:
    return Path(__file__).resolve().parents[1] / PATCH_DIR


def _split_patch_image(
    patch_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    patch = np.asarray(patch_image)
    if patch.ndim != 3:
        raise ValueError("patch image must have shape [H, W, C]")
    if patch.shape[2] == 4:
        bgr = patch[:, :, :3].astype(np.uint8, copy=False)
        alpha = patch[:, :, 3].astype(np.uint8, copy=False)
        return bgr, alpha
    if patch.shape[2] == 3:
        bgr = patch.astype(np.uint8, copy=False)
        alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
        return bgr, alpha
    raise ValueError("patch image must have 3 or 4 channels")


def _prepare_patch_runtime_assets(
    patch_face: PatchFace,
    *,
    control_point_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    patch_bgr, patch_alpha = _split_patch_image(patch_face.image)
    if patch_face.landmarks is None:
        return patch_bgr, patch_alpha, None, None
    control_points = extract_control_points(
        patch_face.landmarks,
        max_count=control_point_count,
    )
    return (
        patch_bgr,
        patch_alpha,
        patch_face.landmarks.astype(np.float32, copy=True),
        control_points[:control_point_count].copy(),
    )


def _flat_patch_from_image(patch_bgr: np.ndarray) -> np.ndarray:
    src = np.asarray(patch_bgr)
    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError("patch_bgr must have shape [H, W, 3]")
    flat_color = np.median(src.reshape(-1, 3), axis=0).astype(np.uint8)
    return np.full(src.shape, flat_color.reshape(1, 1, 3), dtype=np.uint8)


def _select_patch(bank: PatchBank) -> PatchFace:
    patch = bank.random_patch()
    if patch is None:
        return fallback_patch()
    return patch


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = config_from_namespace(args)

    if args.dry_run:
        print(
            "runtime_cuda dry-run OK "
            f"(camera={cfg.camera}, device={cfg.device}, "
            f"input={cfg.input_width}x{cfg.input_height}, camera_fps={cfg.camera_fps}, "
            f"overlay={cfg.debug_overlay}, max_frames={cfg.max_frames}, "
            f"smoke={args.smoke}, "
            f"profile={cfg.profile}, dump_roi_overlay={cfg.dump_roi_overlay}, "
            f"landmark_every={cfg.landmark_every}, "
            f"landmark_scale={cfg.landmark_scale}, "
            f"landmark_smooth={cfg.landmark_smooth}, "
            f"warp_point_count={cfg.warp_point_count}, "
            f"color_match_every={cfg.color_match_every}, "
            f"shading_every={cfg.shading_every}, "
            f"shading_strength={cfg.shading_strength}, "
            f"composite_backend={cfg.composite_backend}, "
            f"composite_cuda_min_area={cfg.composite_cuda_min_area}, "
            f"experimental_dense={args.experimental_dense})"
        )
        return 0

    cap, camera_backend = _open_camera(CAMERA_INDEX)
    if cap is None:
        print(f"failed to open camera index {CAMERA_INDEX}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.input_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.input_height)
    cap.set(cv2.CAP_PROP_FPS, cfg.camera_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"failed to open camera index {CAMERA_INDEX}", file=sys.stderr)
        cap.release()
        return 1
    print(f"runtime_cuda: camera backend={camera_backend}")
    reported_camera_fps = float(cap.get(cv2.CAP_PROP_FPS))

    pipeline = RuntimePipeline(cfg)
    tracker = FaceMeshTracker(refine_landmarks=False)
    patch_bank: PatchBank | None = None
    patch_bgr: np.ndarray | None = None
    patch_alpha: np.ndarray | None = None
    patch_landmarks: np.ndarray | None = None
    patch_control_points: np.ndarray | None = None
    patch_dir = _runtime_patch_dir()
    patch_loader_tracker = FaceMeshTracker(refine_landmarks=False)
    patch_faces = load_patch_faces(str(patch_dir), patch_loader_tracker)
    print(f"runtime_cuda: loaded patch faces={len(patch_faces)} from {patch_dir}")
    patch_bank = PatchBank(patch_faces)
    current_patch = _select_patch(patch_bank)
    patch_bgr, patch_alpha, patch_landmarks, patch_control_points = _prepare_patch_runtime_assets(
        current_patch,
        control_point_count=cfg.control_point_count,
    )
    use_cuda = cfg.device != "cpu"
    frame_count = 0
    dumped_roi = False
    max_frames = cfg.max_frames if cfg.max_frames > 0 else (300 if cfg.dump_roi_overlay else 0)
    if args.smoke > 0 and max_frames == 0:
        max_frames = int(args.smoke)
    last_ts = time.perf_counter()
    ema_frame_ms: float | None = None
    debug_pass_index = -1
    pipeline.set_debug_passes_enabled(False)
    landmarks_ms: float = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            out: np.ndarray | None = None
            stage_timings = None
            roi = None
            anchors = None
            should_update = pipeline.should_update_landmarks(force=cfg.dump_roi_overlay)
            if should_update:
                t_landmarks = time.perf_counter()
                landmarks = tracker.process_bgr(frame, scale=cfg.landmark_scale)
                landmarks_ms = (time.perf_counter() - t_landmarks) * 1000.0
            else:
                landmarks = None
                landmarks_ms = 0.0
            landmark_state = pipeline.update_landmarks(landmarks)
            held_landmarks = landmark_state.landmarks_xy
            if held_landmarks is not None:
                anchors = extract_control_points(
                    held_landmarks,
                    max_count=cfg.control_point_count,
                )[: cfg.control_point_count]
                roi = compute_adaptive_roi(
                    held_landmarks,
                    frame.shape[:2],
                    expand=cfg.roi_expand,
                    min_size=cfg.roi_min_size,
                )
                run_color = (frame_count % cfg.color_match_every) == 0
                run_shading = (frame_count % cfg.shading_every) == 0
                if args.experimental_dense:
                    if (
                        patch_bgr is not None
                        and patch_alpha is not None
                        and patch_control_points is not None
                    ):
                        source_patch = (
                            _flat_patch_from_image(patch_bgr) if args.debug_flat_patch else patch_bgr
                        )
                        n = min(len(anchors), len(patch_control_points))
                        if n >= 3:
                            out, stage_timings = pipeline.process_dense_roi(
                                frame,
                                source_patch,
                                patch_alpha,
                                patch_control_points[:n],
                                anchors[:n],
                                roi,
                                landmarks_xy=held_landmarks,
                                enable_color_match=run_color,
                                enable_shading=run_shading,
                                use_cuda=use_cuda,
                            )
                elif (
                    patch_bgr is not None
                    and patch_alpha is not None
                    and patch_landmarks is not None
                ):
                    source_patch = _flat_patch_from_image(patch_bgr) if args.debug_flat_patch else patch_bgr
                    n = min(len(held_landmarks), len(patch_landmarks))
                    if cfg.warp_point_count > 0:
                        warp_indices = select_landmark_indices(
                            n,
                            max_count=min(cfg.warp_point_count, n),
                        )
                    else:
                        warp_indices = np.arange(n, dtype=np.int32)
                    simplices = mediapipe_simplices_for_indices(tuple(int(i) for i in warp_indices))
                    if len(warp_indices) >= 3 and len(simplices) > 0:
                        out, stage_timings = pipeline.process_triangle_roi(
                            frame,
                            source_patch,
                            patch_alpha,
                            patch_landmarks[warp_indices],
                            held_landmarks[warp_indices],
                            simplices,
                            roi,
                            landmarks_xy=held_landmarks,
                            enable_color_match=run_color,
                            enable_shading=run_shading,
                            use_native=True,
                            use_cuda=use_cuda,
                        )
            else:
                pipeline.reset_postprocess_state()
            if out is None:
                out = pipeline.process(frame)

            if (
                (cfg.debug_overlay == "roi" or cfg.dump_roi_overlay)
                and roi is not None
                and anchors is not None
            ):
                out = draw_roi_overlay(out, roi, anchors)
                if cfg.dump_roi_overlay and not dumped_roi:
                    out_path = Path(cfg.roi_overlay_path)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if cv2.imwrite(str(out_path), out):
                        print(f"wrote {out_path}")
                        dumped_roi = True

            _draw_overlay(out, cfg, pipeline.frame_index)
            debug_pass_name = "final"
            display = out
            debug_passes = []
            if debug_pass_index >= 0:
                debug_passes = pipeline.debug_passes()
                if debug_passes:
                    if debug_pass_index >= len(debug_passes):
                        debug_pass_index = 0
                    debug_pass = debug_passes[debug_pass_index]
                    display = debug_pass.frame_bgr.copy()
                    debug_pass_name = debug_pass.name
                else:
                    debug_pass_index = -1

            now_ts = time.perf_counter()
            frame_ms = (now_ts - last_ts) * 1000.0
            last_ts = now_ts
            alpha = 0.2
            ema_frame_ms = (
                frame_ms
                if ema_frame_ms is None
                else (1.0 - alpha) * ema_frame_ms + alpha * frame_ms
            )
            if cfg.profile:
                fps = (1000.0 / ema_frame_ms) if ema_frame_ms and ema_frame_ms > 0 else 0.0
                _draw_profile_overlay(display, ema_frame_ms, fps, cfg.camera_fps, reported_camera_fps)
                cv2.putText(
                    display,
                    (
                        f"lm_every={cfg.landmark_every} lm_scale={cfg.landmark_scale:.2f} "
                        f"warp_pts={cfg.warp_point_count} cm_every={cfg.color_match_every} "
                        f"sh_every={cfg.shading_every}"
                    ),
                    (12, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (120, 255, 120),
                    1,
                    cv2.LINE_AA,
                )
                if stage_timings is not None:
                    stage_line = (
                        f"map={stage_timings.map_build_ms:.1f} "
                        f"remap={stage_timings.remap_ms:.1f} "
                        f"color={stage_timings.color_ms:.1f} "
                        f"shade={stage_timings.shading_ms:.1f} "
                        f"comp={stage_timings.composite_ms:.1f}ms"
                    )
                    backend_line = (
                        f"backend remap={stage_timings.remap_backend} "
                        f"color={stage_timings.color_backend} "
                        f"comp={stage_timings.composite_backend}"
                    )
                    cv2.putText(
                        display,
                        stage_line,
                        (12, 106),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (120, 220, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        display,
                        backend_line,
                        (12, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (120, 220, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        display,
                        f"landmarks={landmarks_ms:.1f}ms",
                        (12, 154),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (120, 220, 255),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        display,
                        ("no face/patch landmarks"),
                        (12, 106),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (80, 160, 255),
                        1,
                        cv2.LINE_AA,
                    )
            if debug_pass_index >= 0:
                cv2.putText(
                    display,
                    f"debug-pass: {debug_pass_name} ({debug_pass_index + 1}/{len(debug_passes)}) | m=next n=final",
                    (12, 178),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (80, 230, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            frame_count += 1
            if key == 27:
                break
            if key == ord("m"):
                pipeline.set_debug_passes_enabled(True)
                if debug_pass_index < 0:
                    debug_pass_index = 0
                else:
                    debug_passes = pipeline.debug_passes()
                    if debug_passes:
                        debug_pass_index = (debug_pass_index + 1) % len(debug_passes)
                continue
            if key == ord("n"):
                debug_pass_index = -1
                pipeline.set_debug_passes_enabled(False)
                continue
            if key == ord(" ") and args.experimental_dense and patch_bank is not None:
                current_patch = _select_patch(patch_bank)
                patch_bgr, patch_alpha, patch_landmarks, patch_control_points = (
                    _prepare_patch_runtime_assets(
                        current_patch,
                        control_point_count=cfg.control_point_count,
                    )
                )
                pipeline.reset_postprocess_state()
            elif key == ord(" ") and patch_bank is not None:
                current_patch = _select_patch(patch_bank)
                patch_bgr, patch_alpha, patch_landmarks, patch_control_points = (
                    _prepare_patch_runtime_assets(
                        current_patch,
                        control_point_count=cfg.control_point_count,
                    )
                )
                pipeline.reset_postprocess_state()
            if max_frames and frame_count >= max_frames:
                break
            if cfg.dump_roi_overlay and dumped_roi and cfg.max_frames == 0:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if cfg.dump_roi_overlay and not dumped_roi:
        print("failed to dump ROI overlay: no landmarks detected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
