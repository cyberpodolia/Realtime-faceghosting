"""Measure raw camera FPS baseline without any FaceFX processing."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

CAMERA_INDEX = 1


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * (pct / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _backend_const(name: str) -> int:
    mapping = {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "any": cv2.CAP_ANY,
    }
    return mapping[name]


def _decode_fourcc(val: float) -> str:
    ival = int(val)
    chars = [chr((ival >> (8 * i)) & 0xFF) for i in range(4)]
    text = "".join(chars)
    return text if text.strip("\x00") else "----"


def _open_camera_with_fallback(
    camera_index: int, backend_name: str, allow_fallback: bool
) -> tuple[cv2.VideoCapture, int]:
    backend = _backend_const(backend_name)
    cap = cv2.VideoCapture(camera_index, backend)
    if cap.isOpened():
        return cap, camera_index
    if not allow_fallback:
        return cap, camera_index
    cap.release()
    for idx in range(10):
        if idx == camera_index:
            continue
        c = cv2.VideoCapture(idx, backend)
        if c.isOpened():
            return c, idx
        c.release()
    # Return a closed handle for unified caller path.
    return cv2.VideoCapture(camera_index, backend), camera_index


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help="Preferred camera index (default: 1).",
    )
    p.add_argument(
        "--fallback-camera",
        action="store_true",
        help="If requested camera is unavailable, try other indices [0..9].",
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument(
        "--backend",
        choices=("dshow", "msmf", "any"),
        default="dshow",
        help="VideoCapture backend to use.",
    )
    p.add_argument(
        "--request-fps",
        type=float,
        default=0.0,
        help="Requested capture FPS (0 disables explicit request).",
    )
    p.add_argument(
        "--fourcc",
        default="",
        help="Optional 4CC (e.g. MJPG, YUY2). Empty means no explicit format request.",
    )
    p.add_argument(
        "--auto-exposure",
        choices=("keep", "on", "off"),
        default="keep",
        help="Try setting camera auto exposure (Windows/OpenCV dependent).",
    )
    p.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Optional manual exposure value (applied after auto-exposure off).",
    )
    p.add_argument("--warmup-frames", type=int, default=60)
    p.add_argument(
        "--frames", type=int, default=600, help="Measured frames after warmup."
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Show camera preview window during measurement (adds display overhead).",
    )
    p.add_argument(
        "--out-root",
        default=str(Path("scripts") / "baseline_artifacts"),
        help="Output directory for JSON report.",
    )
    p.add_argument("--run-id", default=None)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.width < 1 or args.height < 1:
        raise SystemExit("--width and --height must be >= 1")
    if args.warmup_frames < 0:
        raise SystemExit("--warmup-frames must be >= 0")
    if args.frames < 1:
        raise SystemExit("--frames must be >= 1")
    if args.request_fps < 0:
        raise SystemExit("--request-fps must be >= 0")
    if args.fourcc and len(args.fourcc) != 4:
        raise SystemExit("--fourcc must be exactly 4 characters")

    cap, actual_camera_index = _open_camera_with_fallback(
        int(args.camera), args.backend, bool(args.fallback_camera)
    )
    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc.upper()))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    if args.request_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(args.request_fps))
    if args.auto_exposure == "on":
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    elif args.auto_exposure == "off":
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    if args.exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, float(args.exposure))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(
            f"failed to open camera index {args.camera} (and no fallback index found)",
            file=sys.stderr,
        )
        cap.release()
        return 1

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps_reported = float(cap.get(cv2.CAP_PROP_FPS))
    actual_fourcc = _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    try:
        backend_name = cap.getBackendName()
    except Exception:
        backend_name = args.backend

    read_ms: list[float] = []
    total = args.warmup_frames + args.frames
    wall_start = time.perf_counter()
    last_frame_ts = time.perf_counter()
    ema_frame_ms: float | None = None

    try:
        for idx in range(total):
            t0 = time.perf_counter()
            ok, _frame = cap.read()
            t1 = time.perf_counter()
            if not ok:
                raise RuntimeError(f"camera read failed at frame {idx}")
            read_time_ms = (t1 - t0) * 1000.0
            if idx >= args.warmup_frames:
                read_ms.append(read_time_ms)

            now = time.perf_counter()
            frame_ms = (now - last_frame_ts) * 1000.0
            last_frame_ts = now
            alpha = 0.2
            ema_frame_ms = (
                frame_ms
                if ema_frame_ms is None
                else (1.0 - alpha) * ema_frame_ms + alpha * frame_ms
            )
            if args.preview:
                current_fps = (
                    (1000.0 / ema_frame_ms)
                    if ema_frame_ms and ema_frame_ms > 0
                    else 0.0
                )
                measured_fps = (
                    1000.0 / (sum(read_ms) / len(read_ms)) if read_ms else 0.0
                )
                phase = (
                    f"warmup {idx + 1}/{args.warmup_frames}"
                    if idx < args.warmup_frames
                    else f"measure {idx + 1 - args.warmup_frames}/{args.frames}"
                )
                cv2.putText(
                    _frame,
                    f"raw camera benchmark | {phase}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    _frame,
                    (
                        f"current FPS: {current_fps:5.1f} | read: {read_time_ms:5.1f} ms "
                        f"| reported: {actual_fps_reported:4.1f}"
                    ),
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (120, 255, 120),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    _frame,
                    (
                        f"measured avg FPS: {measured_fps:5.1f} | "
                        f"{actual_w}x{actual_h} {actual_fourcc} {backend_name}"
                    ),
                    (12, 84),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (120, 255, 120),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("Camera Raw FPS", _frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    raise RuntimeError("stopped by user (ESC)")
    finally:
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()

    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    avg_read_ms = sum(read_ms) / len(read_ms)
    summary = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "camera_requested": int(args.camera),
        "camera_used": int(actual_camera_index),
        "requested": {
            "camera_fallback": bool(args.fallback_camera),
            "resolution": {"width": int(args.width), "height": int(args.height)},
            "backend": args.backend,
            "request_fps": float(args.request_fps),
            "fourcc": args.fourcc.upper() if args.fourcc else "",
            "auto_exposure": args.auto_exposure,
            "exposure": args.exposure,
        },
        "actual": {
            "resolution": {"width": actual_w, "height": actual_h},
            "backend": backend_name,
            "reported_fps": actual_fps_reported,
            "fourcc": actual_fourcc,
        },
        "warmup_frames": int(args.warmup_frames),
        "measured_frames": int(args.frames),
        "avg_read_ms": avg_read_ms,
        "p50_read_ms": _percentile(read_ms, 50.0),
        "p95_read_ms": _percentile(read_ms, 95.0),
        "avg_read_fps": (1000.0 / avg_read_ms) if avg_read_ms > 0 else 0.0,
        "wall_fps_including_loop": (total / (wall_ms / 1000.0)) if wall_ms > 0 else 0.0,
    }

    run_id = args.run_id or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "camera_raw_fps.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
