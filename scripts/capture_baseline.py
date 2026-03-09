"""Capture reproducible baseline performance evidence for the current FaceFX pipeline."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CAMERA_INDEX = 1

STAGE_ORDER = (
    "capture",
    "landmarks",
    "mask",
    "warp",
    "color",
    "shading",
    "composite",
    "display",
)


def _avg(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


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


def _pick_screenshot_frame_indices(
    warmup_frames: int, measured_frames: int
) -> list[int]:
    if measured_frames <= 0:
        raise ValueError("measured_frames must be > 0")
    candidates = [
        warmup_frames + measured_frames // 6,
        warmup_frames + measured_frames // 2,
        warmup_frames + (5 * measured_frames) // 6,
    ]
    limit = min(3, measured_frames)
    indices: list[int] = []
    for idx in candidates:
        if idx not in indices:
            indices.append(idx)
        if len(indices) == limit:
            break
    cursor = warmup_frames
    while len(indices) < limit:
        if cursor not in indices:
            indices.append(cursor)
        cursor += 1
    return sorted(indices)


def _summarize_frame_records(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("no frame records")

    frame_ms = [float(r["frame_ms"]) for r in records]
    stage_values: dict[str, list[float]] = {name: [] for name in STAGE_ORDER}
    for rec in records:
        stages = rec["stages"]
        if not isinstance(stages, dict):
            continue
        for stage in STAGE_ORDER:
            val = stages.get(stage)
            if isinstance(val, (int, float)):
                stage_values[stage].append(float(val))

    stage_avg = {k: _avg(v) for k, v in stage_values.items()}
    stage_p95 = {k: _percentile(v, 95.0) for k, v in stage_values.items()}
    bottlenecks = sorted(stage_avg.items(), key=lambda kv: kv[1], reverse=True)[:3]
    avg_frame_ms = _avg(frame_ms)
    avg_fps = (1000.0 / avg_frame_ms) if avg_frame_ms > 0 else 0.0

    return {
        "frames_measured": len(records),
        "avg_fps": avg_fps,
        "avg_frame_ms": avg_frame_ms,
        "p95_frame_ms": _percentile(frame_ms, 95.0),
        "stage_ms_avg": stage_avg,
        "stage_ms_p95": stage_p95,
        "bottlenecks": [{"stage": s, "avg_ms": v} for s, v in bottlenecks],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help="Camera index (ignored: baseline capture uses camera=1).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--input-width", type=int, default=1280)
    parser.add_argument("--input-height", type=int, default=720)
    parser.add_argument(
        "--frames", type=int, default=180, help="Measured frames after warmup."
    )
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--topology", choices=("frozen", "mediapipe"), default="frozen")
    parser.add_argument(
        "--region", choices=("forehead", "eyes", "mouth", "all"), default="all"
    )
    parser.add_argument("--color-match-every", type=int, default=1)
    parser.add_argument("--shading", choices=("on", "off"), default="on")
    parser.add_argument(
        "--out-root",
        default=str(Path("scripts") / "baseline_artifacts"),
        help="Root output directory for run artifacts.",
    )
    parser.add_argument("--run-id", default=None, help="Optional fixed run id.")
    return parser


def _build_runtime_argv(args: argparse.Namespace) -> list[str]:
    return [
        "facefx.main",
        "--camera",
        str(CAMERA_INDEX),
        "--device",
        args.device,
        "--scale",
        str(args.scale),
        "--topology",
        args.topology,
        "--region",
        args.region,
        "--color-match-every",
        str(args.color_match_every),
        "--shading",
        args.shading,
        "--profile",
    ]


def _run_capture(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[int, object], str]:
    import cv2
    import numpy as np

    import facefx.main as app
    import facefx.src.blend as blend_mod

    runtime_argv = _build_runtime_argv(args)
    runtime_cmd = f"{sys.executable} -m " + " ".join(
        shlex.quote(x) for x in runtime_argv
    )
    frame_records: list[dict[str, object]] = []
    screenshots: dict[int, object] = {}
    screenshot_targets = set(
        _pick_screenshot_frame_indices(args.warmup_frames, args.frames)
    )
    frame_state = {"shown": 0, "shading_ms": 0.0}
    total_frames = args.warmup_frames + args.frames

    original_video_capture = app.cv2.VideoCapture
    original_imshow = app.cv2.imshow
    original_destroy_all = app.cv2.destroyAllWindows
    original_wait_key = app.wait_key
    original_profiler = app.FrameProfiler
    original_color_match = app.color_match_lab

    class SizedCapture:
        def __init__(self, *cap_args: object) -> None:
            self._cap = original_video_capture(*cap_args)
            self._cap.set(app.cv2.CAP_PROP_FRAME_WIDTH, args.input_width)
            self._cap.set(app.cv2.CAP_PROP_FRAME_HEIGHT, args.input_height)

        def __getattr__(self, name: str) -> object:
            return getattr(self._cap, name)

    class RecordingProfiler(app.FrameProfiler):
        def start_frame(self) -> None:
            frame_state["shading_ms"] = 0.0
            super().start_frame()

        def end_frame(self) -> None:
            if not self.enabled:
                return
            now = time.perf_counter()
            raw_frame_ms = (now - self._frame_start) * 1000.0
            stage_ms = dict(self._dur_ms)
            super().end_frame()
            shading_ms = float(frame_state["shading_ms"])
            color_match_ms = float(stage_ms.get("color_match", 0.0))
            stage_ms["landmarks"] = float(stage_ms.get("facemesh", 0.0))
            stage_ms["mask"] = float(stage_ms.get("masks", 0.0))
            stage_ms["composite"] = float(stage_ms.get("blend", 0.0))
            stage_ms["shading"] = shading_ms
            stage_ms["color"] = max(0.0, color_match_ms - shading_ms)
            frame_records.append({"frame_ms": raw_frame_ms, "stages": stage_ms})

    def timed_color_match_lab(
        src_bgr: np.ndarray,
        tgt_bgr: np.ndarray,
        mask: np.ndarray,
        *,
        ab_strength: float = 0.5,
        shading: bool = True,
        shading_kernel: int = 51,
        shading_clamp: tuple[float, float] = (0.6, 1.6),
        eps: float = 1e-6,
        use_cuda: bool = False,
    ) -> np.ndarray:
        mask_u8 = blend_mod._mask_to_u8(mask)
        src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        src_mean, src_std = blend_mod._masked_mean_std_lab(src_lab, mask_u8)
        tgt_mean, tgt_std = blend_mod._masked_mean_std_lab(tgt_lab, mask_u8)
        scale = tgt_std / (src_std + eps)
        out_lab = (src_lab - src_mean.reshape(1, 1, 3)) * scale.reshape(
            1, 1, 3
        ) + tgt_mean.reshape(1, 1, 3)
        tgt_l = tgt_lab[:, :, 0]
        src_l = out_lab[:, :, 0]
        tgt_a = tgt_lab[:, :, 1]
        tgt_b = tgt_lab[:, :, 2]
        src_a = out_lab[:, :, 1]
        src_b = out_lab[:, :, 2]
        if ab_strength < 1.0:
            src_a = tgt_a + (src_a - tgt_a) * ab_strength
            src_b = tgt_b + (src_b - tgt_b) * ab_strength
        if shading:
            shade_start = time.perf_counter()
            k = max(3, int(shading_kernel) | 1)
            if use_cuda:
                blur_t = blend_mod._cuda_gaussian_blur_f32(tgt_l, k)
                blur_s = blend_mod._cuda_gaussian_blur_f32(src_l, k)
            else:
                blur_t = cv2.GaussianBlur(tgt_l, (k, k), 0)
                blur_s = cv2.GaussianBlur(src_l, (k, k), 0)
            ratio = (blur_t + eps) / (blur_s + eps)
            ratio = np.clip(ratio, shading_clamp[0], shading_clamp[1])
            src_l = src_l * ratio
            frame_state["shading_ms"] = (time.perf_counter() - shade_start) * 1000.0
        out_lab = cv2.merge(
            [np.clip(src_l, 0, 255), np.clip(src_a, 0, 255), np.clip(src_b, 0, 255)]
        ).astype(np.uint8)
        return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)

    def fake_imshow(_name: str, frame: object) -> None:
        idx = int(frame_state["shown"])
        if idx in screenshot_targets:
            screenshots[idx] = frame.copy()
        frame_state["shown"] = idx + 1

    def fake_wait_key(_delay_ms: int) -> int:
        return 27 if int(frame_state["shown"]) >= total_frames else -1

    old_argv = sys.argv[:]
    try:
        app.cv2.VideoCapture = SizedCapture
        app.cv2.imshow = fake_imshow
        app.cv2.destroyAllWindows = lambda: None
        app.wait_key = fake_wait_key
        app.FrameProfiler = RecordingProfiler
        app.color_match_lab = timed_color_match_lab
        sys.argv = runtime_argv
        rc = app.main()
    finally:
        sys.argv = old_argv
        app.cv2.VideoCapture = original_video_capture
        app.cv2.imshow = original_imshow
        app.cv2.destroyAllWindows = original_destroy_all
        app.wait_key = original_wait_key
        app.FrameProfiler = original_profiler
        app.color_match_lab = original_color_match

    if rc != 0:
        raise RuntimeError(f"facefx.main exited with code {rc}")
    if len(frame_records) < total_frames:
        raise RuntimeError(
            "capture ended before requested frame budget; camera input likely unavailable"
        )
    return frame_records, screenshots, runtime_cmd


def main() -> int:
    args = _build_parser().parse_args()
    run_id = args.run_id or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_records, screenshots, runtime_cmd = _run_capture(args)
    measured = frame_records[args.warmup_frames : args.warmup_frames + args.frames]
    summary = _summarize_frame_records(measured)
    required_screens = min(3, args.frames)
    if len(screenshots) < required_screens:
        raise RuntimeError(
            f"captured {len(screenshots)} screenshots, expected at least {required_screens}"
        )

    saved_screens = []
    for i, frame_idx in enumerate(sorted(screenshots.keys()), start=1):
        path = out_dir / f"screenshot_{i:02d}_frame_{frame_idx:04d}.png"
        import cv2

        cv2.imwrite(str(path), screenshots[frame_idx])
        saved_screens.append(str(path))

    payload = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_id": run_id,
        "resolution": {"width": args.input_width, "height": args.input_height},
        "device": args.device,
        "camera": CAMERA_INDEX,
        "camera_hardcoded": CAMERA_INDEX,
        "warmup_frames": args.warmup_frames,
        "measured_frames": args.frames,
        "runtime_command": runtime_cmd,
        "artifacts": {"screenshots": saved_screens},
        "summary": summary,
    }
    json_path = out_dir / "bench_baseline.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_path = out_dir / "baseline_report.md"
    lines = [
        "# Baseline Report",
        "",
        f"- run_id: `{run_id}`",
        f"- command: `{runtime_cmd}`",
        f"- avg_fps: `{summary['avg_fps']:.2f}`",
        f"- p95_frame_ms: `{summary['p95_frame_ms']:.2f}`",
        "- bottlenecks:",
    ]
    for item in summary["bottlenecks"]:
        lines.append(f"  - `{item['stage']}`: `{item['avg_ms']:.2f} ms`")
    lines.append("")
    lines.append("Screenshots:")
    for path in saved_screens:
        lines.append(f"- `{path}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {json_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
