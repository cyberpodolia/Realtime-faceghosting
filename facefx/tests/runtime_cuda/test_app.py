"""CLI tests for runtime_cuda scaffold app."""

import numpy as np
import pytest
from facefx.runtime_cuda import app


def test_parse_args_valid():
    args = app._parse_args(
        [
            "--camera",
            "9",
            "--device",
            "cuda",
            "--input-width",
            "640",
            "--input-height",
            "480",
            "--camera-fps",
            "30",
            "--debug-overlay",
            "roi",
            "--profile",
            "--dump-roi-overlay",
            "--roi-overlay-path",
            "tmp/roi_overlay.png",
            "--roi-expand",
            "0.2",
            "--roi-min-size",
            "80",
            "--control-point-count",
            "20",
            "--warp-point-count",
            "120",
            "--landmark-every",
            "4",
            "--landmark-scale",
            "0.4",
            "--landmark-smooth",
            "0.7",
            "--color-match-every",
            "2",
            "--smoke",
            "5",
            "--max-frames",
            "25",
        ]
    )
    assert args.camera == 9
    assert args.device == "cuda"
    assert args.input_width == 640
    assert args.input_height == 480
    assert args.camera_fps == 30.0
    assert args.debug_overlay == "roi"
    assert args.profile is True
    assert args.dump_roi_overlay is True
    assert args.roi_overlay_path == "tmp/roi_overlay.png"
    assert args.roi_expand == 0.2
    assert args.roi_min_size == 80
    assert args.control_point_count == 20
    assert args.warp_point_count == 120
    assert args.landmark_every == 4
    assert args.landmark_scale == 0.4
    assert args.landmark_smooth == 0.7
    assert args.color_match_every == 2
    assert args.smoke == 5
    assert args.max_frames == 25


def test_parse_args_default_landmark_every_is_three():
    args = app._parse_args([])
    assert args.landmark_every == 3


def test_parse_args_invalid_size_raises():
    with pytest.raises(SystemExit):
        app._parse_args(["--input-width", "0"])


def test_parse_args_invalid_landmark_scale_raises():
    with pytest.raises(SystemExit):
        app._parse_args(["--landmark-scale", "0.05"])


def test_parse_args_invalid_camera_fps_raises():
    with pytest.raises(SystemExit):
        app._parse_args(["--camera-fps", "0"])


def test_parse_args_invalid_smoke_raises():
    with pytest.raises(SystemExit):
        app._parse_args(["--smoke", "-1"])


def test_parse_args_invalid_color_match_every_raises():
    with pytest.raises(SystemExit):
        app._parse_args(["--color-match-every", "0"])


def test_parse_args_invalid_warp_point_count_raises():
    with pytest.raises(SystemExit):
        app._parse_args(["--warp-point-count", "-1"])


def test_main_dry_run_returns_zero():
    rc = app.main(["--dry-run", "--camera", "9", "--device", "cuda"])
    assert rc == 0


def test_smooth_landmarks_blends_previous_and_current():
    prev = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    current = np.array([[30.0, 30.0], [50.0, 50.0]], dtype=np.float32)
    out = app._smooth_landmarks(prev, current, 0.5)
    assert np.allclose(out, np.array([[20.0, 20.0], [35.0, 35.0]], dtype=np.float32))


def test_split_patch_image_for_rgb_patch():
    patch = np.zeros((8, 10, 3), dtype=np.uint8)
    patch[:, :, 0] = 50
    bgr, alpha = app._split_patch_image(patch)
    assert bgr.shape == (8, 10, 3)
    assert alpha.shape == (8, 10)
    assert np.all(alpha == 255)


def test_split_patch_image_for_rgba_patch():
    patch = np.zeros((6, 7, 4), dtype=np.uint8)
    patch[:, :, :3] = 100
    patch[:, :, 3] = 123
    bgr, alpha = app._split_patch_image(patch)
    assert bgr.shape == (6, 7, 3)
    assert alpha.shape == (6, 7)
    assert int(alpha[0, 0]) == 123


def test_flat_patch_from_image_uses_single_color():
    patch = np.zeros((4, 5, 3), dtype=np.uint8)
    patch[:, :, 0] = 10
    patch[:, :, 1] = 20
    patch[:, :, 2] = 30
    flat = app._flat_patch_from_image(patch)
    assert flat.shape == patch.shape
    assert np.all(flat[:, :, 0] == 10)
    assert np.all(flat[:, :, 1] == 20)
    assert np.all(flat[:, :, 2] == 30)
