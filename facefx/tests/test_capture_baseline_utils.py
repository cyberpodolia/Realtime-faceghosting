"""Tests for scripts/capture_baseline.py helper logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "capture_baseline.py"
    spec = importlib.util.spec_from_file_location("capture_baseline", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pick_screenshot_frame_indices_three_unique():
    mod = _load_module()
    idxs = mod._pick_screenshot_frame_indices(30, 180)
    assert len(idxs) == 3
    assert len(set(idxs)) == 3
    assert idxs[0] >= 30
    assert idxs[-1] < 210


def test_percentile_interpolates():
    mod = _load_module()
    assert mod._percentile([10.0, 20.0, 30.0, 40.0], 50.0) == 25.0
    assert mod._percentile([1.0], 95.0) == 1.0


def test_summarize_frame_records_bottleneck_order():
    mod = _load_module()
    records = [
        {
            "frame_ms": 25.0,
            "stages": {
                "capture": 3.0,
                "landmarks": 8.0,
                "mask": 2.0,
                "warp": 6.0,
                "color": 2.0,
                "shading": 1.0,
                "composite": 2.0,
                "display": 1.0,
            },
        },
        {
            "frame_ms": 30.0,
            "stages": {
                "capture": 4.0,
                "landmarks": 10.0,
                "mask": 2.0,
                "warp": 7.0,
                "color": 3.0,
                "shading": 1.0,
                "composite": 2.0,
                "display": 1.0,
            },
        },
    ]
    summary = mod._summarize_frame_records(records)
    assert summary["frames_measured"] == 2
    assert summary["bottlenecks"][0]["stage"] == "landmarks"
    assert summary["avg_fps"] > 0
