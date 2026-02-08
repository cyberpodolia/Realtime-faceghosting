"""Smoke import checks for core dependencies and modules."""

import importlib


def test_imports():
    for name in ("cv2", "mediapipe", "numpy", "scipy"):
        importlib.import_module(name)

    # Local modules
    for name in (
        "facefx.src.facemesh",
        "facefx.src.regions",
        "facefx.src.triangulation",
        "facefx.src.warp",
        "facefx.src.blend",
        "facefx.src.patchbank",
        "facefx.src.ui",
    ):
        importlib.import_module(name)
