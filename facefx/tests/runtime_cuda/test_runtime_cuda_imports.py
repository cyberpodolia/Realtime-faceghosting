"""Import smoke tests for runtime_cuda package."""

import importlib


def test_runtime_cuda_module_imports():
    for name in (
        "facefx.runtime_cuda",
        "facefx.runtime_cuda.color",
        "facefx.runtime_cuda.config",
        "facefx.runtime_cuda.composite",
        "facefx.runtime_cuda.landmarks",
        "facefx.runtime_cuda.mask",
        "facefx.runtime_cuda.native_backend",
        "facefx.runtime_cuda.pipeline",
        "facefx.runtime_cuda.roi",
        "facefx.runtime_cuda.topology",
        "facefx.runtime_cuda.warp",
        "facefx.runtime_cuda.app",
    ):
        importlib.import_module(name)
