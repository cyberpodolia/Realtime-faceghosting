"""Blend helper tests for perf-oriented refactors."""

from __future__ import annotations

import numpy as np

from facefx.src.blend import _mask_to_u8, color_match_lab


def test_mask_to_u8_accepts_float_and_3d_mask():
    mask = np.zeros((4, 5, 1), dtype=np.float32)
    mask[1:3, 2:4, 0] = 0.5
    out = _mask_to_u8(mask)
    assert out.dtype == np.uint8
    assert out.shape == (4, 5)
    assert out[1, 2] in (127, 128)


def test_color_match_lab_returns_valid_image_shape():
    src = np.full((8, 8, 3), 100, dtype=np.uint8)
    tgt = np.full((8, 8, 3), 140, dtype=np.uint8)
    mask = np.ones((8, 8), dtype=np.float32)
    out = color_match_lab(src, tgt, mask, shading=False)
    assert out.shape == src.shape
    assert out.dtype == np.uint8
    assert np.all(out >= 0)
    assert np.all(out <= 255)
