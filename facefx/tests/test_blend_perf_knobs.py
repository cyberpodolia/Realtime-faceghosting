"""Blend helper tests for perf-oriented refactors."""

from __future__ import annotations

import numpy as np
from facefx.src.blend import (
    _mask_to_u8,
    apply_color_transfer_state_lab,
    apply_shading_state_lab,
    build_color_transfer_state_lab,
    build_shading_state_lab,
    color_match_lab,
)


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


def test_color_transfer_state_build_and_apply():
    src = np.full((16, 16, 3), (120, 60, 30), dtype=np.uint8)
    tgt = np.full((16, 16, 3), (60, 120, 170), dtype=np.uint8)
    mask = np.ones((16, 16), dtype=np.float32)

    state = build_color_transfer_state_lab(src, tgt, mask)
    assert state is not None

    held = apply_color_transfer_state_lab(src, state, ab_strength=0.5)
    assert held.shape == src.shape
    assert held.dtype == np.uint8

    before = float(np.mean(np.abs(src.astype(np.int16) - tgt.astype(np.int16))))
    after = float(np.mean(np.abs(held.astype(np.int16) - tgt.astype(np.int16))))
    assert after < before


def test_shading_state_build_and_apply_hold():
    src = np.full((32, 32, 3), 120, dtype=np.uint8)
    grad = np.linspace(30, 220, 32, dtype=np.uint8)
    tgt_l = np.repeat(grad[None, :], 32, axis=0)
    tgt = np.stack([tgt_l, tgt_l, tgt_l], axis=2)

    state = build_shading_state_lab(src, tgt, shading_kernel=21, use_cuda=False)
    held = apply_shading_state_lab(src, state)
    assert held.shape == src.shape
    assert held.dtype == np.uint8
    assert float(np.std(held[:, :, 0])) > 1.0
