"""Color and shading stage tests for runtime_cuda."""

from __future__ import annotations

import numpy as np
import pytest
from facefx.runtime_cuda.color import (
    build_shading_state_lab,
    match_color_and_shading_roi,
    match_color_and_shading_roi_cached,
)


def test_color_stage_disabled_returns_patch_copy():
    patch = np.full((32, 32, 3), (40, 100, 180), dtype=np.uint8)
    target = np.full((32, 32, 3), (180, 50, 30), dtype=np.uint8)
    alpha = np.full((32, 32), 255, dtype=np.uint8)

    out, stats = match_color_and_shading_roi(
        patch,
        target,
        alpha,
        enable_color_match=False,
        enable_shading=False,
    )
    assert np.array_equal(out, patch)
    assert out is not patch
    assert stats.color_ms == 0.0
    assert stats.shading_ms == 0.0
    assert stats.backend == "off"


def test_color_match_moves_patch_closer_to_target():
    patch = np.full((48, 48, 3), (255, 0, 0), dtype=np.uint8)
    target = np.full((48, 48, 3), (0, 255, 0), dtype=np.uint8)
    alpha = np.full((48, 48), 255, dtype=np.uint8)

    out, stats = match_color_and_shading_roi(
        patch,
        target,
        alpha,
        enable_color_match=True,
        ab_strength=1.0,
        enable_shading=False,
        use_cuda=False,
    )
    before = np.mean(np.abs(patch.astype(np.int16) - target.astype(np.int16)))
    after = np.mean(np.abs(out.astype(np.int16) - target.astype(np.int16)))
    assert after < before
    assert stats.color_ms >= 0.0
    assert stats.shading_ms == 0.0


def test_shading_modifies_luminance_when_enabled():
    patch = np.full((64, 64, 3), 120, dtype=np.uint8)
    ramp = np.linspace(30, 220, 64, dtype=np.uint8)
    target_l = np.repeat(ramp[None, :], 64, axis=0)
    target = np.stack([target_l, target_l, target_l], axis=2)
    alpha = np.full((64, 64), 255, dtype=np.uint8)

    out, stats = match_color_and_shading_roi(
        patch,
        target,
        alpha,
        enable_color_match=False,
        enable_shading=True,
        shading_kernel=21,
        shading_clamp=(0.3, 2.0),
        use_cuda=False,
    )
    assert not np.array_equal(out, patch)
    assert float(np.std(out[:, :, 0])) > 1.0
    assert stats.shading_ms >= 0.0


def test_color_stage_rejects_shape_mismatch():
    patch = np.full((32, 32, 3), 120, dtype=np.uint8)
    target = np.full((16, 16, 3), 90, dtype=np.uint8)
    alpha = np.full((32, 32), 255, dtype=np.uint8)
    with pytest.raises(ValueError):
        match_color_and_shading_roi(patch, target, alpha)


def test_cached_stage_holds_shading_between_refreshes():
    patch = np.full((48, 48, 3), 120, dtype=np.uint8)
    ramp = np.linspace(30, 220, 48, dtype=np.uint8)
    target_l = np.repeat(ramp[None, :], 48, axis=0)
    target = np.stack([target_l, target_l, target_l], axis=2)
    alpha = np.full((48, 48), 255, dtype=np.uint8)

    refreshed, refresh_stats, color_state, shading_state = match_color_and_shading_roi_cached(
        patch,
        target,
        alpha,
        refresh_color=True,
        refresh_shading=True,
        color_match_enabled=False,
        shading_enabled=True,
        shading_kernel=21,
        shading_clamp=(0.3, 2.0),
        use_cuda=False,
    )
    held, held_stats, color_state2, shading_state2 = match_color_and_shading_roi_cached(
        patch,
        target,
        alpha,
        refresh_color=False,
        refresh_shading=False,
        color_state=color_state,
        shading_state=shading_state,
        color_match_enabled=False,
        shading_enabled=True,
        shading_kernel=21,
        shading_clamp=(0.3, 2.0),
        use_cuda=False,
    )
    assert np.array_equal(refreshed, held)
    assert refresh_stats.shading_ms >= 0.0
    assert held_stats.shading_ms == 0.0
    assert color_state2 is None
    assert shading_state2 is shading_state


def test_cached_stage_keeps_soft_edge_closer_to_target():
    patch = np.full((40, 40, 3), (70, 110, 150), dtype=np.uint8)
    target = np.full((40, 40, 3), (210, 230, 250), dtype=np.uint8)
    alpha = np.zeros((40, 40), dtype=np.uint8)
    alpha[6:34, 6:34] = 255
    alpha[4:36, 4:36] = np.maximum(alpha[4:36, 4:36], 96)

    out, _stats, _color_state, _shading_state = match_color_and_shading_roi_cached(
        patch,
        target,
        alpha,
        refresh_color=True,
        refresh_shading=True,
        color_match_enabled=True,
        shading_enabled=False,
        use_cuda=False,
    )
    soft_edge_px = out[5, 20].astype(np.int16)
    center_px = out[20, 20].astype(np.int16)
    patch_px = patch[20, 20].astype(np.int16)
    target_px = target[20, 20].astype(np.int16)
    assert np.mean(np.abs(soft_edge_px - target_px)) < np.mean(np.abs(soft_edge_px - patch_px))
    assert np.mean(np.abs(center_px - target_px)) < np.mean(np.abs(patch_px - target_px))


def test_build_shading_state_lab_ignores_bright_background_outside_mask():
    src = np.full((48, 48, 3), 120, dtype=np.uint8)
    tgt = np.full((48, 48, 3), 120, dtype=np.uint8)
    tgt[:, :12] = 255
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[:, 16:40] = 255

    state = build_shading_state_lab(
        src,
        tgt,
        mask=mask,
        shading_kernel=15,
        shading_clamp=(0.3, 2.0),
        use_cuda=False,
    )
    # Inside the masked center the ratio should stay near neutral, because the
    # bright strip sits fully outside the masked region.
    center_ratio = float(state.ratio_l[24, 28])
    assert abs(center_ratio - 1.0) < 0.1


def test_cached_stage_blends_edge_toward_target_for_seamless_transition():
    patch = np.full((48, 48, 3), (180, 120, 80), dtype=np.uint8)
    target = np.full((48, 48, 3), (70, 80, 95), dtype=np.uint8)
    alpha = np.zeros((48, 48), dtype=np.uint8)
    alpha[6:42, 6:42] = 255
    alpha[4:44, 4:44] = np.maximum(alpha[4:44, 4:44], 128)

    out, _stats, _cstate, _sstate = match_color_and_shading_roi_cached(
        patch,
        target,
        alpha,
        refresh_color=True,
        refresh_shading=False,
        color_match_enabled=True,
        shading_enabled=False,
        use_cuda=False,
    )
    edge_px = out[5, 24].astype(np.int16)
    patch_px = patch[24, 24].astype(np.int16)
    target_px = target[24, 24].astype(np.int16)
    # Edge should be pulled toward live target compared to raw patch color.
    assert np.mean(np.abs(edge_px - target_px)) < np.mean(np.abs(patch_px - target_px))
