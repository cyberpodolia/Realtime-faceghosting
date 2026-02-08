"""Blend helpers."""

from __future__ import annotations

import numpy as np


def blend_with_mask(
    base: np.ndarray,
    overlay: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Blend overlay onto base using mask in [0,1]."""
    if mask.ndim == 2:
        mask = mask[..., None]
    mask = np.clip(mask, 0.0, 1.0)
    out = base.astype(np.float32) * (1.0 - mask) + overlay.astype(np.float32) * mask
    return out.astype(np.uint8)
