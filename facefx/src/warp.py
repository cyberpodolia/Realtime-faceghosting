"""Piecewise affine warp utilities."""

from __future__ import annotations

import cv2
import numpy as np


def warp_triangle(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    t_src: np.ndarray,
    t_dst: np.ndarray,
    dst_mask_accum: np.ndarray | None = None,
) -> None:
    """Warp a source triangle into destination image."""
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))

    t1_rect = []
    t2_rect = []
    for i in range(3):
        t1_rect.append((t_src[i][0] - r1[0], t_src[i][1] - r1[1]))
        t2_rect.append((t_dst[i][0] - r2[0], t_dst[i][1] - r2[1]))

    src_crop = src_img[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
    if src_crop.size == 0:
        return

    mtx = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    dst_w, dst_h = r2[2], r2[3]
    warped = cv2.warpAffine(
        src_crop,
        mtx,
        (dst_w, dst_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((dst_h, dst_w), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), 1.0, lineType=cv2.LINE_AA)

    x, y, w, h = r2
    roi = dst_img[y : y + h, x : x + w]
    if roi.shape[:2] != warped.shape[:2]:
        return

    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - mask) + warped[:, :, c] * mask
    dst_img[y : y + h, x : x + w] = roi

    if dst_mask_accum is not None:
        mroi = dst_mask_accum[y : y + h, x : x + w]
        dst_mask_accum[y : y + h, x : x + w] = np.maximum(mroi, mask)
