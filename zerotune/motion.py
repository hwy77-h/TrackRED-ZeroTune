from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .mask_ops import bbox, clean_mask, mask_center, shift_mask


def _estimate_translation_by_centroid(prev_frame: np.ndarray, curr_frame: np.ndarray, prev_mask: np.ndarray) -> np.ndarray:
    # Conservative fallback when TV-L1 is unavailable or fails: keep the last
    # stable mask in place. This preserves causality and avoids hallucinated jumps.
    _ = prev_frame, curr_frame
    center = mask_center(prev_mask)
    if center is None:
        return np.array([0.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)


def local_tvl1_warp(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    prev_mask: np.ndarray,
    roi_padding: int = 32,
    min_component_area: int = 25,
) -> np.ndarray:
    region = bbox(prev_mask, padding=roi_padding)
    if region is None:
        return np.zeros_like(prev_mask, dtype=np.uint8)

    x0, y0, x1, y1 = region
    prev_roi = prev_frame[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
    curr_roi = curr_frame[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
    mask_roi = (prev_mask[y0 : y1 + 1, x0 : x1 + 1] > 0).astype(np.uint8)

    try:
        from skimage.registration import optical_flow_tvl1

        v, u = optical_flow_tvl1(prev_roi, curr_roi)
        if mask_roi.sum() > 0:
            dx = float(np.median(u[mask_roi > 0]))
            dy = float(np.median(v[mask_roi > 0]))
        else:
            dx = dy = 0.0
        warped_roi = shift_mask(mask_roi, np.array([dx, dy], dtype=np.float32))
    except Exception:
        delta = _estimate_translation_by_centroid(prev_roi, curr_roi, mask_roi)
        warped_roi = shift_mask(mask_roi, delta)

    out = np.zeros_like(prev_mask, dtype=np.uint8)
    out[y0 : y1 + 1, x0 : x1 + 1] = warped_roi
    out = ndi.binary_fill_holes(out > 0).astype(np.uint8)
    return clean_mask(out, min_component_area=min_component_area)

