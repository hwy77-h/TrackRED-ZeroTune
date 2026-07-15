from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def clean_mask(mask: np.ndarray, min_component_area: int = 25) -> np.ndarray:
    mask = ndi.binary_fill_holes(mask > 0)
    labels, n_labels = ndi.label(mask)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    sizes = ndi.sum(mask, labels, index=np.arange(1, n_labels + 1))
    keep_label = int(np.argmax(sizes)) + 1
    if int(sizes[keep_label - 1]) < min_component_area:
        return np.zeros_like(mask, dtype=np.uint8)
    out = labels == keep_label
    out = ndi.binary_closing(out, structure=np.ones((3, 3), dtype=bool), iterations=1)
    out = ndi.binary_fill_holes(out)
    return out.astype(np.uint8)


def bbox(mask: np.ndarray, padding: int = 0) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(w - 1, int(xs.max()) + padding)
    y1 = min(h - 1, int(ys.max()) + padding)
    return x0, y0, x1, y1


def mask_center(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def largest_component_ratio(mask: np.ndarray) -> float:
    labels, n_labels = ndi.label(mask > 0)
    total = int((mask > 0).sum())
    if total == 0:
        return 0.0
    if n_labels == 0:
        return 0.0
    sizes = ndi.sum(mask > 0, labels, index=np.arange(1, n_labels + 1))
    return float(np.max(sizes) / total)


def shift_mask(mask: np.ndarray, delta_xy: np.ndarray) -> np.ndarray:
    shifted = ndi.shift(
        (mask > 0).astype(np.float32),
        shift=(float(delta_xy[1]), float(delta_xy[0])),
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return (shifted > 0.5).astype(np.uint8)

