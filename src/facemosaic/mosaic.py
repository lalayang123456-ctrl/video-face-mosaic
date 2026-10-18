"""Mosaic: pixelate regions in place, directly on the decoded frame's planes (Y/U/V or packed RGB).

There is no YUV <-> RGB round trip, so pixels outside the boxes stay untouched and 10-bit stays 10-bit.
"""
from __future__ import annotations

import math

import numpy as np

# Packed formats whose components on the same plane have different sampling (e.g. yuyv422)
# cannot be handled plane by plane; they are converted to a planar format first.
PLANAR_FALLBACK = {"yuyv422": "yuv422p", "uyvy422": "yuv422p", "yvyu422": "yuv422p"}
# Packed formats with a padding byte: 4 samples per pixel but only 3 components.
PADDED_PACKED = {"rgb0": 4, "bgr0": 4, "0rgb": 4, "0bgr": 4}


class UnsupportedPixFmt(Exception):
    pass


def pixelate(arr: np.ndarray, x1: int, y1: int, x2: int, y2: int, nx: int, ny: int) -> None:
    """Replace arr[y1:y2, x1:x2] in place with nx x ny cells (cell = mean of its pixels). arr is HxW or HxWxC.

    Pure numpy on purpose: cv2.resize occasionally raised "Unknown C++ exception" in batch runs,
    so the mosaic path does not depend on OpenCV.
    """
    region = arr[y1:y2, x1:x2]
    h, w = region.shape[:2]
    if h == 0 or w == 0:
        return
    nx, ny = max(1, min(nx, w)), max(1, min(ny, h))
    xs = np.linspace(0, w, nx + 1).astype(np.int64)  # nx <= w, so the edges are strictly increasing
    ys = np.linspace(0, h, ny + 1).astype(np.int64)
    sums = np.add.reduceat(np.add.reduceat(region.astype(np.float64), ys[:-1], axis=0), xs[:-1], axis=1)
    cnt = np.outer(np.diff(ys), np.diff(xs))
    if region.ndim == 3:
        cnt = cnt[..., None]
    means = np.rint(sums / cnt).astype(region.dtype)
    region[...] = np.repeat(np.repeat(means, np.diff(ys), axis=0), np.diff(xs), axis=1)


def grid_size(w: int, h: int, blocks: int, min_block: int) -> tuple[int, int]:
    block = max(w / max(1, blocks), float(min_block))
    return max(1, round(w / block)), max(1, round(h / block))


def plane_views(frame) -> list[tuple[np.ndarray, int, int]]:
    """Returns [(writable numpy view, x subsampling, y subsampling), ...], one entry per plane."""
    fmt = frame.format
    W, H = frame.width, frame.height
    views = []
    for i, plane in enumerate(frame.planes):
        comps = [c for c in fmt.components if c.plane == i]
        if not comps:
            continue
        if len({(c.width, c.height) for c in comps}) > 1:
            raise UnsupportedPixFmt(fmt.name)
        bits = max(c.bits for c in comps)
        dtype = np.uint16 if bits > 8 else np.uint8
        itemsize = np.dtype(dtype).itemsize
        pw, ph = plane.width, plane.height
        # Samples per pixel: usually the number of components on this plane (2 for NV12's UV plane, 3 for rgb24)
        c = PADDED_PACKED.get(fmt.name, len(comps))
        full = np.frombuffer(plane, dtype).reshape(ph, plane.line_size // itemsize)
        view = full[:, : pw * c].reshape(ph, pw, c) if c > 1 else full[:, :pw]
        sx, sy = max(1, round(W / pw)), max(1, round(H / ph))
        views.append((view, sx, sy))
    return views


def mosaic_frame(frame, boxes, blocks: int = 10, min_block: int = 4) -> None:
    """Pixelate an av.VideoFrame in place. boxes are source pixel coordinates [[x1, y1, x2, y2], ...]."""
    if not boxes:
        return
    views = plane_views(frame)
    ax = max(v[1] for v in views)
    ay = max(v[2] for v in views)
    W, H = frame.width, frame.height
    for x1, y1, x2, y2 in boxes:
        # Snap to the chroma subsampling grid so luma and chroma cells line up
        x1, y1 = (x1 // ax) * ax, (y1 // ay) * ay
        x2, y2 = min(W, math.ceil(x2 / ax) * ax), min(H, math.ceil(y2 / ay) * ay)
        if x2 <= x1 or y2 <= y1:
            continue
        nx, ny = grid_size(x2 - x1, y2 - y1, blocks, min_block)
        for view, sx, sy in views:
            pixelate(view, x1 // sx, y1 // sy, math.ceil(x2 / sx), math.ceil(y2 / sy), nx, ny)


def mosaic_bgr(img: np.ndarray, boxes, blocks: int = 10, min_block: int = 4) -> None:
    """Pixelate a BGR ndarray in place (used for previews)."""
    for x1, y1, x2, y2 in boxes:
        nx, ny = grid_size(x2 - x1, y2 - y1, blocks, min_block)
        pixelate(img, x1, y1, x2, y2, nx, ny)
