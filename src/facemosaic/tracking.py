"""Offline temporal post-processing: link per-frame detections into tracks, confirm real faces,
fill missed frames, extend track ends and enlarge boxes.

Because this runs offline it can use "future" frames: once any frame of a track scores high,
the whole track counts as a face, including the low-score frames before it (dual threshold).
"""
from __future__ import annotations

import numpy as np


def expand_boxes(b: np.ndarray, ratio: float) -> np.ndarray:
    b = np.asarray(b, np.float32).reshape(-1, 4)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    hw, hh = (b[:, 2] - b[:, 0]) * (1 + ratio) / 2, (b[:, 3] - b[:, 1]) * (1 + ratio) / 2
    return np.stack([cx - hw, cy - hh, cx + hw, cy + hh], axis=1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, np.float32).reshape(-1, 4), np.asarray(b, np.float32).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def link_tracks(dets_per_frame, iou_thr: float, max_gap: int, match_expand: float) -> list[dict]:
    """Greedy IoU association. Returns a list of tracks, each {frames, boxes, scores}."""
    tracks: list[dict] = []
    active: list[int] = []
    for f, d in enumerate(dets_per_frame):
        active = [t for t in active if f - tracks[t]["frames"][-1] <= max_gap + 1]
        if d is None or len(d) == 0:
            continue
        d = np.asarray(d, np.float32).reshape(-1, 5)
        used_t, used_d = set(), set()
        if active:
            last = np.array([tracks[t]["boxes"][-1] for t in active], np.float32)
            ious = iou_matrix(expand_boxes(last, match_expand), expand_boxes(d[:, :4], match_expand))
            for flat in np.argsort(-ious, axis=None):
                i, j = divmod(int(flat), ious.shape[1])
                if ious[i, j] < iou_thr:
                    break
                if i in used_t or j in used_d:
                    continue
                tr = tracks[active[i]]
                tr["frames"].append(f)
                tr["boxes"].append(d[j, :4])
                tr["scores"].append(float(d[j, 4]))
                used_t.add(i)
                used_d.add(j)
        for j in range(len(d)):
            if j not in used_d:
                tracks.append({"frames": [f], "boxes": [d[j, :4]], "scores": [float(d[j, 4])]})
                active.append(len(tracks) - 1)
    return tracks


def finalize(dets_per_frame, n_frames: int, width: int, height: int, tcfg: dict) -> tuple[list, dict]:
    """Returns (per-frame mosaic boxes [[x1, y1, x2, y2], ...] in integer pixels, stats)."""
    tracks = link_tracks(dets_per_frame, tcfg["iou_match"], tcfg["max_gap"], tcfg["match_expand"])
    out: list[list] = [[] for _ in range(n_frames)]
    extend = int(tcfg["extend"])
    n_conf = n_filled = 0
    for tr in tracks:
        if max(tr["scores"]) < tcfg["score_high"] or len(tr["frames"]) < tcfg["min_track_len"]:
            continue
        n_conf += 1
        fr = np.asarray(tr["frames"])
        bx = np.asarray(tr["boxes"], np.float32)
        full_f = np.arange(fr[0], fr[-1] + 1)
        full_b = np.stack([np.interp(full_f, fr, bx[:, k]) for k in range(4)], axis=1)
        n_filled += len(full_f) - len(fr)
        pre = np.arange(max(0, fr[0] - extend), fr[0])
        post = np.arange(fr[-1] + 1, min(n_frames, fr[-1] + extend + 1))
        n_filled += len(pre) + len(post)
        frames = np.concatenate([pre, full_f, post])
        boxes = np.concatenate([np.repeat(full_b[:1], len(pre), 0), full_b, np.repeat(full_b[-1:], len(post), 0)])
        boxes = expand_boxes(boxes, tcfg["expand"])
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        for f, b in zip(frames, boxes):
            x1, y1, x2, y2 = int(np.floor(b[0])), int(np.floor(b[1])), int(np.ceil(b[2])), int(np.ceil(b[3]))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                out[int(f)].append([x1, y1, x2, y2])
    stats = {
        "raw_det_frames": sum(1 for d in dets_per_frame if d is not None and len(d)),
        "tracks": len(tracks),
        "tracks_confirmed": n_conf,
        "mosaic_frames": sum(1 for b in out if b),
        "filled_boxes": n_filled,
    }
    return out, stats
