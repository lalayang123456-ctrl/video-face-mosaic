"""Config loading and per-camera profile matching (wrist / head / third, ...).

Merge order: built-in DEFAULTS <- config file <- matched profiles.
See docs/configuration.md for what every parameter does.
"""
from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
from pathlib import Path

import yaml

# Repository root (three levels up from src/facemosaic/config.py). With `pip install -e .`
# this points at the cloned repository.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

DEFAULTS: dict = {
    "detector": {
        "name": "scrfd",  # scrfd | yunet
        "model": "models/det_10g.onnx",
        "device": "cuda",  # cuda | cpu
        "fp16": True,  # use an FP16 model on GPU (auto-converted on first use): same detections, faster
        # Multi-scale detection. size = long side of the detection image (0 = native; never upscales,
        # scales that end up the same size are de-duplicated). border_pad = padding ratio on each side.
        "scales": [
            {"size": 1280, "border_pad": 0.0},  # small, distant faces
            {"size": 640, "border_pad": 0.15},  # medium / large faces
            {"size": 320, "border_pad": 0.15},  # huge faces right in front of the lens
        ],
        "score_low": 0.2,  # raw detections below this score are dropped
        "nms_iou": 0.4,
        "min_face": 0,  # minimum face side in source pixels; smaller detections are dropped
    },
    "tracking": {
        "score_high": 0.5,  # a track counts as a face only if at least one frame reaches this score
        "iou_match": 0.2,  # IoU needed to link boxes in neighbouring frames (after match_expand)
        "match_expand": 0.5,
        "max_gap": 15,  # gaps of up to this many frames inside a track are interpolated
        "extend": 5,  # frames added before the start and after the end of each track
        "expand": 0.25,  # how much larger the mosaic box is than the face box
        "min_track_len": 1,
        # Reject face-like objects (jars, packaging, lids, ...): a confirmed track is kept only if it is tiny,
        # scores very high, or its face box sits in the head region of a person found by a person detector.
        "person_check": {
            "enabled": False,
            "model": "models/yolox_s.onnx",
            "conf": 0.4,  # person detection threshold; cluttered shelves / tables produce spurious "persons" below ~0.4
            "frames": 3,  # check the track's N best-scoring frames; any match keeps the track
            "head_frac": 0.5,  # face center must be in the top part of the person box
            "tiny_face": 22,  # faces this small (px) are kept without a check: distant people's bodies are often not visible
            "trust_score": 0.85,  # tracks scoring this high are kept without a check (e.g. a face filling the frame)
        },
    },
    "mosaic": {
        "blocks": 10,  # number of mosaic cells across the box width
        "min_block": 4,  # minimum cell size in pixels
    },
    "encode": {
        "codec": "auto",  # auto = same as source (h264/hevc/av1); or force h264/hevc/av1
        "fallback_codec": "h264",  # used when the source codec is none of the above
        "hw": True,  # prefer NVENC
        "quality": 19,  # NVENC cq / software crf; lower = better quality
        "preset": None,  # None = encoder-specific default
        "gop": None,  # keyframe interval; None = encoder default
        "max_b_frames": None,
        "extra_options": {},  # raw encoder options passed through, e.g. {"x264-params": "me=hex"}
        "copy_if_no_face": True,  # copy the source file as-is when nothing needs to be masked
        "keep_audio": True,
    },
    "decode": {
        "hwaccel": "cuda",  # NVDEC; None = software. Unsupported codecs fall back to software automatically
        "scale_threads": 4,  # threads used to downscale frames for detection
    },
    "preview": {
        "width": 960,
    },
    "profiles": {},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def resolve_for(cfg: dict, rel_path: str) -> tuple[dict, list[str]]:
    """Effective config for one video: merge every profile whose `match` hits, in order."""
    eff = {k: v for k, v in cfg.items() if k != "profiles"}
    rp = rel_path.replace("\\", "/").lower()
    matched = []
    for name, prof in (cfg.get("profiles") or {}).items():
        patterns = prof.get("match") or []
        if any(fnmatch.fnmatch(rp, p.lower()) for p in patterns):
            eff = deep_merge(eff, {k: v for k, v in prof.items() if k != "match"})
            matched.append(name)
    return eff, matched


def section_hash(*sections: dict) -> str:
    raw = json.dumps(sections, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def resolve_path(p: str | Path) -> Path:
    """Relative paths are looked up in the current directory first, then in the repository root."""
    p = Path(p)
    if p.is_absolute():
        return p
    for base in (Path.cwd(), REPO_ROOT):
        if (base / p).exists():
            return base / p
    return REPO_ROOT / p
