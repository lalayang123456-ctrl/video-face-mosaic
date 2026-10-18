"""Processing pipeline: one video (detect -> temporal post-processing -> mosaic & encode -> verify),
batch runs, and detector comparison."""
from __future__ import annotations

import csv
import gzip
import json
import shutil
import time
import traceback
from pathlib import Path

from tqdm import tqdm

from .config import deep_merge, resolve_for, resolve_path, section_hash
from .detectors import FaceDetector, build_detector
from .models import model_relpath
from .tracking import finalize
from .video import VIDEO_EXTS, background, iter_frames, probe, render, render_preview, verify


def _dump_gz(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


def _load_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


class DetectorPool:
    """Load each detector configuration once and reuse it across videos."""

    def __init__(self):
        self._cache: dict = {}

    def get(self, det_cfg: dict) -> FaceDetector:
        key = (det_cfg["name"], str(resolve_path(det_cfg["model"])), det_cfg.get("device"))
        if key not in self._cache:
            self._cache[key] = build_detector(det_cfg)
        return FaceDetector(det_cfg, self._cache[key])


def _prepared_frames(src, det: FaceDetector, plan, max_frames, dec: dict):
    # Decoding, resizing and preprocessing run on a background thread; the main thread only does
    # GPU inference and post-processing.
    dw, dh = plan[0][:2]
    for p, img in iter_frames(src, dw, dh, max_frames, hwaccel=dec.get("hwaccel"), threads=dec.get("scale_threads", 4)):
        yield p, det.prepare(img, plan)


def detect_video(src: Path, det: FaceDetector, meta: dict, dec: dict, max_frames: int | None = None, desc: str = "detect") -> dict:
    W, H = meta["width"], meta["height"]
    plan = det.plan(W, H)
    pts, dets = [], []
    total = meta.get("frames_header")
    if max_frames:
        total = min(total or max_frames, max_frames)
    t0 = time.perf_counter()
    frames = background(_prepared_frames(src, det, plan, max_frames, dec))
    for p, prep in tqdm(frames, total=total, desc=desc, unit="frame", leave=False):
        d = det.run(prep, W, H)
        pts.append(int(p))
        dets.append([[round(float(v), 1) for v in r[:4]] + [round(float(r[4]), 3)] for r in d])
    elapsed = time.perf_counter() - t0
    return {
        "pts": pts,
        "dets": dets,
        "scales": [f"{w}x{h}" + (f"+pad{p:g}" if p else "") for w, h, p in plan],
        "elapsed_s": round(elapsed, 2),
        "fps": round(len(pts) / max(elapsed, 1e-9), 1),
        "providers": list(getattr(det.backend, "providers", [])),
    }


def process_video(src: Path, dst: Path, work_prefix: Path, cfg: dict, pool: DetectorPool, *, preview: bool = False, force: bool = False) -> dict:
    t_start = time.perf_counter()
    meta = probe(src)
    res: dict = {"src": str(src), "dst": str(dst), "meta": meta}

    # 1) Detection (cached; reused as long as the detector config is unchanged)
    dets_path = Path(str(work_prefix) + ".dets.json.gz")
    det_hash = section_hash(cfg["detector"])
    data = None
    if dets_path.exists() and not force:
        cached = _load_gz(dets_path)
        if cached.get("det_hash") == det_hash:
            data = cached
    if data is None:
        data = detect_video(src, pool.get(cfg["detector"]), meta, cfg["decode"], desc=src.name)
        data["det_hash"] = det_hash
        _dump_gz(dets_path, data)
    res["detect"] = {k: data[k] for k in ("scales", "elapsed_s", "fps", "providers") if k in data}

    # 2) Temporal post-processing
    n = len(data["pts"])
    boxes, tstats = finalize(data["dets"], n, meta["width"], meta["height"], cfg["tracking"])
    res["tracking"] = tstats
    _dump_gz(Path(str(work_prefix) + ".boxes.json.gz"), {"pts": data["pts"], "boxes": boxes})

    # 3) Mosaic + encode, or copy as-is
    ecfg = cfg["encode"]
    if tstats["mosaic_frames"] == 0 and ecfg.get("copy_if_no_face", True):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        res["action"] = "copied"
        res["status"] = "ok"
    else:
        t0 = time.perf_counter()
        res["render"] = render(src, dst, boxes, data["pts"], cfg["mosaic"], ecfg, hwaccel=cfg["decode"].get("hwaccel"),
                               src_pix_fmt=meta["pix_fmt"])
        res["render"]["elapsed_s"] = round(time.perf_counter() - t0, 2)
        res["verify"] = verify(dst, data["pts"], meta["time_base"], meta["width"], meta["height"])
        res["action"] = "rendered"
        res["status"] = "ok" if res["verify"]["ok"] else "verify_failed"

    if preview:
        pv = Path(str(work_prefix) + ".preview.mp4")
        render_preview(src, pv, data["dets"], boxes, meta, cfg["tracking"]["score_high"], cfg["mosaic"], cfg["preview"]["width"],
                       hwaccel=cfg["decode"].get("hwaccel"))
        res["preview"] = str(pv)
    res["elapsed_s"] = round(time.perf_counter() - t_start, 2)
    return res


def find_videos(inp: Path, exclude: list[Path]) -> tuple[list[Path], Path]:
    if inp.is_file():
        return [inp], inp.parent
    ex = [e.resolve() for e in exclude]
    files = []
    for p in sorted(inp.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXTS or ".partial." in p.name:
            continue
        rp = p.resolve()
        if any(rp.is_relative_to(e) for e in ex):
            continue
        files.append(p)
    return files, inp


REPORT_FIELDS = ["file", "status", "action", "profiles", "frames", "raw_det_frames", "mosaic_frames", "tracks_confirmed",
                 "det_fps", "encoder", "elapsed_s", "warnings", "error"]


def _report_row(rel: str, res: dict) -> dict:
    t = res.get("tracking", {})
    return {
        "file": rel,
        "status": res.get("status"),
        "action": res.get("action"),
        "profiles": "+".join(res.get("profiles", [])),
        "frames": res.get("verify", {}).get("frames_in") or res.get("meta", {}).get("frames_header"),
        "raw_det_frames": t.get("raw_det_frames"),
        "mosaic_frames": t.get("mosaic_frames"),
        "tracks_confirmed": t.get("tracks_confirmed"),
        "det_fps": res.get("detect", {}).get("fps"),
        "encoder": res.get("render", {}).get("encoder"),
        "elapsed_s": res.get("elapsed_s"),
        "warnings": " | ".join(res.get("render", {}).get("warnings", [])),
        "error": (res.get("error") or "").strip().splitlines()[-1] if res.get("error") else "",
    }


def run_batch(inp: Path, out_dir: Path, work_dir: Path, cfg: dict, *, preview: bool = False, force: bool = False) -> list[dict]:
    files, base = find_videos(inp, [out_dir, work_dir])
    if not files:
        print(f"No video files found under {inp}")
        return []
    print(f"{len(files)} video(s)  output: {out_dir}  work dir: {work_dir}")
    pool = DetectorPool()
    rows = []
    for i, src in enumerate(files, 1):
        rel = src.relative_to(base).as_posix()
        eff, profiles = resolve_for(cfg, rel)
        dst = out_dir / rel
        work_prefix = work_dir / rel
        result_path = Path(str(work_prefix) + ".result.json")
        cfg_hash = section_hash(eff["detector"], eff["tracking"], eff["mosaic"], eff["encode"])
        if not force and result_path.exists() and dst.exists():
            prev = json.loads(result_path.read_text(encoding="utf-8"))
            if prev.get("status") == "ok" and prev.get("cfg_hash") == cfg_hash:
                print(f"[{i}/{len(files)}] skipped (already done): {rel}")
                rows.append(_report_row(rel, prev))
                continue
        print(f"[{i}/{len(files)}] {rel}" + (f"  profile={'+'.join(profiles)}" if profiles else ""))
        try:
            res = process_video(src, dst, work_prefix, eff, pool, preview=preview, force=force)
        except Exception:  # noqa: BLE001 - one failed video must not stop the batch
            res = {"src": str(src), "dst": str(dst), "status": "failed", "error": traceback.format_exc()}
        res["profiles"] = profiles
        res["cfg_hash"] = cfg_hash
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        row = _report_row(rel, res)
        rows.append(row)
        print(f"    -> {row['status']}  {row['action'] or ''}  masked frames {row['mosaic_frames']}/{row['frames']}"
              f"  detection {row['det_fps']} fps  took {row['elapsed_s']}s" + (f"\n    error: {row['error']}" if row["error"] else ""))

    work_dir.mkdir(parents=True, exist_ok=True)
    report = work_dir / "report.csv"
    with open(report, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r["status"] == "ok" for r in rows)
    print(f"Done: {n_ok} / {len(rows)} succeeded. Report: {report}")
    return rows


def compare(video: Path, cfg: dict, names: list[str], out_dir: Path, *, preview: bool = True, max_frames: int | None = None) -> list[dict]:
    """Compare detectors on one video: frames with detections, frames masked, speed; writes one preview each."""
    meta = probe(video)
    print(f"{video.name}: {meta['width']}x{meta['height']} {meta['codec']} {meta['pix_fmt']} {meta['fps']:.2f}fps {meta['frames_header']} frames")
    rows = []
    for name in names:
        dcfg = deep_merge(cfg["detector"], {"name": name})
        if name != cfg["detector"]["name"]:
            dcfg["model"] = model_relpath(name)
        det = FaceDetector(dcfg)
        data = detect_video(video, det, meta, cfg["decode"], max_frames=max_frames, desc=name)
        n = len(data["pts"])
        boxes, tstats = finalize(data["dets"], n, meta["width"], meta["height"], cfg["tracking"])
        row = {"detector": name, "scales": ",".join(data["scales"]), "frames": n, "fps": data["fps"], **tstats,
               "max_faces": max((len(d) for d in data["dets"]), default=0)}
        if preview:
            pv = out_dir / f"{video.stem}.{name}.preview.mp4"
            render_preview(video, pv, data["dets"], boxes, meta, cfg["tracking"]["score_high"], cfg["mosaic"], cfg["preview"]["width"],
                           hwaccel=cfg["decode"].get("hwaccel"))
            row["preview"] = str(pv)
        rows.append(row)
    keys = ["detector", "scales", "frames", "fps", "raw_det_frames", "mosaic_frames", "tracks", "tracks_confirmed", "filled_boxes", "max_faces"]
    print("\t".join(keys))
    for r in rows:
        print("\t".join(str(r[k]) for k in keys))
    for r in rows:
        if "preview" in r:
            print(f"preview [{r['detector']}]: {r['preview']}")
    return rows
