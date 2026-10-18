"""Video I/O: probing, decoding for detection, mosaic re-encoding (per-frame timestamps kept,
audio copied), output verification and debug previews."""
from __future__ import annotations

import os
import queue
import threading
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from .mosaic import PLANAR_FALLBACK, mosaic_bgr, mosaic_frame

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
SW_ENCODERS = {"h264": "libx264", "hevc": "libx265", "av1": "libsvtav1"}
# Substitutes to try when the encoder does not accept the source pixel format
PIX_ALTERNATIVES = {
    "yuvj420p": ["yuv420p"],
    "yuvj422p": ["yuv422p", "yuv420p"],
    "yuvj444p": ["yuv444p"],
    "yuv420p10le": ["p010le"],
    "yuv422p10le": ["p210le", "yuv420p10le", "p010le"],
    "yuv444p10le": ["yuv444p16le", "yuv444p10msble"],
    "p010le": ["yuv420p10le"],
    "nv12": ["yuv420p"],
}
# Hardware frame formats: encoders here only take frames in system memory, so never pick these
HW_PIX_FMTS = {"cuda", "d3d11", "d3d12", "dxva2_vld", "qsv", "vaapi", "vulkan", "videotoolbox_vld", "drm_prime", "mediacodec", "opencl"}
_END = object()


def background(gen, maxsize: int = 16):
    """Run a generator on a background thread (overlaps decoding with GPU inference / encoding)."""
    q: queue.Queue = queue.Queue(maxsize)
    err: list = []
    stop = threading.Event()

    def run():
        try:
            for item in gen:
                q.put(item)
                if stop.is_set():
                    break
        except BaseException as e:  # noqa: BLE001 - re-raised on the consumer thread
            err.append(e)
        finally:
            q.put(_END)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is _END:
                break
            yield item
    finally:
        stop.set()
        while t.is_alive():
            try:
                q.get(timeout=0.05)
            except queue.Empty:
                pass
    if err:
        raise err[0]


def pick_video_stream(container):
    attached = getattr(getattr(av.stream, "Disposition", None), "attached_pic", 0)
    for s in container.streams.video:
        if attached and (s.disposition & attached):
            continue  # skip cover art
        return s
    raise ValueError("No usable video stream")


def _frame_step(vs) -> int:
    rate = vs.average_rate or vs.guessed_rate or Fraction(30)
    return max(1, round(1 / (Fraction(rate) * vs.time_base)))


def probe(path) -> dict:
    with av.open(str(path)) as c:
        vs = pick_video_stream(c)
        cc = vs.codec_context
        rate = vs.average_rate or vs.guessed_rate
        return {
            "codec": cc.name,
            "width": cc.width,
            "height": cc.height,
            "pix_fmt": cc.format.name if cc.format else None,
            "fps": float(rate) if rate else None,
            "time_base": str(vs.time_base),
            "frames_header": vs.frames or None,
            "duration_s": float(c.duration / av.time_base) if c.duration else None,
            "bit_rate": c.bit_rate,
            "audio": [s.codec_context.name for s in c.streams.audio],
            "other_streams": [f"{s.index}:{s.type}" for s in c.streams if s.index != vs.index and s.type != "audio"],
        }


def open_input(path, hwaccel: str | None = None):
    """Open an input. With hwaccel="cuda" decode on NVDEC; unsupported codecs fall back to software."""
    if hwaccel:
        try:
            from av.codec.hwaccel import HWAccel

            return av.open(str(path), hwaccel=HWAccel(device_type=hwaccel, allow_software_fallback=True))
        except Exception:  # noqa: BLE001
            pass
    return av.open(str(path))


def iter_frames(path, width: int | None = None, height: int | None = None, max_frames: int | None = None,
                hwaccel: str | None = None, threads: int = 4):
    """Decode the main video stream and yield (pts, BGR ndarray); scaling and color conversion run in one swscale pass."""
    with open_input(path, hwaccel) as c:
        vs = pick_video_stream(c)
        vs.thread_type = "AUTO"
        step = _frame_step(vs)
        for i, frame in enumerate(c.decode(vs)):
            if max_frames is not None and i >= max_frames:
                break
            pts = frame.pts if frame.pts is not None else i * step
            img = frame.reformat(width=width, height=height, format="bgr24", interpolation="BILINEAR", threads=threads).to_ndarray()
            yield pts, img


# ---------------------------------------------------------------- encoding


def pick_pix_fmt(src: str, supported: list[str]) -> str:
    supported = [f for f in supported if f not in HW_PIX_FMTS]
    if src in supported:
        return src
    for alt in PIX_ALTERNATIVES.get(src, []):
        if alt in supported:
            return alt
    try:
        high = max(c.bits for c in av.VideoFormat(src).components) > 8
    except Exception:
        high = False
    for alt in (["p010le", "yuv420p10le"] if high else []) + ["yuv420p", "nv12"]:
        if alt in supported:
            return alt
    return supported[0]


def encoder_options(name: str, ecfg: dict) -> dict:
    q = str(ecfg["quality"])
    preset = ecfg.get("preset")
    if name.endswith("_nvenc"):
        return {"preset": str(preset or "p5"), "tune": "hq", "rc": "vbr", "cq": q}
    if name in ("libx264", "libx265"):
        opts = {"crf": q, "preset": str(preset or "medium")}
        if name == "libx265":
            opts["x265-params"] = "log-level=error"
        return opts
    if name == "libsvtav1":
        return {"crf": q, "preset": str(preset if preset is not None else 8)}
    return {}


def _configure(cc, width, height, pix, rate, time_base, ecfg):
    cc.width, cc.height, cc.pix_fmt = width, height, pix
    cc.time_base = time_base
    cc.framerate = rate
    cc.bit_rate = 0  # constant quality via cq / crf
    if ecfg.get("gop"):
        cc.gop_size = int(ecfg["gop"])
    if ecfg.get("max_b_frames") is not None:
        cc.max_b_frames = int(ecfg["max_b_frames"])


def choose_encoder(src_codec: str, src_pix: str, width, height, rate, time_base, ecfg: dict):
    """Pick an encoder of the configured codec family: NVENC first, software if NVENC cannot be opened."""
    family = ecfg["codec"] if ecfg["codec"] != "auto" else src_codec
    if family not in SW_ENCODERS:
        family = ecfg["fallback_codec"]
    names = ([f"{family}_nvenc"] if ecfg.get("hw") else []) + [SW_ENCODERS[family]]
    errors = []
    for name in names:
        if name not in av.codecs_available:
            errors.append(f"{name}: not available")
            continue
        fmts = [f.name for f in (av.Codec(name, "w").video_formats or [])]
        pix = pick_pix_fmt(src_pix, fmts) if fmts else src_pix
        opts = encoder_options(name, ecfg)
        try:
            cc = av.CodecContext.create(name, "w")
            _configure(cc, width, height, pix, rate, time_base, ecfg)
            cc.options = opts
            cc.open()
            del cc
            return name, pix, opts
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
    raise RuntimeError("No usable encoder: " + "; ".join(errors))


def _reset_pict_type(frame):
    # Decoded frames carry their original I/P/B type; clear it so the encoder chooses keyframes itself
    try:
        frame.pict_type = av.video.frame.PictureType.NONE
    except Exception:
        frame.pict_type = 0


def _demux_decode(inp, vs, a_in, step):
    streams = [vs] + ([a_in] if a_in is not None else [])
    i = 0
    for packet in inp.demux(*streams):
        if a_in is not None and packet.stream.index == a_in.index:
            if packet.dts is not None:
                yield "a", packet
            continue
        for frame in packet.decode():
            if frame.pts is None:
                frame.pts = i * step
            i += 1
            yield "v", frame


def render(src, dst, boxes_per_frame, frame_pts, mcfg: dict, ecfg: dict, hwaccel: str | None = None,
           src_pix_fmt: str | None = None) -> dict:
    """Decode -> mosaic per frame -> re-encode. Writes a temporary file and renames it on success.

    src_pix_fmt: the source pixel format (from a software-decoder probe). With hardware decoding the
    decoder context reports "cuda", which must not be used to choose the output format.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".partial" + dst.suffix)
    info: dict = {"warnings": []}
    with open_input(src, hwaccel) as inp:
        vs = pick_video_stream(inp)
        vs.thread_type = "AUTO"
        cc = vs.codec_context
        rate = vs.average_rate or vs.guessed_rate or Fraction(30)
        a_in = inp.streams.audio[0] if (ecfg.get("keep_audio") and inp.streams.audio) else None
        src_pix = src_pix_fmt or (cc.format.name if cc.format and cc.format.name not in HW_PIX_FMTS else "yuv420p")
        name, pix, opts = choose_encoder(cc.name, src_pix, cc.width, cc.height, rate, vs.time_base, ecfg)
        info.update(encoder=name, pix_fmt=pix, options=opts)
        dropped = [f"{s.index}:{s.type}" for s in inp.streams if s.index != vs.index and (a_in is None or s.index != a_in.index)]
        if dropped:
            info["warnings"].append(f"streams not written to output: {', '.join(dropped)}")

        with av.open(str(tmp), "w") as out:
            ovs = out.add_stream(name, rate=rate, options=opts)
            _configure(ovs.codec_context, cc.width, cc.height, pix, rate, vs.time_base, ecfg)
            ovs.time_base = vs.time_base
            for k in ("color_range", "colorspace", "color_primaries", "color_trc"):
                try:
                    setattr(ovs.codec_context, k, getattr(cc, k))
                except Exception:
                    pass
            if name.startswith("hevc") and dst.suffix.lower() in (".mp4", ".mov", ".m4v"):
                ovs.codec_tag = "hvc1"  # QuickTime / Apple compatibility
            oas = None
            if a_in is not None:
                try:
                    oas = out.add_stream_from_template(a_in)
                except Exception as e:  # noqa: BLE001
                    info["warnings"].append(f"audio could not be copied and was dropped: {e}")
                    a_in = None

            n = 0
            blocks, min_block = int(mcfg["blocks"]), int(mcfg["min_block"])
            for kind, item in background(_demux_decode(inp, vs, a_in, _frame_step(vs))):
                if kind == "a":
                    item.stream = oas
                    out.mux(item)
                    continue
                frame = item
                if n >= len(frame_pts):
                    raise RuntimeError(f"Decoded more frames than the {len(frame_pts)} seen during detection")
                if frame.pts != frame_pts[n]:
                    raise RuntimeError(f"Timestamp mismatch at frame {n}: {frame.pts} != {frame_pts[n]}")
                boxes = boxes_per_frame[n]
                if boxes:
                    # Decoded frames share buffers with the decoder's reference frames; writing in place
                    # would leak the mosaic into later frames, so copy first.
                    frame.make_writable()
                    if frame.format.name in PLANAR_FALLBACK:
                        pts, tb = frame.pts, frame.time_base
                        frame = frame.reformat(format=PLANAR_FALLBACK[frame.format.name])
                        frame.pts, frame.time_base = pts, tb
                    mosaic_frame(frame, boxes, blocks, min_block)
                _reset_pict_type(frame)
                for p in ovs.encode(frame):
                    out.mux(p)
                n += 1
            for p in ovs.encode(None):
                out.mux(p)
    if n != len(frame_pts):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Encoded {n} frames but detection saw {len(frame_pts)}")
    os.replace(tmp, dst)
    info["frames_written"] = n
    return info


def verify(dst, expected_pts, in_time_base: str, width: int, height: int) -> dict:
    """Compare the output's packet timestamps (no decoding) with the source, frame by frame."""
    in_tb = Fraction(in_time_base)
    with av.open(str(dst)) as c:
        vs = pick_video_stream(c)
        out_tb = vs.time_base
        size_ok = (vs.codec_context.width, vs.codec_context.height) == (width, height)
        pts = sorted(p.pts for p in c.demux(vs) if p.pts is not None)
    tol = max(in_tb, out_tb)
    mismatch = sum(1 for a, b in zip(expected_pts, pts) if abs(Fraction(a) * in_tb - Fraction(b) * out_tb) > tol)
    ok = size_ok and len(pts) == len(expected_pts) and mismatch == 0
    return {"ok": ok, "frames_in": len(expected_pts), "frames_out": len(pts), "pts_mismatch": mismatch, "size_ok": size_ok}


# ---------------------------------------------------------------- preview


def render_preview(src, dst, raw_dets, final_boxes, meta: dict, score_high: float, mcfg: dict, width: int = 960,
                   hwaccel: str | None = None):
    """Low-res debug preview: final mosaic areas (mosaic + red box), raw detections (green = high score,
    orange = low score) with their scores."""
    W, H = meta["width"], meta["height"]
    pw = min(width, W) // 2 * 2
    ph = round(H * pw / W / 2) * 2
    sx, sy = pw / W, ph / H
    rate = Fraction(meta["fps"]).limit_denominator(1001) if meta.get("fps") else Fraction(30)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    enc = "h264_nvenc" if "h264_nvenc" in av.codecs_available else "libx264"
    with av.open(str(dst), "w") as out:
        s = out.add_stream(enc, rate=rate, options={"preset": "p3", "cq": "28"} if enc == "h264_nvenc" else {"preset": "veryfast", "crf": "26"})
        s.width, s.height, s.pix_fmt = pw, ph, "yuv420p"
        s.codec_context.bit_rate = 0
        for i, (_, img) in enumerate(background(iter_frames(src, pw, ph, hwaccel=hwaccel))):
            img = np.ascontiguousarray(img)
            fb = [[int(x1 * sx), int(y1 * sy), int(np.ceil(x2 * sx)), int(np.ceil(y2 * sy))] for x1, y1, x2, y2 in final_boxes[i]]
            mosaic_bgr(img, fb, int(mcfg["blocks"]), 1)
            for x1, y1, x2, y2 in fb:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            for x1, y1, x2, y2, sc in raw_dets[i]:
                color = (0, 220, 0) if sc >= score_high else (0, 165, 255)
                p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
                cv2.rectangle(img, p1, p2, color, 1)
                cv2.putText(img, f"{sc:.2f}", (p1[0], max(10, p1[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            cv2.putText(img, f"#{i}  raw:{len(raw_dets[i])}  mask:{len(final_boxes[i])}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts = i
            frame.time_base = 1 / rate
            for p in s.encode(frame):
                out.mux(p)
        for p in s.encode(None):
            out.mux(p)
