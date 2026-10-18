from fractions import Fraction

import av
import numpy as np

from facemosaic.config import DEFAULTS
from facemosaic.video import pick_pix_fmt, probe, render, verify

SW_ENCODE = {**DEFAULTS["encode"], "hw": False}  # tests must not need a GPU


def _make_video(path, n=30, w=160, h=120, rate=Fraction(30)):
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=rate)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        a = c.add_stream("aac", rate=48000)
        for i in range(n):
            img = np.full((h, w, 3), (i * 8) % 255, np.uint8)
            f = av.VideoFrame.from_ndarray(img, format="rgb24")
            f.pts = i
            f.time_base = 1 / rate
            for p in s.encode(f):
                c.mux(p)
        samples = np.zeros((1, 1024), np.float32)
        for i in range(n * 48000 // 30 // 1024):
            af = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
            af.sample_rate = 48000
            af.pts = i * 1024
            for p in a.encode(af):
                c.mux(p)
        for st in (s, a):
            for p in st.encode(None):
                c.mux(p)


def _make_static_noise_video(path, n=20, w=128, h=96):
    """Static noise: later frames are almost all skip blocks that copy the reference frame."""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (h, w, 3), np.uint8)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=30, options={"crf": "10", "g": "100"})
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        for i in range(n):
            f = av.VideoFrame.from_ndarray(img, format="rgb24")
            f.pts = i
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode(None):
            c.mux(p)


def _decoded_pts(path):
    with av.open(str(path)) as c:
        return [f.pts for f in c.decode(video=0)]


def test_pick_pix_fmt_never_returns_hw_format():
    nvenc = ["yuv420p", "nv12", "p010le", "yuv444p", "cuda"]
    assert pick_pix_fmt("cuda", nvenc) == "yuv420p"
    assert pick_pix_fmt("yuv420p10le", nvenc) == "p010le"
    assert pick_pix_fmt("yuvj420p", ["yuv420p", "nv12"]) == "yuv420p"


def test_render_keeps_frames_timestamps_and_audio(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out" / "in.mp4"
    _make_video(src)
    meta = probe(src)
    pts = _decoded_pts(src)
    boxes = [[[20, 20, 80, 80]] if i % 2 else [] for i in range(len(pts))]
    info = render(src, dst, boxes, pts, DEFAULTS["mosaic"], SW_ENCODE)
    assert info["frames_written"] == len(pts)
    v = verify(dst, pts, meta["time_base"], meta["width"], meta["height"])
    assert v["ok"], v
    with av.open(str(dst)) as c:
        assert len(c.streams.audio) == 1
    assert not (dst.parent / "in.partial.mp4").exists()


def test_av1_source_stays_av1(tmp_path):
    """The AV1 decoder is called libdav1d; the output must still be AV1, not the fallback codec."""
    src, dst = tmp_path / "av1.mp4", tmp_path / "out.mp4"
    with av.open(str(src), "w") as c:
        s = c.add_stream("libsvtav1", rate=30, options={"crf": "40", "preset": "12"})
        s.width, s.height, s.pix_fmt = 128, 96, "yuv420p"
        for i in range(10):
            f = av.VideoFrame.from_ndarray(np.full((96, 128, 3), i * 20, np.uint8), format="rgb24")
            f.pts = i
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode(None):
            c.mux(p)
    assert probe(src)["codec"] == "av1"
    pts = _decoded_pts(src)
    info = render(src, dst, [[[8, 8, 64, 64]]] * len(pts), pts, DEFAULTS["mosaic"], SW_ENCODE)
    assert info["encoder"] == "libsvtav1"
    assert probe(dst)["codec"] == "av1"


def test_compare_params_accepts_drop_in_replacement_and_flags_codec_change(tmp_path):
    from facemosaic.video import compare_params, stream_params

    src, same, other = tmp_path / "in.mp4", tmp_path / "same.mp4", tmp_path / "other.mp4"
    _make_video(src)
    pts = _decoded_pts(src)
    boxes = [[[20, 20, 80, 80]]] * len(pts)
    render(src, same, boxes, pts, DEFAULTS["mosaic"], {**SW_ENCODE, "quality": 23})
    render(src, other, boxes, pts, DEFAULTS["mosaic"], {**SW_ENCODE, "codec": "hevc"})
    s = stream_params(src)
    diffs = [d for d in compare_params(s, stream_params(same)) if not d.startswith("bitrate")]
    assert diffs == []
    assert any(d.startswith("codec") for d in compare_params(s, stream_params(other)))


def test_mosaic_does_not_leak_into_later_frames_via_reference(tmp_path):
    """Decoded frames share buffers with the decoder's reference frames: they must be copied before
    pixelating, otherwise the mosaic leaks into later frames."""
    src, dst = tmp_path / "noise.mp4", tmp_path / "out.mp4"
    _make_static_noise_video(src)
    with av.open(str(src)) as c:
        orig = [f.to_ndarray(format="gray") for f in c.decode(video=0)]
    pts = _decoded_pts(src)
    boxes = [[[16, 16, 80, 80]]] + [[] for _ in range(len(pts) - 1)]
    render(src, dst, boxes, pts, DEFAULTS["mosaic"], {**SW_ENCODE, "quality": 0})  # x264 crf 0 = lossless
    with av.open(str(dst)) as c:
        out = [f.to_ndarray(format="gray") for f in c.decode(video=0)]
    assert not np.array_equal(out[0], orig[0])  # frame 0 is masked
    for i in range(1, len(out)):
        assert np.array_equal(out[i], orig[i]), f"frame {i} was polluted by an earlier mosaic"
