"""Command-line interface: facemosaic <run | compare | download-models | check>."""
from __future__ import annotations

import argparse
import platform
import sys
from fractions import Fraction
from pathlib import Path

from . import __version__
from .config import DEFAULT_CONFIG, REPO_ROOT, deep_merge, load_config, resolve_path


def _load(config: Path | None) -> dict:
    path = config or (DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    if config and not config.exists():
        sys.exit(f"Config file not found: {config}")
    print(f"Config: {path or 'built-in defaults (configs/default.yaml not found)'}")
    return load_config(path)


def cmd_run(a) -> int:
    from .pipeline import run_batch

    cfg = _load(a.config)
    if not a.input.exists():
        sys.exit(f"Input not found: {a.input}")
    work = a.work or (a.output / "_work")
    rows = run_batch(a.input, a.output, work, cfg, preview=a.preview, force=a.force)
    return 0 if rows and all(r["status"] == "ok" for r in rows) else 1


def cmd_compare(a) -> int:
    from .pipeline import compare

    cfg = _load(a.config)
    if a.scales:
        scales = []
        for tok in a.scales:
            size, _, pad = tok.partition(":")
            scales.append({"size": int(size), "border_pad": float(pad or 0)})
        cfg["detector"] = deep_merge(cfg["detector"], {"scales": scales})
    compare(a.video, cfg, a.detectors, a.output, preview=not a.no_preview, max_frames=a.max_frames)
    return 0


def cmd_download(a) -> int:
    from .models import MODELS, download

    names = list(MODELS) if a.all else ["scrfd"]
    for name in names:
        download(name, a.dir, force=a.force)
    return 0


def cmd_check(a) -> int:
    """Check the runtime: dependency versions, model files, GPU inference, NVENC encoding, NVDEC decoding."""
    import av
    import cv2
    import numpy as np

    cfg = _load(a.config)
    ok = True

    def line(flag: str, msg: str):
        print(f"[{flag}] {msg}")

    print(f"facemosaic {__version__}  Python {platform.python_version()}  {platform.system()} {platform.release()}")
    print(f"numpy {np.__version__}  opencv {cv2.__version__}  av {av.__version__}")
    try:
        import onnxruntime as ort

        line("OK", f"onnxruntime {ort.__version__}, providers: {ort.get_available_providers()}")
    except ImportError:
        line("!!", "onnxruntime is not installed: pip install -e \".[gpu]\" (or \".[cpu]\")")
        return 1

    dcfg = cfg["detector"]
    model = resolve_path(dcfg["model"])
    if model.exists():
        line("OK", f"model: {model}")
    else:
        line("!!", f"model missing: {model} -- run: facemosaic download-models")
        ok = False

    if model.exists() and dcfg["name"] == "scrfd":
        from .detectors import build_detector

        try:
            det = build_detector(dcfg)
            det.detect(np.zeros((480, 640, 3), np.uint8), 0.5, 0.4)
            fp16 = " (FP16)" if dcfg.get("fp16") and dcfg.get("device") == "cuda" else ""
            line("OK", f"detector {dcfg['name']} works, providers: {det.providers}{fp16}")
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0]
            if dcfg.get("device") == "cuda":
                line("!!", f"GPU inference unavailable: {msg}\n     Without an NVIDIA GPU, set detector.device: cpu in the config (much slower)")
            else:
                line("!!", f"detector unavailable: {msg}")
            ok = False

    nvenc = []
    for name in ("h264_nvenc", "hevc_nvenc", "av1_nvenc"):
        if name not in av.codecs_available:
            continue
        try:
            cc = av.CodecContext.create(name, "w")
            cc.width, cc.height, cc.pix_fmt = 256, 256, "yuv420p"
            cc.time_base = Fraction(1, 30)
            cc.open()
            nvenc.append(name)
        except Exception:  # noqa: BLE001
            pass
    if nvenc:
        line("OK", f"NVENC hardware encoding: {', '.join(nvenc)}")
    else:
        line("--", "NVENC unavailable; software encoders will be used (libx264 / libx265 / libsvtav1, slower)")

    try:
        from av.codec.hwaccel import hwdevices_available

        hw = hwdevices_available()
    except Exception:  # noqa: BLE001
        hw = []
    if "cuda" in hw:
        line("OK", "NVDEC hardware decoding available")
    else:
        line("--", "NVDEC unavailable; software decoding will be used (or set decode.hwaccel: null)")

    print("Environment check passed" if ok else "Environment check FAILED -- fix the [!!] items above")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="facemosaic",
        description="Batch face detection + mosaic for videos. Output keeps the source frame count and timestamps.",
    )
    ap.add_argument("--version", action="version", version=f"facemosaic {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="<command>")

    r = sub.add_parser("run", help="process a directory (recursively) or a single video file")
    r.add_argument("input", type=Path, help="input directory (searched recursively for videos) or a single video file")
    r.add_argument("-o", "--output", type=Path, required=True, help="output directory; mirrors the input directory layout")
    r.add_argument("-c", "--config", type=Path, help=f"config file (default: {DEFAULT_CONFIG.relative_to(REPO_ROOT).as_posix()})")
    r.add_argument("-w", "--work", type=Path, help="directory for intermediate results and the report (default: <output>/_work)")
    r.add_argument("--preview", action="store_true", help="also write a low-res preview with detection boxes, for spot checks")
    r.add_argument("--force", action="store_true", help="ignore previous results and reprocess everything")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="try detectors / detection scales on one video; prints stats and writes previews")
    c.add_argument("video", type=Path)
    c.add_argument("-o", "--output", type=Path, default=Path("compare_output"), help="preview directory (default: ./compare_output)")
    c.add_argument("-c", "--config", type=Path)
    c.add_argument("--detectors", nargs="+", default=["scrfd"], choices=["scrfd", "yunet"], help="default: scrfd only")
    c.add_argument("--scales", nargs="+", metavar="SIZE[:PAD]", help="override detection scales, e.g. 1280 640:0.15 320:0.15")
    c.add_argument("--max-frames", type=int, help="only process the first N frames")
    c.add_argument("--no-preview", action="store_true")
    c.set_defaults(func=cmd_compare)

    d = sub.add_parser("download-models", help="download models into models/ (SHA256-verified)")
    d.add_argument("--dir", type=Path, default=REPO_ROOT / "models", help="download directory (default: <repo>/models)")
    d.add_argument("--all", action="store_true", help="also download YuNet (only used by `compare`)")
    d.add_argument("--force", action="store_true", help="download again even if present")
    d.set_defaults(func=cmd_download)

    k = sub.add_parser("check", help="check the environment (GPU, encoders/decoders, models)")
    k.add_argument("-c", "--config", type=Path)
    k.set_defaults(func=cmd_check)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
