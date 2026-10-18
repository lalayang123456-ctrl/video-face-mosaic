"""Model registry and download (with SHA256 check). Model files are not committed to the repository;
`facemosaic download-models` fetches them into models/."""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from tqdm import tqdm

MODELS: dict[str, dict] = {
    "scrfd": {
        "file": "det_10g.onnx",
        "url": "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx",
        "sha256": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
        "license": "InsightFace pretrained weights, non-commercial research use only",
    },
    "yunet": {
        "file": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "license": "OpenCV Zoo, MIT",
    },
}


def model_relpath(name: str) -> str:
    return f"models/{MODELS[name]['file']}"


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mirror(url: str) -> str:
    # If HF_ENDPOINT is set (e.g. https://hf-mirror.com), download Hugging Face files from that mirror
    endpoint = os.environ.get("HF_ENDPOINT")
    prefix = "https://huggingface.co"
    if endpoint and url.startswith(prefix):
        return endpoint.rstrip("/") + url[len(prefix):]
    return url


def download(name: str, dest_dir: Path, force: bool = False) -> Path:
    info = MODELS[name]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / info["file"]
    if dest.exists() and not force:
        if sha256sum(dest) == info["sha256"]:
            print(f"[OK] {name}: already present, checksum OK: {dest}")
            return dest
        print(f"[!!] {name}: present but checksum does not match, downloading again")
    url = _mirror(info["url"])
    tmp = dest.with_name(dest.name + ".download")
    print(f"Downloading {name}: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "facemosaic"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0) or None
        with tqdm(total=total, unit="B", unit_scale=True, leave=False) as bar:
            for chunk in iter(lambda: r.read(1 << 16), b""):
                f.write(chunk)
                bar.update(len(chunk))
    digest = sha256sum(tmp)
    if digest != info["sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: checksum mismatch, expected {info['sha256']}, got {digest}")
    os.replace(tmp, dest)
    print(f"[OK] {name}: {dest} ({info['license']})")
    return dest
