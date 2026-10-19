"""Face detectors: SCRFD (onnxruntime, GPU) and YuNet (OpenCV, CPU).

detect() returns a float32 array of shape (N, 5): x1, y1, x2, y2, score, in input-image pixels.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np

EMPTY = np.zeros((0, 5), np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.asarray(keep, np.int64)


class SCRFD:
    """InsightFace SCRFD (det_10g.onnx). Accepts any input size that is a multiple of 32."""

    strides = (8, 16, 32)

    def __init__(self, model_path: str | Path, device: str = "cuda"):
        import onnxruntime as ort

        ort.set_default_logger_severity(3)  # dynamic input sizes make ORT spam harmless "shape mismatch" warnings
        if device == "cuda":
            try:
                ort.preload_dlls()  # load the CUDA / cuDNN runtime installed via pip
            except Exception:
                pass
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._new_session = lambda: ort.InferenceSession(str(model_path), so, providers=providers)
        session = self._new_session()
        self.providers = session.get_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in self.providers:
            raise RuntimeError(f"CUDA is not available, got providers: {self.providers}")
        self.input_name = session.get_inputs()[0].name
        self.output_names = [o.name for o in session.get_outputs()]
        if len(self.output_names) not in (6, 9):
            raise ValueError(f"Not an SCRFD model (expected 6 or 9 outputs, got {len(self.output_names)})")
        # One session per input shape: switching shapes inside one CUDA session forces cuDNN to
        # re-plan the convolutions on every call, which is >10x slower.
        self._sessions: dict = {}
        self._spare = session
        self._centers: dict = {}

    def _session(self, shape):
        s = self._sessions.get(shape)
        if s is None:
            s = self._spare or self._new_session()
            self._spare = None
            self._sessions[shape] = s
        return s

    def _anchor_centers(self, h: int, w: int, stride: int, num_anchors: int) -> np.ndarray:
        key = (h, w, stride, num_anchors)
        c = self._centers.get(key)
        if c is None:
            c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            c = (c * stride).reshape(-1, 2)
            if num_anchors > 1:
                c = np.repeat(c, num_anchors, axis=0)
            self._centers[key] = c
        return c

    def prepare(self, img: np.ndarray) -> np.ndarray:
        """CPU preprocessing (pad to a multiple of 32 + normalize). Can run on the decode thread."""
        h, w = img.shape[:2]
        ph, pw = math.ceil(h / 32) * 32, math.ceil(w / 32) * 32
        if (ph, pw) != (h, w):
            canvas = np.zeros((ph, pw, 3), np.uint8)
            canvas[:h, :w] = img
            img = canvas
        return cv2.dnn.blobFromImage(img, 1.0 / 128.0, (pw, ph), (127.5, 127.5, 127.5), swapRB=True)

    def detect(self, img: np.ndarray, score_thr: float, nms_iou: float) -> np.ndarray:
        return self.infer(self.prepare(img), score_thr, nms_iou)

    def infer(self, blob: np.ndarray, score_thr: float, nms_iou: float) -> np.ndarray:
        ph, pw = blob.shape[2:]
        outs = self._session((ph, pw)).run(self.output_names, {self.input_name: blob})
        boxes_all, scores_all = [], []
        for i, s in enumerate(self.strides):
            scores = outs[i].reshape(-1)
            fh, fw = ph // s, pw // s
            num_anchors = scores.size // (fh * fw)
            pos = np.where(scores >= score_thr)[0]
            if pos.size == 0:
                continue
            dist = outs[i + 3].reshape(-1, 4)[pos] * s
            ctr = self._anchor_centers(fh, fw, s, num_anchors)[pos]
            boxes_all.append(np.concatenate([ctr - dist[:, :2], ctr + dist[:, 2:]], axis=1))
            scores_all.append(scores[pos])
        if not boxes_all:
            return EMPTY
        boxes = np.concatenate(boxes_all).astype(np.float32)
        scores = np.concatenate(scores_all).astype(np.float32)
        keep = nms(boxes, scores, nms_iou)
        return np.concatenate([boxes[keep], scores[keep, None]], axis=1)


class YuNet:
    """OpenCV FaceDetectorYN. The model is loaded from memory because OpenCV cannot open non-ASCII paths."""

    def __init__(self, model_path: str | Path):
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass
        buf = np.fromfile(str(model_path), np.uint8)
        self.det = cv2.FaceDetectorYN.create("onnx", buf, np.array([], np.uint8), (320, 320), 0.5, 0.3, 5000)
        self.providers = ["opencv-cpu"]
        self._size = None

    def prepare(self, img: np.ndarray) -> np.ndarray:
        return img

    def detect(self, img: np.ndarray, score_thr: float, nms_iou: float) -> np.ndarray:
        return self.infer(img, score_thr, nms_iou)

    def infer(self, img: np.ndarray, score_thr: float, nms_iou: float) -> np.ndarray:
        h, w = img.shape[:2]
        if self._size != (w, h):
            self.det.setInputSize((w, h))
            self._size = (w, h)
        self.det.setScoreThreshold(score_thr)
        self.det.setNMSThreshold(nms_iou)
        _, faces = self.det.detect(img)
        if faces is None or len(faces) == 0:
            return EMPTY
        x, y, bw, bh = faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 3]
        return np.stack([x, y, x + bw, y + bh, faces[:, 14]], axis=1).astype(np.float32)


class PersonDetector:
    """YOLOX (COCO, ONNX) person detector, used to check that a face track belongs to a person.

    detect() returns float32 (N, 5): x1, y1, x2, y2, score for class "person", in input-image pixels.
    """

    size = 640

    def __init__(self, model_path: str | Path, device: str = "cuda"):
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            try:
                ort.preload_dlls()
            except Exception:
                pass
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        grids, strides = [], []
        for s in (8, 16, 32):
            g = self.size // s
            yv, xv = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((g * g, 1), s))
        self._grids = np.concatenate(grids).astype(np.float32)
        self._strides = np.concatenate(strides).astype(np.float32)

    def detect(self, img: np.ndarray, conf: float = 0.25, nms_iou: float = 0.45) -> np.ndarray:
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[: int(h * r), : int(w * r)] = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        out = self.session.run(None, {self.input_name: canvas.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        score = out[:, 4] * out[:, 5]  # objectness * P(person); class 0 = person
        keep = score >= conf
        if not keep.any():
            return EMPTY
        xy = (out[keep, :2] + self._grids[keep]) * self._strides[keep]
        wh = np.exp(out[keep, 2:4]) * self._strides[keep]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r
        k = nms(boxes, score[keep], nms_iou)
        return np.concatenate([boxes[k], score[keep][k, None]], 1).astype(np.float32)


def face_in_person(face, persons: np.ndarray, head_frac: float = 0.5, margin: float = 0.1, max_area: float = 0.8) -> bool:
    """True if the face box sits in the head region (upper part) of some person box."""
    fx, fy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
    fa = (face[2] - face[0]) * (face[3] - face[1])
    for x1, y1, x2, y2, _ in persons:
        pw, ph = x2 - x1, y2 - y1
        if (x1 - margin * pw <= fx <= x2 + margin * pw and y1 - margin * ph <= fy <= y1 + head_frac * ph
                and fa <= max_area * pw * ph):
            return True
    return False


def ensure_fp16(model: Path) -> Path:
    """Convert the FP32 model to FP16 on first use (inputs/outputs stay FP32) and cache it next to it."""
    out = model.with_name(model.stem + "_fp16.onnx")
    if not out.exists():
        import onnx
        from onnxconverter_common import float16

        m = float16.convert_float_to_float16(onnx.load(str(model)), keep_io_types=True)
        tmp = out.with_name(out.name + ".tmp")
        onnx.save(m, str(tmp))
        os.replace(tmp, out)
    return out


def build_detector(det_cfg: dict):
    from .config import resolve_path

    name = det_cfg["name"].lower()
    model = resolve_path(det_cfg["model"])
    if not model.exists():
        raise FileNotFoundError(f"Model file not found: {model}\nRun: facemosaic download-models")
    if name == "scrfd":
        device = det_cfg.get("device", "cuda")
        if det_cfg.get("fp16") and device == "cuda":
            model = ensure_fp16(model)
        return SCRFD(model, device)
    if name == "yunet":
        return YuNet(model)
    raise ValueError(f"Unknown detector: {name}")


def det_input_size(src_w: int, src_h: int, det_size: int) -> tuple[int, int]:
    """Detection resolution: scale the long side down to det_size (never up), keep even dimensions."""
    long_side = max(src_w, src_h)
    if not det_size or det_size >= long_side:
        return src_w, src_h
    s = det_size / long_side
    return max(2, round(src_w * s / 2) * 2), max(2, round(src_h * s / 2) * 2)


def scale_plan(src_w: int, src_h: int, scales: list[dict]) -> list[tuple[int, int, float]]:
    """Multi-scale plan [(det_w, det_h, border_pad), ...], largest first; duplicate sizes keep the first entry."""
    plan, seen = [], set()
    for s in scales:
        wh = det_input_size(src_w, src_h, int(s.get("size") or 0))
        if wh not in seen:
            seen.add(wh)
            plan.append((wh[0], wh[1], float(s.get("border_pad") or 0)))
    return sorted(plan, key=lambda p: -p[0] * p[1])


class FaceDetector:
    """Multi-scale detection: large scales catch distant faces, small scales catch faces close to the lens.

    Handles border padding and maps boxes back to source-video pixel coordinates.
    """

    def __init__(self, det_cfg: dict, backend=None):
        self.cfg = det_cfg
        self.backend = backend or build_detector(det_cfg)

    def plan(self, src_w: int, src_h: int) -> list[tuple[int, int, float]]:
        return scale_plan(src_w, src_h, self.cfg["scales"])

    def prepare(self, img: np.ndarray, plan) -> list:
        """CPU part (resize per scale + padding + preprocessing); img must have the size of plan[0]."""
        out = []
        for dw, dh, pad in plan:
            im = img if img.shape[1] == dw and img.shape[0] == dh else cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
            px, py = int(round(dw * pad)), int(round(dh * pad))
            if px or py:
                im = cv2.copyMakeBorder(im, py, py, px, px, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            out.append((self.backend.prepare(im), (dw, dh, px, py)))
        return out

    def detect(self, img: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
        """Detect on a single image (the source frame or any downscaled copy of it)."""
        plan = [(w, h, p) for w, h, p in self.plan(img.shape[1], img.shape[0])]
        if plan[0][:2] != (img.shape[1], img.shape[0]):
            img = cv2.resize(img, plan[0][:2], interpolation=cv2.INTER_AREA)
        return self.run(self.prepare(img, plan), src_w, src_h)

    def run(self, prepared: list, src_w: int, src_h: int) -> np.ndarray:
        parts = []
        for payload, (w, h, px, py) in prepared:
            d = self.backend.infer(payload, self.cfg["score_low"], self.cfg["nms_iou"])
            if len(d) == 0:
                continue
            d = d.copy()
            d[:, [0, 2]] = (d[:, [0, 2]] - px) * (src_w / w)
            d[:, [1, 3]] = (d[:, [1, 3]] - py) * (src_h / h)
            parts.append(d)
        if not parts:
            return EMPTY
        d = np.concatenate(parts)
        d[:, [0, 2]] = d[:, [0, 2]].clip(0, src_w)
        d[:, [1, 3]] = d[:, [1, 3]].clip(0, src_h)
        bw, bh = d[:, 2] - d[:, 0], d[:, 3] - d[:, 1]
        min_face = float(self.cfg.get("min_face") or 0)
        d = d[(bw > 1) & (bh > 1) & (np.minimum(bw, bh) >= min_face)]
        if len(prepared) > 1 and len(d) > 1:
            d = d[nms(d[:, :4], d[:, 4], self.cfg["nms_iou"])]
        return d
