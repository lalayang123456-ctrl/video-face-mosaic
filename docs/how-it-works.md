# How it works

This page is for people who change or maintain the code. To just use the tool, the [README](../README.md) is enough.

## Pipeline

Each video is processed in two passes:

```
Pass 1: detection (pipeline.detect_video)
  decode (NVDEC) -> resize to the largest detection scale (swscale, multi-threaded) -> per-scale preprocessing   <- background thread
  -> SCRFD inference (GPU, FP16) -> merge scales + NMS                                                          <- main thread
  -> raw per-frame detections written to <video>.dets.json.gz (also the cache)

Temporal post-processing (tracking.finalize)
  greedy IoU linking into tracks -> dual-threshold confirmation -> interpolate missed frames -> extend track ends -> enlarge boxes
  -> final per-frame mosaic boxes written to <video>.boxes.json.gz

Pass 2: mosaic and encode (video.render)
  decode -> for frames with boxes: copy the buffer, then pixelate in place on the YUV planes -> encode (NVENC)
  -> written with the source timestamps; audio copied as-is -> temporary file, renamed on success

Verification (video.verify)
  read only the output's packet timestamps (no decoding) and compare frame count and times with the source
```

If no frame needs masking, pass 2 is skipped and the source file is copied.

Why two passes instead of masking while detecting:

- Temporal post-processing needs "future" frames: if a track first scores high at frame 10, its low-score detections in frames 1–9 must be masked as well.
- Cached detections mean that changing tracking, mosaic or encoding settings does not require detecting again.
- The final per-frame mosaic boxes are saved separately in `.boxes.json.gz` for auditing. Note: this file is regenerated from the detections on every run, so manual edits to it currently have no effect (see "Manual review tool" at the end).

## Modules

| Module | Responsibility |
|---|---|
| `cli.py` | Command-line arguments and the entry point of each subcommand |
| `pipeline.py` | `process_video` (full pipeline for one video), `run_batch` (batches, resuming, report), `compare` |
| `detectors.py` | `SCRFD` / `YuNet` backends (`prepare` = preprocessing, `infer` = inference); `FaceDetector` handles multi-scale, padding and coordinate mapping |
| `tracking.py` | `link_tracks` links detections into tracks; `finalize` produces the final per-frame mosaic boxes |
| `mosaic.py` | `plane_views` maps a frame's planes to writable numpy views; `mosaic_frame` pixelates in place |
| `video.py` | Opening inputs (incl. hardware decoding), decoding for detection, `render`, `verify`, `render_preview`, encoder selection |
| `config.py` | Defaults, config merging, profile matching, config hashing |
| `models.py` | Model registry (URL, SHA256, license) and download |

## Key design decisions and pitfalls

### Frame count and timestamps must not change

Robot datasets need cameras aligned with each other and with action data, frame by frame. Therefore:

- Decoding and encoding go through PyAV rather than piping raw video through the ffmpeg CLI. Raw video pipes lose per-frame pts, which breaks audio sync and frame alignment for variable-frame-rate sources.
- Every encoded frame keeps the source pts, and the output stream uses the source time_base.
- Pass 2 checks every frame's pts against the pts recorded in pass 1 and fails immediately on a mismatch.
- After encoding, `verify` reads the output's packet timestamps and compares them with the source. The tolerance is the coarser of the two time_bases.

### Mosaic directly on the YUV planes

Frames are not converted to RGB and back. Each plane (Y/U/V, NV12's interleaved UV plane, packed RGB, ...) is pixelated directly in the corresponding region.

- Benefits: pixels outside the boxes do not change at all (apart from encoding loss); 10-bit stays 10-bit; two color conversions are saved.
- Box coordinates are snapped to the chroma subsampling grid (even coordinates for 4:2:0), so luma and chroma cells line up.
- A few packed formats (e.g. `yuyv422`) are converted to a planar format first.

**`frame.make_writable()` is mandatory**: decoder output frames share buffers with the decoder's internal reference frames. Writing in place would leak the mosaic into later frames that are predicted from them. `tests/test_video.py::test_mosaic_does_not_leak_into_later_frames_via_reference` covers this.

**The mosaic is pure numpy**: when detection and masking ran in the same process, `cv2.resize` occasionally raised "Unknown C++ exception" (opencv-python 5.0.0 on Windows). It could not be reproduced in isolation and the root cause is unknown. The problem disappeared after switching to a numpy block-mean implementation.

### Dual threshold + offline tracks

- Detection only filters with a low threshold (`score_low` = 0.2) to keep as many candidates as possible.
- A track is confirmed as a face only if at least one frame scores ≥ `score_high` (0.5). The whole confirmed track, including its low-score frames, is masked.
- Sporadic false positives on objects usually stay low-score for the whole track and are dropped, which limits accidental masking of training data.

### Multi-scale detection

SCRFD's largest anchors correspond to faces of roughly 512 px. On a 1280-wide detection image, huge faces right in front of the lens (600+ px and cut off by the frame edge) are essentially missed, with at most a 0.2–0.3 score on part of the face. Downscaled to 640 or 320 they score 0.7–0.87. So by default every frame is detected at 1280 / 640 / 320, and the merged results go through one more NMS pass.

**One ONNX Runtime session per input shape**: when a single session alternates between input shapes on every frame, the CUDA execution provider re-plans the convolutions each time; with three scales this went from about 28 ms to about 345 ms per frame.

### Performance

- Decoding, resizing and per-scale preprocessing run on a background thread, in parallel with GPU inference on the main thread (`video.background`).
- NVDEC hardware decoding, with 4 swscale threads for resizing (single-threaded scaling is the bottleneck at 2.7K 10-bit).
- FP16 model: same detections as FP32 (on a test set, all 131 face boxes matched, with a maximum score difference of 0.0008), but faster inference.
- With hardware decoding, the decoder context reports the pixel format as `cuda`, which must not be used to choose the output format. The format from the software `probe` is used instead.

### Resuming

- `<video>.result.json` stores the status and a config hash (detector + tracking + mosaic + encode). If the status is ok, the hash matches and the output exists, the video is skipped.
- `<video>.dets.json.gz` stores a hash of the detector section; the detections are reused as long as the detector settings are unchanged.
- Output is written to `<name>.partial.<ext>` first and renamed with `os.replace` on success.

## Intermediate file formats

`<video>.dets.json.gz`

```json
{
  "pts": [0, 1001, 2002],
  "dets": [[[x1, y1, x2, y2, score]], [], []],
  "scales": ["1280x720", "640x360+pad0.15", "320x180+pad0.15"],
  "elapsed_s": 36.2, "fps": 64.1,
  "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "det_hash": "..."
}
```

`<video>.boxes.json.gz`

```json
{ "pts": [0, 1001, 2002], "boxes": [[[x1, y1, x2, y2]], [], []] }
```

All coordinates are source-video pixels; `pts` is in units of the source video stream's time_base.

## Possible improvements

- **Half faces at the frame edge**: when the upper half of a face is outside the frame, face detectors miss it. A human pose model (e.g. RTMPose) could derive a head box from nose / eye / ear keypoints as a fallback.
- **Parallel videos**: videos are currently processed one at a time. When GPU utilization is low, processing 2–3 videos concurrently would help.
- **Manual review tool**: a small tool to step through frames and add or remove boxes in `.boxes.json.gz`, then re-run only pass 2.
