# Configuration reference

Configuration is merged in this order, later entries overriding earlier ones:

1. Built-in defaults (`DEFAULTS` in `src/facemosaic/config.py`)
2. The config file (`configs/default.yaml` by default, or the file passed with `-c`)
3. Matching profiles (see the end of this page)

Anything a config file does not set falls back to the built-in default, so a custom config only needs the parts you want to change.

---

## detector: face detection

| Parameter | Default | Description |
|---|---|---|
| `name` | `scrfd` | Detector. `scrfd`: accurate and fast on GPU (recommended). `yunet`: lightweight, CPU only, noticeably more false positives; for comparison only |
| `model` | `models/det_10g.onnx` | Model path. Relative paths are looked up in the current directory first, then in the repository root |
| `device` | `cuda` | `cuda` or `cpu`. Only used by scrfd |
| `fp16` | `true` | Use an FP16 model on GPU; `*_fp16.onnx` is generated automatically on the first run. Same detections as FP32, faster |
| `scales` | see below | List of detection scales, each with `size` and `border_pad` |
| `score_low` | `0.2` | Raw detections below this score are dropped |
| `nms_iou` | `0.4` | IoU threshold for merging overlapping boxes (within a scale and across scales) |
| `min_face` | `0` | Minimum face side in source pixels; smaller detections are dropped. 0 = no limit |

### scales: multi-scale detection

SCRFD only detects faces within a certain size range. At a single scale it either misses small, distant faces or huge faces right in front of the lens, so each frame is resized to several sizes, detected at each, and the results are merged.

```yaml
scales:
  - {size: 1280, border_pad: 0.0}    # small, distant faces
  - {size: 640,  border_pad: 0.15}   # medium / large faces
  - {size: 320,  border_pad: 0.15}   # huge faces right in front of the lens
```

- `size`: long side of the detection image in pixels; `0` means native resolution. **Never upscales**: a scale larger than the source is treated as native resolution, and scales that end up the same size are de-duplicated (the first one wins). For example, a 640×480 video with the config above is detected at 640×480 and 320×240 only.
- `border_pad`: black padding added on every side before detection, as a ratio of the detection image size. Helps detect faces cut off by the frame edge, at the cost of more computation.
- Every scale is one more inference pass, but small scales are cheap: 320 costs about 1/16 of the computation of 1280.

## tracking: temporal post-processing

Per-frame detection has misses (a face not detected in a few frames) and false positives (objects detected as faces). This step links detections in neighbouring frames into tracks and then decides which frames to mask.

| Parameter | Default | Description |
|---|---|---|
| `score_high` | `0.5` | Track confirmation threshold: a track is a face only if at least one of its frames scores ≥ this value; then the whole track is masked, including its low-score frames. **Lower: fewer misses, more false positives; higher: the opposite** |
| `iou_match` | `0.2` | IoU needed to link a box to a track in the next frame |
| `match_expand` | `0.5` | Boxes are enlarged by this ratio before computing IoU, to tolerate fast motion |
| `max_gap` | `15` | Largest gap (in frames) allowed inside a track; missing frames are filled by linear interpolation (~0.5 s at 30 fps) |
| `extend` | `5` | Extra frames masked before the start and after the end of every track, so faces do not show while entering or leaving the frame |
| `expand` | `0.25` | The mosaic box is the face box enlarged by this ratio (width and height each +25%), covering hair and ears and absorbing box jitter |
| `min_track_len` | `1` | Tracks with fewer detected frames than this are dropped |

## mosaic

| Parameter | Default | Description |
|---|---|---|
| `blocks` | `10` | Number of cells across the mosaic box width. Scales with face size, so large and small faces are equally obscured |
| `min_block` | `4` | Minimum cell size in pixels; takes over for very small faces |

## encode: output encoding

| Parameter | Default | Description |
|---|---|---|
| `codec` | `auto` | `auto` keeps the source codec (h264 / hevc / av1); or force `h264` / `hevc` / `av1` |
| `fallback_codec` | `h264` | Used when the source codec is none of the three (e.g. mjpeg) |
| `hw` | `true` | Prefer NVENC; falls back to software encoders (libx264 / libx265 / libsvtav1) when NVENC cannot be opened |
| `quality` | `19` | NVENC cq or software crf. Lower means better quality and larger files. `0` is lossless with libx264 |
| `preset` | `null` | Encoder preset. When null: NVENC `p5`, x264 / x265 `medium`, SVT-AV1 `8` |
| `gop` | `null` | Keyframe interval; null = encoder default. Some data loaders require a fixed value (LeRobot typically uses 2) |
| `max_b_frames` | `null` | Maximum number of B-frames; null = encoder default |
| `extra_options` | `{}` | Raw encoder options passed straight through, applied last. Useful to reproduce how the originals were encoded, e.g. `{x264-params: "me=hex:b-adapt=1"}` |
| `copy_if_no_face` | `true` | Copy the source file unchanged when no frame needs masking |
| `keep_audio` | `true` | Copy the first audio track as-is (no re-encoding) |

The pixel format follows the source wherever possible: 10-bit sources stay 10-bit. When the encoder does not accept the source format, the closest equivalent is used (e.g. NVENC takes `p010le` instead of `yuv420p10le`; the two are equivalent). Color metadata (range / primaries / transfer / matrix) is copied to the output.

## decode

| Parameter | Default | Description |
|---|---|---|
| `hwaccel` | `cuda` | NVDEC hardware decoding; `null` = software. Codecs the hardware cannot decode fall back to software automatically |
| `scale_threads` | `4` | Threads used to downscale frames for detection |

## preview

| Parameter | Default | Description |
|---|---|---|
| `width` | `960` | Width of the `--preview` videos (height keeps the aspect ratio) |

---

## profiles: per-camera overrides

```yaml
profiles:
  wrist:
    match: ["*wrist*"]
    detector:
      scales:
        - {size: 1280, border_pad: 0.1}
        - {size: 640,  border_pad: 0.25}
        - {size: 320,  border_pad: 0.25}
```

- `match`: list of wildcard patterns matched against the video's **path relative to the input directory** (always `/`-separated, case-insensitive). `*` matches any characters including `/`, so a camera name in a directory name or in a file name both match.
- Every other key has the same structure as the top-level config; only write what you want to override.
- If several profiles match, they are applied in the order written.
- **List values (such as `scales`) replace the default list entirely**; they are not merged item by item.
- The profiles that matched are shown in the `profiles` column of `report.csv` and in the run log.

### Example: custom camera names

For a dataset whose cameras are called `cam_left_wrist`, `cam_right_wrist`, `cam_high` and `cam_low`:

```yaml
profiles:
  wrist:
    match: ["*wrist*"]
    detector:
      scales:
        - {size: 1280, border_pad: 0.1}
        - {size: 640,  border_pad: 0.25}
        - {size: 320,  border_pad: 0.25}
  overview:
    match: ["*cam_high*", "*cam_low*"]
    detector:
      scales:
        - {size: 0,   border_pad: 0.0}
        - {size: 640, border_pad: 0.15}
        - {size: 320, border_pad: 0.15}
```

### Example: LeRobot datasets

```yaml
encode:
  codec: auto        # LeRobot uses AV1 by default; auto writes AV1 (av1_nvenc needs an RTX 40 series GPU or newer, otherwise libsvtav1)
  gop: 2             # match LeRobot's encoding settings so random frame access stays fast
```

Use the `videos/` directory as the input and a new directory as the output, then replace the original `videos/` directory with the output. Since frame counts and timestamps do not change, `meta/` should not need any changes.

> This workflow has not been verified on a real LeRobot dataset yet. Before processing a whole dataset, mask a few episodes and load them with LeRobot to confirm they read correctly.
