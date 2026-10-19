# video-face-mosaic

Batch face mosaicking for multi-camera robot-arm videos (wrist / head / third-person cameras).

Every frame of every video is run through a face detector and the faces are pixelated. The output keeps the source's frame count, per-frame timestamps and resolution (checked automatically after each video), so masked videos can replace the originals in a dataset without breaking the alignment between cameras or between video and action data.

## Features

- **Every frame, multiple scales**: the SCRFD face detector (GPU) runs at several scales per frame, so it finds both small, distant faces and huge faces right in front of the lens.
- **No flickering faces**: detections are linked into tracks across frames. Frames where a face is briefly missed are filled in by interpolation, and a few extra frames are masked at the start and end of each track.
- **Few false positives**: a track only counts as a face if at least one of its frames scores high. This keeps objects being grasped from being masked by mistake, which would corrupt training data.
- **Data stays intact**: only face pixels are changed. The source codec (H.264 / HEVC / AV1), 10-bit depth, color metadata and audio track are kept. Videos with no faces are copied unchanged.
- **Built for batches**: the input directory layout is mirrored; interrupted runs resume where they stopped; one failed video does not stop the batch; a CSV report is written; optional preview videos with detection boxes for spot checks.
- **Per-camera settings**: wrist / head / third-person cameras are recognized from the file path and can use different parameters.
- **Hardware acceleration**: NVDEC decoding, NVENC encoding and FP16 inference. Runs on CPU too, much more slowly.

## Quick start

```powershell
git clone https://github.com/lalayang123456-ctrl/video-face-mosaic.git
cd video-face-mosaic
python -m venv .venv
.venv\Scripts\activate                 # Linux: source .venv/bin/activate
pip install -e ".[gpu]"                # no NVIDIA GPU: pip install -e ".[cpu]"
facemosaic download-models             # download the face detection model
facemosaic check                       # the last line should read "Environment check passed"
facemosaic run D:\data\episodes -o D:\data\episodes_masked --preview
```

When it finishes:

- The masked videos are in `D:\data\episodes_masked\`, with the same layout as the input.
- The report is `D:\data\episodes_masked\_work\report.csv`.
- Preview videos for spot checks are at `D:\data\episodes_masked\_work\**\*.preview.mp4`.

## Installation

### Requirements

| | |
|---|---|
| OS | Windows 10/11 or Linux |
| Python | 3.10 or newer (tested on 3.11) |
| GPU (recommended) | An NVIDIA GPU with a recent driver. You do **not** need to install the CUDA Toolkit: the CUDA / cuDNN runtime is installed by pip. |
| Without a GPU | Works, but detection is about 20x slower; only practical for a small number of videos. |

You do not need to install ffmpeg; PyAV ships with its own.

### Steps

1. Create and activate a virtual environment (recommended: `.venv` inside the repository).

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate          # Windows PowerShell
   ```
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # Linux
   ```

2. Install. `-e` is an editable install, so the tool reads `configs/` and `models/` from the repository.

   ```bash
   pip install -e ".[gpu]"         # NVIDIA GPU
   pip install -e ".[cpu]"         # no NVIDIA GPU
   ```

   With the CUDA / cuDNN runtime, the GPU install takes about 2 GB.

3. Download the model (about 17 MB, verified against a SHA256 checksum).

   ```bash
   facemosaic download-models
   ```

   If Hugging Face is not reachable from your network, set a mirror first: `$env:HF_ENDPOINT="https://hf-mirror.com"` (PowerShell) or `export HF_ENDPOINT=https://hf-mirror.com` (Linux). You can also download the file manually; see [models/README.md](models/README.md).

4. Check the environment.

   ```bash
   facemosaic check
   ```

   Each line starts with a status: `[OK]` works; `[--]` an optional feature is unavailable and a slower fallback will be used; `[!!]` must be fixed.

   For the CPU install, also set `detector.device` to `cpu` in the config file; see [Configuration](#configuration).

## Usage

### 1. Batch processing: `facemosaic run`

```bash
facemosaic run <input dir or video> -o <output dir> [-c config.yaml] [--preview] [--force]
```

| Option | Description |
|---|---|
| `<input>` | Input directory (searched recursively for `.mp4 .mov .mkv .avi .webm .m4v`) or a single video file |
| `-o, --output` | Output directory. Mirrors the input layout; file names are unchanged |
| `-c, --config` | Config file. Default: `configs/default.yaml` |
| `-w, --work` | Directory for intermediate results and the report. Default: `<output>/_work` |
| `--preview` | Also write a low-resolution preview with detection boxes, for spot checks |
| `--force` | Ignore previous results and reprocess everything |

Example input and output:

```
D:\data\episodes\                      D:\data\episodes_masked\
├── episode_0001\                      ├── episode_0001\
│   ├── wrist.mp4          ──────►     │   ├── wrist.mp4          (masked)
│   ├── head.mp4                       │   ├── head.mp4
│   └── third_person.mp4               │   └── third_person.mp4
└── episode_0002\                      ├── episode_0002\ ...
    └── ...                            └── _work\                 (report and intermediate results)
                                           ├── report.csv
                                           └── episode_0001\
                                               ├── wrist.mp4.result.json     per-video details
                                               ├── wrist.mp4.dets.json.gz    raw detections (cache)
                                               ├── wrist.mp4.boxes.json.gz   final mosaic boxes per frame
                                               └── wrist.mp4.preview.mp4     preview (with --preview)
```

LeRobot datasets (`videos/chunk-000/observation.images.wrist/episode_000000.mp4`) work as-is: use the `videos` directory as the input. Camera names in directory names are matched too.

### 2. Checking the results

**`_work/report.csv`** has one row per video (opens in Excel):

| Column | Meaning |
|---|---|
| `status` | `ok` succeeded; `verify_failed` the output's frame count or timestamps differ from the source (should never happen; please report it); `failed` an error occurred, see `error` |
| `action` | `rendered` masked and re-encoded; `copied` no faces, the source file was copied unchanged |
| `profiles` | Camera profiles that matched (wrist / head / third) |
| `frames` / `mosaic_frames` | Total frames / frames with at least one mosaic |
| `raw_det_frames` | Frames where the detector fired at all (including low-score detections later rejected as false positives) |
| `det_fps` | Detection speed in frames per second |
| `encoder` | Encoder used, e.g. `hevc_nvenc` |
| `warnings` | E.g. metadata streams or cover art in the source that were not written to the output |

**Preview videos** (`--preview`):

- Red box + mosaic: final masked area
- Green box: high-score detection (≥ `score_high`)
- Orange box: low-score detection. Only masked when it belongs to the same track as a high-score detection.
- Top-left: frame number / raw detections in this frame / masked boxes in this frame

Things worth spot-checking: close-up faces, half faces at the frame edge, small distant faces, and videos whose `mosaic_frames` is unusually high (possible false positives) or low (possible misses).

### 3. Interrupting, resuming, re-running

- **Resume after an interruption**: run the same command again. Finished videos are reported as "skipped (already done)"; failed ones are retried.
- **Re-run after changing parameters**: without `--force`, videos whose effective settings changed are reprocessed. If you only changed `tracking`, `mosaic` or `encode` settings, the cached detections are reused, which is fast.
- **`--force`**: reprocess everything from scratch, including detection.
- Output is written to a `*.partial.*` file first and renamed only on success, so you never get a half-written video.

### 4. Tuning and comparing: `facemosaic compare`

Try different detectors or detection scales on one video. Prints statistics and writes previews; does not write masked videos.

```bash
facemosaic compare some.mp4                                   # default config
facemosaic compare some.mp4 --scales 1280 640:0.25 320:0.25   # other scales and padding (size:padding)
facemosaic compare some.mp4 --detectors scrfd yunet           # compare with YuNet (needs download-models --all)
facemosaic compare some.mp4 --max-frames 300 -o my_compare    # first 300 frames only, previews in my_compare/
```

### 5. Environment check: `facemosaic check`

Checks dependency versions, the model file, GPU inference, NVENC encoding and NVDEC decoding. Run it after moving to a new machine or updating the GPU driver.

## Configuration

The default config is [configs/default.yaml](configs/default.yaml); every option is commented there. The full reference is [docs/configuration.md](docs/configuration.md).

To customize, copy `default.yaml`, edit it and pass it with `-c`:

```bash
facemosaic run D:\data\episodes -o D:\data\out -c configs\my_dataset.yaml
```

### Per-camera settings (profiles)

`profiles` are matched against each video's **path relative to the input directory** (case-insensitive, `*` is a wildcard). A matching profile overrides the defaults. The three default profiles:

| Profile | Matches | Adjustment |
|---|---|---|
| `wrist` | `*wrist*` | Close-up faces, often cut off by the frame edge: more padding at detection time |
| `head` | `*head*` | Default settings |
| `third` | `*third*` | Many small, distant faces: the largest detection scale uses the native resolution |

**If your cameras are named differently** (e.g. `cam_left`, `top_camera`), edit `match`; otherwise those videos use the default settings. The `profiles` column of `report.csv` shows what matched.

### Common parameters

| Parameter | Default | When to change it |
|---|---|---|
| `tracking.score_high` | 0.5 | Lower it (e.g. 0.4) if faces are missed; raise it (e.g. 0.6) if objects get masked |
| `tracking.expand` | 0.25 | Raise it if the mosaic does not cover hair or ears |
| `tracking.max_gap` / `extend` | 15 / 5 | Raise them if a face flickers into view for single frames |
| `tracking.person_check.enabled` | false | Turn on if objects in view (jars, packaging, lids, shadows) get masked; drops face tracks that do not belong to a person. Needs `facemosaic download-models --all` |
| `mosaic.blocks` | 10 | Lower it if masked faces are still recognizable |
| `encode.quality` | 19 | Lower it for better output quality (larger files) |
| `encode.gop` | encoder default | Set it when your data loader needs a specific keyframe interval (LeRobot typically uses 2) |
| `detector.device` | cuda | Set to `cpu` on machines without an NVIDIA GPU |

## Performance

Measured on an RTX 4060 Laptop GPU + Intel i9-13900H with the default config.

| Video | Detection speed | Total time for a 30 s video |
|---|---|---|
| 640×480 H.264 (wrist / head) | ~100–110 frames/s | ~10 s |
| 1280×720 H.264 (third-person, native-resolution scale) | ~48 frames/s | ~20 s |
| 2688×1512 HEVC 10-bit | ~45 frames/s | ~25 s |
| 640×480, CPU only (`device: cpu`) | ~5 frames/s | ~3 min |

Detection dominates the run time; mosaic plus re-encoding is fast (about 190 frames/s with NVENC on 2.7K video). `--preview` decodes the video once more and encodes a preview, which adds some time.

## Known limitations

- **Faces whose upper half is outside the frame** (only the mouth and chin visible) are generally not detected; this is a limitation of face detectors in general. They are covered when neighbouring frames detect the face.
- **Faces seen from behind or heavily occluded** are not detected either. A face seen from behind usually does not identify a person.
- **Non audio/video streams are not kept**: metadata streams, timecode tracks and cover art in the source are not written to re-encoded outputs; the `warnings` column says so. Copied videos are unaffected.
- **Only the first video stream and the first audio stream of each file are processed.**
- Automatic masking cannot guarantee that no face is ever missed. Spot-check the previews before publishing data.

## FAQ

**Objects get masked (jars, food packaging, cup lids, shadows)**
The face detector sometimes fires on face-like objects, and in robot videos those objects can stay in view for a long time. Run `facemosaic download-models --all` and set `tracking.person_check.enabled: true`: face tracks that do not belong to a detected person are dropped. See [docs/configuration.md](docs/configuration.md#trackingperson_check-rejecting-face-like-objects).

**PowerShell refuses to activate the virtual environment ("running scripts is disabled on this system")**
Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again. Or skip activation and use the full path: `.venv\Scripts\facemosaic.exe run ...`.

**`facemosaic check` reports that GPU inference is unavailable**
Make sure you installed `.[gpu]`, not `.[cpu]`, and that the GPU driver is recent (`nvidia-smi` should work). Do not install `onnxruntime` and `onnxruntime-gpu` in the same environment; if both are installed, uninstall both and run `pip install -e ".[gpu]"` again.

**NVENC / NVDEC is reported as unavailable**
Software encoding / decoding is used automatically. The results are the same, just slower.

**The masked video does not play**
If the source is HEVC 10-bit, so is the output. The built-in Windows player may not support it; use VLC or PotPlayer.

**Model file not found**
Run `facemosaic download-models` in the repository root. Relative `models/...` paths in the config are looked up in the current directory first, then in the repository root.

**How do I mask a camera's videos completely, or skip a camera?**
Exclude that directory from the `run` input. There is no "mask the whole frame" option at the moment.

## Repository layout

```
video-face-mosaic/
├── README.md                 this file
├── pyproject.toml            package metadata, dependencies, `facemosaic` command
├── configs/
│   └── default.yaml          default config (incl. wrist / head / third profiles)
├── models/                   model directory (models are downloaded, not committed)
│   └── README.md
├── docs/
│   ├── configuration.md      every config parameter
│   └── how-it-works.md       pipeline and key design decisions (for maintainers)
├── src/facemosaic/
│   ├── cli.py                command line: run / compare / download-models / check
│   ├── pipeline.py           per-video pipeline, batch runs, report
│   ├── detectors.py          SCRFD / YuNet detectors, multi-scale detection
│   ├── tracking.py           track linking, gap filling, extension, box enlargement
│   ├── mosaic.py             in-place mosaic on YUV planes
│   ├── video.py              decoding, encoding, timestamp verification, previews
│   ├── models.py             model download and checksum
│   └── config.py             config loading and profile matching
└── tests/                    unit tests (no GPU needed)
```

## Development and tests

```bash
pip install -e ".[gpu,dev]"
pytest
```

The tests need neither a GPU nor the model files. They cover gap filling, mosaicking for many pixel formats, profile matching, and preservation of frame count and timestamps. The pipeline and key design decisions are described in [docs/how-it-works.md](docs/how-it-works.md).

## Models and licenses

| Model | Purpose | License |
|---|---|---|
| SCRFD-10G (`det_10g.onnx`, InsightFace) | Default detector | Pretrained weights are **for non-commercial research use only**; check licensing before commercial use |
| YOLOX-s (Megvii) | Optional person detector for `tracking.person_check` | Apache-2.0 |
| YuNet (OpenCV Zoo) | Optional, for comparison | MIT |

Both models are trained on the WIDER FACE dataset, which has its own terms of use. Check compliance before using this commercially or publishing data.
