# models/

Model files are not committed. From the repository root, run:

```bash
facemosaic download-models          # SCRFD (required)
facemosaic download-models --all    # also YOLOX-s (person check) and YuNet (only used by `facemosaic compare`)
```

| File | Purpose | Source | License |
|---|---|---|---|
| `det_10g.onnx` | SCRFD-10G face detector (default) | InsightFace `buffalo_l` | Pretrained weights are **for non-commercial research use only** |
| `det_10g_fp16.onnx` | FP16 version of the model above | Generated automatically on the first GPU run | Same as above |
| `yolox_s.onnx` | YOLOX-s person detector (optional, for `tracking.person_check`) | Megvii YOLOX | Apache-2.0 |
| `face_detection_yunet_2023mar.onnx` | YuNet face detector (optional, for comparison) | OpenCV Zoo | MIT |

Downloads are verified against a SHA256 checksum. If Hugging Face is not reachable from your network, point the
download at a mirror first:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"   # Windows PowerShell
```
```bash
export HF_ENDPOINT=https://hf-mirror.com      # Linux
```

You can also download the files manually and put them in this directory with the same file names.
