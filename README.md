# Dual-Drone YOLO Detection Pipeline

Detect people (and optionally weapons) from drone footage. In **dual-drone mode**, the pipeline associates detections across two synchronized drones and triangulates a fused position.

## ✅ Pipeline (step by step)

### 0) Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### (Optional) Summarize logs into LaTeX tables

If you have console logs under `logs/`, you can export the detection + RMSE tables (LaTeX) into a `.txt` file:

```bash
python summarize_logs.py --log-dir logs --pattern run_*_console.log --format latex --out output/log_tables.txt
```

To export a single run log:

```bash
python summarize_logs.py --log-dir logs --pattern run_20260205_125250_console.log --format latex --out output/log_tables_20260205_125250.txt
```

Models in this repo:
- Person model: `models/people/yolo11n.pt`
- Weapon model: `models/weapons/best.pt`

### 1) Organize raw data

Put your raw videos under `inputs/raw/`. Any nested folder structure is OK and will be preserved by preprocessing.

Example layout:

```text
inputs/raw/
   drone1/45/<video>.mp4
   drone2/45/<video>.mp4
   drone1/90/<video>.mp4
   drone2/90/<video>.mp4
```

### 2) (Optional) Extract telemetry subtitles (SRT)

If your drone embeds telemetry as subtitle streams, you can extract `.srt` files and generate a small `.txt` (first GPS + gimbal angles):

```bash
python extract_srt.py
```

What it does:
- Walks `inputs/` and tries to extract the **first subtitle stream** from each MP4 (`ffmpeg -map 0:s:0`) into a sibling `.srt`.
- Walks `inputs/raw/` for `.srt` and writes a `.txt` next to each one with:
   - `pitch`, `roll`, `yaw` (from `G.PRY`)
   - `lat`, `lon` (from `GPS(...)`)

Limitations:
- Assumes telemetry is in subtitle stream `0:s:0`.
- Parses only the **first** matching GPS and `G.PRY` values per `.srt`.

### 3) Preprocess raw videos → clips → sampled frames

The detector consumes **frame sample folders** under `inputs/samples/`.

Create clips from raw videos and extract 1 JPEG every `W` frames:

```bash
python preprocess_videos.py \
   --raw inputs/raw \
   --clips inputs/clips \
   --samples inputs/samples \
   -X 10 -Z 1080p -W 10 \
   -C balanced -F 10
```

If you already have clips under `inputs/clips/`, extract frames only:

```bash
python preprocess_videos.py \
   --clips-only \
   --clips inputs/clips \
   --samples inputs/samples \
   -W 10
```

Important flags:
- `-X, --clip-duration`: clip length in seconds
- `-Z, --resolution`: `1080p|720p|480p|360p|240p` (keeps aspect ratio)
- `-W, --frame-interval`: sample every W frames
- `-C, --compression`: `none|high_quality|balanced|compressed|very_compressed`
- `-F, --fps`: cap FPS (reduces file size when lower than original)
- `-B, --max-bitrate`: bitrate cap like `2M` or `1000k`

Notes:
- Clip creation uses `ffmpeg` (and falls back to OpenCV if `ffmpeg` fails).
- The script preserves your directory structure (e.g. `inputs/raw/drone1/45/...` → `inputs/samples/drone1/45/...`).

### 4) Run detection

#### Single-drone

```bash
python src/main.py \
   --input inputs/samples/drone1 \
   --output output/single_drone
```

Run only a single angle (when your input is organized like `inputs/samples/drone1/45/...` and `inputs/samples/drone1/90/...`):

```bash
python src/main.py \
   --input inputs/samples/drone1 \
   --angle 90 \
   --output output/single_drone
```

Common options:
- `--person-confidence 0.6`
- `--weapon-confidence 0.3`
- `--no-weapons`
- `--sample-majority-threshold 3`

#### Dual-drone (fusion + triangulation)

```bash
python src/main.py \
   --dual-drone \
   --input-drone1 inputs/samples/drone1 \
   --input-drone2 inputs/samples/drone2 \
   --output output/dual_drone \
   --association-threshold 100.0
```

Run only a single angle (filters to the matching common angle):

```bash
python src/main.py \
   --dual-drone \
   --input-drone1 inputs/samples/drone1 \
   --input-drone2 inputs/samples/drone2 \
   --angle 90 \
   --output output/dual_drone \
   --association-threshold 100.0
```

### 5) Inspect outputs

Results are written under `output/`. Expect per-angle subfolders like:

```text
output/
   angle_45/
      drone1_detections/
      drone2_detections/
      fused_detections/
   angle_90/
      ...
```

Ground truth is inferred from filenames:
- `real*` = has weapons
- `falso*` = no weapons

## 🐳 Docker (optional)

```bash
docker build -t drone-detector .
docker run -v $(pwd):/workspace drone-detector
```

## 📁 Minimal project map

```text
src/main.py           # CLI entrypoint (single + dual drone)
preprocess_videos.py  # raw/clips → samples (frames)
extract_srt.py        # extract .srt + write first telemetry values to .txt
inputs/raw/           # raw .mp4
inputs/clips/         # generated clips
inputs/samples/       # generated frame samples (what the pipeline reads)
models/               # YOLO weights
output/               # results
```
