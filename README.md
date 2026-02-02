# Drone YOLO Detection Pipeline

A computer vision pipeline for detecting people in drone footage with optional weapon detection using YOLOv11 and YOLOv8 models.

## 🎯 Features

- **Person Detection**: YOLOv11 for high-performance person detection
- **Distance Estimation**: Estimates distance from camera using drone intrinsic parameters (EVO 2 Dual V2)
- **Weapon Detection**: Optional YOLOv8-based weapon detection on person crops
- **Majority Voting**: Sample-level classification using configurable frame threshold to reduce false positives
- **Comprehensive Metrics**: Per-frame, per-sample, per-class, per-distance, and per-height statistics with RMSE tracking
- **Batch Processing**: Process single images, directories, or entire sample collections
- **Organized Output**: Structured results with separate folders for detections, crops, and weapon analysis
- **Docker Support**: Containerized deployment option

## 📁 Project Structure

```
drone_yolo_detection/
├── src/                     # 📁 Source code
│   ├── __init__.py         
│   ├── config.py           # ⚙️ Configuration settings
│   ├── detector.py         # 🧠 Main detection logic
│   ├── estimation.py       # 📏 Distance estimation logic
│   ├── weapon_detector.py  # 🔫 Weapon detection logic
│   └── main.py             # 🚀 CLI interface
├── models/
│   ├── people/
│   │   └── yolo11n.pt      # 🤖 YOLOv11 model for person detection
│   └── weapons/
│       └── yolov8guns.pt   # 🔫 YOLOv8 fine-tuned model for weapon detection
├── inputs/
│   ├── raw/                # Original video files (.mp4)
│   ├── clips/              # Processed video clips
│   └── samples/            # Frame samples extracted from clips
├── output/                 # 💾 Results will be saved here
│   ├── detections/         # Images with bounding boxes
│   ├── crops/              # Individual person crops
│   └── weapon_detections/  # Person crops with weapon analysis
├── Dockerfile              # 🐳 Docker configuration
├── requirements.txt        # 📦 Python dependencies
├── preprocess_videos.py    # 📹 Video preprocessing
├── analyze_metrics.py      # 📊 Metrics analysis and evaluation
└── README.md               # 📖 This documentation
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python src/main.py

# With Docker
docker build -t people-detector .
docker run -v $(pwd):/workspace people-detector
```

## 📊 Usage Examples

```bash
# Process all sample directories
python src/main.py --input inputs/samples --output output/batch_results

# Custom confidence thresholds
python src/main.py --person-confidence 0.6 --weapon-confidence 0.3

# Adjust majority voting (require 3+ frames with weapons to classify sample)
python src/main.py --sample-majority-threshold 3

# Disable weapon detection
python src/main.py --no-weapons

# Full configuration
python src/main.py --model models/people/yolo11n.pt \
                   --input inputs/samples \
                   --output output/detections \
                   --person-confidence 0.6 \
                   --weapon-confidence 0.25 \
                   --sample-majority-threshold 2 \
                   --save-crops
```

## 🔧 Configuration

### Command Line Arguments
- `--model`: Path to YOLO person detection model (default: `models/people/yolo11n.pt`)
- `--input`: Input directory containing sample folders (default: `inputs/samples`)
- `--input_with_weapons`: Input directory with weapon samples (optional)
- `--input_without_weapons`: Input directory without weapon samples (optional)
- `--output`: Output directory (default: `output/detections`)
- `--person-confidence`: Person detection threshold, 0.0-1.0 (default: 0.7)
- `--weapon-confidence`: Weapon detection threshold, 0.0-1.0 (default: 0.2)
- `--sample-majority-threshold`: Frames with weapons needed to classify sample (default: 1)
- `--save-crops`: Save person crops (default: enabled)
- `--no-crops`: Disable saving person crops
- `--no-weapons`: Disable weapon detection
- `--log-level`: Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--log-dir`: Directory to write log files (default: `<output>/logs`)

### Configuration File
Edit `src/config.py` to customize defaults:
- Model path
- Confidence threshold
- Bounding box colors and thickness
- Supported image formats
- Crop padding and minimum size
- Output folder names

## 📁 Output Structure

```
output/
├── detections/                    # Images with person bounding boxes
│   ├── image1_detected.jpg
│   └── image2_detected.jpg
├── crops/                         # Individual person crops
│   ├── image1_person_01_conf_0.85.jpg
│   └── image1_person_02_conf_0.92.jpg
└── weapon_detections/             # Person crops with weapon analysis
    ├── image1_person_01_conf_0.85_weapon_check.jpg
    └── image1_person_02_conf_0.92_weapon_check.jpg
```

## 📹 Video Preprocessing

Process raw videos into clips and frame samples using `preprocess_videos.py`:

```bash
# Basic usage
python preprocess_videos.py

# Custom parameters
python preprocess_videos.py -X 15 -Z 1080p -W 60 -C compressed -F 15 -B 1M
```

### Parameters
- `-X, --clip-duration`: Clip duration in seconds (default: 10)
- `-Z, --resolution`: Target resolution - 1080p, 720p, 480p, 360p, 240p (default: 720p)
- `-W, --frame-interval`: Extract 1 frame every W frames (default: 30)
- `-C, --compression`: Quality preset - high_quality, balanced, compressed, very_compressed (default: balanced)
- `-F, --fps`: Target FPS (optional)
- `-B, --max-bitrate`: Maximum bitrate limit, e.g., '2M', '1000k' (optional)

## 📈 Metrics Analysis

Analyze detection performance and generate comprehensive metrics using `analyze_metrics.py`:

```bash
# Basic usage (Unique frames mode with full validation)
python analyze_metrics.py

# Individual crops mode
python analyze_metrics.py B

# Custom validation modes
python analyze_metrics.py A errors  # Show only errors
python analyze_metrics.py A silent  # No validation output
python analyze_metrics.py B full    # Full validation report
```

### Parameters
- **Option** (positional): Analysis mode
  - `A`: Unique frames mode (default) - counts each frame once
  - `B`: Individual crops mode - counts each person detection separately
- **Validation** (positional): Validation output level
  - `full`: Complete validation report (default)
  - `errors`: Show only validation errors
  - `silent`: Run validation without output
  - `none`: Skip validation

### Output
- **Console**: Detailed metrics breakdown by confidence and fraction thresholds
- **Excel File**: `metricas_analise_[A/B]_[timestamp].xlsx` with two sheets:
  - Análise por Frame: Frame-level or crop-level metrics
  - Análise por Sample: Sample-level metrics with majority voting

### Metrics Calculated
- **True Positives (TP)**: Correctly identified weapon detections
- **True Negatives (TN)**: Correctly identified non-weapon cases
- **False Positives (FP)**: Incorrectly identified as having weapons
- **False Negatives (FN)**: Missed weapon detections
- **Accuracy**: Overall correctness (TP + TN) / Total
- **Precision**: Reliability of positive predictions TP / (TP + FP)
- **Recall**: Sensitivity to actual positives TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall

> **⚠️ Note**: This script contains hard-coded values and paths to accommodate manual labeling of results (e.g., `MANUAL_FALSE_POSITIVES`, `PERSON_DETECTIONS_PER_SAMPLE`, `CROPS_DIR`, `WEAPON_DETECTIONS_DIR`). These must be adjusted according to each trial and your specific directory structure.

## 🏗️ Pipeline Components

### 1. Person Detection (YOLOv11)
- **Model**: YOLOv11n (nano) for fast inference
- **Target**: COCO person class (ID: 0)
- **Output**: Bounding boxes with confidence scores

### 2. Distance Estimation
Uses drone camera intrinsic parameters to estimate distance:
- **Camera**: EVO 2 Dual V2
- **Sensor**: 6.4mm x 4.8mm
- **Focal Length**: 25.6mm (35mm equivalent)
- **Resolution**: 1920x1080 pixels
- **Method**: Calculates distance from person pixel height and camera focal length
- **Assumption**: Average person height of 1.7m
- **Formula**: `distance = (real_height × focal_length) / (pixel_height × pixel_size)`
- **Accuracy**: Typically within 20-30% of actual distance
- **Output**: Distance logs saved to `person_distances.log`

### 3. Crop Extraction
- **Padding**: 10% padding around person bounding boxes
- **Minimum Size**: 32x32 pixels
- **Format**: JPEG files with metadata in filename

### 4. Weapon Detection (YOLOv8)
- **Model**: YOLOv8 custom-trained for weapon detection
- **Input**: Person crops from step 3
- **Majority Voting**: Sample classified as having weapons if ≥ threshold frames contain weapons
- **Threshold**: Configurable via `--majority-threshold` (default: 1)
- **Output**: Annotated crops with weapon bounding boxes

## 📊 Metrics Tracking

The pipeline tracks comprehensive metrics:

- **Frame-Level**: Per-frame confusion matrix (TP, TN, FP, FN)
- **Sample-Level**: Per-sample confusion matrix using majority voting
- **Per-Class**: Metrics segmented by sample class (e.g., 'real', 'falso')
- **Per-Distance**: Metrics segmented by real distance
- **Per-Height**: Metrics segmented by camera height
- **Distance RMSE**: Root Mean Squared Error for distance estimation

Ground truth is determined from filenames: 'real*' = has weapons, 'falso*' = no weapons.

## 🐳 Docker

```bash
# Build
docker build -t people-detector .

# Run with default settings
docker run -v $(pwd):/workspace people-detector

# Run with custom parameters
docker run -v $(pwd):/workspace people-detector \
    python src/main.py --confidence 0.7 --output custom_results/
```
