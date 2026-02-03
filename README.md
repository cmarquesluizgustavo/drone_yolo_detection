# Dual-Drone YOLO Detection Pipeline

An advanced computer vision pipeline for detecting and localizing people using synchronized dual-drone footage with geometric triangulation, position estimation, and optional weapon detection using YOLO models.

## 🎯 Features

- **Dual-Drone Fusion**: Synchronized processing of two drone video streams with cross-drone detection association
- **Geometric Triangulation**: Precise position estimation by combining measurements from two viewpoints
- **Person Detection**: YOLO-based high-performance person detection
- **Position Estimation**: 3D position estimation using camera intrinsics, gimbal angles, and GPS data
- **Geographic Coordinates**: Target position output in latitude/longitude with distance RMSE tracking
- **Weapon Detection**: Optional YOLO-based weapon detection on person crops
- **Majority Voting**: Sample-level classification using configurable frame threshold to reduce false positives
- **Comprehensive Metrics**: Per-frame, per-sample, per-angle, and fused detection statistics with RMSE tracking
- **Ground Plane Projection**: Conversion between geographic coordinates and local Cartesian ground plane
- **Batch Processing**: Process synchronized sample collections from multiple drones
- **Organized Output**: Structured results with separate folders for single-drone and fused detections
- **Docker Support**: Containerized deployment option

## 📁 Project Structure

```
drone_yolo_detection/
├── src/                           # 📁 Source code
│   ├── __init__.py         
│   ├── camera.py                 # 📷 Camera intrinsics and gimbal parameters
│   ├── detection_pipeline.py    # 🔍 Single-drone detection pipeline
│   ├── dual_drone_pipeline.py   # 🔗 Dual-drone synchronized processing
│   ├── detection_fusion.py      # 🎯 Cross-drone detection association
│   ├── position_estimation.py   # � Distance, bearing, and position estimation
│   ├── geoconverter.py          # 🌍 Geographic-to-Cartesian coordinate conversion
│   ├── detection_statistics.py  # 📊 Metrics tracking and reporting
│   ├── people_detector.py       # 👤 Person detection with YOLO
│   ├── weapon_detector.py       # 🔫 Weapon detection logic
│   └── main.py                  # 🚀 CLI interface
├── models/
│   ├── people/
│   │   ├── yolo11n.pt           # 🤖 YOLO11 nano model for person detection
│   │   
│   └── weapons/
│       └── best.pt              # 🔫 Fine-tuned model for weapon detection
├── inputs/
│   ├── raw/                     # Original video files (.mp4)
│   │   ├── drone1/              # Drone 1 raw footage
│   │   │   ├── 45/              # 45-degree angle recordings
│   │   │   └── 90/              # 90-degree angle recordings
│   │   └── drone2/              # Drone 2 raw footage
│   │       ├── 45/
│   │       └── 90/
│   ├── clips/                   # Processed video clips
│   │   ├── drone1/
│   │   └── drone2/
│   └── samples/                 # Frame samples extracted from clips
│       ├── drone1/
│       │   ├── 45/
│       │   └── 90/
│       └── drone2/
│           ├── 45/
│           └── 90/
├── output/                      # 💾 Results will be saved here
│   ├── angle_45/                # Results for 45° angle
│   │   ├── drone1_detections/  # Drone 1 individual detections
│   │   ├── drone2_detections/  # Drone 2 individual detections
│   │   └── fused_detections/   # Merged dual-drone results
│   └── angle_90/                # Results for 90° angle
│       ├── drone1_detections/
│       ├── drone2_detections/
│       └── fused_detections/
├── logs/                        # 📝 Execution logs
├── Dockerfile                   # 🐳 Docker configuration
├── requirements.txt             # 📦 Python dependencies
├── preprocess_videos.py         # 📹 Video preprocessing
└── README.md                    # 📖 This documentation
```

## 🚀 Quick Start

### Single-Drone Mode
```bash
# Install dependencies
pip install -r requirements.txt

# Run single-drone pipeline
python src/main.py --input inputs/samples/drone1 --output output/single_drone

# With custom confidence thresholds
python src/main.py --person-confidence 0.6 --weapon-confidence 0.3
```

### Dual-Drone Mode
```bash
# Run dual-drone pipeline with fusion
python src/main.py --dual-drone \
                   --input-drone1 inputs/samples/drone1 \
                   --input-drone2 inputs/samples/drone2 \
                   --output output/dual_drone \
                   --association-threshold 100.0

# With Docker
docker build -t drone-detector .
docker run -v $(pwd):/workspace drone-detector
```

## 📊 Usage Examples

### Single-Drone Detection
```bash
# Process all sample directories from one drone
python src/main.py --input inputs/samples/drone1 --output output/drone1_results

# Custom confidence thresholds
python src/main.py --person-confidence 0.6 --weapon-confidence 0.3

# Adjust majority voting (require 3+ frames with weapons to classify sample)
python src/main.py --sample-majority-threshold 3

# Disable weapon detection
python src/main.py --no-weapons

# Process only specific clips (0, 2, 7)
python src/main.py --filter-clips
```

### Dual-Drone Detection with Fusion
```bash
# Basic dual-drone processing
python src/main.py --dual-drone \
                   --input-drone1 inputs/samples/drone1 \
                   --input-drone2 inputs/samples/drone2 \
                   --output output/fused_results

# Full configuration with association threshold
python src/main.py --dual-drone \
                   --model models/people/yolo11n.pt \
                   --input-drone1 inputs/samples/drone1 \
                   --input-drone2 inputs/samples/drone2 \
                   --output output/dual_detections \
                   --person-confidence 0.5 \
                   --weapon-confidence 0.5 \
                   --sample-majority-threshold 1 \
                   --association-threshold 100.0 \
                   --save-crops

# Custom association threshold (distance in meters for matching detections)
python src/main.py --dual-drone \
                   --association-threshold 50.0  # Stricter matching
```

## 🔧 Configuration

### Command Line Arguments

#### General Options
- `--model`: Path to YOLO person detection model (default: `models/people/yolo11n.pt`)
- `--input`: Input directory for single-drone mode (default: `inputs/samples`)
- `--output`: Output directory (default: `output/detections`)
- `--log-level`: Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--log-dir`: Directory to write log files (default: `logs/`)

#### Detection Thresholds
- `--person-confidence`: Person detection threshold, 0.0-1.0 (default: 0.5)
- `--weapon-confidence`: Weapon detection threshold, 0.0-1.0 (default: 0.5)
- `--sample-majority-threshold`: Frames with weapons needed to classify sample (default: 1)

#### Output Options
- `--save-crops`: Save person crops (enabled by default)
- `--no-crops`: Disable saving person crops
- `--no-weapons`: Disable weapon detection

#### Dual-Drone Mode
- `--dual-drone`: Enable dual-drone processing mode
- `--input-drone1`: Input directory for drone 1 (default: `inputs/samples/drone1`)
- `--input-drone2`: Input directory for drone 2 (default: `inputs/samples/drone2`)
- `--association-threshold`: Distance threshold in meters for associating detections across drones (default: 100.0)

#### Processing Options
- `--filter-clips`: Process only clips 0, 2, and 7 (specific gimbal angles: 0°, 45°, -45°)
- `--input_with_weapons`: Input directory with weapon samples (optional)
- `--input_without_weapons`: Input directory without weapon samples (optional)

## 📁 Output Structure

### Single-Drone Mode
```
output/
├── detections/                    # Images with person bounding boxes
│   ├── sample_name/
│   │   ├── frame_000.jpg
│   │   └── frame_001.jpg
├── crops/                         # Individual person crops
│   ├── sample_name/
│   │   ├── frame_000_person_01_conf_0.85.jpg
│   │   └── frame_000_person_02_conf_0.92.jpg
└── weapon_detections/             # Person crops with weapon analysis
    ├── sample_name/
        ├── frame_000_person_01_weapon_check.jpg
        └── frame_000_person_02_weapon_check.jpg
```

### Dual-Drone Mode
```
output/
├── angle_45/                           # Results for 45° camera angle
│   ├── drone1_detections/              # Drone 1 individual results
│   │   ├── sample_name/
│   │   │   ├── frame_000.jpg
│   │   │   └── frame_000.json         # Detection metadata
│   ├── drone2_detections/              # Drone 2 individual results
│   │   ├── sample_name/
│   └── fused_detections/               # Merged dual-drone results
│       ├── sample_name/
│           ├── frame_000_fused.jpg    # Visualization with both drones
│           └── frame_000_fused.json   # Fused detection data with triangulation
├── angle_90/                           # Results for 90° camera angle
│   ├── drone1_detections/
│   ├── drone2_detections/
│   └── fused_detections/
└── logs/
    └── run_YYYYMMDD_HHMMSS_console.log
```

### JSON Output Format
Each detection includes:
- Bounding box coordinates
- Person confidence score
- Weapon detection status and confidence
- Estimated distance (single drone)
- Bearing angle
- Geographic coordinates (lat/lon)
- Triangulated position (dual-drone fused detections)
- RMSE for position estimation

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

## 🏗️ Pipeline Components

### 1. Person Detection (YOLO)
- **Models**: YOLO11n for fast inference
- **Target**: COCO person class (ID: 0)
- **Output**: Bounding boxes with confidence scores
- **Processing**: Applied independently to each drone's footage

### 2. Camera Model & Position Estimation
Uses drone camera intrinsic parameters and real-time telemetry:
- **Camera**: EVO 2 Dual V2 / Autel EVO II
- **Sensor**: 6.4mm x 4.8mm
- **Focal Length**: 25.6mm (35mm equivalent)
- **Resolution**: 1920x1080 pixels
- **Telemetry Data**: GPS coordinates, altitude, gimbal pitch/yaw angles

#### Distance Estimation Methods
1. **Pixel-Height Method**: `distance = (real_height × focal_length) / (pixel_height × pixel_size)`
   - Assumes average person height of 1.7m
   - Used for initial estimates

2. **Pitch-Angle Method**: `distance = altitude / tan(pitch_angle + pixel_angle)`
   - Uses gimbal pitch angle and pixel position
   - More accurate for known camera height
   - Primary method for position estimation

3. **Bearing Estimation**: Calculates horizontal angle from camera center
   - Uses gimbal yaw and pixel horizontal position
   - Provides target direction relative to true north

### 3. Geographic Coordinate System
- **Ground Plane Projection**: Converts GPS (lat/lon) to local Cartesian coordinates (x, y)
- **Origin**: Dynamically set based on first drone position
- **Coordinate System**: 
  - X-axis: East direction (meters)
  - Y-axis: North direction (meters)
- **Conversion**: Uses `pyproj` for accurate geodetic transformations

### 4. Dual-Drone Fusion
Combines measurements from two drones for improved accuracy:

#### Detection Association
- **Input**: Distance and bearing measurements from both drones
- **Method**: Projects each detection to ground plane coordinates
- **Matching**: Pairs detections within association threshold distance
- **Threshold**: Configurable (default: 100.0m)

#### Geometric Triangulation
- **Method**: Intersects bearing rays from two drone positions
- **Output**: Single fused position estimate (lat/lon)
- **Confidence**: Combined from both detections using probabilistic fusion
- **Validation**: Checks consistency between observations

#### Position Refinement
- Calculates triangulated geographic coordinates
- Computes RMSE against individual estimates
- Provides confidence metrics for final position

### 5. Crop Extraction
- **Padding**: 10% padding around person bounding boxes
- **Minimum Size**: 32x32 pixels
- **Format**: JPEG files with metadata in filename
- **Metadata**: Confidence scores, frame IDs, detection IDs

### 6. Weapon Detection
- **Model**: Custom-trained YOLO model for weapon detection
- **Input**: Person crops from step 5
- **Majority Voting**: Sample classified as having weapons if ≥ threshold frames contain weapons
- **Threshold**: Configurable via `--sample-majority-threshold` (default: 1)
- **Output**: Annotated crops with weapon bounding boxes
- **Confidence Fusion**: Combines weapon detections from both drones

## 📊 Metrics Tracking

The pipeline tracks comprehensive metrics at multiple levels:

### Single-Drone Metrics
- **Frame-Level**: Per-frame confusion matrix (TP, TN, FP, FN)
- **Sample-Level**: Per-sample confusion matrix using majority voting
- **Per-Class**: Metrics segmented by sample class ('real', 'falso')
- **Per-Distance**: Performance by ground truth distance
- **Per-Height**: Performance by camera altitude
- **Distance RMSE**: Root Mean Squared Error for distance estimation

### Dual-Drone Metrics
- **Individual Drone Stats**: Separate metrics for each drone
- **Fused Stats**: Combined metrics after detection fusion
- **Position RMSE**: Accuracy of geometric triangulation
- **Association Rate**: Percentage of detections successfully matched across drones
- **Per-Angle Analysis**: Metrics broken down by camera angle (45°, 90°)
- **Confidence Improvement**: Comparison of single-drone vs fused confidence scores

### Output Reports
Statistics are printed to console and saved to logs:
- Overall accuracy, precision, recall, F1-score
- Confusion matrices (frame-level and sample-level)
- Distance estimation errors
- Per-angle performance comparison
- Fusion quality metrics

Ground truth is determined from filenames: `real*` = has weapons, `falso*` = no weapons.

## 🐳 Docker

```bash
# Build
docker build -t drone-detector .

# Run single-drone mode
docker run -v $(pwd):/workspace drone-detector

# Run dual-drone mode with custom parameters
docker run -v $(pwd):/workspace drone-detector \
    python src/main.py --dual-drone \
                       --input-drone1 inputs/samples/drone1 \
                       --input-drone2 inputs/samples/drone2 \
                       --person-confidence 0.5 \
                       --association-threshold 50.0
```

## 🔬 Technical Details

### Coordinate Systems
1. **Image Coordinates**: Pixel coordinates (0,0) at top-left
2. **Ground Plane Coordinates**: Local Cartesian (x, y) in meters
   - Origin: First drone GPS position
   - X-axis: East
   - Y-axis: North
3. **Geographic Coordinates**: WGS84 latitude/longitude

### Triangulation Algorithm
1. Convert both drone GPS positions to ground plane coordinates
2. For each detection, calculate:
   - Distance estimate (from pitch angle method)
   - Bearing angle (from image position + gimbal yaw)
3. Project detection positions to ground plane
4. Match detections from both drones within association threshold
5. Calculate intersection of bearing rays
6. Convert fused position back to geographic coordinates
7. Compute RMSE between individual and fused estimates

### Error Sources
- **Camera Calibration**: Focal length and sensor size assumptions
- **GPS Accuracy**: ±3-5m typical accuracy
- **Altitude Accuracy**: ±0.5-1m typical
- **Gimbal Angle Accuracy**: ±0.1° typical
- **Person Height Assumption**: Fixed 1.7m (varies in reality)
- **Wind/Vibration**: Affects camera stability
- **Synchronization**: Frame timing between drones

### Performance Considerations
- **Processing Speed**: ~10-30 FPS depending on resolution and model
- **Memory Usage**: ~2-4 GB RAM for YOLO models
- **GPU Acceleration**: Recommended for real-time processing
- **Storage**: ~1-2 MB per processed frame with crops and JSON

## 📦 Dependencies

Core dependencies (see `requirements.txt`):
- `opencv-python-headless>=4.5.0` - Image processing
- `numpy>=1.20.0` - Numerical computations
- `ultralytics>=8.0.0` - YOLO model inference
- `torch>=1.11.0` - Deep learning framework
- `torchvision>=0.12.0` - Vision utilities
- `Pillow>=8.0.0` - Image manipulation
- `pandas>=1.3.0` - Data analysis
- `openpyxl>=3.0.0` - Excel file generation
- `scipy>=1.8.0` - Scientific computing
- `pyproj>=3.3.0` - Geodetic coordinate transformations

## 🚧 Current Limitations

- Fixed person height assumption (1.7m) affects distance accuracy
- Requires synchronized frame capture between drones
- Association threshold must be tuned per scenario
- GPS accuracy limits triangulation precision
- Manual ground truth labeling from filenames
- Processing is offline (not real-time streaming)

## 🔮 Future Improvements

- [ ] Real-time streaming support
- [ ] Automatic person height estimation
- [ ] Multi-target tracking across frames
- [ ] Adaptive association thresholds
- [ ] IMU data integration for better orientation
- [ ] Machine learning-based position refinement
- [ ] Support for 3+ drones
- [ ] Web-based visualization dashboard
- [ ] Automatic camera calibration
- [ ] Real-time performance optimization

## 📄 License

This project is part of research work on multi-drone surveillance systems.

## 👥 Contributors

Developed for drone-based person and weapon detection research.

---

**Note**: This system is designed for research purposes. Ensure compliance with local regulations regarding drone operation and surveillance.
