"""
Configuration file for people detection pipeline.
"""

# Model configuration
MODEL_PATH = "models/yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.5

# Input/Output paths
INPUT_DIR = "inputs/samples"
OUTPUT_DIR = "output/detections"

# Detection settings
PERSON_CLASS_ID = 0  # COCO class ID for 'person'
BOX_COLOR = (0, 255, 0)  # Green bounding boxes (BGR format)
BOX_THICKNESS = 2
FONT_SCALE = 0.5
FONT_THICKNESS = 2

# Supported image formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']

# Crop settings
CROP_PADDING = 0.1  # 10% padding around bounding boxes
SAVE_CROPS = True   # Whether to save individual person crops
CROP_MIN_SIZE = 32  # Minimum crop size in pixels

# Output organization
DETECTIONS_FOLDER = "detections"  # Folder name for images with bounding boxes
CROPS_FOLDER = "crops"           # Folder name for individual person crops