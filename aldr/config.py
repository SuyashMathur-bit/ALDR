"""
ALDR Configuration Module
AI-Driven Law & Road Monitoring (ALDR)
"""

import os
from pathlib import Path

# Environment detection: Google Colab vs Local Windows / G: Drive
def get_default_data_root():
    if os.path.exists("/content/drive/MyDrive"):
        return Path("/content/drive/MyDrive")
    elif os.path.exists(r"G:\My Drive"):
        return Path(r"G:\My Drive")
    else:
        # Fallback to current directory
        return Path(".")

DATA_ROOT = get_default_data_root()

# Project directory paths
ALDR_PART2_DIR = DATA_ROOT / "ALDR_Part2"
ALDR_REID_DIR = DATA_ROOT / "ALDR_ReID"
VERI_ROOT_DIR = DATA_ROOT / "VeRi"
INDIAN_VEHICLE_DATASET_DIR = DATA_ROOT / "Indian Vehicle Dataset by Amit Juyal.v2i.yolov11extracted"
MODIFICATION_DATASET_DIR = DATA_ROOT / "ALDR_Modification_11class"

# Model weights paths
VEHICLE_DETECTOR_WEIGHTS = ALDR_PART2_DIR / "vehicle_baseline-3" / "weights" / "best.pt"
if not VEHICLE_DETECTOR_WEIGHTS.exists():
    VEHICLE_DETECTOR_WEIGHTS = ALDR_PART2_DIR / "vehicle_baseline_50ep" / "weights" / "best.pt"
MODIFICATION_DETECTOR_WEIGHTS = ALDR_PART2_DIR / "modification_unified_11class" / "weights" / "best.pt"
REID_CHECKPOINT_DIR = ALDR_REID_DIR / "checkpoints"
REID_EMBEDDING_MODEL_PATH = ALDR_REID_DIR / "reid_embedding_model.keras"

# Output directory for reports, cropped evidence, and predictions
OUTPUT_DIR = Path("./aldr_output")

# Dimensions and Re-ID parameters
REID_IMG_HEIGHT = 224
REID_IMG_WIDTH = 112
REID_EMBEDDING_DIM = 512
REID_BATCH_SIZE = 32

# Detection thresholds
VEHICLE_CONF_THRESHOLD = 0.25
MODIFICATION_CONF_THRESHOLD = 0.25

# Cross-view association parameters
REID_SIMILARITY_THRESHOLD = 0.65  # Calibrated cosine similarity threshold
TEMPORAL_WINDOW_SECONDS = 300.0   # Maximum time window for cross-view matches (if timestamps present)
CAMERA_DISTANCE_MATRIX = None     # Optional distance matrix between cameras

# 11 Fine-grained modification categories (preserving exact order and indexing)
MODIFICATION_CLASSES = [
    "bumper",
    "wheel",
    "aftermarket_exhaust",
    "additional_lights",
    "aftermarket_lights",
    "colored_headlight",
    "bull_bar",
    "aftermarket_spoiler",
    "aftermarket_tyre",
    "modification_kit",
    "illegal_number_plate"
]

MODIFICATION_ID_TO_NAME = {i: name for i, name in enumerate(MODIFICATION_CLASSES)}
MODIFICATION_NAME_TO_ID = {name: i for i, name in enumerate(MODIFICATION_CLASSES)}

# Vehicle detector classes from Indian Vehicle Dataset (17 classes)
VEHICLE_CLASSES = [
    "ambulance", "auto", "barrier", "bus", "car", "cyclist", "eriksha",
    "illegal vehicle", "loader", "motorbike", "pedestrian", "scooter",
    "tracktor", "traffic cone", "traffic police stand", "truck", "vikram"
]

# Vehicle classes considered motor vehicles for road law compliance & Re-ID
TARGET_VEHICLE_CLASSES = {
    "ambulance", "auto", "bus", "car", "eriksha", "illegal vehicle",
    "loader", "motorbike", "scooter", "tracktor", "truck", "vikram"
}

# Non-vehicle classes to ignore
NON_VEHICLE_CLASSES = {"barrier", "pedestrian", "traffic cone", "traffic police stand", "cyclist"}
