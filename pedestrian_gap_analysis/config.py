# config.py — centralized pipeline constants

import os

# ── Input / Output ──────────────────────────────────────────────────────────
VIDEO_PATH: str = os.environ.get("VIDEO_PATH", "input_video.mp4")
OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "output")

# ── YOLOv8 Detection ────────────────────────────────────────────────────────
YOLO_MODEL: str = "yolov8n.pt"
YOLO_CONF_THRESHOLD: float = 0.3
# COCO class IDs: person=0, motorcycle=3, car=2, bus=5, truck=7
YOLO_CLASSES: list = [0, 2, 3, 5, 7]
VEHICLE_CLASSES: list = [2, 3, 5, 7]   # non-pedestrian targets
PERSON_CLASS_ID: int = 0

# ── ByteTrack ────────────────────────────────────────────────────────────────
BYTETRACK_MAX_AGE: int = 30  # frames before a lost track is dropped

# ── Gap Classification ───────────────────────────────────────────────────────
ROLLING_GAP_SPEED_RATIO: float = 0.40   # fraction of mean speed → slow threshold (raised for sensitivity)
ROLLING_GAP_MIN_DURATION: float = 0.3   # seconds of continuous slow movement (lowered for sensitivity)

# ── Age Bucketing ────────────────────────────────────────────────────────────
AGE_YOUNG_MAX: int = 29   # ≤ 29 → Young
AGE_MIDDLE_MAX: int = 59  # 30–59 → Middle; ≥ 60 → Old

# ── Regression ───────────────────────────────────────────────────────────────
MIN_RECORDS_FOR_REGRESSION: int = 30

# ── Visualization ────────────────────────────────────────────────────────────
PLOT_DPI: int = 150

# ── Annotated Video ──────────────────────────────────────────────────────────
OUTPUT_VIDEO_CODEC: str = "mp4v"
OUTPUT_VIDEO_FILENAME: str = "annotated_output.mp4"

# ── GPU / Device ─────────────────────────────────────────────────────────────
# "0" = first CUDA GPU, "cpu" = force CPU, "0,1" = multi-GPU
DEVICE: str = "0"

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_FILENAME: str = "gap_acceptance_dataset.csv"
MODEL_SUMMARY_FILENAME: str = "logit_model_summary.txt"
