"""
config.py
=========
Central configuration for the Deepfake Detection project.
Edit the DATASET_ROOT to point to your participant package dataset.
All other scripts import from here — no hardcoded paths elsewhere.
"""

import os
import torch
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — Edit DATASET_ROOT to match your machine
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(
    r"D:\RVCE\Sem 3\Deepfake ML Hackathon"
    r"\Deepfake ML Hackathon - Participant Package-20261105T145705Z-1-001"
    r"\Deepfake ML Hackathon - Participant Package"
    r"\hackathon_dataset\hackathon_dataset"
)

TRAIN_REAL_DIR  = DATASET_ROOT / "train" / "real"
TRAIN_FAKE_DIR  = DATASET_ROOT / "train" / "fake"
TEST_DIR        = DATASET_ROOT / "test" / "images"
TRAIN_META_CSV  = DATASET_ROOT / "train_metadata.csv"
SAMPLE_SUB_CSV  = DATASET_ROOT / "sample_submission.csv"

PROJECT_ROOT    = Path(__file__).resolve().parent
DATA_DIR        = PROJECT_ROOT / "data"
WEIGHTS_DIR     = PROJECT_ROOT / "weights"
OUTPUTS_DIR     = PROJECT_ROOT / "outputs"

TRAIN_SPLIT_CSV = DATA_DIR / "train_split.csv"
VAL_SPLIT_CSV   = DATA_DIR / "val_split.csv"

# ─────────────────────────────────────────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 224          # Must match training if you change inference script
BATCH_SIZE   = 32           # Training batch size
INFER_BATCH  = 64           # Inference batch size (larger = faster)
NUM_WORKERS  = 4
THRESHOLD    = 0.5          # Fake probability threshold for binary label

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS         = 25
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
SEED           = 42
VAL_SPLIT      = 0.20       # 80/20 split
BACKBONE       = "efficientnet_b4"   # Options: efficientnet_b4, efficientnet_b2, convnext_tiny

# ImageNet normalisation stats (used for pre-trained backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────────────────────
# MIXED PRECISION
# ─────────────────────────────────────────────────────────────────────────────
# AMP only works on CUDA — auto-disabled on CPU to suppress warnings
USE_AMP = torch.cuda.is_available()

# ─────────────────────────────────────────────────────────────────────────────
# BEST WEIGHTS FILE (used by inference_script.py)
# ─────────────────────────────────────────────────────────────────────────────
BEST_WEIGHTS = WEIGHTS_DIR / "best_model.pth"

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE — resolved once here so all scripts can import it
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
