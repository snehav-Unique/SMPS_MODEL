"""
src/augmentations.py
====================
Member 1 Task — Augmentation Pipelines
Defines albumentations transforms for train, validation, and inference.

Key Design Choices:
  - ImageCompression: teaches model to ignore JPEG artifacts (critical per README)
  - GaussianBlur + GaussNoise: robustness against image degradation
  - CoarseDropout: occlusion robustness
  - NO geometric distortion beyond flip — face alignment must be preserved
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms() -> A.Compose:
    """
    Heavy augmentation pipeline for training.
    Targets robustness across compression, lighting, and blur — all
    mentioned as real-world noise in the hackathon README.
    """
    return A.Compose([
        # Spatial
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4),

        # Colour / brightness
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),

        # Blur / noise — teaches model to not rely on pixel-perfect artifacts
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

        # JPEG compression artifact simulation — CRITICAL for generalisation
        # Deepfake detectors often overfit to JPEG block patterns; this prevents it
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.4),

        # Occlusion robustness
        A.CoarseDropout(
            max_holes=4, max_height=32, max_width=32,
            min_holes=1, min_height=8,  min_width=8,
            fill_value=0, p=0.2
        ),

        # Normalise + convert to tensor (always last)
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """
    Minimal transforms for validation — only resize + normalise.
    No augmentation so AUC is measured on clean images.
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_inference_transforms() -> A.Compose:
    """
    Identical to val_transforms — used by inference_script.py.
    Must match val_transforms exactly for reproducibility.
    """
    return get_val_transforms()


def get_tta_transforms() -> list:
    """
    Test-Time Augmentation (TTA) transforms.
    Run inference with each and average predictions for better AUC.
    Optional — use if time permits.
    """
    return [
        # Original
        get_val_transforms(),
        # Horizontal flip
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
        # Slight brightness increase
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=0, p=1.0),
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
    ]
