"""
src/dataset.py
==============
Member 1 Task — PyTorch Dataset classes for Deepfake Detection.

Classes:
    DeepfakeDataset  — Training / validation dataset (reads from split CSV)
    TestDataset      — Inference dataset (flat directory, no labels)

Both classes accept albumentations transforms.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

from config import TRAIN_REAL_DIR, TRAIN_FAKE_DIR, TEST_DIR, IMAGE_SIZE


class DeepfakeDataset(Dataset):
    """
    Dataset for training and validation.

    Reads image_id + label from a CSV (output of create_split.py).
    Images are loaded from TRAIN_REAL_DIR (label=0) or TRAIN_FAKE_DIR (label=1).

    Args:
        csv_path  : Path to split CSV (image_id, label columns).
        transform : Albumentations Compose pipeline (train or val transforms).
        base_real : Path to real images folder (default: config.TRAIN_REAL_DIR).
        base_fake : Path to fake images folder (default: config.TRAIN_FAKE_DIR).
    """

    def __init__(self, csv_path, transform=None, base_real=None, base_fake=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.base_real = Path(base_real) if base_real else TRAIN_REAL_DIR
        self.base_fake = Path(base_fake) if base_fake else TRAIN_FAKE_DIR

        # Verify columns
        assert "image_id" in self.df.columns, "CSV must have 'image_id' column"
        assert "label"    in self.df.columns, "CSV must have 'label' column"

        # Pre-build full image paths
        self.image_paths = []
        self.labels = []
        missing = []

        for _, row in self.df.iterrows():
            img_id = row["image_id"]
            label  = int(row["label"])
            folder = self.base_fake if label == 1 else self.base_real
            path   = folder / f"{img_id}.jpg"

            if not path.exists():
                missing.append(str(path))
                continue

            self.image_paths.append(path)
            self.labels.append(label)

        if missing:
            print(f"[WARNING] {len(missing)} images not found and skipped.")
            if len(missing) <= 5:
                for m in missing:
                    print(f"  Missing: {m}")

        print(f"[Dataset] Loaded {len(self.image_paths)} images "
              f"(Real: {self.labels.count(0)}, Fake: {self.labels.count(1)})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label    = self.labels[idx]

        # Load image — PIL → numpy (H, W, C) for albumentations
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]   # Returns torch.Tensor after ToTensorV2

        return img, torch.tensor(label, dtype=torch.float32)


class TestDataset(Dataset):
    """
    Dataset for inference on the test set (no labels).

    Reads all .jpg / .jpeg / .png files from a flat directory.
    Returns (tensor_image, image_id_string) per sample.

    Args:
        image_dir : Path to test/images/ folder.
        transform : Albumentations Compose pipeline (inference transforms).
    """

    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(
            list(self.image_dir.glob("*.jpg"))
            + list(self.image_dir.glob("*.jpeg"))
            + list(self.image_dir.glob("*.png"))
        )
        self.transform = transform

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        print(f"[TestDataset] Found {len(self.image_paths)} images in {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_id = img_path.stem   # UUID filename without .jpg

        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        return img, image_id


def get_dataloader(dataset, batch_size, shuffle, num_workers=4, pin_memory=True):
    """
    Convenience wrapper for DataLoader creation.
    """
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(shuffle),   # Drop last incomplete batch only during training
    )
