"""
eda.py
======
Member 1 Task — Exploratory Data Analysis

Analyses the training dataset and generates:
  - Class distribution
  - Per-channel mean/std statistics
  - Sample image grid (10 Real + 10 Fake)
  - Brightness & contrast distributions
  - Any corrupted / unreadable image detection

Usage:
    python eda.py

Outputs saved to: outputs/eda/
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TRAIN_META_CSV, TRAIN_REAL_DIR, TRAIN_FAKE_DIR,
    TEST_DIR, OUTPUTS_DIR, SEED
)

EDA_DIR = OUTPUTS_DIR / "eda"


def load_image_safe(path):
    """Load image, return None if corrupted."""
    try:
        img = Image.open(path).convert("RGB")
        img.verify()
        img = Image.open(path).convert("RGB")  # Re-open after verify
        return img
    except Exception:
        return None


def compute_channel_stats(image_paths, sample_n=1000):
    """Compute per-channel mean and std on a random sample of images."""
    random.seed(SEED)
    sample = random.sample(image_paths, min(sample_n, len(image_paths)))

    means, stds = [], []
    for p in tqdm(sample, desc="Computing channel stats", ncols=80):
        img = load_image_safe(p)
        if img is None:
            continue
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        means.append(arr.mean(axis=(0, 1)))
        stds.append(arr.std(axis=(0, 1)))

    mean = np.array(means).mean(axis=0)
    std  = np.array(stds).mean(axis=0)
    return mean, std


def plot_sample_grid(real_paths, fake_paths, n=10, save_path=None):
    """Plot n real and n fake images in a 2-row grid."""
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 5))
    fig.suptitle("Top Row: Real (label=0) | Bottom Row: Fake (label=1)", fontsize=13, y=1.02)

    random.seed(SEED)
    sample_real = random.sample(real_paths, n)
    sample_fake = random.sample(fake_paths, n)

    for col, (rp, fp) in enumerate(zip(sample_real, sample_fake)):
        for row, (path, title) in enumerate([(rp, "Real"), (fp, "Fake")]):
            img = load_image_safe(path)
            if img:
                axes[row][col].imshow(img)
            else:
                axes[row][col].set_facecolor("red")
                axes[row][col].text(0.5, 0.5, "CORRUPT", ha="center", va="center", color="white")
            axes[row][col].axis("off")
            if col == 0:
                axes[row][col].set_title(title, fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[EDA] Sample grid saved → {save_path}")
    plt.close()


def plot_brightness_distribution(real_paths, fake_paths, save_path=None):
    """Plot mean brightness histogram for real vs fake images."""
    def get_brightness(paths, label, n=500):
        random.seed(SEED)
        sample = random.sample(paths, min(n, len(paths)))
        brightness = []
        for p in tqdm(sample, desc=f"Brightness ({label})", ncols=80):
            img = load_image_safe(p)
            if img:
                brightness.append(np.array(img).mean())
        return brightness

    real_b = get_brightness(real_paths, "Real")
    fake_b = get_brightness(fake_paths, "Fake")

    plt.figure(figsize=(10, 4))
    plt.hist(real_b, bins=50, alpha=0.6, label="Real", color="steelblue", density=True)
    plt.hist(fake_b, bins=50, alpha=0.6, label="Fake", color="tomato",    density=True)
    plt.xlabel("Mean Pixel Brightness (0–255)")
    plt.ylabel("Density")
    plt.title("Brightness Distribution: Real vs Fake")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[EDA] Brightness distribution saved → {save_path}")
    plt.close()


def check_corrupted_images(all_paths):
    """Scan all images and log corrupted ones."""
    corrupted = []
    for p in tqdm(all_paths, desc="Checking for corruption", ncols=80):
        if load_image_safe(p) is None:
            corrupted.append(str(p))
    return corrupted


def main():
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  Deepfake Detection — Exploratory Data Analysis")
    print("="*60 + "\n")

    # ── 1. Load metadata ───────────────────────────────────────────────────
    meta = pd.read_csv(TRAIN_META_CSV)
    print(f"[EDA] Training metadata: {len(meta)} entries")
    print(meta["label"].value_counts().rename({0: "Real (0)", 1: "Fake (1)"}).to_string())
    print()

    # ── 2. Build path lists ────────────────────────────────────────────────
    real_paths = sorted(TRAIN_REAL_DIR.glob("*.jpg"))
    fake_paths = sorted(TRAIN_FAKE_DIR.glob("*.fake.jpg")) or sorted(TRAIN_FAKE_DIR.glob("*.jpg"))
    test_paths = sorted(TEST_DIR.glob("*.jpg"))

    print(f"[EDA] Real images found : {len(real_paths)}")
    print(f"[EDA] Fake images found : {len(fake_paths)}")
    print(f"[EDA] Test images found : {len(test_paths)}")
    print()

    # ── 3. Image resolution check ─────────────────────────────────────────
    # All should be 224×224 per README
    sample_img = load_image_safe(real_paths[0]) if real_paths else None
    if sample_img:
        print(f"[EDA] Sample resolution : {sample_img.size}  (expected 224×224)")
    print()

    # ── 4. Corrupted image check ──────────────────────────────────────────
    print("[EDA] Scanning for corrupted images...")
    all_train_paths = real_paths + fake_paths
    corrupted = check_corrupted_images(all_train_paths)
    if corrupted:
        print(f"[WARNING] {len(corrupted)} corrupted images found:")
        for c in corrupted[:10]:
            print(f"  {c}")
        corrupt_log = EDA_DIR / "corrupted_images.txt"
        with open(corrupt_log, "w") as f:
            f.write("\n".join(corrupted))
        print(f"  Full list → {corrupt_log}")
    else:
        print("[EDA] No corrupted images found. ✓")
    print()

    # ── 5. Channel mean/std ────────────────────────────────────────────────
    print("[EDA] Computing channel statistics (sample of 1000 images)...")
    mean, std = compute_channel_stats(all_train_paths, sample_n=1000)
    print(f"  Dataset Mean : R={mean[0]:.4f} G={mean[1]:.4f} B={mean[2]:.4f}")
    print(f"  Dataset Std  : R={std[0]:.4f}  G={std[1]:.4f}  B={std[2]:.4f}")
    print(f"  (ImageNet:    Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225])")
    print()

    # ── 6. Sample image grid ───────────────────────────────────────────────
    print("[EDA] Generating sample image grid...")
    plot_sample_grid(
        real_paths, fake_paths, n=10,
        save_path=EDA_DIR / "sample_grid.png"
    )

    # ── 7. Brightness distribution ────────────────────────────────────────
    print("[EDA] Plotting brightness distribution...")
    plot_brightness_distribution(
        real_paths, fake_paths,
        save_path=EDA_DIR / "brightness_distribution.png"
    )

    # ── 8. Summary report ─────────────────────────────────────────────────
    summary = {
        "total_train": len(meta),
        "real_count":  int((meta["label"] == 0).sum()),
        "fake_count":  int((meta["label"] == 1).sum()),
        "test_count":  len(test_paths),
        "corrupted_count": len(corrupted),
        "dataset_mean_R": float(mean[0]), "dataset_mean_G": float(mean[1]), "dataset_mean_B": float(mean[2]),
        "dataset_std_R":  float(std[0]),  "dataset_std_G":  float(std[1]),  "dataset_std_B":  float(std[2]),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(EDA_DIR / "eda_summary.csv", index=False)

    print(f"\n[EDA COMPLETE] All outputs saved to: {EDA_DIR}")
    print(f"  - sample_grid.png")
    print(f"  - brightness_distribution.png")
    print(f"  - eda_summary.csv")
    if corrupted:
        print(f"  - corrupted_images.txt")


if __name__ == "__main__":
    main()
