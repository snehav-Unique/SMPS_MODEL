"""
create_split.py
===============
Member 1 Task — Step 2
Creates a stratified 80/20 train/validation split from train_metadata.csv.

Usage:
    python create_split.py

Outputs:
    data/train_split.csv   — 80% of training images
    data/val_split.csv     — 20% of training images (held-out for validation)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from config import TRAIN_META_CSV, TRAIN_SPLIT_CSV, VAL_SPLIT_CSV, VAL_SPLIT, SEED, DATA_DIR


def create_split():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading metadata from: {TRAIN_META_CSV}")
    meta = pd.read_csv(TRAIN_META_CSV)

    print(f"[INFO] Total samples: {len(meta)}")
    print(f"[INFO] Class distribution:\n{meta['label'].value_counts().to_string()}")
    print(f"[INFO]   Real (0): {(meta['label'] == 0).sum()} | Fake (1): {(meta['label'] == 1).sum()}")

    train_df, val_df = train_test_split(
        meta,
        test_size=VAL_SPLIT,
        stratify=meta["label"],
        random_state=SEED,
    )

    train_df.to_csv(TRAIN_SPLIT_CSV, index=False)
    val_df.to_csv(VAL_SPLIT_CSV, index=False)

    print(f"\n[DONE] Split created:")
    print(f"  Train → {TRAIN_SPLIT_CSV}  ({len(train_df)} images)")
    print(f"  Val   → {VAL_SPLIT_CSV}  ({len(val_df)} images)")
    print(f"  Train fake/real: {(train_df['label']==1).sum()} / {(train_df['label']==0).sum()}")
    print(f"  Val   fake/real: {(val_df['label']==1).sum()} / {(val_df['label']==0).sum()}")


if __name__ == "__main__":
    create_split()
