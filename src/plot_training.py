"""
src/plot_training.py
====================
Member 4 uses this after training completes.
Plots training loss curves and AUC from outputs/train_log.csv.

Usage:
    python src/plot_training.py
    python src/plot_training.py --log outputs/train_log.csv
"""

import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUTS_DIR


def plot_curves(log_path: str):
    df = pd.read_csv(log_path)
    out_dir = Path(log_path).parent

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Training Progress", fontsize=14)

    # Loss
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="steelblue")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val Loss",   color="tomato")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # AUC-ROC
    axes[1].plot(df["epoch"], df["train_auc"], label="Train AUC", color="steelblue")
    axes[1].plot(df["epoch"], df["val_auc"],   label="Val AUC",   color="tomato")
    axes[1].axhline(y=0.90, linestyle="--", color="green",  alpha=0.5, label="AUC=0.90 target")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("AUC-ROC Curves"); axes[1].legend(); axes[1].grid(alpha=0.3)

    # Learning Rate
    if "lr" in df.columns:
        axes[2].plot(df["epoch"], df["lr"], color="purple")
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("LR Schedule"); axes[2].grid(alpha=0.3)
    else:
        axes[2].axis("off")

    plt.tight_layout()
    out_path = out_dir / "training_curves.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Training curves → {out_path}")

    # Print final stats
    best_row = df.loc[df["val_auc"].idxmax()]
    print(f"\n[Best Epoch] Epoch {int(best_row['epoch'])}")
    print(f"  Val AUC  : {best_row['val_auc']:.4f}")
    print(f"  Val F1   : {best_row['val_f1']:.4f}")
    print(f"  Val Acc  : {best_row['val_acc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=str(OUTPUTS_DIR / "train_log.csv"))
    args = parser.parse_args()
    plot_curves(args.log)
