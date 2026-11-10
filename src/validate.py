"""
src/validate.py
===============
Member 4 owns this file. Member 1 provides the scaffold.

Standalone validation script — run on saved weights to get AUC, F1, Acc.
Also generates predictions CSV on the validation set for threshold analysis.

Usage:
    python src/validate.py --weights weights/best_model.pth
    python src/validate.py --weights weights/best_model.pth --backbone convnext_tiny
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    roc_curve, classification_report
)

from config import (
    VAL_SPLIT_CSV, OUTPUTS_DIR, NUM_WORKERS,
    BATCH_SIZE, THRESHOLD, BACKBONE
)
from src.dataset import DeepfakeDataset, get_dataloader
from src.augmentations import get_val_transforms
from src.model import build_model, load_model_weights


def find_optimal_threshold(labels, probs):
    """Find threshold that maximises F1 on the validation set."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.3, 0.7, 0.01):
        preds = (np.array(probs) >= thr).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


@torch.no_grad()
def run_validation(weights_path: str, backbone: str = None, threshold: float = THRESHOLD):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone or BACKBONE

    print(f"\n[Validate] Loading model: {backbone} from {weights_path}")
    model = build_model(backbone=backbone, pretrained=False)
    model = load_model_weights(model, weights_path)

    val_ds     = DeepfakeDataset(VAL_SPLIT_CSV, transform=get_val_transforms())
    val_loader = get_dataloader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_probs, all_labels, all_ids = [], [], []

    for batch_imgs, batch_labels in val_loader:
        batch_imgs = batch_imgs.to(device)
        logits = model(batch_imgs)
        probs  = torch.sigmoid(logits).cpu().squeeze(1).numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch_labels.numpy().tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels, dtype=int)

    # ── Metrics ───────────────────────────────────────────────────────────
    auc = roc_auc_score(all_labels, all_probs)
    preds_bin = (all_probs >= threshold).astype(int)
    f1  = f1_score(all_labels, preds_bin)
    acc = accuracy_score(all_labels, preds_bin)

    opt_thr, opt_f1 = find_optimal_threshold(all_labels, all_probs)

    print(f"\n{'='*50}")
    print(f"  Validation Results ({weights_path})")
    print(f"{'='*50}")
    print(f"  AUC-ROC   : {auc:.4f}   ← Primary metric")
    print(f"  Accuracy  : {acc:.4f}   (threshold={threshold})")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Optimal F1: {opt_f1:.4f} (threshold={opt_thr:.2f})")
    print(f"{'='*50}")
    print(classification_report(all_labels, preds_bin, target_names=["Real", "Fake"]))

    # ── Save val predictions ───────────────────────────────────────────────
    out_path = OUTPUTS_DIR / "val_predictions.csv"
    pd.DataFrame({
        "label":      all_labels,
        "confidence": all_probs,
        "predicted":  preds_bin,
    }).to_csv(out_path, index=False)
    print(f"[Saved] Validation predictions → {out_path}")

    return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate deepfake detection model")
    parser.add_argument("--weights",   required=True,              help="Path to .pth weights file")
    parser.add_argument("--backbone",  default=None,               help="Backbone name (default: from config)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Classification threshold")
    args = parser.parse_args()

    run_validation(args.weights, args.backbone, args.threshold)
