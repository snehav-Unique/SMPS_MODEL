"""
src/train.py
============
Member 2 owns this file. Member 1 provides the full pipeline scaffold.

Training loop for the Deepfake Detection model.

Usage:
    python src/train.py
    python src/train.py --backbone efficientnet_b2 --epochs 30 --lr 5e-5

Saves:
    weights/best_model.pth        — Best validation AUC checkpoint
    weights/last_model.pth        — Last epoch checkpoint
    outputs/train_log.csv         — Epoch-wise loss/AUC log
"""

import sys
import os
import argparse
import random
import numpy as np
import pandas as pd
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from config import (
    TRAIN_SPLIT_CSV, VAL_SPLIT_CSV,
    WEIGHTS_DIR, OUTPUTS_DIR,
    BATCH_SIZE, NUM_WORKERS, EPOCHS, LR, WEIGHT_DECAY,
    SEED, BACKBONE, USE_AMP, THRESHOLD
)
from src.dataset import DeepfakeDataset, get_dataloader
from src.augmentations import get_train_transforms, get_val_transforms
from src.model import build_model, count_parameters


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING EPOCH
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for batch_imgs, batch_labels in loader:
        batch_imgs   = batch_imgs.to(device)
        batch_labels = batch_labels.to(device).unsqueeze(1)  # (B,) → (B,1)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            logits = model(batch_imgs)             # (B, 1)
            loss   = criterion(logits, batch_labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_imgs.size(0)
        probs = torch.sigmoid(logits).detach().cpu().squeeze(1).numpy()
        all_preds.extend(probs.tolist())
        all_labels.extend(batch_labels.cpu().squeeze(1).numpy().tolist())

    avg_loss = running_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_preds)
    return avg_loss, auc


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION EPOCH
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for batch_imgs, batch_labels in loader:
        batch_imgs   = batch_imgs.to(device)
        batch_labels = batch_labels.to(device).unsqueeze(1)

        with autocast(enabled=use_amp):
            logits = model(batch_imgs)
            loss   = criterion(logits, batch_labels)

        running_loss += loss.item() * batch_imgs.size(0)
        probs = torch.sigmoid(logits).cpu().squeeze(1).numpy()
        all_preds.extend(probs.tolist())
        all_labels.extend(batch_labels.cpu().squeeze(1).numpy().tolist())

    avg_loss = running_loss / len(loader.dataset)
    auc      = roc_auc_score(all_labels, all_preds)
    preds_bin = (np.array(all_preds) >= threshold).astype(int)
    f1       = f1_score(all_labels, preds_bin)
    acc      = accuracy_score(all_labels, preds_bin)

    return avg_loss, auc, f1, acc


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    set_seed(SEED)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # AMP only helps on CUDA; resolve once and pass explicitly everywhere
    use_amp = USE_AMP and (device.type == "cuda")
    # Windows: multiprocessing with num_workers > 0 requires __main__ guard
    # already satisfied since we're in main(); safe to use NUM_WORKERS
    n_workers = NUM_WORKERS if device.type == "cuda" else 0

    print(f"\n{'='*60}")
    print(f"  Deepfake Detection Training")
    print(f"  Backbone : {args.backbone}")
    print(f"  Device   : {device}  {'(GPU ✓)' if device.type == 'cuda' else '(CPU — consider Colab/Kaggle for GPU)'}")
    print(f"  AMP/FP16 : {'enabled' if use_amp else 'disabled (CPU)'}")
    print(f"  Epochs   : {args.epochs}  |  LR: {args.lr}  |  Batch: {args.batch_size}")
    print(f"{'='*60}\n")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = DeepfakeDataset(TRAIN_SPLIT_CSV, transform=get_train_transforms())
    val_ds   = DeepfakeDataset(VAL_SPLIT_CSV,   transform=get_val_transforms())

    train_loader = get_dataloader(train_ds, args.batch_size, shuffle=True,  num_workers=n_workers)
    val_loader   = get_dataloader(val_ds,   args.batch_size, shuffle=False, num_workers=n_workers)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(backbone=args.backbone, pretrained=True).to(device)
    print(f"[Model] Parameters: {count_parameters(model):,}")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler(enabled=use_amp)

    # ── Training ──────────────────────────────────────────────────────────
    best_auc   = 0.0
    log_rows   = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss, val_auc, val_f1, val_acc = validate(
            model, val_loader, criterion, device, use_amp, threshold=THRESHOLD
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"TrainLoss={train_loss:.4f} TrainAUC={train_auc:.4f} | "
            f"ValLoss={val_loss:.4f} ValAUC={val_auc:.4f} "
            f"ValF1={val_f1:.4f} ValAcc={val_acc:.4f} | "
            f"Time={elapsed:.1f}s"
        )

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_auc": train_auc,
            "val_loss": val_loss,     "val_auc": val_auc,
            "val_f1": val_f1,         "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # ── Save best model ────────────────────────────────────────────
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), WEIGHTS_DIR / "best_model.pth")
            print(f"  ★ New best AUC: {best_auc:.4f} — model saved.")

    # ── Save last model + log ──────────────────────────────────────────────
    torch.save(model.state_dict(), WEIGHTS_DIR / "last_model.pth")
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(OUTPUTS_DIR / "train_log.csv", index=False)

    print(f"\n[DONE] Best Val AUC: {best_auc:.4f}")
    print(f"  Best weights → {WEIGHTS_DIR / 'best_model.pth'}")
    print(f"  Training log → {OUTPUTS_DIR / 'train_log.csv'}")


if __name__ == "__main__":
    # Required on Windows: DataLoader workers spawn new processes,
    # which re-import this module — the guard prevents infinite spawning.
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Deepfake Detection Training")
    parser.add_argument("--backbone",   type=str,   default=BACKBONE,   help="Model backbone name")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS,     help="Number of epochs")
    parser.add_argument("--lr",         type=float, default=LR,         help="Learning rate")
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()
    main(args)
