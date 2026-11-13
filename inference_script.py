"""
inference_script.py
===================
Official inference script for the Deepfake ML Hackathon (IIIT Bangalore).
Filled from inference_template.py by Member 1.

Usage (as required by the hackathon):
    python inference_script.py \
        --test_dir ./hackathon_dataset/test/images/ \
        --weights  ./weights/best_model.pth \
        --output   predictions.csv

Optional flags:
    --backbone     Model backbone name (default: efficientnet_b4)
    --batch_size   Inference batch size (default: 64)
    --threshold    Fake probability threshold (default: 0.5)
    --fp16         Use FP16 inference (faster on T4, default: enabled if CUDA)
    --tta          Use test-time augmentation (3x slower but potentially better AUC)

All paths are via CLI arguments - no hardcoded paths (required by hackathon rules).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# -----------------------------------------------------------------------------
# CONSTANTS (overrideable via CLI)
# -----------------------------------------------------------------------------
IMAGE_SIZE   = 224
BATCH_SIZE   = 64
# Windows: num_workers>0 with DataLoader requires spawn context - safe inside
# __main__ guard. Set to 0 on CPU to avoid overhead; GPU gets full workers.
NUM_WORKERS  = 4 if torch.cuda.is_available() else 0
THRESHOLD    = 0.5
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE     = "efficientnet_b4"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------------
# DATASET - Flat directory of test images, no labels
# -----------------------------------------------------------------------------
class TestDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(
            list(self.image_dir.glob("*.jpg"))
            + list(self.image_dir.glob("*.jpeg"))
            + list(self.image_dir.glob("*.png"))
        )
        self.transform = transform

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in: {self.image_dir}")
        print(f"[INFO] Found {len(self.image_paths)} test images in {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_id = img_path.stem   # UUID without .jpg extension

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Could not read {img_path}: {e}. Using blank image.")
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, image_id


# -----------------------------------------------------------------------------
# MODEL - Architecture must match what was trained
# -----------------------------------------------------------------------------
def build_model(backbone: str) -> nn.Module:
    """Build the model architecture (without loading weights)."""

    if backbone == "efficientnet_b4":
        from torchvision.models import efficientnet_b4
        model = efficientnet_b4(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 1),
        )

    elif backbone == "efficientnet_b2":
        from torchvision.models import efficientnet_b2
        model = efficientnet_b2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1),
        )

    elif backbone == "convnext_tiny":
        try:
            import timm
            model = timm.create_model("convnext_tiny", pretrained=False, num_classes=1)
        except ImportError:
            raise ImportError("timm is required for convnext_tiny. Run: pip install timm")

    elif backbone in ("efficientnetv2_s", "tf_efficientnetv2_s"):
        try:
            import timm
            model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=1)
        except ImportError:
            raise ImportError("timm is required. Run: pip install timm")

    elif backbone in ("swin_tiny", "swin_tiny_patch4_window7_224"):
        try:
            import timm
            model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=1)
        except ImportError:
            raise ImportError("timm is required. Run: pip install timm")

    elif backbone == "densenet121":
        try:
            import timm
            model = timm.create_model("densenet121", pretrained=False, num_classes=1)
        except ImportError:
            raise ImportError("timm is required. Run: pip install timm")

    elif backbone in ("maxvit_tiny", "maxvit_tiny_tf_224"):
        try:
            import timm
            model = timm.create_model("maxvit_tiny_tf_224", pretrained=False, num_classes=1)
        except ImportError:
            raise ImportError("timm is required. Run: pip install timm")

    else:
        raise ValueError(f"Unknown backbone: {backbone!r}")

    return model


class EnsembleRunner:
    """Wrapper that runs the multi-model ensemble with horizontal-flip TTA."""
    def __init__(self, models_and_weights, device):
        self.models_and_weights = models_and_weights
        self.device = device

    def to(self, device):
        for m, _ in self.models_and_weights:
            m.to(device)
        self.device = device
        return self

    def eval(self):
        for m, _ in self.models_and_weights:
            m.eval()
        return self

    def half(self):
        for m, _ in self.models_and_weights:
            m.half()
        return self

    def __call__(self, x):
        total = None
        for model, weight in self.models_and_weights:
            p = (
                torch.sigmoid(model(x)).squeeze(-1) +
                torch.sigmoid(model(x.flip(-1))).squeeze(-1)
            ) / 2.0
            if total is None:
                total = weight * p
            else:
                total = total + weight * p
        return total


def load_model(weights_path: str, backbone: str, device: str):
    """Load model and weights onto device (supports single model, JSON config, or ensemble bundle)."""
    import json
    w_path = Path(weights_path)

    # Check if directory or JSON config
    if w_path.is_dir():
        config_file = w_path / "ensemble_config.json"
        if config_file.exists():
            w_path = config_file
        else:
            pths = sorted(list(w_path.glob("*.pth")))
            if pths:
                w_path = pths[0]
            else:
                raise FileNotFoundError(f"No model checkpoint or ensemble_config.json found in {weights_path}")

    if w_path.suffix.lower() == ".json":
        with open(w_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"[INFO] Loading Super-Ensemble ({len(cfg['models'])} models) via {w_path.name} on {device} ...")
        models_and_weights = []
        base_dir = w_path.parent
        for m_info in cfg["models"]:
            m = build_model(m_info["arch"])
            m_pth = base_dir / m_info["file"]
            state = torch.load(m_pth, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            elif isinstance(state, dict) and "model" in state:
                state = state["model"]
            m.load_state_dict(state)
            m.to(device)
            m.eval()
            models_and_weights.append((m, m_info["weight"]))
            print(f"  Loaded {m_info['name']} ({m_info['arch']}) with weight {m_info['weight']}")
        return EnsembleRunner(models_and_weights, device)

    state = torch.load(w_path, map_location=device)

    if isinstance(state, dict) and state.get("is_ensemble"):
        print(f"[INFO] Detected unified ensemble bundle with {len(state['models'])} models from {weights_path}")
        models_and_weights = []
        for (name, arch), weight in zip(state["models"], state["weights"]):
            m = build_model(arch)
            s = state["state_dicts"][name]
            m.load_state_dict(s)
            m.to(device)
            m.eval()
            models_and_weights.append((m, weight))
            print(f"  Loaded ensemble model {name} ({arch}) with weight {weight}")
        return EnsembleRunner(models_and_weights, device)

    model = build_model(backbone)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    elif isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state  # Full model saved

    model.to(device)
    model.eval()
    print(f"[INFO] Model ({backbone}) loaded from {weights_path} on {device}")
    return model


# -----------------------------------------------------------------------------
# TRANSFORMS - MUST match val/inference transforms used during training
# -----------------------------------------------------------------------------
def get_inference_transforms():
    """Standard inference transforms - matches val transforms in augmentations.py."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_transforms():
    """
    Three TTA variants: original, horizontal flip, brightness shift.
    If --tta is set, inference averages predictions from all three.
    """
    return [
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
    ]


# -----------------------------------------------------------------------------
# INFERENCE LOOP
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_inference_single(model, test_dir: str, transform, batch_size: int,
                         device: str, use_fp16: bool) -> pd.DataFrame:
    """Run inference with a single transform (standard or one TTA variant)."""
    dataset = TestDataset(test_dir, transform=transform)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    all_ids, all_probs = [], []
    total_batches = len(loader)

    for i, (batch_imgs, batch_ids) in enumerate(loader):
        batch_imgs = batch_imgs.to(device)

        if use_fp16 and device == "cuda":
            batch_imgs = batch_imgs.half()

        if isinstance(model, EnsembleRunner):
            probs = model(batch_imgs).float().cpu().numpy()
        else:
            logits = model(batch_imgs)              # (B, 1) or (B,)
            probs  = torch.sigmoid(logits).squeeze(-1).float().cpu().numpy()

        all_ids.extend(list(batch_ids))
        all_probs.extend(probs.tolist())

        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            print(f"  Batch [{i+1:4d}/{total_batches}] | "
                  f"Processed {(i+1)*batch_size} images", end="\r")

    print()
    return pd.DataFrame({"image_id": all_ids, "confidence": all_probs})


def run_inference(model, test_dir: str, batch_size: int, device: str,
                  use_fp16: bool, use_tta: bool, threshold: float) -> pd.DataFrame:
    """
    Main inference runner. Supports standard and TTA inference.
    Returns DataFrame with columns: image_id, label, confidence
    """
    if use_fp16 and device == "cuda":
        model = model.half()
        print("[INFO] FP16 inference enabled.")

    if use_tta:
        print(f"[INFO] Running TTA inference (3 augmentations)...")
        tta_transforms = get_tta_transforms()
        all_dfs = []
        for idx, tf in enumerate(tta_transforms):
            print(f"  TTA pass {idx+1}/3...")
            df = run_inference_single(model, test_dir, tf, batch_size, device, use_fp16)
            all_dfs.append(df.set_index("image_id")["confidence"])

        # Average all TTA predictions
        avg_probs = pd.concat(all_dfs, axis=1).mean(axis=1).reset_index()
        avg_probs.columns = ["image_id", "confidence"]
        result = avg_probs
    else:
        print(f"[INFO] Running standard inference...")
        result = run_inference_single(
            model, test_dir,
            get_inference_transforms(),
            batch_size, device, use_fp16
        )

    result["label"]      = (result["confidence"] >= threshold).astype(int)
    result["confidence"] = result["confidence"].round(6)
    result = result[["image_id", "label", "confidence"]]  # Required column order

    return result


# -----------------------------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------------------------
def validate_output(df: pd.DataFrame, expected_count: int = None):
    """Validate that predictions.csv meets hackathon requirements."""
    errors = []
    if "image_id"   not in df.columns: errors.append("Missing column: image_id")
    if "label"      not in df.columns: errors.append("Missing column: label")
    if "confidence" not in df.columns: errors.append("Missing column: confidence")

    if not errors:
        if not df["confidence"].between(0, 1).all():
            errors.append("Confidence values outside [0, 1]")
        if not df["label"].isin([0, 1]).all():
            errors.append("Labels must be 0 or 1 only")
        if df["image_id"].duplicated().any():
            errors.append(f"Duplicate image_ids: {df['image_id'].duplicated().sum()}")
        if expected_count and len(df) != expected_count:
            errors.append(f"Row count mismatch: got {len(df)}, expected {expected_count}")

    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        raise ValueError("Output validation failed. See errors above.")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Deepfake ML Hackathon - Inference Script (IIIT Bangalore)"
    )
    parser.add_argument("--test_dir",   required=True,
                        help="Path to test images directory (flat, no subfolders)")
    parser.add_argument("--weights",    required=True,
                        help="Path to model weights file (.pth)")
    parser.add_argument("--output",     default="predictions.csv",
                        help="Output CSV path (default: predictions.csv)")
    parser.add_argument("--backbone",   default=BACKBONE,
                        help=f"Model backbone (default: {BACKBONE})")
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE,
                        help=f"Inference batch size (default: {BATCH_SIZE})")
    parser.add_argument("--threshold",  type=float, default=THRESHOLD,
                        help=f"Fake probability threshold (default: {THRESHOLD})")
    parser.add_argument("--fp16",       action="store_true", default=True,
                        help="Use FP16 inference on CUDA (default: True)")
    parser.add_argument("--tta",        action="store_true", default=False,
                        help="Use test-time augmentation (slower, may improve AUC)")
    args = parser.parse_args()

    device = DEVICE
    print(f"\n{'='*60}")
    print(f"  Deepfake ML Hackathon - Inference")
    print(f"  Test dir  : {args.test_dir}")
    print(f"  Weights   : {args.weights}")
    print(f"  Output    : {args.output}")
    print(f"  Backbone  : {args.backbone}")
    print(f"  Device    : {device}")
    print(f"  Batch     : {args.batch_size}")
    print(f"  Threshold : {args.threshold}")
    print(f"  FP16      : {args.fp16 and device == 'cuda'}")
    print(f"  TTA       : {args.tta}")
    print(f"{'='*60}\n")

    # -- Load model --------------------------------------------------------
    model = load_model(args.weights, args.backbone, device)

    # -- Run inference -----------------------------------------------------
    predictions = run_inference(
        model,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        device=device,
        use_fp16=args.fp16,
        use_tta=args.tta,
        threshold=args.threshold,
    )

    # -- Validate ----------------------------------------------------------
    validate_output(predictions)

    # -- Save --------------------------------------------------------------
    predictions.to_csv(args.output, index=False)

    print(f"\n[DONE] Predictions saved to: {args.output}")
    print(f"  Total images   : {len(predictions)}")
    print(f"  Predicted FAKE : {predictions['label'].sum()} ({predictions['label'].mean()*100:.1f}%)")
    print(f"  Predicted REAL : {(predictions['label']==0).sum()}")
    print(f"  Mean confidence: {predictions['confidence'].mean():.4f}")
    print(f"  Min confidence : {predictions['confidence'].min():.4f}")
    print(f"  Max confidence : {predictions['confidence'].max():.4f}")


if __name__ == "__main__":
    main()
