"""
inference_template.py
=====================
OFFICIAL INFERENCE SCRIPT TEMPLATE — Deepfake ML Hackathon
============================================================

Fill in the sections marked with TODO.
This script MUST work exactly as shown when called:

    python inference_script.py \
        --test_dir ./test/images/ \
        --weights ./model_weights.pt \
        --output predictions.csv

Requirements:
    - Processes ALL images in --test_dir
    - Outputs CSV with columns: image_id, label, confidence
    - image_id = filename WITHOUT .jpg extension
    - label    = 0 (Real) or 1 (Fake)
    - confidence = float [0.0, 1.0] (probability of being FAKE)
    - Must run within 60 minutes on an NVIDIA T4 GPU
    - No hardcoded paths
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ── PyTorch imports (remove if using TensorFlow) ──────────────────────────────
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — Adjust these to match your training setup
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 224      # Must match what you trained with
BATCH_SIZE   = 64       # Increase for speed; decrease if OOM
NUM_WORKERS  = 4        # DataLoader workers
THRESHOLD    = 0.5      # Confidence threshold for binary label
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# DATASET — Reads test images from a flat directory
# ─────────────────────────────────────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(
            list(self.image_dir.glob("*.jpg")) +
            list(self.image_dir.glob("*.jpeg")) +
            list(self.image_dir.glob("*.png"))
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        image_id = img_path.stem  # Filename without extension
        return img, image_id


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — TODO: Replace this with YOUR model architecture
# ─────────────────────────────────────────────────────────────────────────────
def build_model() -> nn.Module:
    """
    TODO: Replace this with your actual model architecture.
    Must return a PyTorch nn.Module.

    Example below uses EfficientNet-B4 with a binary head.
    Replace with whatever you trained.
    """
    # ── EXAMPLE (replace with your architecture) ──
    from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
    model = efficientnet_b4(weights=None)  # No pretrained weights — loaded separately
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 1),  # Single logit → sigmoid → probability
    )
    return model
    # ── END EXAMPLE ──


def load_model(weights_path: str) -> nn.Module:
    """
    TODO: Adjust if your model saving format differs.
    Common patterns:
        torch.load(path)                        → full model object
        model.load_state_dict(torch.load(path)) → state dict only
    """
    model = build_model()
    state = torch.load(weights_path, map_location=DEVICE)
    # Handle both full model and state_dict saves
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state  # Full model saved directly
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Model loaded from {weights_path} on {DEVICE}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS — TODO: Use the SAME transforms as your training (no augmentation)
# ─────────────────────────────────────────────────────────────────────────────
def get_inference_transforms():
    """
    TODO: Use the same normalization you used during training.
    If you normalized with ImageNet stats, keep as-is.
    If you used different stats, update mean and std.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet default
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model: nn.Module, test_dir: str) -> pd.DataFrame:
    transform = get_inference_transforms()
    dataset   = TestDataset(test_dir, transform=transform)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)

    all_ids         = []
    all_labels      = []
    all_confidences = []

    print(f"[INFO] Running inference on {len(dataset)} images (batch_size={BATCH_SIZE})...")

    with torch.no_grad():
        for batch_imgs, batch_ids in loader:
            batch_imgs = batch_imgs.to(DEVICE)

            # ── TODO: Adjust forward pass if your model output shape differs ──
            logits = model(batch_imgs)          # Shape: (B, 1) or (B,)
            probs  = torch.sigmoid(logits).squeeze(-1).cpu().numpy()  # Fake probability

            labels = (probs >= THRESHOLD).astype(int)

            all_ids.extend(list(batch_ids))
            all_labels.extend(labels.tolist())
            all_confidences.extend(probs.tolist())

    return pd.DataFrame({
        "image_id":   all_ids,
        "label":      all_labels,
        "confidence": all_confidences,
    })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Deepfake ML Hackathon — Inference Script")
    parser.add_argument("--test_dir", required=True,       help="Path to test images directory")
    parser.add_argument("--weights",  required=True,       help="Path to model weights file")
    parser.add_argument("--output",   default="predictions.csv", help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threshold",  type=float, default=THRESHOLD)
    args = parser.parse_args()

    global BATCH_SIZE, THRESHOLD
    BATCH_SIZE = args.batch_size
    THRESHOLD  = args.threshold

    # Load model
    model = load_model(args.weights)

    # Run inference
    predictions = run_inference(model, args.test_dir)

    # Validate output
    assert "image_id"   in predictions.columns, "Missing image_id column"
    assert "label"      in predictions.columns, "Missing label column"
    assert "confidence" in predictions.columns, "Missing confidence column"
    assert predictions["confidence"].between(0, 1).all(), "Confidence values must be in [0, 1]"
    assert predictions["label"].isin([0, 1]).all(),       "Labels must be 0 or 1"

    # Save
    predictions.to_csv(args.output, index=False)
    print(f"[DONE] Predictions saved to {args.output}")
    print(f"  Total images: {len(predictions)}")
    print(f"  Predicted FAKE: {predictions['label'].sum()} ({predictions['label'].mean()*100:.1f}%)")
    print(f"  Predicted REAL: {(predictions['label']==0).sum()}")
    print(f"  Mean confidence: {predictions['confidence'].mean():.4f}")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# ── TENSORFLOW ALTERNATIVE ──
# If using TensorFlow instead of PyTorch, replace the code above with:
#
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# from pathlib import Path
#
# def load_model_tf(weights_path):
#     if weights_path.endswith('.h5'):
#         return tf.keras.models.load_model(weights_path)
#     else:
#         return tf.saved_model.load(weights_path)
#
# def run_inference_tf(model, test_dir, batch_size=64):
#     test_dir = Path(test_dir)
#     image_paths = sorted(test_dir.glob("*.jpg"))
#     all_ids, all_labels, all_confs = [], [], []
#     for i in range(0, len(image_paths), batch_size):
#         batch = image_paths[i:i+batch_size]
#         imgs = []
#         for p in batch:
#             img = tf.keras.preprocessing.image.load_img(str(p), target_size=(224, 224))
#             img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
#             imgs.append(img)
#         imgs = np.stack(imgs)
#         preds = model(imgs, training=False).numpy().flatten()
#         labels = (preds >= 0.5).astype(int)
#         all_ids.extend([p.stem for p in batch])
#         all_labels.extend(labels.tolist())
#         all_confs.extend(preds.tolist())
#     return pd.DataFrame({"image_id": all_ids, "label": all_labels, "confidence": all_confs})
# ─────────────────────────────────────────────────────────────────────────────
