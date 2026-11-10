"""
src/model.py
============
Member 2 owns this file, but Member 1 provides the base skeleton.

Defines build_model() — the single source of truth for the architecture.
Both train.py and inference_script.py import from here.

Supported backbones (set in config.py → BACKBONE):
    "efficientnet_b4"   — Primary model (best accuracy vs speed trade-off)
    "efficientnet_b2"   — Lighter, faster inference (use if B4 OOMs on T4)
    "convnext_tiny"     — Alternative for ensemble diversity
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import timm

from config import BACKBONE, DEVICE


def build_model(backbone: str = None, pretrained: bool = True) -> nn.Module:
    """
    Builds and returns the deepfake detection model.

    Args:
        backbone  : Backbone name (default: config.BACKBONE).
        pretrained: Load ImageNet weights (True during training, False at inference
                    since weights are loaded separately from .pth file).

    Returns:
        nn.Module with a single binary output logit.
    """
    backbone = backbone or BACKBONE

    # ── EfficientNet-B4 (Primary) ──────────────────────────────────────────
    if backbone == "efficientnet_b4":
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b4(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 1),   # Single logit → sigmoid → fake probability
        )
        return model

    # ── EfficientNet-B2 (Lighter alternative) ─────────────────────────────
    elif backbone == "efficientnet_b2":
        from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1),
        )
        return model

    # ── ConvNeXt-Tiny (Ensemble diversity via timm) ────────────────────────
    elif backbone == "convnext_tiny":
        model = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=1,       # Binary output directly
        )
        return model

    # ── EfficientNetV2-S (via timm, strong alternative) ───────────────────
    elif backbone == "efficientnetv2_s":
        model = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=pretrained,
            num_classes=1,
        )
        return model

    else:
        raise ValueError(f"Unknown backbone: {backbone!r}. "
                         f"Choose from: efficientnet_b4, efficientnet_b2, "
                         f"convnext_tiny, efficientnetv2_s")


def load_model_weights(model: nn.Module, weights_path: str) -> nn.Module:
    """
    Loads a saved state dict into the model.
    Handles both plain state_dicts and checkpoint dicts with 'state_dict' key.

    Args:
        model        : An uninitialised model (from build_model(pretrained=False)).
        weights_path : Path to the .pth file.

    Returns:
        model loaded onto DEVICE in eval mode.
    """
    state = torch.load(weights_path, map_location=DEVICE)

    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
        print(f"[INFO] Loaded 'state_dict' key from checkpoint: {weights_path}")
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
        print(f"[INFO] Loaded 'model' key from checkpoint: {weights_path}")
    elif isinstance(state, dict):
        model.load_state_dict(state)
        print(f"[INFO] Loaded state dict from: {weights_path}")
    else:
        # Full model saved — replace directly
        model = state
        print(f"[INFO] Loaded full model object from: {weights_path}")

    model.to(DEVICE)
    model.eval()
    return model


def count_parameters(model: nn.Module) -> int:
    """Returns the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check — run: python src/model.py
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    for name in ["efficientnet_b4", "efficientnet_b2", "convnext_tiny"]:
        m = build_model(backbone=name, pretrained=False)
        params = count_parameters(m)
        x = torch.randn(2, 3, 224, 224)
        out = m(x)
        print(f"  {name:25s} | params={params:,} | output shape: {out.shape}")
