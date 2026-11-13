"""
src/gradcam_vis.py
==================
Member 3 owns this file. Member 1 provides the scaffold.

Generates Grad-CAM visualizations on a sample of test images.
Earns +2 bonus points per evaluation metric doc.

Usage:
    python src/gradcam_vis.py \
        --weights weights/best_model.pth \
        --test_dir hackathon_dataset/test/images/ \
        --n_samples 20

Outputs:
    outputs/gradcam/   — PNG overlays for n_samples images
"""

import sys
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torchvision import transforms

from config import (
    BACKBONE, OUTPUTS_DIR, WEIGHTS_DIR, TEST_DIR,
    IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, SEED, DEVICE
)
from src.model import build_model, load_model_weights

GRADCAM_DIR = OUTPUTS_DIR / "gradcam"


def get_target_layer(model, backbone: str):
    """
    Returns the last convolutional layer for Grad-CAM hook.
    This varies by architecture.
    """
    if backbone in ("efficientnet_b4", "efficientnet_b2"):
        return model.features[-1]       # Last conv block
    elif backbone == "convnext_tiny":
        return model.stages[-1]         # Last ConvNeXt stage
    elif backbone == "efficientnetv2_s":
        return model.blocks[-1]
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


def generate_gradcam(model, img_tensor, target_layer, device):
    """
    Generates a Grad-CAM heatmap for the given image tensor.

    Returns:
        heatmap: np.ndarray of shape (H, W) in [0,1]
    """
    activations = []
    gradients   = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(device)
    img_tensor.requires_grad_(False)

    logit = model(img_tensor)
    prob  = torch.sigmoid(logit).item()

    model.zero_grad()
    logit.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    if not activations or not gradients:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE)), prob

    act = activations[0].squeeze(0)   # (C, H, W)
    grd = gradients[0].squeeze(0)     # (C, H, W)

    weights = grd.mean(dim=(1, 2))    # GAP over spatial dims → (C,)
    cam = (weights[:, None, None] * act).sum(dim=0)  # (H, W)
    cam = torch.relu(cam)

    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize to IMAGE_SIZE×IMAGE_SIZE
    cam_np = cam.cpu().numpy()
    cam_pil = Image.fromarray((cam_np * 255).astype(np.uint8)).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
    )
    heatmap = np.array(cam_pil) / 255.0

    return heatmap, prob


def overlay_heatmap(original_img_pil, heatmap, alpha=0.4):
    """Overlay a Grad-CAM heatmap on the original image."""
    cmap    = plt.get_cmap("jet")
    heatmap_rgb = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    original_np = np.array(original_img_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))
    overlay = (alpha * heatmap_rgb + (1 - alpha) * original_np).astype(np.uint8)
    return Image.fromarray(overlay)


def main(args):
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)
    print(f"[GradCAM] Loading model: {args.backbone} from {args.weights}")
    model = build_model(backbone=args.backbone, pretrained=False)
    model = load_model_weights(model, args.weights)

    target_layer = get_target_layer(model, args.backbone)

    # ── Sample images ──────────────────────────────────────────────────────
    test_dir = Path(args.test_dir)
    img_paths = sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.png"))
    random.seed(SEED)
    sample_paths = random.sample(img_paths, min(args.n_samples, len(img_paths)))

    # ── Transforms (same as inference) ────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    print(f"[GradCAM] Generating {len(sample_paths)} visualizations...")

    for i, img_path in enumerate(sample_paths):
        original = Image.open(img_path).convert("RGB")
        tensor   = transform(original)

        heatmap, prob = generate_gradcam(model, tensor, target_layer, device)
        overlay = overlay_heatmap(original, heatmap)
        label   = "FAKE" if prob >= 0.5 else "REAL"

        # Save side-by-side plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"{img_path.stem[:20]}...  |  Pred: {label} (conf={prob:.3f})", fontsize=11)

        axes[0].imshow(original.resize((IMAGE_SIZE, IMAGE_SIZE)))
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = GRADCAM_DIR / f"gradcam_{i+1:03d}_{label}_{prob:.2f}.png"
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()

        print(f"  [{i+1:3d}/{len(sample_paths)}] {label} conf={prob:.3f} -> {save_path.name}")

    print(f"\n[DONE] Grad-CAM visualizations saved to: {GRADCAM_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM visualizer for deepfake detection")
    parser.add_argument("--weights",   required=True, help="Path to model weights .pth")
    parser.add_argument("--test_dir",  default=str(TEST_DIR), help="Path to test images dir")
    parser.add_argument("--backbone",  default=BACKBONE, help="Model backbone name")
    parser.add_argument("--n_samples", type=int, default=20, help="Number of images to visualize")
    args = parser.parse_args()
    main(args)
