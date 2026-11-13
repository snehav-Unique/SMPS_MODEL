"""
create_ensemble_bundle.py
=========================
Consolidates the 4 best model weights into a single unified ensemble bundle:
  1. EfficientNetV2-S (tf_efficientnetv2_s)  - weight: 0.35
  2. ConvNeXt-Tiny    (convnext_tiny)        - weight: 0.30
  3. Swin-Tiny        (swin_tiny_patch4_window7_224) - weight: 0.20
  4. DenseNet-121     (densenet121)          - weight: 0.15

Also copies individual checkpoints into `weights/` for unified packaging.
Allows `inference_script.py` to seamlessly accept a single `--weights weights/ensemble_bundle.pth`
file and run the full ensemble pipeline, avoiding any CLI mismatches.
"""

import shutil
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "name": "efficientnetv2_s",
        "arch": "tf_efficientnetv2_s",
        "src": PROJECT_ROOT / "efficientnetv2_s_results/efficientnetv2_s_results/efficientnetv2_s_best.pth",
        "dst": WEIGHTS_DIR / "efficientnetv2_s_best.pth",
        "weight": 0.40,
    },
    {
        "name": "convnext_tiny",
        "arch": "convnext_tiny",
        "src": WEIGHTS_DIR / "convnext_tiny_best.pth",
        "dst": WEIGHTS_DIR / "convnext_tiny_best.pth",
        "weight": 0.25,
    },
    {
        "name": "swin_tiny",
        "arch": "swin_tiny_patch4_window7_224",
        "src": PROJECT_ROOT / "swin_results/swin_results/swin_tiny_best.pth",
        "dst": WEIGHTS_DIR / "swin_tiny_best.pth",
        "weight": 0.15,
    },
    {
        "name": "maxvit_tiny",
        "arch": "maxvit_tiny_tf_224",
        "src": WEIGHTS_DIR / "maxvit_tiny_best.pth",
        "dst": WEIGHTS_DIR / "maxvit_tiny_best.pth",
        "weight": 0.12,
    },
    {
        "name": "densenet121",
        "arch": "densenet121",
        "src": PROJECT_ROOT / "densenet121_results/densenet121_results/densenet121_best.pth",
        "dst": WEIGHTS_DIR / "densenet121_best.pth",
        "weight": 0.08,
    },
]

def bundle():
    print("[INFO] Consolidating model checkpoints into weights/ ...")
    bundle_dict = {
        "is_ensemble": True,
        "models": [],
        "weights": [],
        "state_dicts": {},
    }

    for m in MODELS:
        src = m["src"]
        dst = m["dst"]
        if src != dst and not dst.exists():
            print(f"  Copying {src.name} -> {dst}")
            shutil.copy2(src, dst)
        
        print(f"  Loading state dict for {m['name']} ({m['arch']})...")
        state = torch.load(dst, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]

        bundle_dict["models"].append((m["name"], m["arch"]))
        bundle_dict["weights"].append(m["weight"])
        bundle_dict["state_dicts"][m["name"]] = state

    bundle_path = WEIGHTS_DIR / "ensemble_bundle.pth"
    print(f"\n[INFO] Saving unified bundle to: {bundle_path} ...")
    torch.save(bundle_dict, bundle_path)
    size_mb = bundle_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Bundle created! Size: {size_mb:.1f} MB (well below 5 GB hackathon limit)")

if __name__ == "__main__":
    bundle()
