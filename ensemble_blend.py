"""
ensemble_blend.py
=================
Super-Ensemble Blending across Top 5 Paradigms (Secures +2 Bonus Points for Diverse Architectural Ensemble)

Performs performance-weighted probability blending across our 5 top models:
  1. EfficientNetV2-S (Fused-MBConv CNN)          - Weight: 0.40  (Peak Val AUC: 0.9025)
  2. ConvNeXt-Tiny (Modern 7x7 ConvNet)            - Weight: 0.25  (Peak Val AUC: 0.8973)
  3. Swin-Tiny (Hierarchical ViT)                  - Weight: 0.15  (Peak Val AUC: 0.8953)
  4. MaxViT-Tiny (Multi-Axis Vision Transformer)   - Weight: 0.12  (Peak Val AUC: 0.8902)
  5. DenseNet-121 (Dense Feature-Reuse)           - Weight: 0.08  (Peak Val AUC: 0.8905)

Outputs:
  predictions.csv          — Final official hackathon submission file (3,000 test images)
  predictions_ensemble.csv — Benchmark reference copy
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

def find_file(relative_paths):
    for rel in relative_paths:
        p = PROJECT_ROOT / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find any of: {relative_paths}")

def run_ensemble():
    v2_path = find_file([
        Path("efficientnetv2_s_results/efficientnetv2_s_results/predictions_efficientnetv2_s.csv"),
        Path("efficientnetv2_s_results/predictions_efficientnetv2_s.csv"),
    ])
    conv_path = find_file([
        Path("predictions_convnext.csv"),
        Path("convnext_results/convnext_results/predictions_convnext.csv"),
        Path("convnext_results/predictions_convnext.csv"),
    ])
    swin_path = find_file([
        Path("swin_results/swin_results/predictions_swin.csv"),
        Path("swin_results/predictions_swin.csv"),
    ])
    maxvit_path = find_file([
        Path("predictions_maxvit.csv"),
        Path("maxvit_results/content/maxvit_results/predictions_maxvit.csv"),
    ])
    dense_path = find_file([
        Path("densenet121_results/densenet121_results/predictions_densenet121.csv"),
        Path("densenet121_results/predictions_densenet121.csv"),
    ])

    print("[INFO] Loading individual model predictions for 5-Model Super-Ensemble:")
    print(f"  - EffNetV2-S (0.40) : {v2_path}")
    print(f"  - ConvNeXt   (0.25) : {conv_path}")
    print(f"  - Swin-Tiny  (0.15) : {swin_path}")
    print(f"  - MaxViT     (0.12) : {maxvit_path}")
    print(f"  - DenseNet   (0.08) : {dense_path}")

    v2     = pd.read_csv(v2_path).set_index("image_id")
    conv   = pd.read_csv(conv_path).set_index("image_id")
    swin   = pd.read_csv(swin_path).set_index("image_id")
    maxvit = pd.read_csv(maxvit_path).set_index("image_id")
    dense  = pd.read_csv(dense_path).set_index("image_id")

    # Align IDs across all 5 models
    common_ids = v2.index.intersection(conv.index).intersection(swin.index).intersection(maxvit.index).intersection(dense.index)
    assert len(common_ids) == 3000, f"Expected 3000 common IDs, got {len(common_ids)}"

    v2     = v2.loc[common_ids]
    conv   = conv.loc[common_ids]
    swin   = swin.loc[common_ids]
    maxvit = maxvit.loc[common_ids]
    dense  = dense.loc[common_ids]

    # Performance-weighted 5-model probability blending
    ensemble_conf = (
        0.40 * v2["confidence"] +
        0.25 * conv["confidence"] +
        0.15 * swin["confidence"] +
        0.12 * maxvit["confidence"] +
        0.08 * dense["confidence"]
    )

    ensemble_df = pd.DataFrame({
        "image_id":   ensemble_conf.index,
        "label":      (ensemble_conf >= 0.5).astype(int),
        "confidence": ensemble_conf.round(6),
    }).reset_index(drop=True)

    # Canary validation checks (prevents hackathon disqualification)
    assert len(ensemble_df) == 3000, f"Row count mismatch: {len(ensemble_df)}"
    assert list(ensemble_df.columns) == ["image_id", "label", "confidence"], f"Bad columns: {ensemble_df.columns}"
    assert ensemble_df["confidence"].between(0.0, 1.0).all(), "Confidence values outside [0, 1]"
    assert ensemble_df.isnull().sum().sum() == 0, "Missing/null values detected"
    assert ensemble_df["image_id"].nunique() == 3000, "Duplicate image_ids found"

    # Save to both predictions.csv (submission) and predictions_ensemble.csv
    sub_path = PROJECT_ROOT / "predictions.csv"
    ens_path = PROJECT_ROOT / "predictions_ensemble.csv"
    ensemble_df.to_csv(sub_path, index=False)
    ensemble_df.to_csv(ens_path, index=False)

    print(f"\n[SUCCESS] Saved 5-Model Super-Ensemble to:")
    print(f"  1. {sub_path} (Official Submission)")
    print(f"  2. {ens_path} (Ensemble Archive)")
    print(f"  Total samples : {len(ensemble_df)}")
    print(f"  Fake count (1): {int(ensemble_df['label'].sum())} ({ensemble_df['label'].mean()*100:.1f}%)")
    print(f"  Real count (0): {int((ensemble_df['label'] == 0).sum())} ({(1-ensemble_df['label'].mean())*100:.1f}%)")
    print(f"  Confidence range: [{ensemble_df['confidence'].min():.4f}, {ensemble_df['confidence'].max():.4f}], mean={ensemble_df['confidence'].mean():.4f}")

if __name__ == "__main__":
    run_ensemble()
