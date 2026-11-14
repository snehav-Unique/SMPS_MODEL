
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

v2 = pd.read_csv(ROOT / "outputs/models/efficientnetv2_s/predictions_efficientnetv2_s.csv").set_index("image_id")["confidence"].rename("EffNetV2")
conv = pd.read_csv(ROOT / "outputs/predictions/predictions_convnext.csv").set_index("image_id")["confidence"].rename("ConvNeXt")
swin = pd.read_csv(ROOT / "outputs/models/swin_tiny/predictions_swin.csv").set_index("image_id")["confidence"].rename("Swin")
dense = pd.read_csv(ROOT / "outputs/models/densenet121/predictions_densenet121.csv").set_index("image_id")["confidence"].rename("DenseNet")
maxvit = pd.read_csv(ROOT / "outputs/predictions/predictions_maxvit.csv").set_index("image_id")["confidence"].rename("MaxViT")

df_m = pd.read_csv(ROOT / "outputs/predictions/predictions_maxvit.csv")
print("=== MaxViT-Tiny Summary ===")
print("  Total rows  :", len(df_m))
print("  Fake count  :", int(df_m["label"].sum()), f"({df_m['label'].mean()*100:.1f}%)")
print("  Real count  :", int((df_m["label"] == 0).sum()), f"({(1-df_m['label'].mean())*100:.1f}%)")
print(f"  Confidence range: [{df_m['confidence'].min():.4f}, {df_m['confidence'].max():.4f}], mean={df_m['confidence'].mean():.4f}")

merged = pd.concat([v2, conv, swin, dense, maxvit], axis=1)
print("\n=== Pearson Correlation Matrix (3,000 Test Predictions) ===")
print(merged.corr().round(4))
