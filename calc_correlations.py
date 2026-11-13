import pandas as pd
from pathlib import Path

ROOT = Path(r"D:\RVCE\Sem 3\Deepfake ML Hackathon\SMPS-Model\SMPS_MODEL")

v2 = pd.read_csv(ROOT / "efficientnetv2_s_results/efficientnetv2_s_results/predictions_efficientnetv2_s.csv").set_index("image_id")["confidence"].rename("EffNetV2")
conv = pd.read_csv(ROOT / "predictions_convnext.csv").set_index("image_id")["confidence"].rename("ConvNeXt")
swin = pd.read_csv(ROOT / "swin_results/swin_results/predictions_swin.csv").set_index("image_id")["confidence"].rename("Swin")
dense = pd.read_csv(ROOT / "densenet121_results/densenet121_results/predictions_densenet121.csv").set_index("image_id")["confidence"].rename("DenseNet")
maxvit = pd.read_csv(ROOT / "predictions_maxvit.csv").set_index("image_id")["confidence"].rename("MaxViT")

df_m = pd.read_csv(ROOT / "predictions_maxvit.csv")
print("=== MaxViT-Tiny Summary ===")
print("  Total rows  :", len(df_m))
print("  Fake count  :", int(df_m["label"].sum()), f"({df_m['label'].mean()*100:.1f}%)")
print("  Real count  :", int((df_m["label"] == 0).sum()), f"({(1-df_m['label'].mean())*100:.1f}%)")
print(f"  Confidence range: [{df_m['confidence'].min():.4f}, {df_m['confidence'].max():.4f}], mean={df_m['confidence'].mean():.4f}")

merged = pd.concat([v2, conv, swin, dense, maxvit], axis=1)
print("\n=== Pearson Correlation Matrix (3,000 Test Predictions) ===")
print(merged.corr().round(4))
