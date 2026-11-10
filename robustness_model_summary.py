"""Add individual-model statistics to the completed robustness report."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

out = Path(__file__).resolve().parent / 'outputs' / 'robustness'
models = ['efficientnetv2_s', 'convnext_tiny', 'swin_tiny', 'densenet121']
conditions = ['clean', 'jpeg_q50', 'jpeg_q70', 'blur_sigma1.5', 'noise_sigma15']
rows = []
for model in models:
    clean = pd.read_csv(out / f'predictions_{model}_clean.csv')
    assert len(clean) == 200
    baseline = roc_auc_score(clean.true_label, clean.confidence)
    for condition in conditions:
        df = pd.read_csv(out / f'predictions_{model}_{condition}.csv')
        assert len(df) == 200 and df.image_id.tolist() == clean.image_id.tolist()
        y, p = df.true_label, df.confidence
        labels = p >= .5
        tn, fp, fn, tp = confusion_matrix(y, labels).ravel()
        auc = roc_auc_score(y, p)
        rows.append(dict(model=model, condition=condition, auc=auc, auc_drop=baseline-auc,
            accuracy=accuracy_score(y, labels), precision=precision_score(y,labels,zero_division=0),
            recall=recall_score(y,labels), f1=f1_score(y,labels), tn=tn, fp=fp, fn=fn, tp=tp))
stats = pd.DataFrame(rows)
stats.to_csv(out / 'per_model_statistics.csv', index=False)
text = '\n## Individual-model comparison\n\nAUC values on the same fixed sample; full classification metrics are in per_model_statistics.csv.\n\n'
text += '| Model | Clean | JPEG 50 | JPEG 70 | Blur | Noise |\n|---|---|---|---|---|---|\n'
for model in models:
    values = stats[stats.model == model].set_index('condition').loc[conditions, 'auc']
    text += '| ' + model + ' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |\n'
ensemble = pd.read_csv(out / 'statistics.csv')
text += '\n## AUC-drop uncertainty and interpretation\n\n| Condition | Absolute AUC drop | Paired 95% CI for drop |\n|---|---|---|\n'
for row in ensemble.iloc[1:].itertuples():
    text += f'| {row.condition} | {row.auc_drop:.4f} | {row.drop_ci_low:.4f} to {row.drop_ci_high:.4f} |\n'
text += '\nAll four observed drops are below 0.02. The upper confidence bounds for blur and noise exceed 0.02, so this sample does not establish that the underlying drops are below the target. Noise shows the clearest degradation: its paired interval excludes zero. JPEG improvements are small and their drop intervals include zero; they do not establish that compression improves performance.\n\nThe ensemble clean AUC (0.8813) is lower than EfficientNetV2-S alone (0.8861) on this sample. Ensemble superiority is therefore not demonstrated here; weights were kept fixed rather than tuned to this sample. Noise lowers ensemble accuracy from 87.0% to 83.5% and fake recall from 87.0% to 82.0%. These threshold-based metrics describe a separate effect from AUC.\n'
report = out / 'REPORT.md'
base = report.read_text(encoding='utf-8').split('\n## Individual-model comparison')[0]
report.write_text(base + text + '\nRegenerate this section with `python SMPS/robustness_model_summary.py` after the main evaluation.\n', encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
ensemble_stats = pd.read_csv(out / 'statistics.csv').set_index('condition')
for ax, condition in zip(axes, conditions):
    row = ensemble_stats.loc[condition]
    matrix = [[int(row.tn), int(row.fp)], [int(row.fn), int(row.tp)]]
    ax.imshow(matrix, cmap='Blues', vmin=0, vmax=100)
    for y in range(2):
        for x in range(2):
            ax.text(x, y, str(matrix[y][x]), ha='center', va='center', color='white' if matrix[y][x]>50 else 'black')
    ax.set_xticks([0,1], ['Real','Fake']); ax.set_yticks([0,1], ['Real','Fake'])
    ax.set_title(condition, fontsize=10); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
fig.tight_layout(); fig.savefig(out/'confusion_matrices.png', dpi=180); plt.close(fig)
fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
for ax, condition in zip(axes, conditions):
    ax.imshow(Image.open(out/f'example_{condition}.png')); ax.set_title(condition, fontsize=10); ax.axis('off')
fig.tight_layout(); fig.savefig(out/'corruption_examples.png', dpi=180); plt.close(fig)
with report.open('a', encoding='utf-8') as f:
    f.write('\n## Charts\n\n![AUC comparison](auc_comparison.png)\n\n![Confusion matrices](confusion_matrices.png)\n\n![Corruption examples](corruption_examples.png)\n')
print(stats.to_string(index=False))
