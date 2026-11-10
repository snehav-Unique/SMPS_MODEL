# Evaluation Metric Guide

**Deepfake ML Hackathon — IIIT Bangalore**

---

## How Evaluation Works

This hackathon uses a **non-leaderboard, sealed evaluation** model:

- All submissions are collected until the deadline.  
- **No intermediate scores are published** during the hackathon.  
- All results are announced simultaneously on Results Day.

This design rewards genuine model quality over leaderboard gaming.

---

## Primary Metric: AUC-ROC

Your model is evaluated using **Area Under the Receiver Operating Characteristic Curve (AUC-ROC)**.

### Why AUC-ROC?

| Property | Benefit |
| :---- | :---- |
| Threshold-independent | Evaluates the full ranking of predictions, not a single cutoff |
| Game-proof | Cannot be maximized by always predicting one class |
| Confidence-sensitive | Rewards models that are *confidently correct* |
| Imbalance-robust | Works well even with slight class imbalance |

### How to Interpret

| AUC-ROC | Meaning |
| :---- | :---- |
| 1.0 | Perfect classifier |
| 0.9 – 1.0 | Excellent |
| 0.7 – 0.9 | Good |
| 0.5 – 0.7 | Weak (close to random) |
| 0.5 | Random guessing |
| \< 0.5 | Worse than random |

### What It Uses From Your predictions.csv

The `confidence` column drives the AUC score — it is the predicted probability that an image is **FAKE** (label=1). This must be a float in \[0.0, 1.0\].

```py
# How we compute your score:
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(true_labels, your_confidence_column)
```

---

## Full Scoring Breakdown (100 pts Total)

### 1\. Accuracy on Test Set — 50 pts

Computed from your `predictions.csv` against our private `test_labels.csv`.

```
normalized_score = max(0, (AUC - 0.5) / 0.5) × 50
```

&nbsp;

| AUC-ROC | Score |
| :---- | :---- |
| 1.00 | 50 / 50 |
| 0.90 | 40 / 50 |
| 0.80 | 30 / 50 |
| 0.70 | 20 / 50 |
| 0.60 | 10 / 50 |
| 0.50 | 0 / 50 |

---

### 2\. Model Efficiency — 20 pts

Measured by running your `inference_script.py` on our GPU (NVIDIA T4) against the full test set.

| Inference Time | Score |
| :---- | :---- |
| \< 5 minutes | 20 / 20 |
| 5 – 10 minutes | 15 / 20 |
| 10 – 20 minutes | 10 / 20 |
| 20 – 40 minutes | 5 / 20 |
| \> 40 minutes or Out-of-Memory | 0 / 20 |

**Tips:**

- Use batched inference (don't process one image at a time)  
- Use GPU-optimized preprocessing (`torchvision.transforms`)  
- Consider FP16 inference (`model.half()` in PyTorch)  
- Avoid loading the full model for every image

---

### 3\. Generalization & Robustness — 20 pts

Your model is evaluated on **multiple variants** of the test set to assess stability under real-world conditions. The exact variants are not disclosed — this is intentional.

```
robustness_score = weighted AUC across test variants (normalized to 20 pts)
```

**What earns a high robustness score:**

- Model performance that doesn't collapse under minor image degradations  
- Predictions based on semantic deepfake artifacts, not pixel-perfect memorization  
- Confidence distributions that are stable across image quality levels

**What hurts robustness:**

- Models that rely solely on JPEG block artifacts or specific noise patterns  
- Overfit models that memorize training set statistics

---

### 4\. Documentation Quality — 10 pts (Manual)

Judged manually by the organizing panel using this rubric:

| Criteria | Points |
| :---- | :---- |
| Clear problem statement and approach rationale | 2 pts |
| Model architecture described (diagram/table welcome) | 2 pts |
| Hyperparameter selection justified (ablation or evidence) | 2 pts |
| Training pipeline explained (splits, augmentation, loss, optimizer) | 2 pts |
| Challenges \+ honest reflection on limitations | 1 pt |
| Reproducibility instructions complete and clear | 1 pt |

---

## Bonus Points

| Bonus | Points |
| :---- | :---- |
| Grad-CAM / SHAP / attention map visualizations | \+2 |
| Ensemble of diverse architectures (well-documented) | \+2 |
| Model exported for both PyTorch and TensorFlow | \+1 |

---

## Penalties

| Violation | Penalty |
| :---- | :---- |
| Inference script doesn't reproduce submitted predictions | –5 pts |
| Model OOM or timeout during inference | Efficiency score \= 0 |
| Missing rows in predictions.csv | Scored as incorrect |

---

## Disqualification Triggers

- Code plagiarism (detected via automated tools \+ manual review)  
- Use of external training datasets  
- predictions.csv appears to be hardcoded (not model-generated)  
- Submission fails all canary checks (internal integrity test)

---

## Tiebreaker Order

If two teams have the same final score:

1. Higher Robustness sub-score  
2. Higher Efficiency sub-score  
3. Higher Documentation sub-score  
4. Earlier submission timestamp (last Drive upload before deadline)

---

## Recommended Validation Strategy

Since there is no public leaderboard, create your own held-out validation set from the training data:

```py
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    train_metadata,
    test_size=0.2,
    stratify=train_metadata['label'],
    random_state=42
)
```

Track your AUC-ROC on this local validation set as your primary development metric.

---

## Sample Scoring Code (For Your Own Validation)

```py
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# Your validation predictions
val_preds = pd.read_csv('my_val_predictions.csv')
val_truth = pd.read_csv('my_val_labels.csv')

merged = val_preds.merge(val_truth, on='image_id')

auc = roc_auc_score(merged['true_label'], merged['confidence'])
f1  = f1_score(merged['true_label'], merged['label'])
acc = accuracy_score(merged['true_label'], merged['label'])

print(f"AUC-ROC : {auc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {acc:.4f}")
```

---

*Deepfake ML Hackathon — IIIT Bangalore Organizing Team*