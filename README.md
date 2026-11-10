# Deepfake Detection — IIIT Bangalore Hackathon
# Team: SMPS

## Completed robustness evaluation

See [robustness results and reproduction instructions](ROBUSTNESS.md), the
[full statistics report](outputs/robustness/REPORT.md), and
[completed checkpoints](outputs/robustness/CHECKPOINTS.md).
Dataset images and model checkpoints are kept outside Git; obtain them separately.

## Project Structure

```
SMPS/
├── config.py               ← Central config — edit DATASET_ROOT here
├── create_split.py         ← Step 1: creates data/train_split.csv + val_split.csv
├── eda.py                  ← Step 2: exploratory data analysis
├── inference_script.py     ← Official hackathon inference script (self-contained)
├── requirements.txt        ← pip dependencies
├── src/
│   ├── augmentations.py    ← Train / val / TTA transforms
│   ├── dataset.py          ← DeepfakeDataset + TestDataset classes
│   ├── model.py            ← build_model() — EfficientNet-B4 / ConvNeXt-Tiny
│   ├── train.py            ← Full training loop
│   ├── validate.py         ← Standalone validation + optimal threshold finder
│   ├── gradcam_vis.py      ← Grad-CAM visualizer (+2 bonus pts)
│   └── plot_training.py    ← Loss/AUC curve plotter
├── data/                   ← Generated CSV splits (gitignore large files)
├── weights/                ← Saved model .pth files (gitignore)
└── outputs/                ← CSV logs, plots, predictions
    ├── eda/
    ├── gradcam/
    └── train_log.csv
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure paths
Edit `config.py` → set `DATASET_ROOT` to your local hackathon_dataset path.

---

## Step-by-Step Execution

### Step 1 — Create train/val split
```bash
python create_split.py
```
Outputs: `data/train_split.csv`, `data/val_split.csv`

### Step 2 — Run EDA
```bash
python eda.py
```
Outputs: `outputs/eda/sample_grid.png`, `brightness_distribution.png`, `eda_summary.csv`

### Step 3 — Verify model architecture
```bash
python src/model.py
```

### Step 4 — Train the model
```bash
python src/train.py
```
Or with custom args:
```bash
python src/train.py --backbone efficientnet_b4 --epochs 25 --lr 1e-4 --batch_size 32
```
Outputs: `weights/best_model.pth`, `outputs/train_log.csv`

### Step 5 — Plot training curves
```bash
python src/plot_training.py
```
Outputs: `outputs/training_curves.png`

### Step 6 — Validate on held-out val set
```bash
python src/validate.py --weights weights/best_model.pth
```

### Step 7 — Generate Grad-CAM visualizations (bonus)
```bash
python src/gradcam_vis.py --weights weights/best_model.pth --test_dir <path/to/test/images> --n_samples 20
```

### Step 8 — Run official inference (generates predictions.csv)
```bash
python inference_script.py \
    --test_dir "D:\RVCE\Sem 3\Deepfake ML Hackathon\Deepfake ML Hackathon - Participant Package-20261105T145705Z-1-001\Deepfake ML Hackathon - Participant Package\hackathon_dataset\hackathon_dataset\test\images" \
    --weights weights/best_model.pth \
    --output predictions.csv
```

---

## Evaluation Metric

**Primary:** AUC-ROC on `confidence` column  
**Scoring:** `max(0, (AUC - 0.5) / 0.5) × 50 pts`

| AUC | Score |
|:----|:------|
| 1.00 | 50/50 |
| 0.90 | 40/50 |
| 0.80 | 30/50 |

---

## Rules

- Only provided dataset used for training ✓
- Pre-trained ImageNet weights allowed ✓
- No hardcoded paths in inference script ✓
- Reproducible via `inference_script.py` ✓
