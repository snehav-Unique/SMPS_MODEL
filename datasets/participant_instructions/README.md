# Deepfake ML Hackathon — Participant Dataset README

**International Institute of Information Technology (IIIT), Bangalore**

---

## Overview

Welcome\! This package contains everything you need to participate in the Deepfake ML Hackathon. Your goal is to train a binary classifier that distinguishes **real face images (label=0)** from **deepfake face images (label=1)**.

---

## Dataset Structure

```
hackathon_dataset/
├── train/
│   ├── real/          ← Real face images  (label = 0)
│   └── fake/          ← Deepfake images   (label = 1)
├── test/
│   └── images/        ← Unlabelled test images (NO subfolders)
├── train_metadata.csv ← image_id, label for training set
├── sample_submission.csv
├── README.md          ← You are here
└── evaluation_metric.md
```

> **Important:** The `test/images/` folder contains no label information — filenames are random UUIDs. You must generate predictions for every image in this folder.

---

## Label Convention

| Label | Meaning |
| :---- | :---- |
| `0` | REAL — genuine, unmanipulated face |
| `1` | FAKE — synthetically generated or manipulated deepfake |

---

## Training Data

- `train/real/` and `train/fake/` — face-cropped images, 224×224 pixels, JPEG format.  
- `train_metadata.csv` — maps each `image_id` (filename without `.jpg`) to its binary label.  
- All images are pre-aligned (face-centered and cropped).  
- **Note:** Real-world training data contains inherent imperfections — noise, compression artifacts, and occasional annotation ambiguity. This is intentional and reflects production conditions. Your model should be robust to such noise.

---

## Test Data

- `test/images/` — 224×224 JPEG face images with UUID filenames.  
- **No labels are provided.** You must generate predictions for all images.  
- Generate your `predictions.csv` by running your model on this folder.

---

## Required Submission Format

Your `predictions.csv` must have **exactly this structure** (including header):

```
image_id,label,confidence
3f9a2c1d-...,1,0.87
a8b4e7f2-...,0,0.13
```

&nbsp;

| Column | Type | Description |
| :---- | :---- | :---- |
| `image_id` | string | Filename WITHOUT `.jpg` extension (match test set exactly) |
| `label` | int (0 or 1\) | Binary prediction: 0=Real, 1=Fake |
| `confidence` | float \[0.0–1.0\] | Probability that the image is FAKE |

> A `sample_submission.csv` is provided as a format reference (with dummy predictions). **Do not submit it as-is.**

- Every test image must have exactly one row.  
- Missing image IDs will be scored as incorrect.

---

## Required Deliverables (Google Drive Folder)

Upload to a single Google Drive folder, shared as "Anyone with the link can view":

| \# | File/Folder | Required |
| :---- | :---- | :---- |
| 1 | `predictions.csv` | Required |
| 2 | Model weights (`.pt` / `.pth` for PyTorch, `.h5` or SavedModel dir for TensorFlow) | Required |
| 3 | `inference_script.py` (see spec below) | Required |
| 4 | `requirements.txt` or `environment.yml` | Required |
| 5 | Source code (GitHub link or zipped folder with `README.md`) | Required |
| 6 | Technical Report (PDF or Markdown, min 4 pages) | Required |
| 7 | Training logs / loss curves | Optional |
| 8 | Grad-CAM / attention visualizations | Optional (+bonus) |

---

## `inference_script.py` Specification

Your inference script **must** accept the following CLI arguments:

```sh
python inference_script.py \
  --test_dir ./test/images/ \
  --weights ./model_weights.pt \
  --output predictions.csv
```

&nbsp;

```py
# Minimum required interface
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', required=True,  help='Path to test images folder')
    parser.add_argument('--weights',  required=True,  help='Path to model weight file')
    parser.add_argument('--output',   default='predictions.csv', help='Output CSV path')
    args = parser.parse_args()
    # Load model → run inference → write predictions.csv

if __name__ == '__main__':
    main()
```

> **We re-run this script on our infrastructure to verify your results. Hardcoded paths will fail.**

---

## Rules (Summary)

1. **Only the provided dataset** may be used for training (no external deepfake datasets).  
2. **Pre-trained ImageNet models** (e.g., EfficientNet, ViT, ResNet) are permitted.  
3. **Fine-tuning must be on the provided dataset only** — no deepfake-specific pre-trained detectors.  
4. **Data augmentation** on provided training images is allowed.  
5. **Test images** may not be used for training in any way (supervised or unsupervised).  
6. **One submission per team** — you may update your Drive folder before the deadline; only the latest upload is evaluated.  
7. Code plagiarism results in immediate disqualification.

---

## Evaluation Criteria

| Criterion | Weight | How It's Measured |
| :---- | :---- | :---- |
| Accuracy on Test Set | 50 pts | AUC-ROC on private held-out test labels |
| Model Efficiency | 20 pts | Inference time \+ memory on our GPU |
| Generalization & Robustness | 20 pts | AUC stability across test set variants |
| Documentation Quality | 10 pts | Manual judge rubric |

See `evaluation_metric.md` for full details.

---

## Framework Requirements

- **Python 3.10**  
- **PyTorch 2.x** or **TensorFlow 2.x** only  
- Max model size: **5 GB**  
- Inference must complete within **60 minutes** on an NVIDIA T4 GPU

---

## Tips

- Create your own validation split from the training data (e.g., 80/20). There is no public leaderboard — you will not receive intermediate scores.  
- The test set is designed to test generalization across different deepfake generation methods. A model that only learns surface-level artifacts from one method may underperform.  
- Report your AUC-ROC, F1, and threshold selection in your technical report.  
- Document your architecture, loss function, and any data augmentation choices clearly.

---

## Questions?

Contact the event PoC via the official channel. Questions about specific team approaches or intermediate scores cannot be answered.

---

*Good luck\! — Deepfake ML Hackathon Organizing Team, IIIT Bangalore*