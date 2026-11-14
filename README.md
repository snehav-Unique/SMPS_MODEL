# SMPS | Deepfake Detection

**IIIT Bangalore Deepfake ML Hackathon · Team SMPS**

A five-model ensemble classifies aligned faces as **Real (0)** or **Fake (1)**. Fine-tuning uses the official dataset only. The final CSV contains 3,000 official test image IDs; confidence is the probability of **Fake**.

## Project navigation

| Deliverable | Location |
|---|---|
| Final test predictions | [predictions.csv](predictions.csv) |
| Official inference CLI | [inference_script.py](inference_script.py) and [worker controller](ensemble_worker_control.py) |
| Trained models and ensemble configuration | [weights/](weights/) |
| Official dataset and event instructions | [datasets/](datasets/) |
| Final technical report, four pages | [final_technical_report.pdf](docs/reports/final_technical_report.pdf) |
| Preserved earlier detailed report | [report_detailed.pdf](docs/reports/report_detailed.pdf) |
| Concise technical report, two pages | [report.pdf](docs/reports/report.pdf) |
| Final measured status | [FINAL_STATUS.md](docs/FINAL_STATUS.md) |
| Offline results dashboard | [results.html](presentation/results.html) |
| Existing explanatory video | [Explanatory Video.mp4](presentation/Explanatory%20Video.mp4) |
| Submission requirements | [Submission guide](docs/SUBMISSION.md) |

Use the final technical report for submission. Earlier reports are preserved as history and contain superseded performance/score claims. The final report and final status distinguish the measured five-model results from earlier runs. See the [setup audit](docs/SETUP_AUDIT.md) for verification scope and remaining target-runtime checks.

## Repository structure

```text
SMPS/
|-- inference_script.py           Required hackathon CLI
|-- ensemble_worker_control.py    Sequential ensemble workers
|-- config.py                     Training and dataset configuration
|-- requirements.txt              Recorded final runtime dependencies
|-- predictions.csv               Final verified test predictions
|-- weights/                      Checkpoints, ensemble config and exports
|-- datasets/                     Official archive and participant instructions
|-- data/                         Saved train/validation splits
|-- src/                          Models, datasets, augmentation and training
|-- scripts/                      Preparation, evaluation and reporting tools
|-- notebooks/
|   |-- SMPS_Final_Five_Model_Verification.ipynb
|   |-- SMPS_T4_Efficiency_Benchmark.ipynb
|   `-- training/                 All original training/result notebooks
|-- outputs/
|   |-- final_verification/       Final FP32 replay and timing evidence
|   |-- final_validation/         Five-model robustness statistics
|   |-- models/                   Results grouped by architecture
|   |-- predictions/              Individual-model and earlier predictions
|   |-- gradcam/                  Explainability visualizations
|   `-- ...                       Preserved earlier runs, figures and logs
|-- docs/                         Setup, protocols, reports and handover
`-- presentation/                 Existing dashboard and video
```

**Every original tracked file is retained.** The [reorganization manifest](docs/REORGANIZATION_MANIFEST.json) maps original paths to current paths and records original SHA-256 hashes. Historical evidence, final predictions, inference sources and binary assets retain their contents. The [previous README](docs/README_BEFORE_REORGANIZATION.md) is also preserved.

## Reproduce inference

Run from the repository root. Download actual model binaries with Git LFS; pointer files cannot be used for inference.

```bash
git lfs install
git lfs pull --include="weights/*,datasets/hackathon_dataset.zip"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python inference_script.py --test_dir PATH/test/images --weights weights/ensemble_config.json --output fresh_predictions.csv
```

The CLI accepts the required `--test_dir`, `--weights`, and `--output` arguments. It defaults to FP32 and sequential model processes to limit memory use. Keep the worker controller beside the entry point and all five checkpoints beside the ensemble configuration.

Use a new Python 3.10 virtual environment. The original measured Colab environment is recorded in [environment.txt](outputs/final_verification/environment.txt), and its original pins are preserved in [requirements_original_colab.txt](docs/environments/requirements_original_colab.txt). The current requirements use Python-3.10-compatible NumPy/PyMuPDF pins. This compatibility adjustment requires fresh verification in the target environment; exact recorded GPU results are not automatically transferable to a changed runtime. Earlier training notebooks retain their architecture-specific dependencies and hyperparameters.

### Verify and evaluate

```bash
python scripts/check_delivery.py
python scripts/verify_final.py --test-dir PATH/test/images
python scripts/validate_final.py --dataset-root PATH/hackathon_dataset
python scripts/build_results.py
```

Fresh evaluation writes to `outputs/reproduction/verification/` and `outputs/reproduction/validation/`. Choose a new `--output-dir` for another run; existing completed runs are protected. `build_results.py` rebuilds the dashboard from the recorded final evidence, not fresh reproduction folders. Open `presentation/results.html` in a browser. CPU runs do not establish T4 timing.

For Colab, upload [the final verification notebook](notebooks/SMPS_Final_Five_Model_Verification.ipynb), select a Tesla T4 runtime and run its cells. It downloads official assets, generates a fresh candidate, independently replays inference and measures robustness. Its results download excludes large weights and datasets.

## Dataset and training

Extract `datasets/hackathon_dataset.zip` into `datasets/extracted/`. The default root is `datasets/extracted/hackathon_dataset`, containing `train_metadata.csv`, `train/real`, `train/fake`, and `test/images`. For another extraction layout, set the directory containing the metadata explicitly:

```powershell
$env:SMPS_DATASET_ROOT = "D:\path\to\hackathon_dataset"
python -m pip install -r requirements-training.txt
python scripts/create_split.py
python scripts/eda.py
python -m src.train
```

The generic training pipeline in `src/` is a baseline scaffold, not a one-command reproduction of every final architecture. Architecture-specific hyperparameters, training outputs and challenges remain in [training notebooks](notebooks/training/) and [model results](outputs/models/). Saved splits contain 5,600 training and 1,400 validation images. Test images must not be used for training or tuning. Training dependencies add the Albumentations 1.x API used by this scaffold.

## Final ensemble

| Model | Fixed blend weight | Recorded individual peak validation AUC |
|---|---:|---:|
| EfficientNetV2-S | 0.40 | 0.9025 |
| ConvNeXt-Tiny | 0.25 | 0.8973 |
| Swin-Tiny | 0.15 | 0.8953 |
| MaxViT-Tiny | 0.12 | 0.8902 |
| DenseNet-121 | 0.08 | 0.8905 |

Each model averages original and horizontal-flip probabilities before weighted blending. Blend weights are heuristic. Individual peaks and final validation use different samples, so these numbers do not establish an ensemble gain over the best individual model.

## Measured final results

| Check | Recorded result |
|---|---|
| Independent FP32 replay | All 3,000 labels and probabilities matched exactly |
| Tesla T4 process-wall time | 172.34 seconds |
| Peak PyTorch allocated GPU memory | 2,796.93 MiB; excludes other GPU allocations |
| Clean AUC on 200 balanced validation images | 0.8817; 95% bootstrap interval 0.8243–0.9311 |
| Accuracy on that subset at threshold 0.5 | 87.5% |
| Largest observed corruption AUC drop | 0.0110 under Gaussian noise |

See [final replay evidence](outputs/final_verification/verification.json), [validation statistics](outputs/final_validation/statistics.csv), and [final status](docs/FINAL_STATUS.md). Noise-drop uncertainty extends above 0.02. Private test labels, official scores and population robustness guarantees remain unknown. Grad-CAM is qualitative attribution rather than proof of correct reasoning.

## Hackathon submission

Supply one Google Drive folder shared as **Anyone with the link can view**, containing source or a public repository link, trained weights, official test predictions and detailed documentation. The event requests an optional 3–5 minute explanatory video; the existing video is preserved and its duration has not been revalidated. See the [submission guide](docs/SUBMISSION.md) for file locations and original event instructions.

Evaluation weights: test performance **50%**, efficiency **20%**, robustness **20%**, documentation **10%**. Fine-tune exclusively on the provided dataset, use original and reproducible code, and provide only one final team submission; only the latest upload is evaluated.
