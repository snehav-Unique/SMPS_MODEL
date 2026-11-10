# Robustness evaluation

The completed evaluation uses 200 validation images (100 real, 100 fake) and a fixed four-model ensemble. No training or weight tuning was performed during evaluation.

| Condition | Ensemble AUC | Absolute AUC drop |
|---|---:|---:|
| Clean | 0.8813 | 0.0000 |
| JPEG quality 50 | 0.8874 | -0.0061 |
| JPEG quality 70 | 0.8854 | -0.0041 |
| Gaussian blur, sigma 1.5 | 0.8740 | 0.0073 |
| Gaussian noise, sigma 15 | 0.8684 | 0.0129 |

All four observed drops are below 0.02. Confidence intervals for blur and noise extend above 0.02, so this sample does not establish that the underlying drops are below the target. These results do not establish hidden-test performance or T4 efficiency.

- [Full report and charts](outputs/robustness/REPORT.md)
- [Completed checkpoints](outputs/robustness/CHECKPOINTS.md)
- [Ensemble statistics](outputs/robustness/statistics.csv)
- [Individual model statistics](outputs/robustness/per_model_statistics.csv)
- [Verification results](outputs/robustness/verification.json)

## Local assets

The complete participant dataset archive and original trained model checkpoints are tracked with Git LFS. Run `git lfs pull` after cloning; see [dataset and weight setup](DATASET_AND_WEIGHTS.md). The exact checkpoint paths are defined in `robustness_test.py`, and their SHA-256 hashes are recorded in `outputs/robustness/run_manifest.json`.

The scripts expect the participant package beside this project directory, containing `hackathon_dataset.zip` or an extracted dataset with `train_metadata.csv`. `config.py` also contains the original author's dataset path; update it before using the original training scripts.

## Reproduce or resume

Install the Python dependencies before running. The recorded evaluation environment used Python 3.10, PyTorch 2.13.0 CPU and timm 1.0.31; the original `requirements.txt` describes the training environment. See `run_manifest.json` for evaluation library versions.

From the parent of this project directory:

```powershell
python SMPS/robustness_test.py
```

For the lower-memory DenseNet resume path:

```powershell
python SMPS/robustness_low_memory.py
python SMPS/robustness_verify.py
python SMPS/robustness_model_summary.py
```

Saved prediction batches are reused. The main runner validates its recorded run signature before reusing results. Keep completed outputs for review; use a separate project copy/output directory for a different protocol. DenseNet CPU runtime settings and the numerical consistency spot check are documented in the report.
